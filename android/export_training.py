"""Export a MobiZO training step for portable CPU or Qualcomm HTP."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from .models.llama2 import Llama2Model, ModelArgs
from .models.training import ZOTrainingStep


def load_checkpoint(model, path):
    """Load Hugging Face TinyLlama weights, converting split-half RoPE layout."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        weights = load_file(str(path))
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)
    args = model.model.params
    for name in list(weights):
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = args.n_heads if name.endswith("q_proj.weight") else args.n_kv_heads
            heads = heads or args.n_heads
            w = weights[name]
            weights[name] = w.reshape(heads, 2, w.shape[0] // heads // 2, w.shape[1]).transpose(1, 2).reshape(w.shape)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [key for key in missing if not key.endswith(".mask")]
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def build_model(params, seq_len, rank, alpha, checkpoint=None, seed=42):
    torch.manual_seed(seed)
    model = Llama2Model(ModelArgs(**params, max_seq_len=seq_len)).eval()
    if checkpoint:
        load_checkpoint(model, checkpoint)
    model.enable_lora(rank, alpha)
    return ZOTrainingStep(model).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("portable", "qnn"), default="qnn")
    parser.add_argument("--params", type=Path, default=Path(__file__).parent / "models/llama2_1b_config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soc", default="SM8650")
    parser.add_argument("--qnn-shards", type=int, default=1,
                        help="Split decoder layers across HTP contexts for large shapes")
    parser.add_argument("--batch-size", type=int, default=1, help="Effective E; physical transformer batch is 2E")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (min(args.batch_size, args.rank) <= 0 or args.seq_len < 2 or args.eps <= 0
            or not all(math.isfinite(v) for v in (args.eps, args.learning_rate, args.alpha))):
        parser.error("Positive batch/rank/eps, sequence length >= 2, and finite optimizer parameters required")
    params = json.loads(args.params.read_text())
    if not 1 <= args.qnn_shards <= params["n_layers"]:
        parser.error("QNN shard count must be between one and the number of decoder layers")
    if args.backend != "qnn" and args.qnn_shards != 1:
        parser.error("--qnn-shards requires --backend qnn")
    args.output.mkdir(parents=True, exist_ok=True)
    step = build_model(params, args.seq_len, args.rank, args.alpha, args.checkpoint, args.seed)
    n = step.model.adapter_numel
    tokens = torch.randint(0, params["vocab_size"], (args.batch_size, args.seq_len), dtype=torch.int32)
    labels = tokens.clone()
    state = torch.zeros(n)
    noise = torch.randn(n)
    lr, eps = torch.tensor([args.learning_rate]), torch.tensor([args.eps])
    inputs = (tokens, labels, state, noise, lr, eps)
    with torch.no_grad():
        if args.backend == "qnn":
            from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
            from executorch.backends.qualcomm.utils.utils import (
                generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec,
            )
            specs = generate_qnn_executorch_compiler_spec(
                soc_model=getattr(QcomChipset, args.soc),
                backend_options=generate_htp_compiler_spec(
                    use_fp16=True, use_multi_contexts=args.qnn_shards > 1),
            )
            from .qnn_training import lower_training
            edge = lower_training(step, inputs, specs, args.qnn_shards)
        else:
            from executorch.exir import to_edge
            edge = to_edge(torch.export.export(step, inputs, strict=True))
        from executorch.exir import ExecutorchBackendConfig
        from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
        program = edge.to_executorch(config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)))
        pte = args.output / "training.pte"
        with pte.open("wb") as f:
            program.write_to_file(f)
        # Example input batch; replace these files with tokenized training data.
        tokens.numpy().tofile(args.output / "tokens.bin")
        labels.numpy().tofile(args.output / "labels.bin")
        state.numpy().tofile(args.output / "initial_state.bin")
    delegates = sum(len(p.delegates) for p in program.executorch_program.execution_plan)
    if args.backend == "qnn" and not delegates:
        raise RuntimeError("QNN export produced no HTP delegates")
    if args.backend == "qnn" and delegates < args.qnn_shards:
        raise RuntimeError(f"Expected at least {args.qnn_shards} HTP contexts; got {delegates}")
    with pte.open("rb") as f:
        pte_hash = hashlib.file_digest(f, "sha256").hexdigest()
    meta = dict(backend=args.backend, soc=args.soc, qnn_shards=args.qnn_shards,
                params=params, rank=args.rank, alpha=args.alpha,
                effective_batch=args.batch_size, physical_batch=2*args.batch_size, seq_len=args.seq_len,
                adapter_numel=n, adapter_layout=step.model.adapter_layout,
                learning_rate=args.learning_rate, eps=args.eps, seed=args.seed,
                checkpoint=str(args.checkpoint),
                torch=torch.__version__, executorch=importlib.metadata.version("executorch"),
                delegates=delegates, optimizer_precision="float32 portable CPU",
                state_abi="explicit FP32 master B input / updated B output",
                pte_sha256=pte_hash)
    (args.output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("backend", "delegates", "adapter_numel", "physical_batch", "pte_sha256")}), flush=True)


if __name__ == "__main__":
    main()
