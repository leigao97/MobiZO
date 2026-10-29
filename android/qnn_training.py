"""HTP model compute with FP32 loss and optimizer arithmetic on portable CPU."""
import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import QnnPassManager
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.utils.utils import qnn_edge_config
from executorch.exir import to_edge_transform_and_lower


class TrainingPartitioner(QnnPartitioner):
    def __init__(self, specs, boundary_layers=()):
        super().__init__(specs)
        self.boundary_paths = {f"model.model.layers.{i}" for i in boundary_layers}

    def generate_partitions(self, edge_program):
        boundaries = {}
        # Keep loss and B updates in FP32 after QNN graph passes.
        for node in edge_program.graph.nodes:
            stack = node.meta.get("nn_module_stack", {})
            # CPU residual adds separate the HTP contexts.
            paths = [path for path, _ in stack.values()]
            if (node.op == "call_function" and paths
                    and paths[-1] in self.boundary_paths
                    and getattr(node.target, "_op", node.target) == torch.ops.aten.add.Tensor):
                boundaries[paths[-1]] = node.name
            in_model = any(path == "model" or path.startswith("model.")
                           for path, _ in stack.values())
            if node.op == "call_function" and not in_model:
                self.skip_node_id_set.add(node.name)
            # Slice B/noise on CPU: HTP InputSlicePad exceeds v75 TCM for full vectors.
            if (node.op == "call_function" and "slice" in str(node.target)
                    and node.args and isinstance(node.args[0], torch.fx.Node)
                    and node.args[0].op == "placeholder"
                    and node.args[0].name in {"state", "noise"}):
                self.skip_node_id_set.add(node.name)
        if boundaries.keys() != self.boundary_paths:
            raise RuntimeError(f"Missing QNN shard boundaries: {self.boundary_paths - boundaries.keys()}")
        self.skip_node_id_set.update(boundaries.values())
        return super().generate_partitions(edge_program)


def lower_training(step, inputs, specs, num_shards=1):
    layers = step.model.model.params.n_layers
    if not 1 <= num_shards <= layers:
        raise ValueError("QNN shard count must be between one and the number of decoder layers")
    boundaries = {layers * i // num_shards - 1 for i in range(1, num_shards)}
    if num_shards > 1:
        # Avoid ExecuTorch 1.0.1 double-deleting shared nonpersistent constants.
        transformer = step.model.model
        for name in ("freqs_cos", "freqs_sin"):
            transformer.register_buffer(name, getattr(transformer, name), persistent=True)
    ep = torch.export.export(step, inputs, strict=True)
    transformed = QnnPassManager().transform_for_export_pipeline(ep)
    edge = to_edge_transform_and_lower(
        transformed,
        transform_passes=QnnPassManager().get_to_edge_transform_passes(ep),
        partitioner=[TrainingPartitioner(specs, boundaries)],
        compile_config=qnn_edge_config(),
    )
    if num_shards > 1:
        from executorch.backends.qualcomm.utils.utils import update_spill_fill_size
        update_spill_fill_size(edge.exported_program())
    return edge
