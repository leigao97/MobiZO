# Android MobiZO training

Tested with TinyLlama on OnePlus 12 (Snapdragon 8 Gen 3), ExecuTorch 1.0.1,
QNN 2.37.0.250724, and Android NDK r26d.

The base model and LoRA A stay frozen. LoRA B is a runtime input/output:
the transformer and LoRA forward run on HTP in FP16; cross-entropy, the
two-sided ZO gradient estimate, and the B update run on CPU in FP32.
The runner generates noise on the phone and saves B in a separate checkpoint.

## Build

On Linux x86_64, install `uv`, Git, GCC, make, and curl. Run from the repository
root (the setup script uses Ubuntu apt packages for local LLVM runtime libraries):

```bash
bash android/setup_host.sh
source android/env.sh
bash android/build.sh
```

## Export

Download `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from Hugging Face into
`.tools/models/TinyLlama-1.1B-Chat-v1.0` (including `model.safetensors`).

```bash
python -m android.export_training \
  --checkpoint .tools/models/TinyLlama-1.1B-Chat-v1.0 \
  --seq-len 128 --batch-size 1 --rank 16 --alpha 32 \
  --output artifacts/tinyllama-qnn
```

`--batch-size` is effective batch E; the positive/negative branches use 2E sequences.
`--qnn-shards` defaults to 1. For large batches, use 2 (S=64/E=16 or S=128/E=8)
or 4 (S=128/E=16) to fit QNN context memory limits.

The bundle includes a synthetic example batch. Replace `tokens.bin` and
`labels.bin` with row-major int32 arrays of shape `[E, S]` for your data.
Labels use unshifted token positions; `-100` masks ignored targets.
This basic runner repeats the supplied batch; dataset iteration is not included.

## Run and resume

Connect the phone with USB debugging enabled, then copy the bundle and runtime:

```bash
adb shell mkdir -p /data/local/tmp/mobizo
adb push artifacts/tinyllama-qnn artifacts/build-android/mobizo_train \
  "$EXECUTORCH_ROOT/build-android/lib/libqnn_executorch_backend.so" \
  "$QNN_SDK_ROOT"/lib/aarch64-android/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV75Stub.so,libQnnHtpPrepare.so,libQnnModelDlc.so} \
  "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so" /data/local/tmp/mobizo/
```

Inside `adb shell`:

```sh
cd /data/local/tmp/mobizo
chmod +x mobizo_train
export LD_LIBRARY_PATH="$PWD:/vendor/lib64"
export ADSP_LIBRARY_PATH="$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp"
./mobizo_train --bundle ./tinyllama-qnn --steps 10 --checkpoint-out adapter.ckpt
./mobizo_train --bundle ./tinyllama-qnn --steps 10 \
  --checkpoint-in adapter.ckpt --checkpoint-out adapter.ckpt
```

Keep the same PTE and manifest when resuming. Updated weights are saved in
`adapter.ckpt`, not written back into the PTE. Each step logs losses, B's change
norm, and latency. Optional `--warmup N` excludes N initial steps from the timing
average; those steps still update B.
