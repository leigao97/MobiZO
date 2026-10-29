#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$EXECUTORCH_ROOT"
args=(--release --job_number "${MOBIZO_BUILD_JOBS:-24}")
# Upstream --no_clean assumes both flatcc build directories already exist.
if [[ -d build-x86/third-party/flatcc && -d build-android/third-party/flatcc ]]; then
  args+=(--no_clean)
fi
bash backends/qualcomm/scripts/build.sh "${args[@]}"
cd "$MOBIZO_ROOT"
python - <<'INSTALL'
from pathlib import Path
import os
import shutil
import executorch.backends.qualcomm.python as qnn_python
for source in (Path(os.environ['EXECUTORCH_ROOT']) / 'build-x86/backends/qualcomm').glob('Py*.so'):
    shutil.copy2(source, Path(next(iter(qnn_python.__path__))) / source.name)
from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec
print('QNN AOT bindings ready')
INSTALL
bash android/build_runner.sh host
bash android/build_runner.sh android
