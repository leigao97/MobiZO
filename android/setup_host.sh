#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
command -v uv >/dev/null || { echo "Install uv before running this script" >&2; exit 1; }
mkdir -p .tools/downloads artifacts/logs
if [[ ! -x .venv-android/bin/python ]]; then uv venv --python 3.11 .venv-android; fi
uv pip install --python .venv-android/bin/python torch==2.9.0+cpu torchvision==0.24.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-android/bin/python -r android/requirements-host.txt
fetch() {
  local url="$1" archive="$2" directory="$3"
  if [[ ! -d ".tools/$directory" ]]; then
    if [[ ! -f ".tools/downloads/$archive" ]]; then
      curl -fL --retry 3 "$url" -o ".tools/downloads/$archive.part"
      mv ".tools/downloads/$archive.part" ".tools/downloads/$archive"
    fi
    python3 android/extract_sdk.py ".tools/downloads/$archive" .tools
  fi
}
fetch https://dl.google.com/android/repository/android-ndk-r26d-linux.zip android-ndk-r26d-linux.zip android-ndk-r26d
fetch https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.37.0.250724/v2.37.0.250724.zip qairt-2.37.0.250724.zip qairt/2.37.0.250724
fetch https://dl.google.com/android/repository/platform-tools-latest-linux.zip platform-tools.zip platform-tools
if [[ ! -d .tools/executorch/.git ]]; then
  git clone --branch v1.0.1 --depth 1 https://github.com/pytorch/executorch.git .tools/executorch
fi
actual="$(git -C .tools/executorch rev-parse HEAD)"
[[ "$actual" == 7b220c6f4ecc0a8b504a9c516e5643a2cf2d4376 ]] || {
  echo "Unexpected ExecuTorch revision: $actual" >&2; exit 1;
}
git -C .tools/executorch submodule update --init --recursive --depth 1 --jobs 12
# QNN host libraries need LLVM's libc++, not only GCC's libstdc++.
if [[ ! -e .tools/host-libs/usr/lib/x86_64-linux-gnu/libc++.so.1 ]]; then
  mkdir -p .tools/host-libs
  (cd .tools/downloads && apt-get download libc++1-18 libc++abi1-18 libunwind-18)
  for package in .tools/downloads/libc++1-18*.deb .tools/downloads/libc++abi1-18*.deb .tools/downloads/libunwind-18*.deb; do
    dpkg-deb -x "$package" .tools/host-libs
  done
fi
echo "Environment ready. Next: bash android/build.sh"
