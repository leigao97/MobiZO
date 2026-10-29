#!/usr/bin/env bash
MOBIZO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MOBIZO_ROOT
export EXECUTORCH_ROOT="$MOBIZO_ROOT/.tools/executorch"
export QNN_SDK_ROOT="$MOBIZO_ROOT/.tools/qairt/2.37.0.250724"
export ANDROID_NDK_ROOT="$MOBIZO_ROOT/.tools/android-ndk-r26d"
export PATH="$MOBIZO_ROOT/.venv-android/bin:$MOBIZO_ROOT/.tools/platform-tools:$PATH"
export LD_LIBRARY_PATH="$MOBIZO_ROOT/.tools/host-libs/usr/lib/x86_64-linux-gnu:$QNN_SDK_ROOT/lib/x86_64-linux-clang:$EXECUTORCH_ROOT/build-x86/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHON_EXECUTABLE="$MOBIZO_ROOT/.venv-android/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
