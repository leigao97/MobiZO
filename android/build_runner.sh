#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
target="${1:-host}"
extra=()
case "$target" in
  host) prefix="$EXECUTORCH_ROOT/build-x86" ;;
  android)
    prefix="$EXECUTORCH_ROOT/build-android"
    extra=(-DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake"
           -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30
           -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH) ;;
  *) echo "Usage: $0 [host|android]" >&2; exit 1 ;;
esac
cmake -S "$MOBIZO_ROOT/android/runner" -B "$MOBIZO_ROOT/artifacts/build-$target" \
  -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_ROOT="$EXECUTORCH_ROOT" \
  -DCMAKE_PREFIX_PATH="$prefix" "${extra[@]}"
cmake --build "$MOBIZO_ROOT/artifacts/build-$target" -j 16
