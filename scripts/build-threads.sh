#!/usr/bin/env bash
# Build the threaded motion-bricks core as a plain wasm module (wasm32-wasip1-threads) for the page's own JS host.
# Requires scripts/build-component.sh's fetched sources (.cache/motion-bricks.cpp, wasi-sdk) and the core build in
# build/threads/core. Output: build/threads/motionbricks-threads.wasm
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache"
SDK="$CACHE/wasi-sdk-34.0-x86_64-linux"
SRC="$CACHE/motion-bricks.cpp"
CORE="$ROOT/build/threads/core"
OUT="$ROOT/build/threads"
JOBS="${JOBS:-$(nproc)}"

[ -d "$SDK" ] || { echo "build-threads: run scripts/build-component.sh first (wasi-sdk missing)" >&2; exit 1; }

export WASI_SDK_PREFIX="$SDK"
cmake -S "$SRC" -B "$CORE" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/wasi-threads.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DBUILD_SHARED_LIBS=OFF \
    -DMOTIONBRICKS_ENABLE_GGML=ON \
    -DMOTIONBRICKS_ENABLE_VULKAN=OFF \
    -DMOTIONBRICKS_ENABLE_LLAMAFILE=OFF \
    -DMOTIONBRICKS_CPU_ALL_VARIANTS=OFF \
    -DMOTIONBRICKS_ENABLE_PHYSICS=OFF \
    -DMOTIONBRICKS_ENABLE_SONIC=OFF \
    -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF \
    -DMOTIONBRICKS_BUILD_TESTS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DGGML_NATIVE=OFF >&2
cmake --build "$CORE" --target motionbricks_static -j "$JOBS" >&2

mkdir -p "$OUT"
CLANG="$SDK/bin/clang"
CFLAGS=(--target=wasm32-wasip1-threads -O2 -msimd128 -matomics -mbulk-memory -pthread -DNDEBUG
        -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS
        -I"$SRC/include" -I"$SRC/ggml/include" -Wall -Wextra -Wno-unused-parameter)
"$CLANG" "${CFLAGS[@]}" -c "$ROOT/component/threads/glue.c" -o "$OUT/glue.o"
"$CLANG" "${CFLAGS[@]}" -c "$ROOT/component/dl-stubs.c" -o "$OUT/dl-stubs.o"

# The exports are the glue's mb_* functions plus the thread entry point the host calls on each new worker.
"$SDK/bin/clang++" --target=wasm32-wasip1-threads -O2 -fno-exceptions -pthread -matomics -mbulk-memory \
    -mexec-model=reactor -Wl,--import-memory -Wl,--shared-memory -Wl,--export-memory \
    -Wl,--max-memory=2147483648 -Wl,--wrap=ggml_backend_graph_compute \
    -Wl,--export-if-defined=mb_alloc -Wl,--export-if-defined=mb_free -Wl,--export-if-defined=mb_error \
    -Wl,--export-if-defined=mb_stage_begin -Wl,--export-if-defined=mb_stage_write -Wl,--export-if-defined=mb_load \
    -Wl,--export-if-defined=mb_joint_parents -Wl,--export-if-defined=mb_neutral_positions \
    -Wl,--export-if-defined=mb_standing -Wl,--export-if-defined=mb_infer -Wl,--export-if-defined=mb_profile \
    -Wl,--export-if-defined=mb_style_speed -Wl,--export-if-defined=mb_agent_new -Wl,--export-if-defined=mb_agent_reset_style \
    -Wl,--export-if-defined=mb_agent_context -Wl,--export-if-defined=mb_command_new -Wl,--export-if-defined=mb_command_free_handle \
    -Wl,--export-if-defined=mb_agent_free_handle -Wl,--export-if-defined=mb_style_free_handle \
    -Wl,--export-if-defined=wasi_thread_start \
    "$OUT/glue.o" "$OUT/dl-stubs.o" "$CORE/libmotionbricks.a" "$CORE/ggml/src/libggml.a" "$CORE/ggml/src/libggml-cpu.a" \
    "$CORE/ggml/src/libggml-base.a" \
    -lwasi-emulated-signal -lwasi-emulated-process-clocks \
    -o "$OUT/motionbricks-threads.wasm" >&2
echo "build-threads: $OUT/motionbricks-threads.wasm" >&2
