#!/usr/bin/env bash
# Build the motion-bricks WASIp3 component and transpile it for the browser (PRD-02, PRD-03).
#
# Usage: scripts/build-component.sh
# Output: build/component/motionbricks.wasm   (component, validated)
#         build/js/                           (jco transpile, async instantiation)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache"
BUILD="$ROOT/build"
JOBS="${JOBS:-$(nproc)}"

log() { echo "build-component: $*" >&2; }

SDK="$("$ROOT/scripts/fetch-wasi-sdk.sh" "$CACHE")"
SRC="$("$ROOT/scripts/fetch-motion-bricks.sh" "$CACHE")"
export WASI_SDK_PREFIX="$SDK"

# Core: static library only (PRD-02). Options match the PRD table.
CORE="$BUILD/core"
cmake -S "$SRC" -B "$CORE" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/wasi-p3.cmake" \
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
log "core built: $CORE/libmotionbricks.a"

# Component glue: WIT bindings generated from wit/ (wit-bindgen 0.63.0), then linked.
COMP="$BUILD/component"
mkdir -p "$COMP"
wit-bindgen c "$ROOT/wit" --world motionbricks --out-dir "$ROOT/component/gen" >&2

CLANG="$SDK/bin/clang"
CFLAGS=(--target=wasm32-wasip3 -O2 -DNDEBUG -D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_PROCESS_CLOCKS
        -I"$SRC/include" -I"$ROOT/component" -Wall -Wextra -Wno-unused-parameter)
"$CLANG" "${CFLAGS[@]}" -c "$ROOT/component/motionbricks.c" -o "$COMP/glue.o"
"$CLANG" "${CFLAGS[@]}" -c "$ROOT/component/gen/motionbricks.c" -o "$COMP/bindings.o"
"$CLANG" "${CFLAGS[@]}" -c "$ROOT/component/dl-stubs.c" -o "$COMP/dl-stubs.o"

"$SDK/bin/clang++" --target=wasm32-wasip3 -O2 -fno-exceptions -mexec-model=reactor \
    "$COMP/glue.o" "$COMP/bindings.o" "$COMP/dl-stubs.o" "$ROOT/component/gen/motionbricks_component_type.o" \
    "$CORE/libmotionbricks.a" "$CORE/ggml/src/libggml.a" "$CORE/ggml/src/libggml-cpu.a" "$CORE/ggml/src/libggml-base.a" \
    -lwasi-emulated-signal -lwasi-emulated-process-clocks \
    -o "$COMP/motionbricks.core.wasm" >&2

# The linker emits a component (wasm-component-ld); validate it as one.
wasm-tools validate --features=component-model "$COMP/motionbricks.core.wasm" >&2
mv "$COMP/motionbricks.core.wasm" "$COMP/motionbricks.wasm"
log "component: $COMP/motionbricks.wasm ($(stat -c %s "$COMP/motionbricks.wasm") bytes)"

# Transpile with the pinned jco (package.json). Async instantiation lets the page choose how
# each core module is fetched and compiled.
JS="$BUILD/js"
rm -rf "$JS"
(cd "$ROOT" && npx --no-install jco transpile "$COMP/motionbricks.wasm" -o "$JS" \
    --name motionbricks --instantiation async >&2)
log "transpiled: $JS"
