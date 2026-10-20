#!/usr/bin/env bash
# PRD-02 acceptance: one G1 inference call through the native build of the same pinned sources.
# Usage: scripts/native-check.sh     (downloads the verified bundle into .cache/model on first run)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache"
BUILD="$ROOT/build/native"
JOBS="${JOBS:-$(nproc)}"

SRC="$("$ROOT/scripts/fetch-motion-bricks.sh" "$CACHE")"

cmake -S "$SRC" -B "$BUILD" -G Ninja \
    -DCMAKE_C_COMPILER="${CC:-clang}" -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DMOTIONBRICKS_ENABLE_GGML=ON -DMOTIONBRICKS_ENABLE_VULKAN=OFF \
    -DMOTIONBRICKS_ENABLE_LLAMAFILE=OFF -DMOTIONBRICKS_CPU_ALL_VARIANTS=OFF \
    -DMOTIONBRICKS_ENABLE_PHYSICS=OFF -DMOTIONBRICKS_ENABLE_SONIC=OFF \
    -DMOTIONBRICKS_DOWNLOAD_MODELS=OFF -DMOTIONBRICKS_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DGGML_OPENMP=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF >&2
cmake --build "$BUILD" --target motionbricks_static -j "$JOBS" >&2

"${CC:-clang}" -std=c11 -O2 -I"$SRC/include" "$ROOT/tests/native/standing.c" -o "$BUILD/standing" \
    "$BUILD/libmotionbricks.a" "$BUILD/ggml/src/libggml.a" "$BUILD/ggml/src/libggml-cpu.a" \
    "$BUILD/ggml/src/libggml-base.a" -lstdc++ -lm -lpthread -ldl

# Fetch the bundle files named by model/manifest.json and verify each one (same contract as the page).
while read -r path bytes sha base; do
    local_path="$CACHE/model/$path"
    mkdir -p "$(dirname "$local_path")"
    if [ ! -f "$local_path" ] || ! echo "$sha  $local_path" | sha256sum -c --status; then
        curl -fsSL --http1.1 --retry 10 --retry-all-errors -C - -o "$local_path.partial" "$base$path"
        mv "$local_path.partial" "$local_path"
    fi
    echo "$sha  $local_path" | sha256sum -c --status || { echo "checksum mismatch: $path" >&2; exit 1; }
done < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    print(f["path"], f["bytes"], f["sha256"], m["base_url"])
' "$ROOT/model/manifest.json")

"$BUILD/standing" "$CACHE/model/g1-f32"

# Ternary bundle (model/ternary): the same standing check on our GGUF files, so a native abort
# prints its ggml message instead of the WASM trap.
TERN="$ROOT/build/ternary-bundle"
rm -rf "$TERN" && mkdir -p "$TERN"
cp "$ROOT/model/ternary/pose.gguf" "$ROOT/model/ternary/vq-decoder.gguf" "$ROOT/model/ternary/support.gguf" "$TERN/"
cp "$ROOT/model/ternary/root.gguf" "$TERN/root.gguf"
"$BUILD/standing" "$TERN" >&2
echo "native ternary standing: PASS" >&2
