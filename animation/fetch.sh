#!/usr/bin/env bash
# Fetches the pinned kimodo.cpp, its verified G1 weights and text encoder, and builds the CPU tools.
# Everything lands under .kimodo/ (git-ignored). Needs git, cmake, ninja, a C++ compiler, and the hf CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
. animation/pin.env

work=.kimodo
src="$work/src"
weights="$work/weights"

if [ ! -d "$src/.git" ]; then
    git clone --no-checkout "$KIMODO_REPO" "$src"
fi
git -C "$src" fetch --depth 1 origin "$KIMODO_COMMIT"
git -C "$src" checkout --quiet --detach "$KIMODO_COMMIT"
git -C "$src" submodule update --init --depth 1 --recursive

# Each file is checked against its repository manifest (size and SHA-256) and resumes if interrupted.
node animation/fetch-weights.mjs

# Vulkan is off so the tools build on any machine; the CPU path is the one the pipeline depends on.
cmake -S "$src" -B "$src/build/release" -G Ninja -DCMAKE_BUILD_TYPE=Release -DKIMODO_ENABLE_VULKAN=OFF
cmake --build "$src/build/release" --target kmd-generate -j"$(nproc 2>/dev/null || echo 4)"

echo "ready: $src/build/release/kmd-generate $weights"
