#!/usr/bin/env bash
# Check out the pinned motion-bricks.cpp source and its GGML dependency. Prints the source directory.
#
# Usage: scripts/fetch-motion-bricks.sh [cache-dir]     (default: .cache under the repo)
#
# Pins (PRD-02): upstream commit 93850c4 and GGML v0.20.2 (8c63e70). Upstream's gitlink
# points GGML at 0a249dd4..., which the GitHub remote does not serve, so the revision named
# in upstream's own reference/README.md is used instead. Deviation recorded in RESULTS.md.
set -euo pipefail

UPSTREAM_URL="https://github.com/localai-org/motion-bricks.cpp"
UPSTREAM_COMMIT="93850c4f68598e7c47722e30972764dad13debde"
GGML_URL="https://github.com/ggml-org/ggml.git"
GGML_COMMIT="8c63e70982c95ceb862e3a1073a2c1beef75d60a"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${1:-$ROOT/.cache}"
SRC="$CACHE/motion-bricks.cpp"
mkdir -p "$CACHE"

log() { echo "fetch-motion-bricks: $*" >&2; }

if [ ! -d "$SRC/.git" ]; then
    # GIT_LFS_SKIP_SMUDGE: the model files are fetched from Hugging Face at runtime, not from LFS.
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$UPSTREAM_URL" "$SRC" >&2
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]; then
    log "upstream HEAD is not $UPSTREAM_COMMIT; refusing to build an unpinned source"
    exit 1
fi

GGML="$SRC/ggml"
if [ ! -f "$GGML/CMakeLists.txt" ] || [ "$(git -C "$GGML" rev-parse HEAD 2>/dev/null || true)" != "$GGML_COMMIT" ]; then
    rm -rf "$GGML"
    git clone --filter=blob:none "$GGML_URL" "$GGML" >&2
    git -C "$GGML" checkout --quiet "$GGML_COMMIT"
fi

# Exception-free WASI core patches (see RESULTS.md, PRD-02). Applied once; re-runs are no-ops.
apply_patches() {
    local repo="$1" dir="$2" patch
    for patch in "$ROOT"/patches/"$dir"/*.patch; do
        if git -C "$repo" apply --check --reverse "$patch" 2>/dev/null; then
            continue
        fi
        git -C "$repo" apply --check "$patch" || { log "patch does not apply: $patch"; exit 1; }
        git -C "$repo" apply "$patch"
        log "applied $(basename "$patch") to $dir"
    done
}
apply_patches "$SRC" motion-bricks
apply_patches "$GGML" ggml

echo "$SRC"
