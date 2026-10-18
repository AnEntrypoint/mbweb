#!/usr/bin/env bash
# Assemble the GitHub Pages site in site/ (PRD-05, PRD-06). Requires scripts/build-component.sh.
#
# site/ holds the page, its bundled JavaScript, the transpiled component core modules and the
# model manifest. The GGUF weights are not part of the site: the page fetches them at runtime (PRD-04).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$ROOT/site"
BUILD_JS="$ROOT/build/js"

[ -f "$BUILD_JS/motionbricks.js" ] || { echo "build-site: run scripts/build-component.sh first" >&2; exit 1; }

rm -rf "$SITE"
mkdir -p "$SITE/component" "$SITE/model"

# jco's default core-module loader has a Node-only branch (node:fs/promises). The page always
# supplies its own loader, so that branch never runs in the browser and stays external.
(cd "$ROOT" && npx --no-install esbuild web/app.mjs web/worker.mjs \
    --bundle --format=esm --platform=browser --target=es2022 --minify \
    --external:node:fs/promises \
    --outdir="$SITE" --entry-names=[name])

cp "$ROOT/web/index.html" "$ROOT/web/style.css" "$SITE/"
cp "$BUILD_JS"/*.wasm "$SITE/component/"
cp "$ROOT/model/manifest.json" "$SITE/model/manifest.json"
# Ternary pose backbone: packed trits + int8 + fp16 blob with a tensor index (qat/export_web.py).
mkdir -p "$SITE/model/ternary"
cp "$ROOT/model/ternary/weights.bin" "$ROOT/model/ternary/manifest.json" "$SITE/model/ternary/"
# Pages serves files as-is; keep the empty marker so no path is filtered by Jekyll.
touch "$SITE/.nojekyll"

echo "build-site: $SITE" >&2
