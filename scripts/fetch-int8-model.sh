#!/usr/bin/env bash
# Stages the int8 model (model/manifest-int8.json) into $1/model/int8 as same-origin parts for Pages.
# Files come from the model-int8-v1 release (the browser cannot read release assets: no CORS), are checked
# against the manifest's SHA-256, and are split into 50 MB parts so every file stays under Pages' file limit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="${1:?usage: fetch-int8-model.sh <site-dir>}"
CACHE="$ROOT/.cache/model/int8"
OUT="$SITE/model/int8"
mkdir -p "$CACHE" "$OUT"

python3 - "$ROOT/model/manifest-int8.json" <<'PY' | while read -r name bytes sha base; do
import json, sys
m = json.load(open(sys.argv[1]))
for f in m["files"]:
    print(f["path"], f["bytes"], f["sha256"], m["source_base_url"])
PY
    local_path="$CACHE/$name"
    if [ ! -f "$local_path" ] || ! echo "$sha  $local_path" | sha256sum -c --status; then
        curl -fsSL --retry 5 -o "$local_path.partial" "$base$name"
        mv "$local_path.partial" "$local_path"
    fi
    echo "$sha  $local_path" | sha256sum -c --status || { echo "checksum mismatch: $name" >&2; exit 1; }
    rm -f "$OUT/$name" "$OUT/$name.part"*
    if [ "$bytes" -gt 50000000 ]; then
        split -b 50000000 -d -a 2 "$local_path" "$OUT/$name.part"
    else
        cp "$local_path" "$OUT/$name"
    fi
done
echo "staged int8 model parts in $OUT"
