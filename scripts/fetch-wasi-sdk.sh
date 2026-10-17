#!/usr/bin/env bash
# Fetch, verify and unpack the pinned wasi-sdk (PRD-01). Prints the SDK directory on success.
#
# Usage: scripts/fetch-wasi-sdk.sh [cache-dir]     (default: .cache under the repo)
# Env:   WASI_SDK_SHA256  overrides the pinned archive digest (for a new release).
set -euo pipefail

VERSION=34.0
ARCHIVE="wasi-sdk-${VERSION}-x86_64-linux.tar.gz"
URL="https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/${ARCHIVE}"
# SHA-256 of the archive as downloaded and checked on 2026-10-17 (upstream publishes no .sha256 asset).
PINNED_SHA256="b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4"
EXPECTED_SHA256="${WASI_SDK_SHA256:-$PINNED_SHA256}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${1:-$ROOT/.cache}"
SDK_DIR="$CACHE/wasi-sdk-${VERSION}-x86_64-linux"
mkdir -p "$CACHE"

log() { echo "fetch-wasi-sdk: $*" >&2; }

if [ ! -x "$SDK_DIR/bin/clang" ]; then
    archive="$CACHE/$ARCHIVE"
    for attempt in 1 2 3 4 5; do
        # -C - resumes a partial archive; --retry covers transient resets inside one attempt.
        if curl -fsSL -C - --retry 3 --retry-delay 2 -o "$archive" "$URL"; then
            break
        fi
        log "download attempt $attempt failed"
        [ "$attempt" -eq 5 ] && { log "giving up on $URL"; exit 1; }
        sleep $((2 ** attempt))
    done

    gzip -t "$archive" || { log "bad gzip: removing $archive"; rm -f "$archive"; exit 1; }

    actual="$(sha256sum "$archive" | cut -d' ' -f1)"
    if [ "$actual" != "$EXPECTED_SHA256" ]; then
        log "sha256 mismatch: expected $EXPECTED_SHA256, got $actual"
        rm -f "$archive"
        exit 1
    fi

    tar -xzf "$archive" -C "$CACHE"
fi

# Each target must compile a trivial object (PRD-01 requirement 3).
probe="$(mktemp -d)"
trap 'rm -rf "$probe"' EXIT
printf 'int mb_probe(int a, int b) { return a + b; }\n' > "$probe/probe.c"
for target in wasip2 wasip3; do
    "$SDK_DIR/bin/clang" --target="wasm32-$target" -c "$probe/probe.c" -o "$probe/probe-$target.o"
    log "wasm32-$target compiles"
done

# libclang-cpp must load from the unpacked SDK (PRD-01 requirement 4).
if ! ldd "$SDK_DIR/bin/clang" | grep -q "not found"; then
    log "clang shared libraries resolve"
else
    log "missing shared library for clang"; exit 1
fi

echo "$SDK_DIR"
