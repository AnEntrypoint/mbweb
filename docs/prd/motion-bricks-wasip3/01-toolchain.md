# PRD-01: Pinned wasi-sdk toolchain

## Problem
The local wasi-sdk 34 download has failed repeatedly (connection resets, truncated archives).
Nothing can be compiled until a verified toolchain exists, and the `wasm32-wasip3` target
has not been confirmed to exist in this SDK.

## Requirements
1. Pin wasi-sdk to an exact release tag and asset name; record its SHA-256.
2. Download in CI and locally with retries and resume (`curl -C -`), and fail on a bad gzip.
3. Verify `clang --target=wasm32-wasip2` and `clang --target=wasm32-wasip3` each compile a
   trivial C file. If wasip3 is absent, record that and select the fallback in PRD-03.
4. Confirm `libclang-cpp` loads from the unpacked SDK (the earlier run failed on a missing
   shared library because the archive was truncated).

## Acceptance
- A script `scripts/fetch-wasi-sdk.sh` exits 0 and prints the SDK path.
- The target check is recorded in `docs/prd/motion-bricks-wasip3/RESULTS.md`.

## Risks
- GitHub release CDN resets; mitigated by resume and checksum.
- wasip3 may not be shipped in wasi-sdk 34; see RISKS.md.
