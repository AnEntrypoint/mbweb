# motion-bricks.cpp WASIp3 + GitHub Pages stick figure: PRD index

Source: https://github.com/localai-org/motion-bricks.cpp (commit 93850c4 at time of writing).

Goal: build the motion-bricks inference path to a WASI Preview 3 component, run it in a
browser page, and publish that page to GitHub Pages via GitHub Actions, where it drives a
stick figure with generated motion.

| PRD | Title | Depends on |
| --- | --- | --- |
| [PRD-01](01-toolchain.md) | Pinned wasi-sdk toolchain and target verification | none |
| [PRD-02](02-wasm-core-build.md) | CPU-only, single-thread WASI core build | PRD-01 |
| [PRD-03](03-component-and-jco.md) | Component wrapping and browser transpile | PRD-02 |
| [PRD-04](04-model-delivery.md) | Runtime model delivery from Hugging Face | PRD-02 |
| [PRD-05](05-stick-figure-page.md) | Stick-figure page rendering motion output | PRD-03, PRD-04 |
| [PRD-06](06-ci-and-pages.md) | GitHub Actions build, test and Pages deploy | PRD-03, PRD-05 |
| [PRD-07](07-oscraft-decision.md) | Decision: no shared libraries with ../oscraft | none |

Open risks are listed in each PRD and summarised in [RISKS.md](RISKS.md).

## Status

Implemented on `main`. Measurements and deviations are in
[RESULTS.md](RESULTS.md); open risks are in [RISKS.md](RISKS.md).

| PRD | Status | Where |
| --- | --- | --- |
| PRD-01 | Done. wasip3 target present in wasi-sdk 34.0 | `scripts/fetch-wasi-sdk.sh`, `cmake/wasi-p3.cmake` |
| PRD-02 | Done, with deviations 1 to 3 (exceptions, SONIC, staged loader) | `scripts/fetch-motion-bricks.sh`, `patches/`, `scripts/native-check.sh` |
| PRD-03 | Done. Async jco transpile, needs JSPI | `wit/`, `component/`, `scripts/build-component.sh` |
| PRD-04 | Done, with deviations 4 to 6 (chunked cache, styles, quantized) | `web/lib/model.mjs`, `model/manifest.json` |
| PRD-05 | Done. Verified in headless Chromium | `web/`, `scripts/smoke.mjs` |
| PRD-06 | Done. Deploys on push to `main`; run 39058280866 passed and Pages serves the page | `.github/workflows/pages.yml` |
| PRD-07 | Decision kept. No code shared with `../oscraft` | none |
