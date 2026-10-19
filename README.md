# mbweb

A browser stick figure driven by [motion-bricks.cpp](https://github.com/localai-org/motion-bricks.cpp)
inference. The C++ core is built to a WASI Preview 3 component, transpiled with `jco`, and run in a
web page that is published to GitHub Pages.

The plan and acceptance criteria live in [docs/prd/motion-bricks-wasip3](docs/prd/motion-bricks-wasip3/README.md).
Measured results and deliberate deviations are in its [RESULTS.md](docs/prd/motion-bricks-wasip3/RESULTS.md).

## Layout

| Path | What it is |
| --- | --- |
| `scripts/fetch-wasi-sdk.sh` | Downloads and verifies the pinned wasi-sdk 34.0 (PRD-01). |
| `scripts/fetch-motion-bricks.sh` | Checks out the pinned upstream source and GGML and applies `patches/` (PRD-02). |
| `cmake/wasi-p3.cmake` | CMake toolchain for the WASIp3 core. |
| `patches/` | The exception-free WASI core changes, applied to the pinned upstream checkout. |
| `wit/motionbricks.wit` | The component interface (PRD-03). |
| `component/` | C glue from the WIT exports to the upstream C ABI, and the `dl*` stubs. |
| `scripts/build-component.sh` | Builds the core, links the component, and transpiles it with `jco`. |
| `web/` | The page: worker, model fetch and verification, and the stick-figure renderer. |
| `scripts/build-site.sh` | Bundles the page and assembles `site/` for Pages. |
| `scripts/smoke.mjs` | Drives the built page in headless Chromium (PRD-04, PRD-05). |
| `scripts/native-check.sh` | One native inference call through the same sources (PRD-02). |
| `model/manifest.json` | The pinned Hugging Face revision with each file's size and SHA-256. |
| `.github/workflows/pages.yml` | CI: build, smoke test, and deploy to Pages on `main` (PRD-06). |

## Build and test

Requirements: Linux x86-64, Node 22, Rust with `cargo`, Ninja, and clang. The first build downloads
about 190 MB of wasi-sdk and compiles the GGML core. The component build also needs the pinned
`wit-bindgen` 0.63.0 and `wasm-tools` 1.262.0 on `PATH`. If `CARGO_HOME` is not writable, install them
into a separate root:

```sh
cargo install wit-bindgen-cli --version 0.63.0 --locked --root "$HOME/.local/mbweb-tools"
cargo install wasm-tools --version 1.262.0 --locked --root "$HOME/.local/mbweb-tools"
export PATH="$HOME/.local/mbweb-tools/bin:$PATH"
```

The smoke test needs a Chromium on `CHROMIUM_PATH`; Playwright's bundled browser is not installed by `npm ci`.

```sh
npm ci
scripts/build-component.sh    # core, component, and jco transpile (build/)
scripts/build-site.sh         # page and site/ for Pages
npm run smoke                 # needs Chromium: CHROMIUM_PATH=... or Playwright's bundled build
scripts/native-check.sh       # optional: native build and one inference call
node --experimental-wasm-jspi scripts/diagnose-chain.mjs http://127.0.0.1:8790/component/
                              # headless: standing stability and chained-clip fidelity (serve site/ first)
```

`npm run smoke` downloads the 733 MB model from Hugging Face on a cold run and checks every file's
SHA-256 before the component sees it.

Node 22 cannot run the component: the transpiled code uses WebAssembly JS Promise Integration
(JSPI), which Chrome and Edge ship from version 137. The page needs JSPI too, so Firefox and Safari
will show an error message instead of a blank page.

## Deploy

Pages deploys from `.github/workflows/pages.yml` on pushes to `main`. In the repository settings,
set Pages to build from GitHub Actions.
