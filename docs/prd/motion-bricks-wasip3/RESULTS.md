# Results: motion-bricks WASIp3 page

Measured on 2026-10-17 in this container: 4 vCPU, 16 GB RAM, Linux, Chromium 141 (headless, from
`/opt/pw-browsers`), Node 22.22, clang 18 (host), clang 23 from wasi-sdk 34.0 (target).

Pinned inputs:

| Input | Pin |
| --- | --- |
| motion-bricks.cpp | `93850c4f68598e7c47722e30972764dad13debde` (`localai-org/motion-bricks.cpp`) |
| GGML | `8c63e70982c95ceb862e3a1073a2c1beef75d60a` (v0.20.2), see deviation 1 |
| wasi-sdk | 34.0 `wasi-sdk-34.0-x86_64-linux.tar.gz`, SHA-256 `b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4` |
| wit-bindgen | 0.63.0 (cargo) |
| wasm-tools | 1.262.0 (cargo) |
| jco | 1.39.0 (npm, exact); preview2-shim 0.30.0, preview3-shim 0.10.0 |
| Model | `LocalAI-io/MotionBricks-G1-GGML` at `cc2a47603dbc203a4f18f35dd06ed3611833f506` |

## PRD-01: toolchain

- `scripts/fetch-wasi-sdk.sh` downloads the archive (192,383,077 bytes) with resume and retries,
  checks it with `gzip -t` and the pinned SHA-256, unpacks it, and prints the SDK path.
- Target check: `clang --target=wasm32-wasip2` and `--target=wasm32-wasip3` each compile a trivial C
  file. **wasip3 is present in wasi-sdk 34.0**, so the wasip2 fallback in PRD-03 was not needed.
- `clang` resolves its shared libraries (`libclang-cpp` loads); `ldd` reports no missing library.
- The earlier 403 responses came from the release web page, not the asset, which returns a redirect.

## PRD-02: CPU core build

- `cmake --build` with the PRD option table (`cmake/wasi-p3.cmake`) builds `libmotionbricks.a`
  (347,698 bytes) and the GGML archives for wasm32-wasip3. Only `motionbricks_static` is built.
- The linked component validates with `wasm-tools validate --features=component-model`.
- Native check (`scripts/native-check.sh`, same sources, host clang): one standing-default call returns
  24 frames, 34 joints, finite values, worst quaternion norm error 1.78e-7. A zeroed pose returns
  48 frames. Both pass the 24..64, multiple-of-4 rule. Output: `RESULT: PASS`.

This is a functional check, not a reference parity check: the upstream reference fixtures are not run.

## PRD-03: component and jco transpile

- World `mb:motionbricks/inference@0.1.0` (`wit/motionbricks.wit`) exports `begin-file`,
  `write-chunk`, `load`, `joint-parents`, `neutral-positions`, `standing-features`, and `infer`.
- Output shape of `infer`: `motion { frames: u32, values: list<f32> }`, with `frames` in 24..64 (multiples
  of 4). Each frame holds 139 floats: root translation xyz (3), then the local XYZW rotations of the
  34 joints (136), frame-major.
- Component: `build/component/motionbricks.wasm`, 5,336,081 bytes, validated.
- `jco transpile --instantiation async` produces `motionbricks.js` (709,521 bytes) and the core modules
  (`motionbricks.core.wasm` 5,319,604 bytes plus small helpers). The page supplies each core module
  through its own loader.
- **JSPI is required.** Every export is async in the transpiled code, and the p3 async lowering needs
  `WebAssembly.Suspending`. Chrome and Edge 137 and later have it. Node 22 (V8 12.4) does not, even with
  `--experimental-wasm-jspi`, so the smoke test runs the built page in headless Chromium instead of Node.

## PRD-04: model delivery

- The page fetches the four GGUF files of `g1-f32/` from the pinned revision (732,632,896 bytes).
  Each file is checked against its SHA-256 from the revision's `SHA256SUMS` before the component sees
  it. A tampered `support.gguf` aborts with the visible error `checksum mismatch for g1-f32/support.gguf`.
- Verified bytes are stored in Cache Storage keyed by revision, split into 64 MiB entries (deviation 4).
- Bytes stream into the component in 4 MiB chunks (`begin-file`, `write-chunk`), then `load` reads them
  through the loader (deviation 3).

Load timings from `scripts/smoke.mjs`, headless Chromium in this container, for three runs in order.
All three use the staged-bytes loader (deviation 3). Run 1 predates the favicon fix, so its only
failed check was the console 404; run 2 and the final run pass every check:

| Run | Cold load (download, verify, stage, load) | Warm load (cache hit, verify, stage, load) |
| --- | --- | --- |
| 1 | 70.6 s | 20.6 s |
| 2 | 35.5 s | 13.5 s |
| 3, final | 53.9 s | 18.1 s |

The spread is the download speed from Hugging Face through the proxy. Warm loads still hash the
732,632,896 bytes, which is most of their time.

## PRD-05: stick-figure page

- `web/`: `index.html`, `style.css`, `app.mjs` (page), `worker.mjs` (component and inference), and
  `lib/` (model fetch, component staging, WASI imports, forward kinematics).
- Forward kinematics (`web/lib/skeleton.mjs`) composes each joint's global rotation from its parent and
  rotates the neutral bone offset into the parent frame. Checked with synthetic inputs: an identity
  chain reproduces the neutral offsets (max error 2.5e-6), and a 90 degree yaw about +Y maps +X to -Z.
- The figure is drawn in a side view (Z horizontal, Y up) at 30 FPS with a pelvis-centred camera.
- Controls: Generate motion (one inference call per click), Play/Pause, and a frame scrubber.
- Headless acceptance (`scripts/smoke.mjs`): the page reaches `ready`, generates a clip of 24 frames,
  the canvas differs between frame 0 and the last frame for two clips, and the console reports no errors.
  The figure was also checked by eye in a screenshot at frame 9 of 24.
- Degradation: without JSPI, or on any worker failure, the status line shows the error text and the
  controls stay disabled.

## PRD-06: CI and Pages

- `.github/workflows/pages.yml` runs on pushes to `main`: build the component, native check, assemble
  `site/`, smoke test in Chromium, then upload with `actions/upload-pages-artifact` and deploy with
  `actions/deploy-pages`. Every action is pinned to a commit SHA, with the tag in a comment.
- The first Pages deployment has not run: the work is on branch `claude/charming-wozniak-c6bhi0`, and the
  workflow triggers only on `main`.

## Deviations

1. **GGML revision.** `motion-bricks.cpp` records GGML `0a249dd4...` in its gitlink. That commit is not
   reachable from `ggml-org/ggml` through the GitHub remote. The revision named in upstream's own
   `reference/README.md` (`8c63e70`, v0.20.2) is used instead.
2. **No C++ exceptions in the WASI core.** `jco` 1.39 cannot translate wasm exception handling, so the
   core is built with `-fno-exceptions` (PRD-02 allows this when the sources need it). Patches in
   `patches/` replace the `try`/`throw` sites this build compiles (`src/error.hpp`, `src/neural_runtime.cpp`,
   `src/transition_debug.hpp`, `ggml/src/gguf.cpp`, `ggml/src/ggml-backend-reg.cpp`). The SONIC sources
   are excluded (`MOTIONBRICKS_ENABLE_SONIC=OFF`), as PRD-02 specifies. In the WASI build an allocation
   failure aborts instead of returning `MB_OUT_OF_MEMORY`.
3. **Weights are staged into the component, not read through a WASI filesystem.** The first attempt
   mounted the bytes as an in-memory preopen. jco's p3 shim (`readableFromPreview2Input`) enqueues each
   byte as its own chunk, and loading the 733 MB model had not finished after 8.5 minutes. The core now
   reads registered buffers (`src/memory_bundle.cpp`, a small patch to the loader), which also matches
   PRD-04's "stream into the component's memory".
4. **Cache Storage entries are chunked.** Headless Chromium rejects single entries of about 400 MB; the
   probe stored and read back 768 MB as 64 MiB entries.
5. **Style bundle not fetched.** The stateless inference API takes no style input, so the `styles/` files
   are not needed by the page. PRD-04 lists them, so this is a scope reduction.
6. **No quantized option.** The pinned revision contains only `g1-f32/`, so there is nothing to select
   (PRD-04 item 5).
7. **Smoke test runs in Chromium.** Node 22 cannot run the component (deviation 2 and PRD-03 notes), so
   `npm run smoke` drives the built page in headless Chromium through `playwright-core`.
8. **Pages deploy uses the Pages artifact.** PRD-06 names a `gh-pages` branch, but `actions/deploy-pages`
   deploys the uploaded artifact, so no branch is written.
9. **Commit attribution (PRD-06 item 3).** The workflow makes no commits, so there is nothing to attribute.
10. **Memory.** Peak wasm memory during load is roughly the staged bytes plus the GGML tensors, about
    1.5 GB. This was estimated from the sizes, not measured.

## Final run

From the last run of `scripts/build-component.sh`, `scripts/build-site.sh`, `npm run smoke`, and
`scripts/native-check.sh` in this container, on the commit that carries this file:

- Smoke: `RESULT: PASS`. Cold 53.9 s, warm 18.1 s, 24 frames, no console errors, checksum error shown
  for a tampered `support.gguf`.
- Native check: `RESULT: PASS`. Standing default 24 frames (worst quaternion norm error 1.78e-7), zeroed
  pose 48 frames.
