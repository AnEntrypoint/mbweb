# PRD-03: Component wrapping and browser transpile

## Problem
A plain core wasm module with the C ABI is not callable from JavaScript with typed values.
WASIp3 is a component-model target, so the module must be wrapped as a component, then
transpiled to JS for the page.

## Requirements
1. Define a WIT world `motionbricks` exposing the inference calls from `inference.h`:
   request create/set features/set mask/set seed, and `infer` returning a flat float array of
   frames x joints x transform values. Use `list<f32>` / `list<u32>` only; no handles leaking
   to JS beyond an opaque resource.
2. Wrap with `wasm-tools component new` (or componentize-js if the C path cannot express
   async). Use the WASIp3 adapter if available (PRD-01); otherwise WASIp2 and mark that as a
   deviation.
3. Transpile with `jco transpile` (pinned version, currently 1.39.0 on npm) into an ES module
   plus `.wasm` core files for the page.
4. Keep `mb_model_infer` synchronous inside the component; expose async only at the JS boundary.

## Acceptance
- `node scripts/smoke.mjs` loads the transpiled component under Node and runs one inference
  call with a zeroed pose, returning the expected frame count (multiple of 4, 24..64).
- Output shape is documented in `docs/prd/motion-bricks-wasip3/RESULTS.md`.

## Risks
- jco support for WASIp3 async is unverified; fallback is a WASIp2 component with a synchronous
  call, which still runs in-page.
- Component wrapping may require `wit-bindgen` for the C side; pin its version too.
