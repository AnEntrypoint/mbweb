# Risks

Status as of the final run recorded in [RESULTS.md](RESULTS.md).

1. **wasip3 availability.** Resolved: wasi-sdk 34.0 ships `wasm32-wasip3`, and the component builds and
   transpiles for it. No wasip2 fallback is in use.
2. **Browser memory.** Still open. Load peaks at roughly 1.5 GB of wasm memory (staged bytes plus GGML
   tensors), estimated and not measured. Devices with little RAM may fail at `load`, which the page
   reports as an error. A quantized GGUF is the next step, but the pinned revision has none (PRD-04).
3. **Download reliability.** Mitigated for the toolchain: resumable fetch and SHA-256 check
   (`scripts/fetch-wasi-sdk.sh`). For the model, each file is verified before use, and a failed
   download is retried on the next load.
4. **Hugging Face runtime dependency.** Open. The page needs Hugging Face reachable and CORS-enabled.
   The headless run fetched the model from a page origin through the CDN redirect, so the redirected
   response passed CORS. Other networks and origins are not tested.
5. **jco and component tooling.** Partly resolved. jco 1.39 translates the async p3 component, but only
   where JSPI is available: Chrome and Edge 137 and later. Node 22 cannot run it. Firefox and Safari
   need JSPI before the page works there.
6. **Exceptions in the WASI core.** New. `jco` 1.39 cannot translate wasm exception handling, so the
   core is built without exceptions and carries source patches (`patches/`). Upstream changes will need
   these patches reapplied; the fetch script fails loudly if they no longer apply.
7. **Chromium Cache Storage limits.** New. Single entries of about 400 MB fail in headless Chromium, so
   the cache stores 64 MiB chunks. Caching is best-effort: a failed write still loads the verified bytes.
8. **Byte-at-a-time WASI streams.** New. The jco 1.39 p3 shim enqueues file reads one byte at a time,
   which made a 733 MB load take minutes. The loader now reads staged buffers instead, so no WASI
   filesystem is involved in model loading.
