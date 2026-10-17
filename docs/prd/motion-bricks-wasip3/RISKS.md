# Risks

1. **wasip3 availability.** Not verified. wasi-sdk 34 may ship only wasip2. Fallback: wasip2
   component, documented as a deviation (PRD-01, PRD-03).
2. **Browser memory.** The 0.73 GB F32 model plus activations in a wasm32 heap. Measure
   peak in PRD-05; if it fails, a quantized GGUF is the next step.
3. **Download reliability.** GitHub release CDN resets during toolchain fetch. Mitigated by
   resumable fetch and checksum (PRD-01).
4. **Hugging Face runtime dependency.** The page needs HF reachable and CORS-enabled (PRD-04).
5. **jco / component tooling maturity** for WASIp3 async (PRD-03).
