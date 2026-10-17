# PRD-04: Runtime model delivery from Hugging Face

## Problem
The F32 G1 bundle is about 0.73 GB (183,148,382 parameters). GitHub Pages rejects files over
100 MB and sites over about 1 GB, so the model cannot be deployed as a static asset.

## Requirements
1. The page fetches the GGUF bundle and style bundle at runtime from the pinned Hugging Face
   revision `cc2a47603dbc203a4f18f35dd06ed3611833f506` in `LocalAI-io/MotionBricks-G1-GGML`.
2. Verify SHA-256 of each file in the browser before handing bytes to the component.
3. Stream with `fetch` into the component's memory; show progress in the page.
4. Cache in `Cache Storage` keyed by revision, so repeat visits skip the download.
5. Offer a smaller option: if a quantized GGUF exists in the same repo, allow selecting it
   with a query flag. Do not produce new quantizations in this PRD.

## Acceptance
- Cold load and warm load timings are recorded in RESULTS.md.
- A checksum mismatch aborts loading with a visible error.

## Risks
- Hugging Face CORS and rate limits; confirm headers before relying on them.
- 0.73 GB in a browser tab is a large memory footprint; see RISKS.md.
