# Published ternary GGUF models

## Which repos ship TQ1_0 / TQ2_0

### Takeaway
Mainline ggml has TQ1_0/TQ2_0, but most public ternary GGUFs use Prism ML's Q2_0 fork format. No ternary GGUF of a non-LLM motion/regression model was found.

### Cited Findings
- TQ1_0 examples: `GeorgyGUF/MAI-DS-R1-tq1_0.gguf` (3 shards, ~143 GB total), `BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked` — [HF search](https://huggingface.co/api/models?search=TQ1_0)
- `ewchampion/Ternary-Bonsai-8B-TQ2_0-GGUF`: `Ternary-Bonsai-8B-TQ2_0.gguf`, 2,656,942,336 B — [tree API](https://huggingface.co/api/models/ewchampion/Ternary-Bonsai-8B-TQ2_0-GGUF/tree/main)
- `prism-ml/Ternary-Bonsai-8B-gguf` (~370k downloads): Q2_0 2,182,184,672 B; Q2_0_g64 2,310,125,920 B; F16 16,383,663,200 B — [tree API](https://huggingface.co/api/models/prism-ml/Ternary-Bonsai-8B-gguf/tree/main)
- Its README: Q2_0 "is not yet in mainline `llama.cpp`. Use our fork" — [README](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/raw/main/README.md)

### Gaps
- Uploaders and sizes checked only for the repos above.

## Conversion tooling

### Cited Findings
- llama.cpp PR #8151 (compilade, merged 2024-09-06) added TQ1_0 (1.6875 bpw) and TQ2_0 (2.0625 bpw); CPU-only at merge — [PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151)
- Commands: `convert_hf_to_gguf.py --outtype tq1_0|tq2_0`, then `llama-quantize` for Q4_K/Q6_K on embeddings/output — [PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151)
- MAI-DS-R1 TQ1_0 was made with `llama-quantize <bf16> <out> TQ1_0` on llama.cpp b5202 — [README](https://huggingface.co/GeorgyGUF/MAI-DS-R1-tq1_0.gguf/raw/main/README.md)

### Gaps
- Whether `llama-quantize` handles non-LLM tensor layouts was not verified from source.

## Non-LLM ternary: parity and quality

### Cited Findings
- `biali/bonsai-image-ternary-4B-FLUX2-klein-GGUF` (FLUX.2 Klein diffusion, not an LLM) ships a Q2_K file (1,361,094,080 B) and says tq1_0/tq2_0 "produce degraded output"; built with stable-diffusion.cpp `sd -M convert --type q2_K` — [tree API](https://huggingface.co/api/models/biali/bonsai-image-ternary-4B-FLUX2-klein-GGUF/tree/main), [README](https://huggingface.co/biali/bonsai-image-ternary-4B-FLUX2-klein-GGUF/raw/main/README.md). No metrics given.
- TerDiT (QAT ternary DiTs, 600M–4.2B): abstract gives no FID numbers — [arXiv 2405.14854](https://arxiv.org/abs/2405.14854)
- Low-bit diffusion errors accumulate across denoising steps — [NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/983591c3e9a0dc94a99134b3238bbe52-Paper-Conference.pdf)

### Inferences
- The one public non-LLM ternary GGUF avoided the native ternary types for quality reasons. For MotionBricks G1, expect to keep sensitive layers at higher precision and gate on a parity check.

### Gaps
- No published ternary quality measurement for a motion or regression network was found.
