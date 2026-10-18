# Ternary quants of MotionBricks G1: HF and public uploads

## Does any ternary quant exist?

### Takeaway
No ternary (TQ1_0, TQ2_0, 1.58-bit, BitNet-style) MotionBricks G1 file was found. Every MotionBricks G1 GGUF found is F32. Ternary search hits on Hugging Face are unrelated LLMs (e.g. prism-ml/Ternary-Bonsai-2-27B-gguf).

### Cited Findings
- LocalAI-io/MotionBricks-G1-GGML, pinned revision cc2a47603dbc203a4f18f35dd06ed3611833f506, contains only `g1-f32/` (pose.gguf 546,369,984 B; root.gguf 136,504,000 B; support.gguf 5,472 B; vq-decoder.gguf 49,753,440 B), `styles/` (15 .mbstyle files plus manifest) and metadata. Source: [tree API at pinned rev](https://huggingface.co/api/models/LocalAI-io/MotionBricks-G1-GGML/tree/cc2a47603dbc203a4f18f35dd06ed3611833f506?recursive=true); the `main` listing is identical ([tree API](https://huggingface.co/api/models/LocalAI-io/MotionBricks-G1-GGML/tree/main?recursive=true)).
- Its README describes the bundle as "Native F32 GGML/GGUF conversion" with 183,148,382 F32 parameters ([README](https://huggingface.co/LocalAI-io/MotionBricks-G1-GGML/resolve/cc2a47603dbc203a4f18f35dd06ed3611833f506/README.md)). Its manifest has no quant-type field ([g1-f32/manifest.json](https://huggingface.co/LocalAI-io/MotionBricks-G1-GGML/resolve/cc2a47603dbc203a4f18f35dd06ed3611833f506/g1-f32/manifest.json)).
- LocalAI-io is the only org with a MotionBricks GGUF repo found via `author=LocalAI-io` ([API](https://huggingface.co/api/models?author=LocalAI-io)). No other LocalAI-io repos are listed.
- chibifire/MotionBricks-G1-GGML (Apache-2.0, 30 tree entries) is a second F32 conversion with the same four GGUF files and the same byte sizes ([tree API](https://huggingface.co/api/models/chibifire/MotionBricks-G1-GGML/tree/main?recursive=true), [README](https://huggingface.co/chibifire/MotionBricks-G1-GGML/resolve/main/README.md)).
- localai-org/motion-bricks.cpp `docs/FORMATS.md` describes all tensors as F32 and lists no quant type ([file](https://github.com/localai-org/motion-bricks.cpp/blob/main/docs/FORMATS.md)). Code search for ternary/TQ/bitnet/1.58 in that repo returned nothing ([repo](https://github.com/localai-org/motion-bricks.cpp)). Its issues/PRs (#1 Windows build, #2 video mention) do not mention quantization.
- NVlabs/GR00T-WholeBodyControl `motionbricks/`: code search for ternary/bitnet/TQ/1.58/quantized release returned no release-quant mention. The only "quantize" hits are VQ quantizers (`quantize_cnn_multihead.py`, `vqvae.py`), which are not weight quantization ([repo](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)). Issue titles scanned show no quant/gguf/ternary request.

### Other public MotionBricks checkpoint uploads (raw PyTorch, not quantized)
All verified via tree API. None is GGUF or quantized; sizes are FP checkpoints.
- suvadityamuk/motionbricks-ckpts: `out/G1-clip.ckpt` 7,761,041 B; `out/motionbricks_pose/.../model-step=2000000.ckpt` 1,639,126,476 B; `out/motionbricks_root/.../model-step=2000000.ckpt` 409,551,754 B; `out/motionbricks_vqvae/.../model-step=2000000.ckpt` 285,607,244 B ([tree](https://huggingface.co/api/models/suvadityamuk/motionbricks-ckpts/tree/main?recursive=true)).
- Aero-Ex/NV_MotionBricks: same four checkpoints at the root and 47 entries total ([tree](https://huggingface.co/api/models/Aero-Ex/NV_MotionBricks/tree/main?recursive=true)).
- ZeyuLing/motius-motionbricks-g1: same four checkpoints under `motionbricks_checkpoint/`, 51 entries ([tree](https://huggingface.co/api/models/ZeyuLing/motius-motionbricks-g1/tree/main?recursive=true)).
- DS yolkarian/motionbricks-checkpoints: dataset, returned by search; not inspected.

### Inferences
- The ternary request has no target: the only GGUF MotionBricks weights are F32 and the upstream release is FP checkpoints. A ternary G1 would need a new conversion from the ckpt files (the pose checkpoint alone is 1.64 GB).

### Gaps
- Search is API-based and anonymous; private repos and unindexed uploads are not visible.
- GitHub code search may miss files in LFS or unindexed branches. Not checked: NVlabs GR00T release notes or Discord.
- Repo description on GitHub for localai-org/motion-bricks.cpp says "GGML/C++" only; no quant claim found there.
