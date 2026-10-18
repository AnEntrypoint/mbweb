# Ternary pose backbone tools

- `ternary_qat.py`: absmean ternary quantizer, straight-through TernaryLinear and TernaryParam (MHA in_proj), addition-only inference form.
- `ternary_pack.py`: 5 trits per byte packing (1.6 bits per weight).
- `train_bones_pose.py`, `train_bones_lazy.py`: QAT on BONES-SEED G1 features with optional KD (`--kd_weight`) and `--save_ckpt`.
- `kd.py`: KL distillation wrapper from the full-precision backbone.
- `export_web.py`: writes `weights.bin` + `manifest.json` (ternary, int8, fp16 kinds; scales as fp16 in the blob).
- `eval_int8_kept.py`: post-training int8 check of kept layers.

Run from the upstream motionbricks checkout with PYTHONPATH set to it. Training data (BONES-SEED) is not in this repo.
