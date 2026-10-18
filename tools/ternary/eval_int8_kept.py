"""Post-training check: store the full-precision (keep_fp) layers as int8 and re-evaluate held-out loss.

Loads a trained backbone checkpoint from train_bones_lazy.py --save_ckpt, applies per-row int8
round-trip to every kept FP layer, and compares held-out loss with the FP16 kept layers.
Run from the upstream motionbricks dir with PYTHONPATH set to it.
"""
import glob, os, sys
import torch
import pytorch_lightning as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
torch.backends.cuda.matmul.allow_tf32 = False

from train_pose import load_config  # noqa: E402
from motionbricks.helper.pl_util import load_motion_rep  # noqa: E402
from ternary_qat import set_alpha  # noqa: E402
from train_bones_lazy import LazyClips, split_indices  # noqa: E402
from train_bones_pose import build, heldout_loss  # noqa: E402

CKPT = sys.argv[1]
POSE = "/workspace/motion-bricks/ckpt/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt"

pl.seed_everything(42)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
keep_fp = tuple(ck["keep_fp"])
conf, _ = load_config("out", 3000)
mr = load_motion_rep(conf)
clips = LazyClips(sorted(glob.glob("/workspace/motion-bricks/bones/snap1/shard_*.npz")), mr)
_, held = split_indices(clips, heldout_cap=100)
held_list = [clips[i] for i in held]
view = type("V", (), {"__getitem__": lambda self, i: held_list[held.index(i)]})()

model, bb, _ = build(conf, mr, POSE, 3e-4, 300, keep_fp=keep_fp)
bb.load_state_dict(ck["backbone"], strict=True)
model = model.cuda()
trainer = pl.Trainer(max_steps=1, devices=1, accelerator="gpu", enable_progress_bar=False,
                     num_sanity_val_steps=0, enable_checkpointing=False, logger=False)
model.trainer = trainer
set_alpha(bb, 1.0)
base, n = heldout_loss(model, view, held)
print(f"kept_fp16 heldout={base:.5f} n={n}", flush=True)

kept_names = []
with torch.no_grad():
    for name, p in bb.named_parameters():
        if not any(s in name for s in keep_fp):
            continue
        if p.dim() != 2 or not (name.endswith(".weight") or name.endswith(".in_proj_weight")):
            continue
        scale = p.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        p.copy_(torch.round(p / scale).clamp(-127, 127) * scale)
        kept_names.append(name)
q8, _ = heldout_loss(model, view, held)
print(f"kept_int8 heldout={q8:.5f} n={n} int8_layers={len(kept_names)}", flush=True)
