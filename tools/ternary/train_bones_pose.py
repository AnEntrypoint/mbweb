"""Ternary QAT for the pretrained pose backbone on BONES-SEED G1 features (upstream training loop).

Data: shard_*.npz written by bones_stream.py (upstream-format 414-dim features, unnormalized).
Normalization: motion_rep.normalize (upstream stats). Clips shorter than MIN_CLIP_FRAMES are skipped.
Split: deterministic hash of clip name; ~5% held out, capped at --heldout_clips.

Held-out comparison (same masks for every eval through fixed seeds):
  fp      : backbone with alpha=0 (pretrained full precision)
  ptq     : ternary conversion with alpha=1, no training
  qat     : ternary after QAT with alpha ramp and latent clipping, alpha=1
Metric: upstream pose loss (cross-entropy over pose tokens).

Usage (from upstream motionbricks/, PYTHONPATH set to it):
    python /workspace/motion-bricks/qat/train_bones_pose.py --max_steps 2000 --lr_scale 1.0
"""
import argparse
import copy
import functools
import glob
import random
import hashlib
import json
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import open_dict
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from train_pose import load_config  # noqa: E402
from motionbricks.data.synthetic_dataset import collate_batch  # noqa: E402
from motionbricks.helper.pl_util import load_motion_rep  # noqa: E402
from ternary_qat import clip_latents, set_alpha, ternarize_linears, ternarize_in_proj, alpha_for_step  # noqa: E402
from bench_ternary_pose import TernaryScheduleCallback  # noqa: E402

MIN_CLIP_FRAMES = 26  # 6 tokens x 4 frames + 1, the shortest window the pose model samples


class ClipDataset(Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return {"keyid": i, "motion": self.clips[i].float()}


def load_clips(feats_dir, motion_rep):
    names, clips = [], []
    for path in sorted(glob.glob(os.path.join(feats_dir, "shard_*.npz"))):
        d = np.load(path)
        offsets = np.concatenate([[0], np.cumsum(d["lengths"])])
        for i, name in enumerate(d["names"]):
            n = int(d["lengths"][i])
            if n < MIN_CLIP_FRAMES:
                continue
            feat = torch.tensor(d["feats"][offsets[i]:offsets[i + 1]].astype(np.float32))
            names.append(str(name))
            clips.append(motion_rep.normalize(feat).half())  # float16 in RAM; cast per batch
    return names, clips


def split(names, heldout_fraction=0.05, heldout_cap=200):
    held, train = [], []
    for i, name in enumerate(names):
        bucket = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 1000
        (held if bucket < heldout_fraction * 1000 else train).append(i)
    return train, held[:heldout_cap]


@torch.no_grad()
def heldout_loss(model, clips, idx, seed=0, attempts=50):
    # the fused transformer fast path bypasses TernaryLinear.forward, so evaluate the real forward
    torch.backends.mha.set_fastpath_enabled(False)
    model.eval()
    losses = []
    for j, i in enumerate(idx):
        batch = collate_batch([{"keyid": i, "motion": clips[i].float()}])
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for attempt in range(attempts):
            torch.manual_seed(seed + 100000 * j + attempt)
            random.seed(seed + 100000 * j + attempt)
            np.random.seed(seed + 100000 * j + attempt)
            meta = {}
            model.inference_step(batch, 0, meta_info=meta)
            if "losses" in meta:
                losses.append(float(meta["losses"]["loss"]))
                break
    model.train()
    return float(np.mean(losses)), len(losses)


def build(conf, motion_rep, pose_ckpt, lr, warmup_steps, keep_fp=(), with_teacher=False):
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        pose_vqvae_net = instantiate(model_conf.pose_vqvae_network,
                                     motion_rep=motion_rep.dual_rep.local_motion_rep)
        backbone_net = instantiate(model_conf.backbone_network, motion_rep=motion_rep,
                                   _recursive_=False)
        state = torch.load(pose_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        backbone_state = {k[len("backbone_net."):]: v for k, v in state.items()
                          if k.startswith("backbone_net.")}
        backbone_net.load_state_dict(backbone_state, strict=True)
        teacher = copy.deepcopy(backbone_net).eval().requires_grad_(False) if with_teacher else None
        replaced = ternarize_linears(backbone_net, keep_fp=keep_fp) + ternarize_in_proj(backbone_net, keep_fp=keep_fp)
        optimizer_fn = instantiate(model_conf.optimizer)
        optimizer_fn = functools.partial(optimizer_fn.func, **{**optimizer_fn.keywords, "lr": lr})
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
        if scheduler_fn is not None:
            scheduler_fn = functools.partial(scheduler_fn.func, **{**scheduler_fn.keywords, "num_warmup_steps": warmup_steps})
        model = instantiate(model_conf, pose_vqvae_network=pose_vqvae_net,
                            root_vqvae_network=None, backbone_network=backbone_net,
                            motion_rep=motion_rep, optimizer=optimizer_fn,
                            scheduler=scheduler_fn, _recursive_=False)
    if with_teacher:
        return model, backbone_net, len(replaced), teacher
    return model, backbone_net, len(replaced)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", default="/workspace/motion-bricks/bones/feats")
    parser.add_argument("--pose_ckpt", default="/workspace/motion-bricks/ckpt/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--heldout_clips", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--summary", default=os.path.join(HERE, "logs", "bones_pose_qat.json"))
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    conf, _ = load_config("out", args.max_steps)
    motion_rep = load_motion_rep(conf)
    names, clips = load_clips(args.feats_dir, motion_rep)
    train_idx, held_idx = split(names, heldout_cap=args.heldout_clips)
    print(f"clips usable={len(clips)} train={len(train_idx)} heldout={len(held_idx)} "
          f"train_frames={sum(clips[i].shape[0] for i in train_idx)}", flush=True)

    warmup = max(1, int(args.warmup_frac * args.max_steps))
    model, backbone_net, replaced = build(conf, motion_rep, args.pose_ckpt, args.lr, warmup)
    model = model.cuda()
    trainer = pl.Trainer(max_steps=args.max_steps, devices=1, accelerator="gpu",
                         precision=conf.trainer.precision, gradient_clip_val=conf.trainer.gradient_clip_val,
                         enable_progress_bar=False, log_every_n_steps=50, num_sanity_val_steps=0,
                         enable_checkpointing=False, logger=False)
    model.trainer = trainer

    class PeriodicEval(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step % args.eval_every == 0 and step > 0:
                set_alpha(backbone_net, 1.0)  # evaluate the fully ternary model, not the ramp
                loss, n = heldout_loss(pl_module, clips, held_idx)
                set_alpha(backbone_net, alpha_for_step(step, args.max_steps))
                print(f"eval step={step} heldout={loss:.4f} n={n}", flush=True)

    set_alpha(backbone_net, 0.0)
    fp, n_fp = heldout_loss(model, clips, held_idx)
    set_alpha(backbone_net, 1.0)
    ptq, _ = heldout_loss(model, clips, held_idx)
    set_alpha(backbone_net, 0.0)
    print(f"fp={fp:.4f} (n={n_fp}) ptq={ptq:.4f}", flush=True)

    schedule = TernaryScheduleCallback(backbone_net, args.max_steps)
    trainer.callbacks.append(schedule)
    if args.eval_every:
        trainer.callbacks.append(PeriodicEval())
    loader = DataLoader(ClipDataset([clips[i] for i in train_idx]), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, collate_fn=collate_batch)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=loader)
    train_s = time.time() - t0
    model.to("cuda")
    set_alpha(backbone_net, 1.0)
    clip_latents(backbone_net, 1.0)
    qat, n_q = heldout_loss(model, clips, held_idx)
    print(f"qat={qat:.4f} (n={n_q})", flush=True)

    summary = {
        "clips_usable": len(clips), "train_clips": len(train_idx), "heldout_clips": len(held_idx),
        "replaced_linear_layers": replaced, "max_steps": args.max_steps, "batch_size": args.batch_size,
        "lr": args.lr, "warmup_steps": warmup, "heldout_loss_fp": round(fp, 5), "heldout_loss_ptq": round(ptq, 5),
        "heldout_loss_qat": round(qat, 5), "train_wall_s": round(train_s, 1),
        "mean_step_s": round(float(np.mean(schedule.step_times[1:])), 4),
        "gpu": torch.cuda.get_device_name(0), "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
