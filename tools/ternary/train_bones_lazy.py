"""Ternary QAT on the full BONES-SEED G1 feature archive with disk-backed shards.

The converted features (74 GB across shard_*.npz) do not fit in RAM, so clips are read from disk
on demand. Batches are formed inside one shard and shard order is shuffled each epoch, so every
shard is read once per pass and the one-shard cache is always hit within a batch.

Uses the same model build, held-out protocol and periodic evaluation as train_bones_pose.py.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from train_pose import load_config  # noqa: E402
from motionbricks.data.synthetic_dataset import collate_batch  # noqa: E402
from motionbricks.helper.pl_util import load_motion_rep  # noqa: E402
from ternary_qat import alpha_for_step, clip_latents, convert_to_add_form, set_alpha  # noqa: E402
from bench_ternary_pose import TernaryScheduleCallback  # noqa: E402
from train_bones_pose import build, heldout_loss, MIN_CLIP_FRAMES  # noqa: E402
from kd import wrap_model_with_kd  # noqa: E402

MAX_FRAMES = 600  # 20 s at 30 fps; longer clips exhaust GPU memory in the pose model


class LazyClips:
    """Clip access by global index; one-shard cache; normalized float32 on access."""

    def __init__(self, shard_paths, motion_rep):
        self.motion_rep = motion_rep
        self.index = []  # (shard_id, clip_in_shard, start_frame, length, name)
        self.shard_paths = shard_paths
        for sid, path in enumerate(shard_paths):
            with np.load(path) as d:
                lengths = d["lengths"]
                names = d["names"]
            offset = 0
            for j, n in enumerate(lengths):
                n = int(n)
                if n >= MIN_CLIP_FRAMES:
                    self.index.append((sid, j, offset, n, str(names[j])))
                offset += n
        self._cache_sid = None
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _shard(self, sid):
        if self._cache_sid != sid:
            with np.load(self.shard_paths[sid]) as d:
                feats = d["feats"]
                lengths = d["lengths"]
            self._cache = (feats, np.concatenate([[0], np.cumsum(lengths)]))
            self._cache_sid = sid
        return self._cache

    def name(self, i):
        return self.index[i][4]

    def __getitem__(self, i, random_window=False, rng=random):
        sid, j, _, n, _ = self.index[i]
        feats, offsets = self._shard(sid)
        start = offsets[j]
        if n > MAX_FRAMES:
            lo = rng.randrange(0, n - MAX_FRAMES + 1) if random_window else 0
            start += lo
            n = MAX_FRAMES
        feat = torch.tensor(feats[start:start + n].astype(np.float32))
        return self.motion_rep.normalize(feat)


class ShardGroupedSampler(Sampler):
    """Batches of indices from one shard at a time, shard order shuffled each epoch."""

    def __init__(self, clips, indices, batch_size, seed=0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        by_shard = {}
        for i in indices:
            by_shard.setdefault(clips.index[i][0], []).append(i)
        self.by_shard = by_shard

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shards = list(self.by_shard)
        rng.shuffle(shards)
        for sid in shards:
            members = list(self.by_shard[sid])
            rng.shuffle(members)
            for k in range(0, len(members), self.batch_size):
                yield members[k:k + self.batch_size]

    def __len__(self):
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.by_shard.values())


class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return {"keyid": i, "motion": self.clips.__getitem__(i, random_window=True)}


def split_indices(clips, heldout_fraction=0.05, heldout_cap=100):
    held, train = [], []
    for i in range(len(clips)):
        bucket = int(hashlib.sha1(clips.name(i).encode()).hexdigest(), 16) % 1000
        (held if bucket < heldout_fraction * 1000 else train).append(i)
    return train, held[:heldout_cap]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", default="/workspace/motion-bricks/bones/feats")
    parser.add_argument("--pose_ckpt", default="/workspace/motion-bricks/ckpt/out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt")
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--heldout_clips", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_fp", default="", help="comma-separated layer-name substrings left in full precision")
    parser.add_argument("--add_inference", action="store_true", help="also evaluate with addition-only ternary layers")
    parser.add_argument("--save_ckpt", default="", help="write the trained backbone state dict here")
    parser.add_argument("--kd_weight", type=float, default=0.0, help="weight of KL distillation from the FP teacher")
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    keep_fp = tuple(s for s in args.keep_fp.split(",") if s)

    pl.seed_everything(args.seed)
    conf, _ = load_config("out", args.max_steps)
    motion_rep = load_motion_rep(conf)
    shard_paths = sorted(glob.glob(os.path.join(args.feats_dir, "shard_*.npz")))
    clips = LazyClips(shard_paths, motion_rep)
    train_idx, held_idx = split_indices(clips, heldout_cap=args.heldout_clips)
    print(f"shards={len(shard_paths)} usable={len(clips)} train={len(train_idx)} heldout={len(held_idx)}", flush=True)

    warmup = max(1, int(args.warmup_frac * args.max_steps))
    if args.kd_weight > 0:
        model, backbone_net, replaced, teacher = build(conf, motion_rep, args.pose_ckpt, args.lr, warmup,
                                                       keep_fp=keep_fp, with_teacher=True)
        teacher = teacher.cuda()
        backbone_net = wrap_model_with_kd(model, teacher, args.kd_weight)
    else:
        model, backbone_net, replaced = build(conf, motion_rep, args.pose_ckpt, args.lr, warmup, keep_fp=keep_fp)
    model = model.cuda()
    trainer = pl.Trainer(max_steps=args.max_steps, devices=1, accelerator="gpu",
                         precision=conf.trainer.precision, gradient_clip_val=conf.trainer.gradient_clip_val,
                         enable_progress_bar=False, log_every_n_steps=50, num_sanity_val_steps=0,
                         enable_checkpointing=False, logger=False)
    model.trainer = trainer

    # evaluation reads held-out clips through the same lazy accessor
    held_clips = {i: clips[i] for i in held_idx}
    held_list = list(held_clips.values())
    held_map = {i: k for k, i in enumerate(held_idx)}
    class HeldView:
        def __getitem__(self, i):
            return held_list[held_map[i]]
    view = HeldView()

    set_alpha(backbone_net, 0.0)
    fp, n_fp = heldout_loss(model, view, held_idx)
    set_alpha(backbone_net, 1.0)
    ptq, _ = heldout_loss(model, view, held_idx)
    set_alpha(backbone_net, 0.0)
    print(f"fp={fp:.4f} (n={n_fp}) ptq={ptq:.4f}", flush=True)

    schedule = TernaryScheduleCallback(backbone_net, args.max_steps)
    trainer.callbacks.append(schedule)

    loader = DataLoader(LazyDataset(clips), batch_sampler=ShardGroupedSampler(clips, train_idx, args.batch_size, args.seed),
                        num_workers=0, collate_fn=collate_batch)
    t0 = time.time()
    trainer.fit(model, train_dataloaders=loader)
    train_s = time.time() - t0
    model.to("cuda")
    set_alpha(backbone_net, 1.0)
    clip_latents(backbone_net, 1.0)
    qat, n_q = heldout_loss(model, view, held_idx)
    print(f"qat={qat:.4f} (n={n_q})", flush=True)
    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
        torch.save({"backbone": backbone_net.state_dict(), "keep_fp": list(keep_fp)}, args.save_ckpt)

    add_loss = None
    if args.add_inference:
        converted = convert_to_add_form(backbone_net)
        add_loss, _ = heldout_loss(model, view, held_idx)
        print(f"add_form={add_loss:.4f} converted={converted}", flush=True)

    summary = {
        "shards": len(shard_paths), "usable_clips": len(clips), "train_clips": len(train_idx), "keep_fp": list(keep_fp), "kd_weight": args.kd_weight,
        "heldout_clips": len(held_idx), "replaced_linear_layers": replaced, "max_steps": args.max_steps,
        "batch_size": args.batch_size, "lr": args.lr, "warmup_steps": warmup,
        "heldout_loss_fp": round(fp, 5), "heldout_loss_ptq": round(ptq, 5), "heldout_loss_qat": round(qat, 5),
        "heldout_loss_add_form": None if add_loss is None else round(add_loss, 5),
        "train_wall_s": round(train_s, 1), "gpu": torch.cuda.get_device_name(0),
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    json.dump(summary, open(args.summary, "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
