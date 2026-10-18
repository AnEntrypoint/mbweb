"""Root model builder for ternary QAT (motionbricks_root, no VQ-VAE: its loss is root regression).

Mirrors train_bones_pose.build for the root MotionModel: loads the pretrained root backbone,
ternarizes its linears and MHA input projections, and returns the Lightning model.
"""
import copy
import functools
import os
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ternary_qat import ternarize_in_proj, ternarize_linears  # noqa: E402

ROOT_KEEP_FP = ("_conv_output", "_proj_num_token_output_logit", "_proj_start_input", "_proj_end_input", "layers.0.")


def load_root_config(result_dir: str, max_steps: int):
    """Same patching as scripts/train_pose.load_config, for the motionbricks_root version."""
    version_dir = os.path.join(result_dir, "motionbricks_root", "version_1")
    conf = OmegaConf.load(os.path.join(version_dir, "hparams.yaml"))
    with open_dict(conf):
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = max_steps
        conf.trainer.accelerator = "auto"
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = True
        conf.trainer.log_every_n_steps = 10
        conf.trainer.val_check_interval = max_steps
        conf.trainer.num_sanity_val_steps = 0
        conf.model.scheduler.num_training_steps = max_steps
        conf.id = "synthetic"
        conf.run_dir = "."
        conf.out_dir = result_dir
    resolved = OmegaConf.to_container(conf, resolve=True)
    return OmegaConf.create(resolved), version_dir


def build_root(conf, motion_rep, root_ckpt, lr, warmup_steps, keep_fp=ROOT_KEEP_FP):
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        backbone_net = instantiate(model_conf.backbone_network, motion_rep=motion_rep, _recursive_=False)
        state = torch.load(root_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        backbone_state = {k[len("backbone_net."):]: v for k, v in state.items() if k.startswith("backbone_net.")}
        backbone_net.load_state_dict(backbone_state, strict=True)
        replaced = ternarize_linears(backbone_net, keep_fp=keep_fp) + ternarize_in_proj(backbone_net, keep_fp=keep_fp)
        optimizer_fn = instantiate(model_conf.optimizer)
        optimizer_fn = functools.partial(optimizer_fn.func, **{**optimizer_fn.keywords, "lr": lr})
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None
        if scheduler_fn is not None:
            scheduler_fn = functools.partial(scheduler_fn.func, **{**scheduler_fn.keywords, "num_warmup_steps": warmup_steps})
        model = instantiate(model_conf, pose_vqvae_network=None, root_vqvae_network=None,
                            backbone_network=backbone_net, motion_rep=motion_rep,
                            optimizer=optimizer_fn, scheduler=scheduler_fn, _recursive_=False)
    return model, backbone_net, len(replaced)
