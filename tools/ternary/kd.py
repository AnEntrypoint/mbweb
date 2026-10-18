"""Distillation from the full-precision backbone into the ternary student.

The student backbone is wrapped so that each training forward also runs the frozen teacher on the
same inputs and returns a KL term between their pose-token distributions. The term is added to
the upstream pose loss through an instance override of model.loss, so the upstream training loop
is unchanged. The teacher lives outside the module tree so Lightning and the optimizer ignore it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KDBackbone(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module):
        super().__init__()
        self.student = student
        self._teacher = [teacher]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["student"], name)

    def forward(self, *args, **kwargs):
        out = self.student(*args, **kwargs)
        if not self.training:
            return out
        with torch.no_grad():
            t_logits = self._teacher[0](*args, **kwargs)["pose_logits"]
        s_logits = out["pose_logits"]
        s_logp = F.log_softmax(s_logits.float(), dim=-1).reshape(-1, s_logits.shape[-1])
        t_prob = F.softmax(t_logits.float(), dim=-1).reshape(-1, t_logits.shape[-1])
        kd = F.kl_div(s_logp, t_prob, reduction="batchmean")
        return {**out, "kd_loss": kd}


def wrap_model_with_kd(model, teacher, weight: float):
    """Install KDBackbone as model.backbone_net and add weight * kd_loss to the upstream loss."""
    student = model.backbone_net
    model.backbone_net = KDBackbone(student, teacher)
    upstream_loss = model.loss

    def loss_with_kd(batch, model_output_batch):
        losses = upstream_loss(batch, model_output_batch)
        if "kd_loss" in model_output_batch:
            losses = dict(losses)
            losses["kd_loss"] = model_output_batch["kd_loss"]
            losses["loss"] = losses["loss"] + weight * model_output_batch["kd_loss"]
        return losses

    model.loss = loss_with_kd
    return model.backbone_net.student
