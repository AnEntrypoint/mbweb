"""Ternary quantization-aware training for nn.Linear layers.

Weights are quantized to {-1, 0, +1} with one absmean scale per output row. Statistics are
computed in FP32 so an all-zero row cannot underflow to NaN in FP16. The forward pass uses
a straight-through estimator: w = latent + alpha * (q - latent).detach(). alpha ramps from
0 (full precision) to 1 (fully ternary) and is a non-persistent buffer, so FP and ternary
checkpoints both load with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

BLOCK = 256


def absmean_ternary(weight: torch.Tensor, eps: float = 1e-8):
    """Return (codes in {-1,0,1} as int8, per-row FP32 scale) for a [out, in] weight."""
    w32 = weight.float()
    scale = w32.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    codes = torch.clamp(torch.round(w32 / scale), -1, 1).to(torch.int8)
    return codes, scale


def ternary_dequant(weight: torch.Tensor) -> torch.Tensor:
    codes, scale = absmean_ternary(weight)
    return (codes.float() * scale).to(weight.dtype)


class TernaryLinear(nn.Module):
    def __init__(self, src: nn.Linear):
        super().__init__()
        self.in_features = src.in_features
        self.out_features = src.out_features
        self.weight = nn.Parameter(src.weight.detach().clone(), requires_grad=src.weight.requires_grad)
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone(), requires_grad=src.bias.requires_grad)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("alpha", torch.zeros((), dtype=torch.float32), persistent=False)

    def effective_weight(self) -> torch.Tensor:
        q = ternary_dequant(self.weight)
        return self.weight + self.alpha.to(self.weight.dtype) * (q - self.weight).detach()

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


def ternarize_linears(module: nn.Module, min_in: int = BLOCK, prefix: str = "", keep_fp=()) -> list:
    """Swap eligible nn.Linear children for TernaryLinear in place. Returns replaced names.

    A layer whose dotted name contains any substring in keep_fp stays full precision.
    """
    replaced = []
    for name, child in list(module.named_children()):
        full = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and child.in_features % min_in == 0:
            if not any(s in full for s in keep_fp):
                setattr(module, name, TernaryLinear(child))
                replaced.append(full)
        else:
            replaced.extend(ternarize_linears(child, min_in, prefix=full + ".", keep_fp=keep_fp))
    return replaced


class TernaryParam(nn.Module):
    """Straight-through ternary parametrization for a raw weight tensor (e.g. MHA in_proj_weight)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("alpha", torch.zeros((), dtype=torch.float32), persistent=False)

    def forward(self, w):
        q = ternary_dequant(w)
        return w + self.alpha.to(w.dtype) * (q - w).detach()


def ternarize_in_proj(module: nn.Module, keep_fp=()) -> list:
    """Attach TernaryParam to every nn.MultiheadAttention in_proj_weight not named in keep_fp."""
    replaced = []
    for name, m in module.named_modules():
        if isinstance(m, nn.MultiheadAttention) and not any(s in name for s in keep_fp):
            if not parametrize.is_parametrized(m, "in_proj_weight"):
                parametrize.register_parametrization(m, "in_proj_weight", TernaryParam())
            replaced.append(f"{name}.in_proj_weight")
    return replaced


def set_alpha(module: nn.Module, value: float) -> None:
    for m in module.modules():
        if isinstance(m, (TernaryLinear, TernaryParam)):
            m.alpha.fill_(float(value))


def alpha_for_step(step: int, total_steps: int, ramp_fraction: float = 0.25) -> float:
    ramp_steps = max(1, int(total_steps * ramp_fraction))
    return min(1.0, step / ramp_steps)


@torch.no_grad()
@torch.no_grad()
def clip_latents(module: nn.Module, limit: float = 1.0) -> None:
    for m in module.modules():
        if isinstance(m, TernaryLinear):
            m.weight.clamp_(-limit, limit)
        elif isinstance(m, nn.MultiheadAttention) and parametrize.is_parametrized(m, "in_proj_weight"):
            m.parametrizations.in_proj_weight.original.clamp_(-limit, limit)


@torch.no_grad()
def export_ternary(module: nn.Module) -> dict:
    """Per-layer int8 codes, FP32 scales and bias. One byte per weight; not packed."""
    out = {}
    for name, m in module.named_modules():
        if isinstance(m, TernaryLinear):
            codes, scale = absmean_ternary(m.weight)
            out[name] = {
                "codes": codes.cpu(),
                "scale": scale.cpu(),
                "bias": None if m.bias is None else m.bias.detach().float().cpu(),
                "in_features": m.in_features,
                "out_features": m.out_features,
            }
    return out


class AddTernaryLinear(nn.Module):
    """Inference form of a ternary layer: y = scale * (P x - N x) + bias.

    P and N are 0/1 sparse masks of the +1 and -1 codes, so the matmuls only add selected
    inputs; the per-row FP32 scale is applied once per output element.
    """

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, bias):
        super().__init__()
        out_features, in_features = codes.shape
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("pos", (codes == 1).to_sparse_csr().float(), persistent=True)
        self.register_buffer("neg", (codes == -1).to_sparse_csr().float(), persistent=True)
        self.register_buffer("scale", scale.reshape(-1).float().clone(), persistent=True)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    def forward(self, x):
        flat = x.reshape(-1, self.in_features).float().t()
        acc = torch.sparse.mm(self.pos, flat) - torch.sparse.mm(self.neg, flat)
        y = (acc * self.scale[:, None]).t()
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


def to_add_form(layer: TernaryLinear) -> AddTernaryLinear:
    """Freeze a TernaryLinear into its addition-only inference form."""
    codes, scale = absmean_ternary(layer.weight.detach())
    return AddTernaryLinear(codes, scale, layer.bias)


def convert_to_add_form(module: nn.Module) -> int:
    """Replace every TernaryLinear under module with its addition-only form. Returns the count.

    Disables PyTorch's fused transformer fast path, which reads layer.weight directly and would
    bypass the addition form. nn.MultiheadAttention's out_proj is also read as a raw weight, so
    it stays in dense ternary form.
    """
    torch.backends.mha.set_fastpath_enabled(False)
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(module, nn.MultiheadAttention) and name == "out_proj":
            continue
        if isinstance(child, TernaryLinear):
            setattr(module, name, to_add_form(child))
            count += 1
        else:
            count += convert_to_add_form(child)
    return count
