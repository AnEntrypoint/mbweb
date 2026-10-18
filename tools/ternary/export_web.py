"""Export a trained ternary backbone for the web target.

Eligible ternary layers (in_features % 256 == 0, same rule as training) are packed at 1.6 bits
per weight with one FP32 scale per row. Layers named in keep_fp, and other linear weights, are
int8 with a per-row FP16 scale (keep_fp layers lose 0.011 held-out nats against FP16). Biases,
norms and embeddings stay FP16. Output is one binary blob plus a JSON manifest, so a browser can
fetch a single brotli-compressed file and index tensors by offset.
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ternary_qat import absmean_ternary  # noqa: E402
from ternary_pack import pack_trits  # noqa: E402


def export(ckpt_path, out_dir):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, keep_fp = ck["backbone"], tuple(ck["keep_fp"])
    os.makedirs(out_dir, exist_ok=True)
    manifest, blob, offset = {}, bytearray(), 0

    def append(raw):
        nonlocal offset
        blob.extend(raw)
        start, offset = offset, offset + len(raw)
        return start

    def add(name, kind, data, extra):
        raw = data.numpy().tobytes()
        scale = extra.pop("scale", None)
        entry = {"kind": kind, "offset": append(raw), "nbytes": len(raw), **extra}
        if scale is not None:
            entry["scale_offset"] = append(scale.numpy().astype("float16").tobytes())
            entry["scale_nbytes"] = scale.numel() * 2
        manifest[name] = entry

    for name, t in sd.items():
        if name.endswith(".parametrizations.in_proj_weight.original"):
            name = name[: -len(".parametrizations.in_proj_weight.original")] + ".in_proj_weight"
        is_linear_weight = name.endswith(".weight") or name.endswith(".in_proj_weight")
        is_linear_weight = is_linear_weight and t.dim() == 2
        kept = any(s in name for s in keep_fp)
        if is_linear_weight and not kept and t.shape[1] % 256 == 0:
            codes, scale = absmean_ternary(t)
            add(name, "ternary", pack_trits(codes), {"shape": list(t.shape), "scale": scale.reshape(-1)})
        elif is_linear_weight:
            scale = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
            add(name, "int8", q, {"shape": list(t.shape), "scale": scale.reshape(-1)})
        else:
            add(name, "fp16", t.detach().half(), {"shape": list(t.shape)})

    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(bytes(blob))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return offset


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = export(a.ckpt, a.out)
    print(f"wrote {n/2**20:.1f} MiB to {a.out}")
