"""Export the trained ternary pose backbone as GGUF v3 for ggml (motion-bricks.cpp).

Layout follows ggml's own block formats, so the C++ runtime's ggml mul_mat reads them directly:
  TQ1_0 (type 34): 256 weights per block, 54 bytes: qs[48] base-3 trits, qh[4] trits, fp16 d.
  Q8_0  (type 8):  32 weights per block, 34 bytes: fp16 d, int8 qs[32].
  F16   (type 1):  dense half precision.

Per-row absmean scales (the trained model's scale) are stored in the fp16 d of every block of
that row, so the decoded weights equal the trained codes times the row scale exactly (TQ1_0)
or within int8 rounding (Q8_0, which uses finer 32-wide scales and so is at least as accurate).

Kept full-precision layers (keep_fp substrings) are Q8_0; other linears are TQ1_0 when
in_features % 256 == 0, else Q8_0 when % 32 == 0, else F16.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ternary_qat import absmean_ternary  # noqa: E402

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
TYPE_F32, TYPE_F16, TYPE_Q8_0, TYPE_TQ1_0 = 0, 1, 8, 34
GGUF_TYPE_UINT32, GGUF_TYPE_STRING = 4, 8
QK_K = 256
POW3 = [1, 3, 9, 27, 81]


def _decode_tq1_byte(b, count, pow3):
    """ggml dequantize_row_tq1_0: the trit at position n of a qs byte is ((b*3^n mod 256)*3)>>8."""
    return tuple(((((b * pow3[n]) & 0xFF) * 3) >> 8) for n in range(count))


def _build_tq1_tables():
    """Map each 5-trit tuple to the qs byte that decodes to it, and each 4-trit tuple to a qh byte."""
    enc5, enc4 = {}, {}
    for b in range(256):
        enc5.setdefault(_decode_tq1_byte(b, 5, POW3), b)
        enc4.setdefault(_decode_tq1_byte(b, 4, POW3), b)
    assert len(enc5) == 243 and len(enc4) == 81, "TQ1_0 trit tables are not bijective"
    return enc5, enc4


ENC5, ENC4 = _build_tq1_tables()


def pack_tq1_0_row(codes_row, row_scale_f16):
    """codes_row: int array in {-1,0,1} of length k (multiple of 256). Returns TQ1_0 bytes."""
    k = len(codes_row)
    out = bytearray()
    digits = [int(c) + 1 for c in codes_row]
    for blk in range(k // QK_K):
        e = digits[blk * QK_K:(blk + 1) * QK_K]
        qs = bytearray(48)
        for m in range(32):  # chunk A: qs[0..31], element m + 32*n for n in 0..4
            qs[m] = ENC5[tuple(e[m + 32 * n] for n in range(5))]
        for m in range(16):  # chunk B: qs[32..47], element 160 + m + 16*n
            qs[32 + m] = ENC5[tuple(e[160 + m + 16 * n] for n in range(5))]
        qh = bytearray(4)
        for j in range(4):   # qh[j] holds elements 240 + 4*n + j, n in 0..3
            qh[j] = ENC4[tuple(e[240 + 4 * n + j] for n in range(4))]
        out += qs + qh + struct.pack("<e", row_scale_f16)
    return bytes(out)


def pack_q8_0_row(values_row, row_scale_f16):
    """values_row: int8 values of length k (multiple of 32). Every block uses the row scale."""
    out = bytearray()
    for blk in range(len(values_row) // 32):
        out += struct.pack("<e", row_scale_f16)
        out += np.asarray(values_row[blk * 32:(blk + 1) * 32], dtype=np.int8).tobytes()
    return bytes(out)


def dequant_tq1_0(data, rows, cols):
    """Reference decoder mirroring ggml dequantize_row_tq1_0, used to verify the packing."""
    out = np.zeros((rows, cols), dtype=np.float32)
    per_row = (cols // QK_K) * 54
    for r in range(rows):
        row = data[r * per_row:(r + 1) * per_row]
        for blk in range(cols // QK_K):
            b = row[blk * 54:(blk + 1) * 54]
            d = struct.unpack("<e", b[52:54])[0]
            qs, qh = b[:48], b[48:52]
            vals = []
            for chunk_start, chunk_len in ((0, 32), (32, 16)):
                for n in range(5):
                    for m in range(chunk_len):
                        vals.append(_decode_tq1_byte(qs[chunk_start + m], 5, POW3)[n])
            for n in range(4):
                for j in range(4):
                    vals.append(_decode_tq1_byte(qh[j], 4, POW3)[n])
            seg = np.array(vals, dtype=np.float32) - 1.0
            out[r, blk * QK_K:(blk + 1) * QK_K] = seg * d
    return out


GGUF_TYPE_UINT64 = 10
UPSTREAM_REVISION = "a0732b642c0333077e127a2f56ab0014c196bca4"
SKELETON = "g1skel34"


def _kv_string(key, value):
    return _encode_str(key) + struct.pack("<I", GGUF_TYPE_STRING) + _encode_str(value)


def _kv_u32(key, value):
    return _encode_str(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)


def _kv_u64(key, value):
    return _encode_str(key) + struct.pack("<I", GGUF_TYPE_UINT64) + struct.pack("<Q", value)


def component_metadata(component, tensors, source_path):
    """Keys the C++ loader reads, written as scripts/convert_to_gguf.py writes them."""
    import hashlib
    with open(source_path, "rb") as f:
        # Provenance marker accepted by patches/motion-bricks/0001 (not upstream's extraction hash).
        source_sha = "ternary:" + hashlib.sha256(f.read()).hexdigest()
    params = sum(int(np.prod(shape)) for _, _, shape, _ in tensors)
    return [
        _kv_string("general.architecture", "motionbricks"),
        _kv_string("general.name", f"NVIDIA MotionBricks G1 ternary {component}"),
        _kv_u32("general.alignment", ALIGNMENT),
        _kv_u32("general.file_type", 0),
        _kv_u32("motionbricks.format_version", 1),
        _kv_string("motionbricks.component", component),
        _kv_string("motionbricks.skeleton", SKELETON),
        _kv_string("motionbricks.upstream_revision", UPSTREAM_REVISION),
        _kv_string("motionbricks.source_sha256", source_sha),
        _kv_u64("motionbricks.parameter_count", params),
    ]


def _encode_str(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _align(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


# Tensors the C++ loader validates by exact type (motion-bricks.cpp src/model.cpp tensor_shape).
F32_NAMES = {
    "_position_emb.embed", "_proj_pose_output_logit.weight", "_pose_token_emb.weight",
    "_proj_num_token_output_logit.weight", "_conv_output.model.6.weight",
    # ggml_repeat sources in the graph (src/pose.cpp, src/root.cpp): quantized types abort there.
    "_input_position_emb.weight", "_proj_num_valid_positions.weight",
}


def choose_type(name, shape, keep_fp):
    rows, cols = shape
    if name in F32_NAMES:
        return TYPE_F32
    if any(s in name for s in keep_fp):
        return TYPE_Q8_0 if cols % 32 == 0 else TYPE_F16
    if cols % QK_K == 0:
        return TYPE_TQ1_0
    if cols % 32 == 0:
        return TYPE_Q8_0
    return TYPE_F16


def build_tensor_bytes(name, tensor, gtype):
    t = tensor.detach().float()
    if gtype == TYPE_F32:
        return t.numpy().astype(np.float32).tobytes()
    if gtype == TYPE_F16:
        return t.half().numpy().tobytes()
    rows, cols = t.shape
    if gtype == TYPE_TQ1_0:
        codes, scale = absmean_ternary(t)
        codes_np = codes.numpy()
        scale16 = scale.reshape(-1).half().float().numpy()
        blob = b"".join(pack_tq1_0_row(codes_np[r], float(scale16[r])) for r in range(rows))
        return blob
    if gtype == TYPE_Q8_0:
        scale = t.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(t / scale).clamp(-127, 127).to(torch.int8).numpy()
        scale16 = scale.reshape(-1).half().float().numpy()
        return b"".join(pack_q8_0_row(q[r], float(scale16[r])) for r in range(rows))
    return t.half().numpy().tobytes()


def tensor_nbytes(gtype, shape):
    if gtype == TYPE_F32:
        return int(np.prod(shape)) * 4
    if gtype == TYPE_TQ1_0:
        rows, cols = shape
        return rows * (cols // QK_K) * 54
    if gtype == TYPE_Q8_0:
        rows, cols = shape
        return rows * (cols // 32) * 34
    return int(np.prod(shape)) * 2


def compact_name(name):
    """Mirror scripts/convert_to_gguf.py: names under GGML_MAX_NAME (64) stay as they are."""
    if len(name.encode("utf-8")) < 64:
        return name
    replacements = (
        ("_root_token_transformer_encoder.layers.", "root.l."),
        ("_shared_transformer_encoder.layers.", "shared.l."),
        ("_transformer_encoder.layers.", "transformer.l."),
        ("self_attn.", "attn."),
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("external_cond_blocks.", "xcond."),
        ("target_cond_blocks.", "tcond."),
    )
    output = name
    for old, new in replacements:
        output = output.replace(old, new)
    if len(output.encode("utf-8")) >= 64:
        suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        output = output.encode("utf-8")[:48].decode("utf-8", "ignore") + "." + suffix
    return output


def write_gguf(path, tensors, metadata):
    """tensors: list of (name, gtype, shape, payload bytes) in write order; metadata: encoded kv pairs."""
    infos = []
    offset = 0
    for name, gtype, shape, payload in tensors:
        assert len(payload) == tensor_nbytes(gtype, shape), name
        infos.append((name, gtype, shape, offset))
        offset = _align(offset + len(payload))
    header = bytearray(GGUF_MAGIC + struct.pack("<I", GGUF_VERSION))
    header += struct.pack("<QQ", len(tensors), len(metadata))
    for kv in metadata:
        header += kv
    for name, gtype, shape, off in infos:
        dims = tuple(reversed(shape))  # GGUF stores dims innermost first
        header += _encode_str(name)
        header += struct.pack("<I", len(dims))
        header += struct.pack(f"<{len(dims)}Q", *dims)
        header += struct.pack("<I", gtype)
        header += struct.pack("<Q", off)
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(b"\0" * (_align(len(header)) - len(header)))
        for _, _, _, payload in tensors:
            f.write(payload)
            f.write(b"\0" * (_align(len(payload)) - len(payload)))


def export(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd, keep_fp = ck["backbone"], tuple(ck["keep_fp"])
    tensors = []
    for name, t in sd.items():
        if name == "initted" or name.endswith(".initted"):
            continue  # upstream bookkeeping flag, dropped by scripts/convert_to_gguf.py
        if name.endswith(".parametrizations.in_proj_weight.original"):
            name = name[: -len(".parametrizations.in_proj_weight.original")] + ".in_proj_weight"
        if t.dim() != 2 or not (name.endswith(".weight") or name.endswith(".in_proj_weight")):
            gtype = TYPE_F32 if name in F32_NAMES else TYPE_F16
            tensors.append((name, gtype, tuple(t.shape), build_tensor_bytes(name, t, gtype)))
            continue
        gtype = choose_type(name, t.shape, keep_fp)
        tensors.append((name, gtype, tuple(t.shape), build_tensor_bytes(name, t, gtype)))
    write_gguf(out_path, tensors, component_metadata("pose", tensors, ckpt_path))
    return sum(len(p) for _, _, _, p in tensors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    total = export(a.ckpt, a.out)
    print(f"wrote {total / 2**20:.1f} MiB payload to {a.out}")


def _quant_type_for(name, shape, mode):
    """mode 'q8': every eligible linear Q8_0, the rest F32/F16. mode 'f32': everything F32."""
    if mode == "f32":
        return TYPE_F32
    if name in F32_NAMES or len(shape) != 2 or not (name.endswith(".weight") or name.endswith(".in_proj_weight")):
        return TYPE_F32 if name in F32_NAMES or len(shape) != 2 else TYPE_F16
    return TYPE_Q8_0 if shape[1] % 32 == 0 else TYPE_F16


def export_lightning(ckpt_path, out_path, component):
    """Export root or vq-decoder from an upstream Lightning checkpoint, selecting tensors as
    scripts/convert_to_gguf.py does, with the prefix stripped."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    if component == "root":
        pick = {k[len("backbone_net."):]: v for k, v in state.items() if k.startswith("backbone_net.")}
        mode = "q8"
    elif component == "vq-decoder":
        pick = {k[len("pose_net."):]: v for k, v in state.items()
                if k.startswith("pose_net.decoder.") or k == "pose_net.quantizer.vq._codebook.embed"}
        mode = "f32"
    else:
        raise ValueError(component)
    tensors = []
    for name, t in sorted(pick.items()):
        if name == "initted":
            continue
        gtype = _quant_type_for(name, tuple(t.shape), mode)
        tensors.append((compact_name(name), gtype, tuple(t.shape), build_tensor_bytes(name, t, gtype)))
    write_gguf(out_path, tensors, component_metadata(component, tensors, ckpt_path))
    return len(tensors), sum(len(p) for _, _, _, p in tensors)
