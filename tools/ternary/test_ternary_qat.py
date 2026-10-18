import torch
import torch.nn as nn

from ternary_qat import (
    TernaryLinear,
    absmean_ternary,
    alpha_for_step,
    clip_latents,
    export_ternary,
    set_alpha,
    ternarize_linears,
)


def test_zero_row_has_no_nan_in_fp16_inputs():
    w = torch.zeros(4, 256, dtype=torch.float16)
    w[1] = 0.5
    codes, scale = absmean_ternary(w)
    assert torch.isfinite(scale).all()
    assert (codes[0] == 0).all()


def test_codes_are_ternary():
    torch.manual_seed(0)
    codes, _ = absmean_ternary(torch.randn(8, 256))
    assert set(codes.unique().tolist()) <= {-1, 0, 1}


def test_alpha_zero_matches_full_precision():
    torch.manual_seed(0)
    lin = nn.Linear(256, 32)
    tl = TernaryLinear(lin)
    x = torch.randn(3, 256)
    assert torch.allclose(tl(x), lin(x), atol=1e-6)


def test_alpha_one_is_ternary_forward():
    torch.manual_seed(0)
    tl = TernaryLinear(nn.Linear(256, 32))
    set_alpha(tl, 1.0)
    codes, scale = absmean_ternary(tl.weight)
    assert torch.allclose(tl.effective_weight(), codes.float() * scale, atol=1e-6)


def test_straight_through_gradient_reaches_latent():
    torch.manual_seed(0)
    tl = TernaryLinear(nn.Linear(256, 8))
    set_alpha(tl, 1.0)
    tl(torch.randn(2, 256)).sum().backward()
    assert tl.weight.grad is not None and tl.weight.grad.abs().sum() > 0


def test_frozen_source_stays_frozen():
    lin = nn.Linear(256, 8)
    lin.weight.requires_grad_(False)
    tl = TernaryLinear(lin)
    assert not tl.weight.requires_grad


def test_only_eligible_layers_replaced():
    net = nn.Sequential(nn.Linear(256, 64), nn.Linear(64, 256), nn.Linear(128, 32))
    replaced = ternarize_linears(net)
    assert replaced == ["0"]
    assert isinstance(net[0], TernaryLinear)
    assert isinstance(net[2], nn.Linear)


def test_alpha_is_not_in_state_dict_so_fp_checkpoints_load_strict():
    net = nn.Sequential(nn.Linear(256, 64))
    fp_state = net.state_dict()
    ternarize_linears(net)
    net.load_state_dict(fp_state, strict=True)
    assert "0.alpha" not in net.state_dict()


def test_ramp_and_clip():
    assert alpha_for_step(0, 100) == 0.0
    assert alpha_for_step(25, 100) == 1.0
    net = nn.Sequential(nn.Linear(256, 4))
    ternarize_linears(net)
    net[0].weight.data.fill_(5.0)
    clip_latents(net, 1.0)
    assert net[0].weight.abs().max().item() <= 1.0


def test_export_shapes():
    net = nn.Sequential(nn.Linear(256, 8), nn.Linear(8, 256))
    ternarize_linears(net)
    packed = export_ternary(net)
    assert set(packed) == {"0"}
    assert packed["0"]["codes"].shape == (8, 256)
    assert packed["0"]["scale"].shape == (8, 1)


def test_pack_roundtrip_all_codes():
    from ternary_pack import pack_trits, unpack_trits
    torch.manual_seed(0)
    codes = torch.randint(-1, 2, (7, 1024), dtype=torch.int8)
    packed = pack_trits(codes)
    assert packed.shape == (7, 1024 // 5 if 1024 % 5 == 0 else 1024 // 5 + 1)
    assert torch.equal(unpack_trits(packed, 1024), codes)


def test_pack_pads_non_multiple_of_five():
    from ternary_pack import pack_trits, unpack_trits
    codes = torch.tensor([[-1, 0, 1, 1, -1, 0, 1]], dtype=torch.int8)
    assert torch.equal(unpack_trits(pack_trits(codes), 7), codes)


def test_add_form_matches_ternary_forward():
    import torch
    from ternary_qat import TernaryLinear, to_add_form
    torch.manual_seed(0)
    src = torch.nn.Linear(256, 96)
    layer = TernaryLinear(src)
    layer.alpha.fill_(1.0)
    x = torch.randn(4, 7, 256)
    with torch.no_grad():
        ref = layer(x)
        got = to_add_form(layer)(x)
    assert torch.allclose(ref, got, atol=1e-4), (ref - got).abs().max()


def test_add_form_has_only_zero_one_masks():
    import torch
    from ternary_qat import TernaryLinear, to_add_form
    torch.manual_seed(1)
    add = to_add_form(TernaryLinear(torch.nn.Linear(256, 32)))
    for mask in (add.pos, add.neg):
        vals = mask.to_dense().unique()
        assert set(vals.tolist()) <= {0.0, 1.0}
    assert not (add.pos.to_dense() * add.neg.to_dense()).any()
