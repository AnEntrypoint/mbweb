"""Pack ternary codes {-1,0,1} at five trits per byte (base 3, 3**5 = 243 < 256).

That is 1.6 bits per weight, against 8 bits for the int8 export. Rows are packed
independently and padded with zero trits to a multiple of five.
"""
import torch

TRITS = 5


def pack_trits(codes: torch.Tensor) -> torch.Tensor:
    """codes: int8 [rows, cols] in {-1,0,1}. Returns uint8 [rows, ceil(cols/5)]."""
    rows, cols = codes.shape
    pad = (-cols) % TRITS
    digits = (codes.to(torch.int16) + 1)
    if pad:
        digits = torch.cat([digits, torch.ones(rows, pad, dtype=torch.int16)], dim=1)
    digits = digits.view(rows, -1, TRITS)
    weights = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)
    packed = (digits * weights).sum(dim=2)
    return packed.to(torch.uint8)


def unpack_trits(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """Inverse of pack_trits. Returns int8 [rows, cols] in {-1,0,1}."""
    rows = packed.shape[0]
    values = packed.to(torch.int16).unsqueeze(2)
    divisors = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int16)
    digits = (values // divisors) % 3
    return (digits.view(rows, -1)[:, :cols] - 1).to(torch.int8)
