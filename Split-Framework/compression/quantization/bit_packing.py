from __future__ import annotations

from typing import Iterable, Sequence

import torch


def shape_numel(shape: Sequence[int] | torch.Size | None) -> int:
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def pack_uint4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")

    flat = values.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if flat.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8), 0
    if bool(((flat < 0) | (flat > 15)).any()):
        raise ValueError("uint4 values must be in [0, 15]")

    original_numel = int(flat.numel())
    if original_numel % 2 != 0:
        flat = torch.cat((flat, torch.zeros((1,), dtype=torch.uint8)))

    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    packed = (lo | hi).contiguous()
    return packed, original_numel


def unpack_uint4(packed: torch.Tensor, num_values: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"packed must be a torch.Tensor, got {type(packed)!r}")

    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    count = int(num_values)
    if count < 0:
        raise ValueError("num_values must be >= 0")
    if count == 0:
        target_device = device if device is not None else "cpu"
        return torch.empty((0,), dtype=torch.uint8, device=target_device)
    if packed_cpu.numel() * 2 < count:
        raise ValueError("packed tensor does not contain enough values")

    out = torch.empty((packed_cpu.numel() * 2,), dtype=torch.uint8)
    out[0::2] = packed_cpu & 0x0F
    out[1::2] = (packed_cpu >> 4) & 0x0F
    target_device = device if device is not None else "cpu"
    return out[:count].to(device=target_device)


def pack_int4(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values)!r}")

    flat = values.detach().to(device="cpu", dtype=torch.int8).reshape(-1)
    if flat.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8), 0
    if bool(((flat < -8) | (flat > 7)).any()):
        raise ValueError("int4 values must be in [-8, 7]")

    encoded = (flat & 0x0F).to(dtype=torch.uint8)
    return pack_uint4(encoded)


def unpack_int4(packed: torch.Tensor, num_values: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    unpacked = unpack_uint4(packed, num_values, device="cpu")
    signed = unpacked.to(dtype=torch.int8)
    negative = signed >= 8
    if bool(negative.any()):
        signed = signed.clone()
        signed[negative] -= 16
    target_device = device if device is not None else "cpu"
    return signed.to(device=target_device)