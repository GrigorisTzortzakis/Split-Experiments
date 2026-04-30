from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from compression.quantization.bit_packing import pack_int4, shape_numel, unpack_int4

Payload = Dict[str, Any]


@dataclass(frozen=True)
class PerChannelInt8Codec:
    """Dynamic symmetric int8 quantization with one multiplier per channel.

    This mirrors the arithmetic int8 codec but computes the scaling multiplier
    independently for each channel instead of once for the whole tensor. For
    rank >= 2 tensors, channel axis 1 is used by default; for rank < 2 tensors,
    axis 0 is used.
    """

    channel_axis: int = 1
    num_bits: int = 8

    def _quant_bounds(self) -> tuple[int, int]:
        bits = int(self.num_bits)
        if bits == 8:
            return -128, 127
        if bits == 4:
            return -8, 7
        raise ValueError("num_bits must be either 8 or 4")

    @staticmethod
    def _as_cpu_f32(x: torch.Tensor) -> torch.Tensor:
        x_detached = x.detach().to(dtype=torch.float32)
        return x_detached.to(device="cpu")

    def _resolve_channel_axis(self, x_cpu: torch.Tensor) -> int:
        if x_cpu.ndim < 2:
            return 0
        axis = int(self.channel_axis)
        if axis < 0:
            axis += x_cpu.ndim
        if axis < 0 or axis >= x_cpu.ndim:
            raise ValueError(f"channel_axis {self.channel_axis} out of range for tensor rank {x_cpu.ndim}")
        return axis

    @staticmethod
    def _move_channel_first(x_cpu: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, Sequence[int], Sequence[int]]:
        if x_cpu.ndim == 0:
            return x_cpu.reshape(1), (0,), (0,)

        perm = (channel_axis, *[idx for idx in range(x_cpu.ndim) if idx != channel_axis])
        inverse_perm = tuple(perm.index(idx) for idx in range(len(perm)))
        return x_cpu.permute(perm).contiguous(), perm, inverse_perm

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = self._as_cpu_f32(x)
        original_shape = tuple(x_cpu.shape)
        resolved_axis = self._resolve_channel_axis(x_cpu)
        _qmin, qmax = self._quant_bounds()

        x_channel_first, _perm, inverse_perm = self._move_channel_first(x_cpu, resolved_axis)
        x_flat = x_channel_first.reshape(x_channel_first.shape[0], -1)

        if x_flat.numel() == 0:
            scale = torch.ones((x_flat.shape[0],), dtype=torch.float32)
        else:
            max_abs = x_flat.abs().max(dim=1).values
            scale = torch.where(max_abs == 0.0, torch.ones_like(max_abs), float(qmax) / max_abs)

        q_flat = torch.round(x_flat * scale.unsqueeze(1))
        qmin, qmax = self._quant_bounds()
        q_flat = torch.clamp(q_flat, qmin, qmax).to(dtype=torch.int8)
        q_channel_first = q_flat.reshape(x_channel_first.shape)

        if x_cpu.ndim == 0:
            q = q_channel_first.reshape(())
        else:
            q = q_channel_first.permute(inverse_perm).contiguous()

        if int(self.num_bits) == 4:
            stored_q, original_numel = pack_int4(q.to(device="cpu", dtype=torch.int8))
        else:
            stored_q = q
            original_numel = int(q.numel())

        return {
            "codec": f"dynamic_symmetric_int{int(self.num_bits)}_per_channel",
            "q": stored_q,
            "scale": scale,
            "shape": original_shape,
            "channel_axis": resolved_axis,
            "num_bits": int(self.num_bits),
            "num_values": original_numel,
        }

    def decode(
        self,
        payload: Payload,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not isinstance(payload, dict):
            raise TypeError(f"decode expects a dict payload, got {type(payload)!r}")
        if "q" not in payload:
            raise KeyError("Invalid payload, missing key: 'q'")

        q = payload["q"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")

        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits == 8:
            if q.dtype != torch.int8:
                raise TypeError("8-bit payload['q'] must be a torch.int8 Tensor")
        elif num_bits == 4:
            if q.dtype != torch.uint8:
                raise TypeError("4-bit payload['q'] must be a packed torch.uint8 Tensor")
        else:
            raise ValueError("payload num_bits must be either 8 or 4")

        scale = payload.get("scale", None)
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or scale.ndim != 1:
            raise TypeError("payload['scale'] must be a 1D torch.float32 Tensor")

        shape = payload.get("shape", None)
        channel_axis = int(payload.get("channel_axis", 0 if q.ndim < 2 else 1))
        target_device = device if device is not None else "cpu"

        if num_bits == 4:
            original_numel = int(payload.get("num_values", shape_numel(shape) if shape is not None else 0))
            q_dev = unpack_int4(q, original_numel, device=target_device)
            if shape is not None:
                q_dev = q_dev.reshape(tuple(int(s) for s in shape))
        else:
            q_dev = q.to(device=target_device)
        scale_dev = scale.to(device=target_device)

        if q_dev.ndim == 0:
            if int(scale_dev.shape[0]) != 1:
                raise ValueError("scalar payload must have exactly one channel scale")
            x = q_dev.to(dtype=torch.float32) / scale_dev[0]
            return x.to(dtype=dtype)

        if channel_axis < 0:
            channel_axis += q_dev.ndim
        if channel_axis < 0 or channel_axis >= q_dev.ndim:
            raise ValueError(f"payload channel_axis {channel_axis} out of range for tensor rank {q_dev.ndim}")

        perm = (channel_axis, *[idx for idx in range(q_dev.ndim) if idx != channel_axis])
        inverse_perm = tuple(perm.index(idx) for idx in range(len(perm)))
        q_channel_first = q_dev.permute(perm).contiguous()
        q_flat = q_channel_first.reshape(q_channel_first.shape[0], -1)

        if int(scale_dev.shape[0]) != int(q_flat.shape[0]):
            raise ValueError("payload scale channel count does not match tensor channel count")

        x_flat = q_flat.to(dtype=torch.float32) / scale_dev.unsqueeze(1)
        x_channel_first = x_flat.reshape(q_channel_first.shape)
        x = x_channel_first.permute(inverse_perm).contiguous()

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            x = x.reshape(tuple(int(s) for s in shape))

        return x.to(dtype=dtype)