from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

Payload = Dict[str, Any]

SUPPORTED_NUM_BITS = (2, 3, 4, 6, 8)


def _normalize_num_bits(num_bits: int) -> int:
    bits = int(num_bits)
    if bits not in SUPPORTED_NUM_BITS:
        raise ValueError(f"num_bits must be one of {SUPPORTED_NUM_BITS}")
    return bits


def _normalize_granularity(granularity: str) -> str:
    normalized = str(granularity or "per_tensor").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tensor": "per_tensor",
        "per_tensor": "per_tensor",
        "channel": "per_channel",
        "per_channel": "per_channel",
        "group": "per_group",
        "per_group": "per_group",
    }
    if normalized not in aliases:
        raise ValueError("granularity must be one of: per_tensor, per_channel, per_group")
    return aliases[normalized]


def _resolve_channel_axis(x_cpu: torch.Tensor, channel_axis: int) -> int:
    if x_cpu.ndim < 2:
        return 0
    axis = int(channel_axis)
    if axis < 0:
        axis += x_cpu.ndim
    if axis < 0 or axis >= x_cpu.ndim:
        raise ValueError(f"channel_axis {channel_axis} out of range for tensor rank {x_cpu.ndim}")
    return axis


def _move_channel_first(x_cpu: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, Sequence[int]]:
    if x_cpu.ndim == 0:
        return x_cpu.reshape(1), (0,)
    perm = (channel_axis, *[idx for idx in range(x_cpu.ndim) if idx != channel_axis])
    return x_cpu.permute(perm).contiguous(), perm


def _split_segments(
    x_cpu: torch.Tensor,
    *,
    granularity: str,
    channel_axis: int,
    group_size: int,
) -> tuple[List[torch.Tensor], Dict[str, Any]]:
    normalized_granularity = _normalize_granularity(granularity)
    shape = tuple(int(dim) for dim in x_cpu.shape)
    if normalized_granularity == "per_tensor":
        return [x_cpu.reshape(-1)], {"shape": shape, "granularity": normalized_granularity}

    resolved_axis = _resolve_channel_axis(x_cpu, channel_axis)
    channel_first, perm = _move_channel_first(x_cpu, resolved_axis)
    inverse_perm = tuple(perm.index(idx) for idx in range(len(perm)))
    channel_flat = channel_first.reshape(channel_first.shape[0], -1)

    if normalized_granularity == "per_channel":
        return [channel_flat[index] for index in range(channel_flat.shape[0])], {
            "shape": shape,
            "granularity": normalized_granularity,
            "channel_axis": resolved_axis,
            "inverse_perm": inverse_perm,
            "channel_count": int(channel_flat.shape[0]),
            "channel_length": int(channel_flat.shape[1]) if channel_flat.ndim == 2 else 1,
        }

    normalized_group_size = max(1, int(group_size))
    segments: List[torch.Tensor] = []
    channel_lengths: List[int] = []
    channel_group_counts: List[int] = []
    for index in range(channel_flat.shape[0]):
        flat_values = channel_flat[index]
        channel_lengths.append(int(flat_values.numel()))
        group_count = 0
        for start in range(0, int(flat_values.numel()), normalized_group_size):
            segments.append(flat_values[start : start + normalized_group_size])
            group_count += 1
        channel_group_counts.append(group_count)

    return segments, {
        "shape": shape,
        "granularity": normalized_granularity,
        "channel_axis": resolved_axis,
        "inverse_perm": inverse_perm,
        "channel_count": int(channel_flat.shape[0]),
        "channel_lengths": channel_lengths,
        "channel_group_counts": channel_group_counts,
        "group_size": normalized_group_size,
    }


def _restore_segments(segments: Sequence[torch.Tensor], metadata: Dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(dim) for dim in metadata.get("shape", ()))
    granularity = _normalize_granularity(metadata.get("granularity", "per_tensor"))
    if granularity == "per_tensor":
        return segments[0].reshape(shape)

    inverse_perm = tuple(int(idx) for idx in metadata.get("inverse_perm", (0,)))
    channel_axis = int(metadata.get("channel_axis", 0))
    channel_first_shape = (int(metadata.get("channel_count", len(segments))), *[dim for idx, dim in enumerate(shape) if idx != channel_axis])

    if granularity == "per_channel":
        channel_tensor = torch.stack(list(segments), dim=0).reshape(channel_first_shape)
        return channel_tensor.permute(inverse_perm).contiguous()

    rebuilt_channels: List[torch.Tensor] = []
    cursor = 0
    channel_lengths = [int(value) for value in metadata.get("channel_lengths", [])]
    channel_group_counts = [int(value) for value in metadata.get("channel_group_counts", [])]
    for channel_length, group_count in zip(channel_lengths, channel_group_counts):
        channel_parts = list(segments[cursor : cursor + group_count])
        cursor += group_count
        rebuilt_channels.append(torch.cat(channel_parts, dim=0)[:channel_length])

    channel_tensor = torch.stack(rebuilt_channels, dim=0).reshape(channel_first_shape)
    return channel_tensor.permute(inverse_perm).contiguous()


def _pack_unsigned_values(values: torch.Tensor, *, num_bits: int) -> tuple[torch.Tensor, int]:
    flat = values.reshape(-1).to(dtype=torch.int64, device="cpu")
    if num_bits >= 8:
        return flat.to(dtype=torch.uint8), int(flat.numel())

    num_values = int(flat.numel())
    total_bits = num_values * int(num_bits)
    packed = torch.zeros((max(1, (total_bits + 7) // 8),), dtype=torch.int64)
    bit_offsets = torch.arange(num_values, dtype=torch.int64) * int(num_bits)
    byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
    bit_positions = torch.remainder(bit_offsets, 8)
    packed.index_add_(0, byte_indices, torch.bitwise_left_shift(flat, bit_positions))

    spill_mask = bit_positions + int(num_bits) > 8
    if torch.any(spill_mask):
        spill_indices = byte_indices[spill_mask] + 1
        spill_values = torch.bitwise_right_shift(flat[spill_mask], 8 - bit_positions[spill_mask])
        packed.index_add_(0, spill_indices, spill_values)

    return torch.bitwise_and(packed, 0xFF).to(dtype=torch.uint8), num_values


def _unpack_unsigned_values(packed: torch.Tensor, *, num_bits: int, num_values: int) -> torch.Tensor:
    packed_flat = packed.reshape(-1).to(dtype=torch.int64, device="cpu")
    if num_bits >= 8:
        return packed_flat[:num_values].to(dtype=torch.int64)

    bit_offsets = torch.arange(int(num_values), dtype=torch.int64) * int(num_bits)
    byte_indices = torch.div(bit_offsets, 8, rounding_mode="floor")
    bit_positions = torch.remainder(bit_offsets, 8)
    unpacked = torch.bitwise_right_shift(packed_flat[byte_indices], bit_positions)

    spill_mask = bit_positions + int(num_bits) > 8
    if torch.any(spill_mask):
        spill = torch.bitwise_left_shift(
            packed_flat[byte_indices[spill_mask] + 1],
            8 - bit_positions[spill_mask],
        )
        unpacked[spill_mask] = torch.bitwise_or(unpacked[spill_mask], spill)

    return torch.bitwise_and(unpacked, (1 << int(num_bits)) - 1)


@dataclass(frozen=True)
class IntCodec:
    """Dynamic symmetric integer quantization with configurable granularity.

    - Supports 2, 3, 4, 6, and 8 bits.
    - Supports per-tensor, per-channel, and per-group scaling.
    - Packs sub-byte widths into uint8 payloads on the wire.
    - Quantize: q = round(x * S)
    - Clip: q in the signed range for the selected bit width.
    - Send one scale per tensor/channel/group.
    - Dequantize: x_hat = q / S
    """

    fixed_scale: Optional[float] = None
    num_bits: int = 8
    granularity: str = "per_tensor"
    channel_axis: int = 1
    group_size: int = 32

    def _quant_bounds(self) -> tuple[int, int]:
        bits = _normalize_num_bits(self.num_bits)
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def _codec_name(self) -> str:
        return "dynamic_symmetric_int"

    @staticmethod
    def _as_cpu_f32(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(dtype=torch.float32, device="cpu")

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")
        if self.fixed_scale == 0:
            raise ValueError("fixed_scale must be non-zero")

        qmin, qmax = self._quant_bounds()

        x_cpu = self._as_cpu_f32(x)
        segments, metadata = _split_segments(
            x_cpu,
            granularity=self.granularity,
            channel_axis=self.channel_axis,
            group_size=self.group_size,
        )

        scale_values: List[float] = []
        q_segments: List[torch.Tensor] = []
        for segment in segments:
            if self.fixed_scale is None:
                max_abs = float(segment.abs().max().item()) if segment.numel() else 0.0
                scale = 1.0 if max_abs == 0.0 else (float(qmax) / max_abs)
            else:
                scale = float(self.fixed_scale)
            scale_values.append(scale)
            q_segment = torch.round(segment * scale)
            q_segments.append(torch.clamp(q_segment, qmin, qmax).to(dtype=torch.int8))

        stored_q = _restore_segments(q_segments, metadata)
        num_bits = _normalize_num_bits(self.num_bits)
        if num_bits < 8:
            unsigned_q = stored_q.to(dtype=torch.int16) - int(qmin)
            q_payload, packed_num_values = _pack_unsigned_values(unsigned_q, num_bits=num_bits)
            packed = True
        else:
            q_payload = stored_q
            packed_num_values = int(stored_q.numel())
            packed = False

        return {
            "codec": self._codec_name(),
            "q": q_payload,
            "scale": torch.tensor(scale_values, dtype=torch.float32),
            "shape": tuple(x_cpu.shape),
            "num_bits": num_bits,
            "packed": packed,
            "packed_num_values": packed_num_values,
            "granularity": metadata["granularity"],
            "channel_axis": metadata.get("channel_axis"),
            "group_size": metadata.get("group_size"),
            "channel_count": metadata.get("channel_count"),
            "channel_length": metadata.get("channel_length"),
            "channel_lengths": metadata.get("channel_lengths"),
            "channel_group_counts": metadata.get("channel_group_counts"),
            "inverse_perm": metadata.get("inverse_perm"),
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

        num_bits = _normalize_num_bits(int(payload.get("num_bits", self.num_bits)))
        scale = payload.get("scale", None)
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32 or scale.ndim != 1:
            raise TypeError("payload['scale'] must be a 1D torch.float32 Tensor")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"

        metadata = {
            "shape": tuple(int(s) for s in shape) if shape is not None else tuple(int(s) for s in q.shape),
            "granularity": payload.get("granularity", self.granularity),
            "channel_axis": payload.get("channel_axis", self.channel_axis),
            "group_size": payload.get("group_size", self.group_size),
            "channel_count": payload.get("channel_count"),
            "channel_length": payload.get("channel_length"),
            "channel_lengths": payload.get("channel_lengths"),
            "channel_group_counts": payload.get("channel_group_counts"),
            "inverse_perm": payload.get("inverse_perm"),
        }

        if bool(payload.get("packed", False)):
            packed_num_values = int(payload.get("packed_num_values", 0) or 0)
            if q.dtype != torch.uint8:
                raise TypeError("packed payload['q'] must be a torch.uint8 Tensor")
            if packed_num_values <= 0:
                raise ValueError("packed payload must include a positive packed_num_values")
            q_unsigned = _unpack_unsigned_values(q, num_bits=num_bits, num_values=packed_num_values)
            q_shaped = (q_unsigned + self._quant_bounds()[0]).to(dtype=torch.int8).reshape(metadata["shape"])
        else:
            if q.dtype != torch.int8:
                raise TypeError("payload['q'] must be a torch.int8 Tensor")
            q_shaped = q.to(device="cpu")

        q_dev = q_shaped.to(device=target_device)
        q_segments, _ = _split_segments(
            q_dev.to(dtype=torch.float32),
            granularity=metadata["granularity"],
            channel_axis=int(metadata.get("channel_axis", self.channel_axis) or 0),
            group_size=int(metadata.get("group_size", self.group_size) or 1),
        )

        if len(q_segments) != int(scale.shape[0]):
            raise ValueError("payload scale segment count does not match quantized tensor segmentation")

        x_segments = [
            q_segment / scale[index].to(device=target_device)
            for index, q_segment in enumerate(q_segments)
        ]
        x = _restore_segments(x_segments, metadata)

        return x.to(dtype=dtype)


@dataclass(frozen=True)
class PerChannelIntCodec(IntCodec):
    granularity: str = "per_channel"


@dataclass(frozen=True)
class PerGroupIntCodec(IntCodec):
    granularity: str = "per_group"


TruncationInt8Codec = IntCodec
PerChannelInt8Codec = PerChannelIntCodec
