from __future__ import annotations

from dataclasses import dataclass, field
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
            "segment_lengths": [int(channel_flat.shape[1])] * int(channel_flat.shape[0]),
            "channel_group_counts": [1] * int(channel_flat.shape[0]),
        }

    normalized_group_size = max(1, int(group_size))
    segments: List[torch.Tensor] = []
    segment_lengths: List[int] = []
    channel_group_counts: List[int] = []
    for index in range(channel_flat.shape[0]):
        flat_values = channel_flat[index]
        group_count = 0
        for start in range(0, int(flat_values.numel()), normalized_group_size):
            segment = flat_values[start : start + normalized_group_size]
            segments.append(segment)
            segment_lengths.append(int(segment.numel()))
            group_count += 1
        channel_group_counts.append(group_count)

    return segments, {
        "shape": shape,
        "granularity": normalized_granularity,
        "channel_axis": resolved_axis,
        "inverse_perm": inverse_perm,
        "segment_lengths": segment_lengths,
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
    channel_first_shape = (int(shape[channel_axis]) if shape else 1, *[dim for idx, dim in enumerate(shape) if idx != channel_axis])

    if granularity == "per_channel":
        channel_tensor = torch.stack(list(segments), dim=0).reshape(channel_first_shape)
        return channel_tensor.permute(inverse_perm).contiguous()

    rebuilt_channels: List[torch.Tensor] = []
    cursor = 0
    channel_group_counts = [int(value) for value in metadata.get("channel_group_counts", [])]
    segment_lengths = [int(value) for value in metadata.get("segment_lengths", [])]
    segment_cursor = 0
    for group_count in channel_group_counts:
        channel_parts: List[torch.Tensor] = []
        for _ in range(group_count):
            channel_parts.append(segments[cursor][: segment_lengths[segment_cursor]])
            cursor += 1
            segment_cursor += 1
        rebuilt_channels.append(torch.cat(channel_parts, dim=0))

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


@dataclass
class NonUniformLoydCodebookCodec:
    """Non-uniform Loyd/Lloyd-Max optimized quantizer with configurable granularity."""

    range_mode: str = "minmax"
    max_iter: int = 50
    tol: float = 1e-4
    num_bits: int = 8
    granularity: str = "per_tensor"
    channel_axis: int = 1
    group_size: int = 32
    _encode_cache: Dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _decode_cache: Dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def _codec_name(self) -> str:
        return "non_uniform_loyd_codebook"

    def _levels(self) -> int:
        return 1 << _normalize_num_bits(self.num_bits)

    @staticmethod
    def _as_cpu_f32(x: torch.Tensor) -> torch.Tensor:
        x_detached = x.detach().to(dtype=torch.float32)
        return x_detached.to(device="cpu")

    def _compute_endpoints(self, x_cpu: torch.Tensor) -> tuple[float, float]:
        mode = str(self.range_mode).strip().lower()
        if mode in ("minmax", "asymmetric", "affine"):
            x_min = float(x_cpu.min().item()) if x_cpu.numel() else 0.0
            x_max = float(x_cpu.max().item()) if x_cpu.numel() else 0.0
            return x_min, x_max
        if mode in ("symmetric", "sym"):
            a = float(x_cpu.abs().max().item()) if x_cpu.numel() else 0.0
            return -a, a
        raise ValueError(f"Unsupported range_mode: {self.range_mode!r}")

    @staticmethod
    def _decision_boundaries(codebook: torch.Tensor) -> torch.Tensor:
        return 0.5 * (codebook[:-1] + codebook[1:])

    def _cache_id(self, metadata: Dict[str, Any], *, levels: int) -> str:
        shape = tuple(int(dim) for dim in metadata.get("shape", ()))
        return (
            f"loyd_{_normalize_granularity(metadata.get('granularity', self.granularity))}"
            f"_{shape}_{int(metadata.get('channel_axis', self.channel_axis) or 0)}"
            f"_{int(metadata.get('group_size', self.group_size) or 1)}"
            f"_{levels}_{str(self.range_mode).strip().lower()}_{int(self.max_iter)}_{float(self.tol)}"
        )

    def _fit_codebook(self, x_flat: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
        levels = self._levels()
        codebook = torch.linspace(x_min, x_max, steps=levels, dtype=torch.float32)

        if x_flat.numel() == 0:
            return codebook

        left_edges = torch.empty(levels, dtype=torch.float32)
        right_edges = torch.empty(levels, dtype=torch.float32)

        for _ in range(max(1, int(self.max_iter))):
            boundaries = self._decision_boundaries(codebook)
            assignments = torch.bucketize(x_flat, boundaries)

            counts = torch.bincount(assignments, minlength=levels)
            sums = torch.bincount(assignments, weights=x_flat, minlength=levels)

            new_codebook = codebook.clone()
            non_empty = counts > 0
            if torch.any(non_empty):
                new_codebook[non_empty] = sums[non_empty] / counts[non_empty].to(dtype=torch.float32)

            left_edges[0] = float(x_min)
            left_edges[1:] = boundaries
            right_edges[:-1] = boundaries
            right_edges[-1] = float(x_max)

            empty = ~non_empty
            if torch.any(empty):
                new_codebook[empty] = 0.5 * (left_edges[empty] + right_edges[empty])

            if torch.max(torch.abs(new_codebook - codebook)).item() <= float(self.tol):
                codebook = new_codebook
                break
            codebook = new_codebook

        return codebook

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = self._as_cpu_f32(x)
        segments, metadata = _split_segments(
            x_cpu,
            granularity=self.granularity,
            channel_axis=self.channel_axis,
            group_size=self.group_size,
        )

        levels = self._levels()
        cache_id = self._cache_id(metadata, levels=levels)
        q_segments: List[torch.Tensor] = []
        if cache_id in self._encode_cache:
            codebook = self._encode_cache[cache_id]
            if int(codebook.shape[0]) != len(segments):
                raise ValueError("cached Lloyd codebook segment count does not match current tensor segmentation")
            cache_write = False
        else:
            codebooks: List[torch.Tensor] = []
            for segment in segments:
                x_min, x_max = self._compute_endpoints(segment)
                if x_max == x_min:
                    codebooks.append(torch.full((levels,), x_min, dtype=torch.float32))
                else:
                    codebooks.append(self._fit_codebook(segment.reshape(-1), x_min, x_max).to(dtype=torch.float32))
            codebook = torch.stack(codebooks, dim=0)
            self._encode_cache[cache_id] = codebook
            cache_write = True

        for index, segment in enumerate(segments):
            boundaries = self._decision_boundaries(codebook[index])
            idx_u8 = torch.bucketize(segment.reshape(-1), boundaries).reshape(segment.shape).to(dtype=torch.uint8)
            q_segments.append(idx_u8)

        q = _restore_segments(q_segments, metadata)
        num_bits = _normalize_num_bits(self.num_bits)
        if num_bits < 8:
            q_payload, packed_num_values = _pack_unsigned_values(q, num_bits=num_bits)
            packed = True
        else:
            q_payload = q
            packed_num_values = int(q.numel())
            packed = False

        payload = {
            "codec": self._codec_name(),
            "q": q_payload,
            "shape": tuple(x_cpu.shape),
            "range_mode": str(self.range_mode).strip().lower(),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "num_bits": num_bits,
            "packed": packed,
            "packed_num_values": packed_num_values,
            "cache_id": cache_id,
            "cache_write": cache_write,
            "granularity": metadata["granularity"],
            "channel_axis": metadata.get("channel_axis"),
            "group_size": metadata.get("group_size"),
            "segment_lengths": metadata.get("segment_lengths"),
            "channel_group_counts": metadata.get("channel_group_counts"),
            "inverse_perm": metadata.get("inverse_perm"),
        }
        if cache_write:
            payload["codebook"] = codebook
        return payload

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
        if not isinstance(q, torch.Tensor) or q.dtype != torch.uint8:
            raise TypeError("payload['q'] must be a torch.uint8 Tensor")

        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits not in SUPPORTED_NUM_BITS:
            raise ValueError(f"payload num_bits must be one of {SUPPORTED_NUM_BITS}")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"
        cache_id = str(payload.get("cache_id") or self._cache_id({
            "shape": tuple(int(s) for s in shape) if shape is not None else tuple(int(s) for s in q.shape),
            "granularity": payload.get("granularity", self.granularity),
            "channel_axis": payload.get("channel_axis", self.channel_axis),
            "group_size": payload.get("group_size", self.group_size),
        }, levels=(1 << num_bits)))

        if "codebook" in payload:
            codebook = payload["codebook"]
            if not isinstance(codebook, torch.Tensor) or codebook.dtype != torch.float32 or codebook.ndim != 2:
                raise TypeError("payload['codebook'] must be a 2D torch.float32 Tensor")
            if int(codebook.shape[1]) != (1 << num_bits):
                raise ValueError(f"payload['codebook'] must have shape (segments, {1 << num_bits})")
            self._decode_cache[cache_id] = codebook
        else:
            codebook = self._decode_cache.get(cache_id)
            if codebook is None:
                raise KeyError(f"Lloyd payload is missing codebook and cache_id {cache_id!r} was not initialized")

        cb_dev = codebook.to(device=target_device)

        metadata = {
            "shape": tuple(int(s) for s in shape) if shape is not None else tuple(int(s) for s in q.shape),
            "granularity": payload.get("granularity", self.granularity),
            "channel_axis": payload.get("channel_axis", self.channel_axis),
            "group_size": payload.get("group_size", self.group_size),
            "segment_lengths": payload.get("segment_lengths"),
            "channel_group_counts": payload.get("channel_group_counts"),
            "inverse_perm": payload.get("inverse_perm"),
        }

        if bool(payload.get("packed", False)):
            packed_num_values = int(payload.get("packed_num_values", 0) or 0)
            if packed_num_values <= 0:
                raise ValueError("packed payload must include a positive packed_num_values")
            q_shaped = _unpack_unsigned_values(q, num_bits=num_bits, num_values=packed_num_values).to(dtype=torch.uint8).reshape(metadata["shape"])
        else:
            q_shaped = q.to(device="cpu")

        q_segments, _ = _split_segments(
            q_shaped.to(device=target_device, dtype=torch.float32),
            granularity=metadata["granularity"],
            channel_axis=int(metadata.get("channel_axis", self.channel_axis) or 0),
            group_size=int(metadata.get("group_size", self.group_size) or 1),
        )
        if len(q_segments) != int(cb_dev.shape[0]):
            raise ValueError("payload codebook segment count does not match quantized tensor segmentation")

        out_segments = []
        for index, q_segment in enumerate(q_segments):
            out_segments.append(cb_dev[index, q_segment.to(dtype=torch.int64)])

        out = _restore_segments(out_segments, metadata)

        return out.to(dtype=dtype)


@dataclass
class NonUniformLoydPerChannelCodebookCodec(NonUniformLoydCodebookCodec):
    granularity: str = "per_channel"


@dataclass
class NonUniformLoydPerGroupCodebookCodec(NonUniformLoydCodebookCodec):
    granularity: str = "per_group"


NonUniformLoydCodebookUInt8Codec = NonUniformLoydCodebookCodec
NonUniformLoydPerChannelCodebookUInt8Codec = NonUniformLoydPerChannelCodebookCodec
LloydMaxCodebookUInt8Codec = NonUniformLoydCodebookCodec