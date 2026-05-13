from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings

import torch

from compression.comparison_papers.paper_top_k import PAPER_TOP_K_CACHE_ID_ATTR

Payload = Dict[str, Any]
SPLIT_FC_CACHE_ID_ATTR = "_split_fc_cache_id"
MAX_LEVEL_COUNT = 1 << 32
_FORWARD_FLAG = 0
_BACKWARD_FLAG = 1


class _BitWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0
        self._used_bits = 0
        self.bit_count = 0

    def write_uint(self, value: int, width: int) -> None:
        if width <= 0:
            return
        masked = int(value) & ((1 << width) - 1)
        for shift in range(width - 1, -1, -1):
            bit = (masked >> shift) & 1
            self._current_byte = (self._current_byte << 1) | bit
            self._used_bits += 1
            self.bit_count += 1
            if self._used_bits == 8:
                self._buffer.append(self._current_byte)
                self._current_byte = 0
                self._used_bits = 0

    def write_bool_list(self, values) -> None:
        for value in values:
            self.write_uint(1 if bool(value) else 0, 1)

    def write_float32(self, value: float) -> None:
        packed = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        self.write_uint(packed, 32)

    def finish(self) -> bytes:
        if self._used_bits:
            self._buffer.append(self._current_byte << (8 - self._used_bits))
            self._current_byte = 0
            self._used_bits = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, payload: bytes, bit_count: int) -> None:
        self._payload = bytes(payload)
        self._bit_count = int(bit_count)
        self._offset = 0

    def read_uint(self, width: int) -> int:
        if width <= 0:
            return 0
        if self._offset + width > self._bit_count:
            raise ValueError("SplitFC bitstream ended unexpectedly")
        value = 0
        for _ in range(width):
            byte_index = self._offset // 8
            bit_index = 7 - (self._offset % 8)
            value = (value << 1) | ((self._payload[byte_index] >> bit_index) & 1)
            self._offset += 1
        return value

    def read_bool_tensor(self, length: int) -> torch.Tensor:
        values = [bool(self.read_uint(1)) for _ in range(max(0, int(length)))]
        return torch.tensor(values, dtype=torch.bool)

    def read_float32(self) -> float:
        raw = self.read_uint(32)
        return float(struct.unpack(">f", struct.pack(">I", raw))[0])


def get_split_fc_cache_id(tensor: Any) -> Optional[str]:
    if not isinstance(tensor, torch.Tensor):
        return None
    cache_id = getattr(tensor, SPLIT_FC_CACHE_ID_ATTR, None)
    if cache_id is None:
        cache_id = getattr(tensor, PAPER_TOP_K_CACHE_ID_ATTR, None)
    return None if cache_id is None else str(cache_id)


def set_split_fc_cache_id(tensor: Any, cache_id: Optional[str]) -> Any:
    if isinstance(tensor, torch.Tensor) and cache_id:
        cache_id = str(cache_id)
        setattr(tensor, SPLIT_FC_CACHE_ID_ATTR, cache_id)
        setattr(tensor, PAPER_TOP_K_CACHE_ID_ATTR, cache_id)
    return tensor


def transfer_split_fc_cache_id(source: Any, target: Any) -> Any:
    return set_split_fc_cache_id(target, get_split_fc_cache_id(source))


def _clamp_level_count(level_count: float | int) -> int:
    return max(2, min(MAX_LEVEL_COUNT, int(round(float(level_count)))))


def _level_bit_width(level_count: float | int) -> int:
    return max(1, int(math.ceil(math.log2(float(_clamp_level_count(level_count))))))


def _integral_dtype(max_value: int) -> torch.dtype:
    if max_value <= 0x100:
        return torch.uint8
    if max_value <= 0x10000:
        return torch.int32
    return torch.int64


def _cube_root(value: float) -> float:
    if value >= 0.0:
        return float(value) ** (1.0 / 3.0)
    return -((-float(value)) ** (1.0 / 3.0))


@dataclass
class SplitFCCodec:
    reduction_ratio: float = 16.0
    feature_bits_per_entry: float = 8.0
    gradient_bits_per_entry: float = 8.0
    endpoint_levels: int = 200
    epsilon: float = 1e-8
    seed: Optional[int] = None
    candidate_ms: Optional[Sequence[int]] = None
    original_activation_shape: Optional[Sequence[int]] = None
    require_channel_shape: bool = False
    _cache_counter: int = field(default=0, init=False, repr=False)
    _state_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _generator: Optional[torch.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reduction_ratio <= 1.0:
            raise ValueError("reduction_ratio must be > 1 for SplitFC")
        if self.feature_bits_per_entry <= 0.0 or self.gradient_bits_per_entry <= 0.0:
            raise ValueError("per-entry bit budgets must be positive")
        if int(self.endpoint_levels) < 2:
            raise ValueError("endpoint_levels must be at least 2")
        self.endpoint_levels = int(self.endpoint_levels)
        if self.seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(self.seed))

    def _next_cache_id(self) -> str:
        cache_id = f"split_fc_{self._cache_counter}"
        self._cache_counter += 1
        return cache_id

    def _cache_id_for_stream(self, stream_id: int) -> str:
        return f"split_fc_{int(stream_id)}"

    def _stream_id_from_cache_id(self, cache_id: str) -> int:
        try:
            return int(str(cache_id).rsplit("_", 1)[1])
        except Exception as exc:
            raise ValueError(f"Invalid SplitFC cache id: {cache_id!r}") from exc

    def _matrix_view(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        shape = tuple(int(dim) for dim in x_cpu.shape)
        if x_cpu.ndim <= 1:
            return x_cpu.reshape(1, -1), shape, 1
        return x_cpu.reshape(int(x_cpu.shape[0]), -1), shape, int(x_cpu.shape[0])

    def _column_count(self, shape: Tuple[int, ...]) -> int:
        if len(shape) <= 1:
            return int(torch.tensor(shape or (0,)).prod().item())
        return int(torch.tensor(shape[1:]).prod().item())

    def _grouping_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.original_activation_shape is None:
            if len(shape) <= 2:
                if self.require_channel_shape:
                    raise ValueError(
                        "original_activation_shape is required for flattened convolutional activations "
                        "to reproduce SplitFC channel-wise normalization."
                    )
                warnings.warn(
                    "SplitFC assumes fully connected grouping for flattened activations unless original_activation_shape is provided",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return shape
        override = tuple(int(dim) for dim in self.original_activation_shape)
        if not override or any(dim <= 0 for dim in override):
            return shape
        column_count = self._column_count(shape)
        if len(override) >= 2 and int(torch.tensor(override[1:]).prod().item()) == column_count:
            return override
        if int(torch.tensor(override).prod().item()) == column_count:
            batch_dim = int(shape[0]) if len(shape) > 1 else 1
            return (batch_dim, *override)
        return shape

    def _infer_groups(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
        grouping_shape = self._grouping_shape(shape)
        column_count = self._column_count(grouping_shape)
        if column_count <= 0:
            return tuple()
        if len(grouping_shape) <= 2:
            return tuple(torch.tensor([index], dtype=torch.int64) for index in range(column_count))
        channel_count = int(grouping_shape[1])
        trailing = int(torch.tensor(grouping_shape[2:]).prod().item()) if len(grouping_shape) > 2 else 1
        groups = []
        for channel_index in range(channel_count):
            start = channel_index * trailing
            stop = min(column_count, start + trailing)
            groups.append(torch.arange(start, stop, dtype=torch.int64))
        return tuple(groups)

    def _normalize_features(self, matrix: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        normalized = torch.zeros_like(matrix)
        for group in self._infer_groups(shape):
            if group.numel() == 0:
                continue
            block = matrix.index_select(1, group)
            group_min = block.min()
            group_max = block.max()
            span = float(group_max - group_min)
            if span <= self.epsilon:
                continue
            normalized.index_copy_(1, group, (block - group_min) / span)
        return normalized

    def _target_keep_count(self, column_count: int) -> float:
        if column_count <= 0:
            return 0.0
        return max(0.0, min(float(column_count), float(column_count) / float(self.reduction_ratio)))

    def _sample_dropout_mask(self, matrix: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        column_count = int(matrix.shape[1])
        if column_count == 0:
            empty_mask = torch.zeros(0, dtype=torch.bool)
            return empty_mask, matrix[:, :0], torch.zeros(0, dtype=matrix.dtype)

        target_keep = self._target_keep_count(column_count)
        normalized = self._normalize_features(matrix, shape)
        sigma = normalized.std(dim=0, unbiased=False)
        sigma_sum = float(sigma.sum().item())
        if sigma_sum <= self.epsilon:
            warnings.warn("SplitFC falling back to uniform dropout probabilities for zero-variance features", RuntimeWarning, stacklevel=2)
            keep_probs = torch.full((column_count,), float(target_keep) / float(column_count), dtype=matrix.dtype)
        else:
            keep_probs = sigma * (float(target_keep) / sigma_sum)
            max_prob = float(keep_probs.max().item())
            if max_prob > 1.0:
                sigma_max = float(sigma.max().item())
                denominator = max(1.0, float(column_count - target_keep))
                lower_bound = (sigma_max * float(target_keep) - sigma_sum) / denominator
                shifted = sigma + max(0.0, lower_bound)
                shifted_sum = float(shifted.sum().item())
                if shifted_sum <= self.epsilon:
                    warnings.warn("SplitFC falling back to uniform dropout probabilities for zero-variance features", RuntimeWarning, stacklevel=2)
                    keep_probs = torch.full((column_count,), float(target_keep) / float(column_count), dtype=matrix.dtype)
                else:
                    keep_probs = shifted * (float(target_keep) / shifted_sum)

        keep_probs = keep_probs.clamp_(0.0, 1.0)
        bernoulli = torch.rand(column_count, generator=self._generator)
        keep_mask = bernoulli < keep_probs
        kept_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
        kept_matrix = matrix.index_select(1, kept_indices)
        kept_probs = keep_probs.index_select(0, kept_indices).clamp_min(self.epsilon)
        scaled_kept = kept_matrix / kept_probs.unsqueeze(0)
        return keep_mask, scaled_kept, keep_probs

    def _available_bits(self, *, batch_size: int, original_columns: int, forward: bool) -> float:
        per_entry = self.feature_bits_per_entry if forward else self.gradient_bits_per_entry
        budget = float(batch_size) * float(original_columns) * float(per_entry)
        if forward:
            budget -= float(original_columns)
        return max(0.0, budget)

    def _shape_header_bits(self, shape: Tuple[int, ...]) -> int:
        return 8 + (32 * len(shape))

    def _actual_available_bits(self, *, batch_size: int, original_columns: int, forward: bool, shape: Tuple[int, ...]) -> float:
        return self._available_bits(batch_size=batch_size, original_columns=original_columns, forward=forward)

    def _mean_error_term(self, ranges: torch.Tensor, indices: torch.Tensor, batch_size: int) -> float:
        if indices.numel() == 0:
            return 0.0
        selected = ranges.index_select(0, indices)
        return float(((selected * selected) * (float(batch_size) / 2.0)).sum().item())

    def _endpoint_plan(self, matrix: torch.Tensor, sorted_indices: torch.Tensor, count: int) -> Dict[str, Any]:
        if count <= 0:
            empty_int = torch.empty(0, dtype=torch.int64)
            empty_float = torch.empty(0, dtype=matrix.dtype)
            return {
                "positions": empty_int,
                "codes_min": empty_int,
                "codes_max": empty_int,
                "quant_mins": empty_float,
                "quant_maxs": empty_float,
                "ranges": empty_float,
                "global_min": 0.0,
                "global_max": 0.0,
            }

        positions = sorted_indices[:count]
        selected = matrix.index_select(1, positions)
        mins = selected.min(dim=0).values
        maxs = selected.max(dim=0).values
        global_min = float(mins.min().item())
        global_max = float(maxs.max().item())
        span = global_max - global_min
        if span <= self.epsilon:
            codes_min = torch.zeros(count, dtype=torch.int64)
            codes_max = torch.zeros(count, dtype=torch.int64)
            quant_mins = torch.full((count,), global_min, dtype=matrix.dtype)
            quant_maxs = quant_mins.clone()
        else:
            delta = span / float(self.endpoint_levels - 1)
            codes_min = torch.floor((mins - global_min) / delta).clamp_(0, self.endpoint_levels - 1).to(dtype=torch.int64)
            codes_max = torch.ceil((maxs - global_min) / delta).clamp_(0, self.endpoint_levels - 1).to(dtype=torch.int64)
            quant_mins = global_min + codes_min.to(dtype=matrix.dtype) * delta
            quant_maxs = global_min + codes_max.to(dtype=matrix.dtype) * delta
        return {
            "positions": positions.to(dtype=torch.int64),
            "codes_min": codes_min,
            "codes_max": codes_max,
            "quant_mins": quant_mins,
            "quant_maxs": quant_maxs,
            "ranges": (quant_maxs - quant_mins).clamp_min(0.0),
            "global_min": global_min,
            "global_max": global_max,
        }

    def _allocation_coefficients(self, *, batch_size: int, endpoint_ranges: torch.Tensor, residual_count: int, residual_mean_range: float) -> Tuple[torch.Tensor, torch.Tensor]:
        coeffs = [float((span.item() ** 2) * float(batch_size) / 4.0) for span in endpoint_ranges]
        weights = [int(batch_size)] * int(endpoint_ranges.numel())
        if residual_count > 0:
            coeffs.append(float((residual_mean_range ** 2) * float(batch_size) * float(residual_count) / 2.0))
            weights.append(int(residual_count))
        return torch.tensor(coeffs, dtype=torch.float64), torch.tensor(weights, dtype=torch.float64)

    def _objective_for_levels(self, coeffs: torch.Tensor, levels: torch.Tensor) -> float:
        if int(coeffs.numel()) == 0:
            return 0.0
        safe_levels = levels.to(dtype=torch.float64).clamp_min(2.0)
        terms = torch.where(
            coeffs.to(dtype=torch.float64) > self.epsilon,
            coeffs.to(dtype=torch.float64) / ((safe_levels - 1.0) ** 2),
            torch.zeros_like(coeffs, dtype=torch.float64),
        )
        return float(terms.sum().item())

    def _budget_bits_for_levels(self, levels: torch.Tensor, weights: torch.Tensor) -> float:
        if int(levels.numel()) == 0:
            return 0.0
        return float((weights.to(dtype=torch.float64) * torch.log2(levels.to(dtype=torch.float64).clamp_min(2.0))).sum().item())

    def _continuous_level_for_multiplier(self, coefficient: float, weight: float, multiplier: float) -> float:
        if coefficient <= self.epsilon or weight <= 0.0:
            return 2.0
        ratio = float(coefficient) / max(float(multiplier) * float(weight), self.epsilon)
        if ratio <= self.epsilon:
            return 2.0
        discriminant = (ratio * ratio / 4.0) - (ratio * ratio * ratio / 27.0)
        if discriminant >= 0.0:
            root_term = math.sqrt(discriminant)
            shifted = _cube_root((ratio / 2.0) + root_term) + _cube_root((ratio / 2.0) - root_term)
        else:
            cosine_arg = max(-1.0, min(1.0, 1.5 * math.sqrt(3.0 / ratio)))
            shifted = 2.0 * math.sqrt(ratio / 3.0) * math.cos(math.acos(cosine_arg) / 3.0)
        return float(min(MAX_LEVEL_COUNT, max(2.0, 1.0 + shifted)))

    def _continuous_levels_for_multiplier(self, coeffs: torch.Tensor, weights: torch.Tensor, multiplier: float) -> torch.Tensor:
        values = [
            self._continuous_level_for_multiplier(float(coefficient.item()), float(weight.item()), multiplier)
            for coefficient, weight in zip(coeffs, weights)
        ]
        return torch.tensor(values, dtype=torch.float64)

    def _solve_continuous_level_allocation(self, coeffs: torch.Tensor, weights: torch.Tensor, budget_bits: float) -> Optional[torch.Tensor]:
        if int(coeffs.numel()) == 0:
            return torch.empty(0, dtype=torch.float64)
        if float(weights.sum().item()) > float(budget_bits) + 1e-9:
            return None
        low_multiplier = 1e-12
        low_levels = self._continuous_levels_for_multiplier(coeffs, weights, low_multiplier)
        if self._budget_bits_for_levels(low_levels, weights) <= float(budget_bits) + 1e-9:
            return low_levels
        high_multiplier = 1.0
        high_levels = self._continuous_levels_for_multiplier(coeffs, weights, high_multiplier)
        while self._budget_bits_for_levels(high_levels, weights) > float(budget_bits) + 1e-9:
            high_multiplier *= 2.0
            high_levels = self._continuous_levels_for_multiplier(coeffs, weights, high_multiplier)
            if high_multiplier > 1e24:
                return None
        for _ in range(80):
            mid = (low_multiplier + high_multiplier) / 2.0
            mid_levels = self._continuous_levels_for_multiplier(coeffs, weights, mid)
            if self._budget_bits_for_levels(mid_levels, weights) > float(budget_bits):
                low_multiplier = mid
            else:
                high_multiplier = mid
                high_levels = mid_levels
        return high_levels

    def _round_levels_to_budget(self, *, continuous_levels: torch.Tensor, coeffs: torch.Tensor, weights: torch.Tensor, budget_bits: float) -> torch.Tensor:
        if int(continuous_levels.numel()) == 0:
            return torch.empty(0, dtype=torch.int64)
        discrete_levels = torch.floor(continuous_levels).to(dtype=torch.int64).clamp_min_(2)
        used_bits = self._budget_bits_for_levels(discrete_levels.to(dtype=torch.float64), weights)

        while used_bits > float(budget_bits) + 1e-9:
            best_index: Optional[int] = None
            best_penalty = float("inf")
            for index, current_level in enumerate(discrete_levels.tolist()):
                if current_level <= 2:
                    continue
                reduced_level = current_level - 1
                freed_bits = float(weights[index].item()) * (math.log2(current_level) - math.log2(reduced_level))
                if freed_bits <= self.epsilon:
                    continue
                current_error = 0.0 if float(coeffs[index].item()) <= self.epsilon else float(coeffs[index].item()) / ((current_level - 1.0) ** 2)
                reduced_error = 0.0 if float(coeffs[index].item()) <= self.epsilon else float(coeffs[index].item()) / ((reduced_level - 1.0) ** 2)
                penalty = (reduced_error - current_error) / freed_bits
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_index = index
            if best_index is None:
                break
            discrete_levels[best_index] -= 1
            used_bits = self._budget_bits_for_levels(discrete_levels.to(dtype=torch.float64), weights)

        while True:
            remaining_bits = float(budget_bits) - used_bits
            best_index = None
            best_gain = 0.0
            for index, current_level in enumerate(discrete_levels.tolist()):
                if current_level >= MAX_LEVEL_COUNT:
                    continue
                increased_level = min(MAX_LEVEL_COUNT, current_level + 1)
                extra_bits = float(weights[index].item()) * (math.log2(increased_level) - math.log2(current_level))
                if extra_bits > remaining_bits + 1e-9:
                    continue
                current_error = 0.0 if float(coeffs[index].item()) <= self.epsilon else float(coeffs[index].item()) / ((current_level - 1.0) ** 2)
                improved_error = 0.0 if float(coeffs[index].item()) <= self.epsilon else float(coeffs[index].item()) / ((increased_level - 1.0) ** 2)
                gain = (current_error - improved_error) / max(extra_bits, self.epsilon)
                if gain > best_gain:
                    best_gain = gain
                    best_index = index
            if best_index is None or best_gain <= 0.0:
                break
            discrete_levels[best_index] += 1
            used_bits = self._budget_bits_for_levels(discrete_levels.to(dtype=torch.float64), weights)
        return discrete_levels

    def _optimal_level_allocation(self, *, kept_count: int, batch_size: int, budget_bits: float, endpoint_ranges: torch.Tensor, residual_count: int, residual_mean_range: float, endpoint_overhead_bits: float) -> Optional[Dict[str, Any]]:
        base_bits = float(kept_count) + 128.0 + float(endpoint_overhead_bits)
        if base_bits > budget_bits:
            return None
        coeffs, weights = self._allocation_coefficients(
            batch_size=batch_size,
            endpoint_ranges=endpoint_ranges,
            residual_count=residual_count,
            residual_mean_range=residual_mean_range,
        )
        if int(coeffs.numel()) == 0:
            return {
                "entry_levels": torch.empty(0, dtype=torch.int64),
                "mean_levels": 0,
                "used_bits": base_bits,
                "coefficients": coeffs,
                "weights": weights,
            }
        continuous_levels = self._solve_continuous_level_allocation(coeffs, weights, budget_bits - base_bits)
        if continuous_levels is None:
            return None
        discrete_levels = self._round_levels_to_budget(
            continuous_levels=continuous_levels,
            coeffs=coeffs,
            weights=weights,
            budget_bits=budget_bits - base_bits,
        )
        entry_count = int(endpoint_ranges.numel())
        used_bits = base_bits + self._budget_bits_for_levels(discrete_levels.to(dtype=torch.float64), weights)
        return {
            "entry_levels": discrete_levels[:entry_count].to(dtype=torch.int64),
            "mean_levels": int(discrete_levels[entry_count].item()) if residual_count > 0 else 0,
            "used_bits": used_bits,
            "coefficients": coeffs,
            "weights": weights,
        }

    def _max_two_stage_count(self, *, kept_count: int, batch_size: int, budget_bits: float) -> int:
        endpoint_width = math.log2(float(self.endpoint_levels))
        numerator = float(budget_bits) - (2.0 * float(kept_count)) - 128.0
        denominator = float(batch_size) + (2.0 * endpoint_width) - 1.0
        if numerator < 0.0 or denominator <= 0.0:
            return 0
        return max(0, min(kept_count, int(math.floor(numerator / denominator))))

    def _candidate_values_for_budget(self, *, kept_count: int, batch_size: int, budget_bits: float) -> Tuple[int, ...]:
        if kept_count <= 0:
            return (0,)
        if self.candidate_ms is not None:
            candidates = sorted({max(0, min(kept_count, int(value))) for value in self.candidate_ms}, reverse=True)
            return tuple(candidates) if candidates else (0,)
        dmax = self._max_two_stage_count(kept_count=kept_count, batch_size=batch_size, budget_bits=budget_bits)
        candidates = {
            max(0, min(kept_count, int(math.floor(dmax * n / 10.0))))
            for n in range(1, 11)
        }
        return tuple(sorted(candidates, reverse=True))

    def _quantize_uniform_codes(self, values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, level_counts: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.empty_like(values, dtype=torch.int64)
        levels = level_counts.to(dtype=values.dtype).clamp_min(2.0).unsqueeze(0)
        deltas = (upper - lower).unsqueeze(0) / (levels - 1.0).clamp_min(1.0)
        scaled = torch.zeros_like(values)
        nonzero = (upper - lower) > self.epsilon
        if bool(nonzero.any()):
            scaled[:, nonzero] = (values[:, nonzero] - lower.unsqueeze(0)[:, nonzero]) / deltas[:, nonzero]
        max_codes = (levels - 1.0).to(dtype=torch.int64)
        return scaled.round().to(dtype=torch.int64).clamp_min_(0).minimum(max_codes)

    def _dequantize_uniform_codes(self, codes: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, level_counts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if codes.numel() == 0:
            return torch.empty_like(codes, dtype=dtype)
        levels = level_counts.to(dtype=dtype).clamp_min(2.0).unsqueeze(0)
        deltas = (upper - lower).to(dtype=dtype).unsqueeze(0) / (levels - 1.0).clamp_min(1.0)
        return lower.to(dtype=dtype).unsqueeze(0) + codes.to(dtype=dtype) * deltas

    def _plan_quantization(self, matrix: torch.Tensor, *, budget_bits: float) -> Dict[str, Any]:
        batch_size = int(matrix.shape[0])
        kept_count = int(matrix.shape[1])
        if kept_count == 0:
            return {
                "kept_count": 0,
                "indicator": torch.empty(0, dtype=torch.bool),
                "two_stage_positions": torch.empty(0, dtype=torch.int64),
                "mean_positions": torch.empty(0, dtype=torch.int64),
                "two_stage_codes": torch.empty((batch_size, 0), dtype=torch.int64),
                "two_stage_levels": torch.empty(0, dtype=torch.int64),
                "endpoint_min_codes": torch.empty(0, dtype=torch.int64),
                "endpoint_max_codes": torch.empty(0, dtype=torch.int64),
                "mean_codes": torch.empty(0, dtype=torch.int64),
                "mean_levels": 0,
                "endpoint_global_min": 0.0,
                "endpoint_global_max": 0.0,
                "mean_global_min": 0.0,
                "mean_global_max": 0.0,
                "used_bits": 0.0,
                "objective": 0.0,
            }

        ranges = matrix.max(dim=0).values - matrix.min(dim=0).values
        means = matrix.mean(dim=0)
        sorted_indices = torch.argsort(ranges, descending=True)
        previous_plan: Optional[Dict[str, Any]] = None
        previous_objective = float("inf")
        endpoint_width = math.log2(float(self.endpoint_levels))

        for selected_m in self._candidate_values_for_budget(kept_count=kept_count, batch_size=batch_size, budget_bits=budget_bits):
            endpoint = self._endpoint_plan(matrix, sorted_indices, selected_m)
            mean_positions = sorted_indices[selected_m:]
            residual_count = int(mean_positions.numel())
            mean_values = means.index_select(0, mean_positions) if residual_count > 0 else means.new_empty((0,))
            mean_global_min = float(mean_values.min().item()) if residual_count > 0 else 0.0
            mean_global_max = float(mean_values.max().item()) if residual_count > 0 else 0.0
            allocation = self._optimal_level_allocation(
                kept_count=kept_count,
                batch_size=batch_size,
                budget_bits=budget_bits,
                endpoint_ranges=endpoint["ranges"],
                residual_count=residual_count,
                residual_mean_range=max(0.0, mean_global_max - mean_global_min),
                endpoint_overhead_bits=2 * selected_m * endpoint_width,
            )
            if allocation is None:
                continue

            indicator = torch.zeros(kept_count, dtype=torch.bool)
            if selected_m > 0:
                indicator.scatter_(0, endpoint["positions"], True)

            top_matrix = matrix.index_select(1, endpoint["positions"]) if selected_m > 0 else matrix[:, :0]
            top_codes = self._quantize_uniform_codes(top_matrix, endpoint["quant_mins"], endpoint["quant_maxs"], allocation["entry_levels"])
            if residual_count > 0:
                if mean_global_max - mean_global_min <= self.epsilon:
                    mean_codes = torch.zeros(residual_count, dtype=torch.int64)
                else:
                    mean_levels = max(2, int(allocation["mean_levels"]))
                    delta = (mean_global_max - mean_global_min) / max(1.0, float(mean_levels - 1))
                    mean_codes = ((mean_values - mean_global_min) / delta).round().to(dtype=torch.int64).clamp_(0, mean_levels - 1)
            else:
                mean_codes = torch.empty(0, dtype=torch.int64)

            objective = self._objective_for_levels(
                allocation["coefficients"],
                torch.cat([
                    allocation["entry_levels"].to(dtype=torch.float64),
                    torch.tensor([float(max(2, int(allocation["mean_levels"])))], dtype=torch.float64) if residual_count > 0 else torch.empty(0, dtype=torch.float64),
                ]),
            )
            objective += self._mean_error_term(ranges, mean_positions, batch_size)

            plan = {
                "kept_count": kept_count,
                "indicator": indicator,
                "two_stage_positions": endpoint["positions"].to(dtype=torch.int64),
                "mean_positions": mean_positions.to(dtype=torch.int64),
                "two_stage_codes": top_codes.to(dtype=_integral_dtype(max(1, max((int(level.item()) for level in allocation["entry_levels"]), default=1)))),
                "two_stage_levels": allocation["entry_levels"].to(dtype=torch.int64),
                "endpoint_min_codes": endpoint["codes_min"].to(dtype=_integral_dtype(self.endpoint_levels)),
                "endpoint_max_codes": endpoint["codes_max"].to(dtype=_integral_dtype(self.endpoint_levels)),
                "mean_codes": mean_codes.to(dtype=_integral_dtype(max(1, int(allocation["mean_levels"])))) if residual_count > 0 else mean_codes,
                "mean_levels": int(allocation["mean_levels"]),
                "endpoint_global_min": float(endpoint["global_min"]),
                "endpoint_global_max": float(endpoint["global_max"]),
                "mean_global_min": mean_global_min,
                "mean_global_max": mean_global_max,
                "used_bits": float(allocation["used_bits"]),
                "objective": float(objective),
            }
            if previous_plan is not None and float(plan["objective"]) > previous_objective:
                return previous_plan
            previous_plan = plan
            previous_objective = float(plan["objective"])

        if previous_plan is None:
            raise ValueError("SplitFC quantization budget is too small for the requested tensor")
        return previous_plan

    def _order_plan_for_wire(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        indicator = plan["indicator"].reshape(-1).to(dtype=torch.bool)
        ordered_two_stage = torch.nonzero(indicator, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        ordered_mean = torch.nonzero(~indicator, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        two_stage_positions = [int(pos) for pos in plan["two_stage_positions"].reshape(-1).tolist()]
        mean_positions = [int(pos) for pos in plan["mean_positions"].reshape(-1).tolist()]
        two_stage_map = {pos: idx for idx, pos in enumerate(two_stage_positions)}
        mean_map = {pos: idx for idx, pos in enumerate(mean_positions)}

        endpoint_min_codes = torch.tensor([int(plan["endpoint_min_codes"][two_stage_map[int(pos.item())]].item()) for pos in ordered_two_stage], dtype=torch.int64)
        endpoint_max_codes = torch.tensor([int(plan["endpoint_max_codes"][two_stage_map[int(pos.item())]].item()) for pos in ordered_two_stage], dtype=torch.int64)
        two_stage_levels = torch.tensor([int(plan["two_stage_levels"][two_stage_map[int(pos.item())]].item()) for pos in ordered_two_stage], dtype=torch.int64)
        if int(ordered_two_stage.numel()) > 0:
            two_stage_codes = torch.stack([plan["two_stage_codes"][:, two_stage_map[int(pos.item())]] for pos in ordered_two_stage], dim=1)
        else:
            two_stage_codes = torch.empty((plan["two_stage_codes"].shape[0], 0), dtype=torch.int64)
        mean_codes = torch.tensor([int(plan["mean_codes"][mean_map[int(pos.item())]].item()) for pos in ordered_mean], dtype=torch.int64) if int(ordered_mean.numel()) > 0 else torch.empty(0, dtype=torch.int64)
        return {
            **plan,
            "two_stage_positions": ordered_two_stage,
            "mean_positions": ordered_mean,
            "endpoint_min_codes": endpoint_min_codes,
            "endpoint_max_codes": endpoint_max_codes,
            "two_stage_levels": two_stage_levels,
            "two_stage_codes": two_stage_codes,
            "mean_codes": mean_codes,
        }

    def _pack_quantized_payload(self, *, shape: Tuple[int, ...], stream_id: int, dropout_mask: Optional[torch.Tensor], plan: Dict[str, Any], forward: bool) -> Tuple[bytes, int]:
        ordered = self._order_plan_for_wire(plan)
        writer = _BitWriter()
        writer.write_uint(_FORWARD_FLAG if forward else _BACKWARD_FLAG, 1)
        writer.write_uint(int(stream_id), 32)
        if forward:
            if dropout_mask is None:
                raise ValueError("Forward SplitFC payload requires dropout mask")
            writer.write_uint(len(shape), 8)
            for dim in shape:
                writer.write_uint(int(dim), 32)
            writer.write_bool_list(dropout_mask.reshape(-1).tolist())
        writer.write_bool_list(ordered["indicator"].reshape(-1).tolist())
        writer.write_float32(ordered["endpoint_global_min"])
        writer.write_float32(ordered["endpoint_global_max"])
        writer.write_float32(ordered["mean_global_min"])
        writer.write_float32(ordered["mean_global_max"])

        endpoint_width = _level_bit_width(self.endpoint_levels)
        for code in ordered["endpoint_min_codes"].reshape(-1).tolist():
            writer.write_uint(int(code), endpoint_width)
        for code in ordered["endpoint_max_codes"].reshape(-1).tolist():
            writer.write_uint(int(code), endpoint_width)

        entry_levels = ordered["two_stage_levels"].reshape(-1).tolist()
        for column_index, level in enumerate(entry_levels):
            width = _level_bit_width(level)
            for row_index in range(int(ordered["two_stage_codes"].shape[0])):
                writer.write_uint(int(ordered["two_stage_codes"][row_index, column_index].item()), width)

        mean_width = _level_bit_width(ordered["mean_levels"]) if int(ordered["mean_codes"].numel()) > 0 else 0
        for code in ordered["mean_codes"].reshape(-1).tolist():
            writer.write_uint(int(code), mean_width)
        return writer.finish(), int(writer.bit_count)

    def _build_plan_from_wire(self, *, shape: Tuple[int, ...], kept_count: int, indicator: torch.Tensor, endpoint_min_codes: torch.Tensor, endpoint_max_codes: torch.Tensor, endpoint_global_min: float, endpoint_global_max: float, mean_global_min: float, mean_global_max: float, batch_size: int, budget_bits: float) -> Dict[str, Any]:
        two_stage_positions = torch.nonzero(indicator, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        mean_positions = torch.nonzero(~indicator, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        selected_m = int(two_stage_positions.numel())
        if selected_m > 0 and endpoint_global_max - endpoint_global_min > self.epsilon:
            delta = (endpoint_global_max - endpoint_global_min) / float(self.endpoint_levels - 1)
            quant_mins = endpoint_global_min + endpoint_min_codes.to(dtype=torch.float32) * delta
            quant_maxs = endpoint_global_min + endpoint_max_codes.to(dtype=torch.float32) * delta
        elif selected_m > 0:
            quant_mins = torch.full((selected_m,), float(endpoint_global_min), dtype=torch.float32)
            quant_maxs = quant_mins.clone()
        else:
            quant_mins = torch.empty(0, dtype=torch.float32)
            quant_maxs = torch.empty(0, dtype=torch.float32)

        allocation = self._optimal_level_allocation(
            kept_count=kept_count,
            batch_size=batch_size,
            budget_bits=budget_bits,
            endpoint_ranges=(quant_maxs - quant_mins).clamp_min(0.0),
            residual_count=int(mean_positions.numel()),
            residual_mean_range=max(0.0, float(mean_global_max) - float(mean_global_min)),
            endpoint_overhead_bits=2.0 * float(selected_m) * math.log2(float(self.endpoint_levels)),
        )
        if allocation is None:
            raise ValueError("Failed to reconstruct SplitFC quantization plan from packed payload")
        return {
            "kept_count": kept_count,
            "indicator": indicator,
            "two_stage_positions": two_stage_positions,
            "mean_positions": mean_positions,
            "two_stage_levels": allocation["entry_levels"].to(dtype=torch.int64),
            "mean_levels": int(allocation["mean_levels"]),
            "endpoint_min_codes": endpoint_min_codes.to(dtype=torch.int64),
            "endpoint_max_codes": endpoint_max_codes.to(dtype=torch.int64),
            "endpoint_global_min": float(endpoint_global_min),
            "endpoint_global_max": float(endpoint_global_max),
            "mean_global_min": float(mean_global_min),
            "mean_global_max": float(mean_global_max),
            "shape": shape,
        }

    def _unpack_quantized_payload(self, payload: Payload) -> Tuple[Dict[str, Any], Tuple[int, ...], str, bool]:
        raw_payload = payload.get("q")
        if not isinstance(raw_payload, (bytes, bytearray, memoryview)):
            raise TypeError("SplitFC payload['q'] must be packed bytes")
        bit_count = int(payload.get("actual_bit_count") or 0)
        if bit_count <= 0:
            raise ValueError("SplitFC payload must include actual_bit_count")
        reader = _BitReader(bytes(raw_payload), bit_count)
        forward = reader.read_uint(1) == _FORWARD_FLAG
        stream_id = reader.read_uint(32)
        cache_id = self._cache_id_for_stream(stream_id)

        if forward:
            ndim = reader.read_uint(8)
            shape = tuple(int(reader.read_uint(32)) for _ in range(int(ndim)))
            column_count = self._column_count(shape)
            dropout_mask = reader.read_bool_tensor(column_count)
            keep_indices = torch.nonzero(dropout_mask, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        else:
            cached = self._state_cache.get(cache_id)
            if cached is None:
                raise KeyError(f"Missing cached SplitFC forward state for {cache_id!r}")
            shape = tuple(int(dim) for dim in cached["shape"])
            dropout_mask = cached["keep_mask"].to(dtype=torch.bool)
            keep_indices = cached["keep_indices"].to(dtype=torch.int64)

        kept_count = int(keep_indices.numel())
        indicator = reader.read_bool_tensor(kept_count)
        endpoint_global_min = reader.read_float32()
        endpoint_global_max = reader.read_float32()
        mean_global_min = reader.read_float32()
        mean_global_max = reader.read_float32()
        selected_m = int(indicator.sum().item())
        endpoint_width = _level_bit_width(self.endpoint_levels)
        endpoint_min_codes = torch.tensor([reader.read_uint(endpoint_width) for _ in range(selected_m)], dtype=torch.int64)
        endpoint_max_codes = torch.tensor([reader.read_uint(endpoint_width) for _ in range(selected_m)], dtype=torch.int64)

        batch_size = 1 if len(shape) <= 1 else int(shape[0])
        original_columns = self._column_count(shape)
        budget_bits = self._actual_available_bits(batch_size=batch_size, original_columns=original_columns, forward=forward, shape=shape)
        plan = self._build_plan_from_wire(
            shape=shape,
            kept_count=kept_count,
            indicator=indicator,
            endpoint_min_codes=endpoint_min_codes,
            endpoint_max_codes=endpoint_max_codes,
            endpoint_global_min=endpoint_global_min,
            endpoint_global_max=endpoint_global_max,
            mean_global_min=mean_global_min,
            mean_global_max=mean_global_max,
            batch_size=batch_size,
            budget_bits=budget_bits,
        )

        entry_levels = plan["two_stage_levels"].reshape(-1).tolist()
        if entry_levels:
            entry_codes = torch.empty((batch_size, len(entry_levels)), dtype=torch.int64)
            for column_index, level in enumerate(entry_levels):
                width = _level_bit_width(level)
                for row_index in range(batch_size):
                    entry_codes[row_index, column_index] = int(reader.read_uint(width))
        else:
            entry_codes = torch.empty((batch_size, 0), dtype=torch.int64)

        residual_count = int(plan["mean_positions"].numel())
        mean_width = _level_bit_width(plan["mean_levels"]) if residual_count > 0 else 0
        mean_codes = torch.tensor([reader.read_uint(mean_width) for _ in range(residual_count)], dtype=torch.int64) if residual_count > 0 else torch.empty(0, dtype=torch.int64)
        plan["two_stage_codes"] = entry_codes.to(dtype=_integral_dtype(max(1, max(entry_levels, default=1))))
        plan["mean_codes"] = mean_codes.to(dtype=_integral_dtype(max(1, int(plan["mean_levels"])))) if residual_count > 0 else mean_codes
        if forward:
            self._state_cache[cache_id] = {
                "shape": shape,
                "keep_mask": dropout_mask.to(device="cpu", dtype=torch.bool),
                "keep_indices": keep_indices.to(device="cpu", dtype=torch.int64),
            }
        return plan, shape, cache_id, forward

    def _dequantize_quantized_matrix(self, payload: Payload, *, batch_size: int, kept_count: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        if kept_count == 0:
            return torch.zeros((batch_size, 0), dtype=dtype, device=device)
        out = torch.zeros((batch_size, kept_count), dtype=dtype, device=device)
        two_stage_positions = payload["two_stage_positions"].to(device=device, dtype=torch.int64)
        if int(two_stage_positions.numel()) > 0:
            endpoint_global_min = float(payload["endpoint_global_min"])
            endpoint_global_max = float(payload["endpoint_global_max"])
            min_codes = payload["endpoint_min_codes"].to(dtype=torch.int64)
            max_codes = payload["endpoint_max_codes"].to(dtype=torch.int64)
            if endpoint_global_max - endpoint_global_min <= self.epsilon:
                lowers = torch.full((int(two_stage_positions.numel()),), endpoint_global_min, dtype=dtype)
                uppers = lowers.clone()
            else:
                delta = (endpoint_global_max - endpoint_global_min) / float(self.endpoint_levels - 1)
                lowers = endpoint_global_min + min_codes.to(dtype=dtype) * delta
                uppers = endpoint_global_min + max_codes.to(dtype=dtype) * delta
            values = self._dequantize_uniform_codes(
                payload["two_stage_codes"].to(device=device, dtype=torch.int64),
                lowers.to(device=device, dtype=dtype),
                uppers.to(device=device, dtype=dtype),
                payload["two_stage_levels"].to(device=device, dtype=torch.int64),
                dtype,
            )
            out.index_copy_(1, two_stage_positions, values)

        mean_positions = payload["mean_positions"].to(device=device, dtype=torch.int64)
        if int(mean_positions.numel()) > 0:
            if payload["mean_global_max"] - payload["mean_global_min"] <= self.epsilon:
                means = torch.full((int(mean_positions.numel()),), float(payload["mean_global_min"]), dtype=dtype, device=device)
            else:
                levels = float(max(2, int(payload["mean_levels"])))
                delta = (float(payload["mean_global_max"]) - float(payload["mean_global_min"])) / max(1.0, levels - 1.0)
                means = float(payload["mean_global_min"]) + payload["mean_codes"].to(device=device, dtype=dtype) * delta
            out.index_copy_(1, mean_positions, means.unsqueeze(0).expand(batch_size, -1))
        return out

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")
        matrix, shape, batch_size = self._matrix_view(x)
        original_columns = int(matrix.shape[1])
        cache_id = get_split_fc_cache_id(x)

        if cache_id is not None:
            cached = self._state_cache.pop(cache_id, None)
            if cached is None:
                raise KeyError(f"Missing cached SplitFC forward state for {cache_id!r}")
            if tuple(shape) != tuple(cached["shape"]):
                raise ValueError("Backward gradient shape does not match cached SplitFC forward shape")
            keep_indices = cached["keep_indices"]
            kept = matrix.index_select(1, keep_indices) if int(keep_indices.numel()) > 0 else matrix[:, :0]
            plan = self._plan_quantization(
                kept,
                budget_bits=self._actual_available_bits(batch_size=batch_size, original_columns=original_columns, forward=False, shape=shape),
            )
            packed_bytes, bit_count = self._pack_quantized_payload(
                shape=shape,
                stream_id=self._stream_id_from_cache_id(cache_id),
                dropout_mask=None,
                plan=plan,
                forward=False,
            )
            return {
                "codec": "split_fc",
                "q": packed_bytes,
                "actual_bit_count": int(bit_count),
                "paper_bit_count": float(plan["used_bits"]),
            }

        keep_mask, kept_matrix, keep_probs = self._sample_dropout_mask(matrix, shape)
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1).to(dtype=torch.int64)
        cache_id = self._next_cache_id()
        self._state_cache[cache_id] = {
            "shape": shape,
            "keep_mask": keep_mask.to(dtype=torch.bool),
            "keep_indices": keep_indices,
            "keep_probs": keep_probs.to(dtype=torch.float32),
        }
        plan = self._plan_quantization(
            kept_matrix,
            budget_bits=self._actual_available_bits(batch_size=batch_size, original_columns=original_columns, forward=True, shape=shape),
        )
        packed_bytes, bit_count = self._pack_quantized_payload(
            shape=shape,
            stream_id=self._stream_id_from_cache_id(cache_id),
            dropout_mask=keep_mask.to(dtype=torch.bool),
            plan=plan,
            forward=True,
        )
        return {
            "codec": "split_fc",
            "q": packed_bytes,
            "actual_bit_count": int(bit_count),
            "paper_bit_count": float(original_columns) + float(plan["used_bits"]),
        }

    def decode(self, payload: Payload, *, device: Optional[torch.device | str] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not isinstance(payload, dict):
            raise TypeError(f"decode expects a dict payload, got {type(payload)!r}")
        if "q" not in payload:
            raise KeyError("Invalid SplitFC payload, missing 'q'")
        target_device = device if device is not None else "cpu"
        quantized, shape, cache_id, forward = self._unpack_quantized_payload(payload)
        batch_size = 1 if len(shape) <= 1 else int(shape[0])
        column_count = self._column_count(shape)

        if forward:
            keep_indices = self._state_cache[cache_id]["keep_indices"].to(dtype=torch.int64)
            kept = self._dequantize_quantized_matrix(
                quantized,
                batch_size=batch_size,
                kept_count=int(keep_indices.numel()),
                dtype=dtype,
                device=target_device,
            )
            full = torch.zeros((batch_size, column_count), dtype=dtype, device=target_device)
            if int(keep_indices.numel()) > 0:
                full.index_copy_(1, keep_indices.to(device=target_device), kept)
            return set_split_fc_cache_id(full.reshape(shape), cache_id)

        cached = self._state_cache.pop(cache_id, None)
        if cached is None:
            raise KeyError(f"Missing cached SplitFC forward state for {cache_id!r}")
        keep_indices = cached["keep_indices"].to(dtype=torch.int64)
        kept = self._dequantize_quantized_matrix(
            quantized,
            batch_size=batch_size,
            kept_count=int(keep_indices.numel()),
            dtype=dtype,
            device=target_device,
        )
        full = torch.zeros((batch_size, column_count), dtype=dtype, device=target_device)
        if int(keep_indices.numel()) > 0:
            full.index_copy_(1, keep_indices.to(device=target_device), kept)
        return set_split_fc_cache_id(full.reshape(shape), cache_id)
