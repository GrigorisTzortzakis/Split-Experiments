from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from compression.quantization.bit_packing import pack_uint4, shape_numel, unpack_uint4

Payload = Dict[str, Any]


@dataclass(frozen=True)
class NonUniformLoydPerChannelCodebookUInt8Codec:
    range_mode: str = "minmax"
    max_iter: int = 50
    tol: float = 1e-4
    channel_axis: int = 1
    num_bits: int = 8

    def _codec_name(self) -> str:
        return f"non_uniform_loyd_per_channel_codebook_uint{int(self.num_bits)}"

    def _levels(self) -> int:
        bits = int(self.num_bits)
        if bits not in (4, 8):
            raise ValueError("num_bits must be either 8 or 4")
        return 1 << bits

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

    def _compute_endpoints(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mode = str(self.range_mode).strip().lower()
        if mode in ("minmax", "asymmetric", "affine"):
            x_min = x_flat.min(dim=1).values if x_flat.numel() else torch.zeros((x_flat.shape[0],), dtype=torch.float32)
            x_max = x_flat.max(dim=1).values if x_flat.numel() else torch.zeros((x_flat.shape[0],), dtype=torch.float32)
            return x_min, x_max
        if mode in ("symmetric", "sym"):
            a = x_flat.abs().max(dim=1).values if x_flat.numel() else torch.zeros((x_flat.shape[0],), dtype=torch.float32)
            return -a, a
        raise ValueError(f"Unsupported range_mode: {self.range_mode!r}")

    @staticmethod
    def _move_channel_first(x_cpu: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, Sequence[int], Sequence[int]]:
        if x_cpu.ndim == 0:
            return x_cpu.reshape(1), (0,), (0,)

        perm = (channel_axis, *[idx for idx in range(x_cpu.ndim) if idx != channel_axis])
        inverse_perm = tuple(perm.index(idx) for idx in range(len(perm)))
        return x_cpu.permute(perm).contiguous(), perm, inverse_perm

    @staticmethod
    def _decision_boundaries(codebook: torch.Tensor) -> torch.Tensor:
        return 0.5 * (codebook[:-1] + codebook[1:])

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
        original_shape = tuple(x_cpu.shape)
        resolved_axis = self._resolve_channel_axis(x_cpu)

        x_channel_first, _perm, inverse_perm = self._move_channel_first(x_cpu, resolved_axis)
        x_flat = x_channel_first.reshape(x_channel_first.shape[0], -1)
        x_min, x_max = self._compute_endpoints(x_flat)
        levels = self._levels()

        q_flat = torch.zeros_like(x_flat, dtype=torch.uint8)
        codebook = torch.zeros((x_flat.shape[0], levels), dtype=torch.float32)

        for channel_idx in range(x_flat.shape[0]):
            channel_values = x_flat[channel_idx]
            channel_min = float(x_min[channel_idx].item())
            channel_max = float(x_max[channel_idx].item())
            if channel_max == channel_min:
                codebook[channel_idx] = channel_min
                continue

            fitted = self._fit_codebook(channel_values, channel_min, channel_max)
            boundaries = self._decision_boundaries(fitted)
            codebook[channel_idx] = fitted
            q_flat[channel_idx] = torch.bucketize(channel_values, boundaries).to(dtype=torch.uint8)

        q_channel_first = q_flat.reshape(x_channel_first.shape)
        if x_cpu.ndim == 0:
            q_indices = q_channel_first.reshape(())
        else:
            q_indices = q_channel_first.permute(inverse_perm).contiguous()

        if int(self.num_bits) == 4:
            q, original_numel = pack_uint4(q_indices.to(device="cpu", dtype=torch.uint8))
        else:
            q = q_indices
            original_numel = int(q_indices.numel())

        return {
            "codec": self._codec_name(),
            "q": q,
            "codebook": codebook.to(dtype=torch.float32),
            "shape": original_shape,
            "range_mode": str(self.range_mode).strip().lower(),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
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
        if "codebook" not in payload:
            raise KeyError("Invalid payload, missing key: 'codebook'")

        q = payload["q"]
        if not isinstance(q, torch.Tensor) or q.dtype != torch.uint8:
            raise TypeError("payload['q'] must be a torch.uint8 Tensor")

        codebook = payload["codebook"]
        if not isinstance(codebook, torch.Tensor) or codebook.dtype != torch.float32 or codebook.ndim != 2:
            raise TypeError("payload['codebook'] must be a 2D torch.float32 Tensor")
        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits not in (4, 8):
            raise ValueError("payload num_bits must be either 8 or 4")
        if int(codebook.shape[1]) != (1 << num_bits):
            raise ValueError(f"payload['codebook'] must have shape (channels, {1 << num_bits})")

        shape = payload.get("shape", None)
        channel_axis = int(payload.get("channel_axis", 0 if q.ndim < 2 else 1))
        target_device = device if device is not None else "cpu"

        if num_bits == 4:
            original_numel = int(payload.get("num_values", shape_numel(shape) if shape is not None else 0))
            q_dev = unpack_uint4(q, original_numel, device=target_device)
            if shape is not None:
                q_dev = q_dev.reshape(tuple(int(s) for s in shape))
        else:
            q_dev = q.to(device=target_device)
        cb_dev = codebook.to(device=target_device)

        if q_dev.ndim == 0:
            if int(cb_dev.shape[0]) != 1:
                raise ValueError("scalar payload must have a single channel codebook")
            out = cb_dev[0, q_dev.reshape(1).to(dtype=torch.int64)].reshape(())
            return out.to(dtype=dtype)

        if channel_axis < 0:
            channel_axis += q_dev.ndim
        if channel_axis < 0 or channel_axis >= q_dev.ndim:
            raise ValueError(f"payload channel_axis {channel_axis} out of range for tensor rank {q_dev.ndim}")

        perm = (channel_axis, *[idx for idx in range(q_dev.ndim) if idx != channel_axis])
        inverse_perm = tuple(perm.index(idx) for idx in range(len(perm)))
        q_channel_first = q_dev.permute(perm).contiguous()
        q_flat = q_channel_first.reshape(q_channel_first.shape[0], -1).to(dtype=torch.int64)

        if int(cb_dev.shape[0]) != int(q_flat.shape[0]):
            raise ValueError("payload codebook channel count does not match tensor channel count")

        out_flat = torch.gather(cb_dev, dim=1, index=q_flat)
        out_channel_first = out_flat.reshape(q_channel_first.shape)
        out = out_channel_first.permute(inverse_perm).contiguous()

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)