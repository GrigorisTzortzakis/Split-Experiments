from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from compression.quantization.bit_packing import pack_uint4, shape_numel, unpack_uint4

Payload = Dict[str, Any]


@dataclass(frozen=True)
class NonUniformLoydCodebookUInt8Codec:
    """Non-uniform Loyd/Lloyd-Max optimized 8-bit quantizer with a codebook."""

    range_mode: str = "minmax"
    max_iter: int = 50
    tol: float = 1e-4
    num_bits: int = 8

    def _codec_name(self) -> str:
        return f"non_uniform_loyd_codebook_uint{int(self.num_bits)}"

    def _levels(self) -> int:
        bits = int(self.num_bits)
        if bits not in (4, 8):
            raise ValueError("num_bits must be either 8 or 4")
        return 1 << bits

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
        x_min, x_max = self._compute_endpoints(x_cpu)

        levels = self._levels()
        if x_max == x_min:
            codebook = torch.full((levels,), x_min, dtype=torch.float32)
            idx_u8 = torch.zeros_like(x_cpu, dtype=torch.uint8)
            if int(self.num_bits) == 4:
                q, original_numel = pack_uint4(idx_u8)
            else:
                q = idx_u8
                original_numel = int(idx_u8.numel())
            return {
                "codec": self._codec_name(),
                "q": q,
                "codebook": codebook,
                "shape": tuple(x_cpu.shape),
                "range_mode": str(self.range_mode).strip().lower(),
                "max_iter": int(self.max_iter),
                "tol": float(self.tol),
                "num_bits": int(self.num_bits),
                "num_values": original_numel,
            }

        x_flat = x_cpu.reshape(-1)
        codebook = self._fit_codebook(x_flat, x_min, x_max)
        boundaries = self._decision_boundaries(codebook)
        idx = torch.bucketize(x_flat, boundaries).reshape(x_cpu.shape)
        idx_u8 = idx.to(dtype=torch.uint8)

        if int(self.num_bits) == 4:
            q, original_numel = pack_uint4(idx_u8)
        else:
            q = idx_u8
            original_numel = int(idx_u8.numel())

        return {
            "codec": self._codec_name(),
            "q": q,
            "codebook": codebook.to(dtype=torch.float32),
            "shape": tuple(x_cpu.shape),
            "range_mode": str(self.range_mode).strip().lower(),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
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
        if not isinstance(codebook, torch.Tensor) or codebook.dtype != torch.float32 or codebook.ndim != 1:
            raise TypeError("payload['codebook'] must be a 1D torch.float32 Tensor")
        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits not in (4, 8):
            raise ValueError("payload num_bits must be either 8 or 4")
        if int(codebook.numel()) != (1 << num_bits):
            raise ValueError(f"payload['codebook'] must have {1 << num_bits} entries")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"

        if num_bits == 4:
            original_numel = int(payload.get("num_values", shape_numel(shape) if shape is not None else 0))
            q_dev = unpack_uint4(q, original_numel, device=target_device)
            if shape is not None:
                q_dev = q_dev.reshape(tuple(int(s) for s in shape))
        else:
            q_dev = q.to(device=target_device)
        cb_dev = codebook.to(device=target_device)

        idx = q_dev.to(dtype=torch.int64)
        out = cb_dev[idx]

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)


LloydMaxCodebookUInt8Codec = NonUniformLoydCodebookUInt8Codec