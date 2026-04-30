from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from compression.quantization.bit_packing import pack_uint4, shape_numel, unpack_uint4

Payload = Dict[str, Any]


@dataclass(frozen=True)
class MuLawCodebookUInt8Codec:
    """Non-uniform 8-bit (256-level) quantizer with explicit codewords.

    This codec uses μ-law companding to create *non-uniform* reconstruction
    levels while still storing a single byte per value (uint8 indices):

    - Normalize x to [-1, 1] based on selected range.
    - Compand with μ-law: y = sign(x) * log(1 + μ|x|) / log(1 + μ).
    - Uniformly quantize y into 256 bins => q in [0, 255] (uint8).
    - Decode via a 256-entry codebook generated from inverse μ-law.

    Range modes:
    - range_mode="minmax" (default): endpoints are [min(x), max(x)] (centered normalization).
    - range_mode="symmetric": endpoints are [-a, a], a=max(abs(x)).

    Notes:
    - Codebook is generated per encoded tensor (per call to encode).
    - If the range collapses (all values equal), q is all zeros and the codebook is constant.
    """

    range_mode: str = "minmax"
    mu: float = 255.0
    num_bits: int = 8

    def _codec_name(self) -> str:
        return f"mulaw_codebook_uint{int(self.num_bits)}"

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

    def _compand(self, x_norm: torch.Tensor) -> torch.Tensor:
        mu = float(self.mu)
        if mu <= 0:
            raise ValueError("mu must be > 0")

        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        ax = x_norm.abs()
        log_mu = torch.log1p(torch.tensor(mu, dtype=torch.float32))
        return torch.sign(x_norm) * (torch.log1p(mu * ax) / log_mu)

    def _expand(self, y: torch.Tensor) -> torch.Tensor:
        mu = float(self.mu)
        if mu <= 0:
            raise ValueError("mu must be > 0")

        y = torch.clamp(y, -1.0, 1.0)
        ay = y.abs()
        log_mu = torch.log1p(torch.tensor(mu, dtype=torch.float32))
        return torch.sign(y) * (torch.expm1(ay * log_mu) / mu)

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
                "mu": float(self.mu),
                "num_bits": int(self.num_bits),
                "num_values": original_numel,
            }

        mid = 0.5 * (x_min + x_max)
        half = 0.5 * (x_max - x_min)

        x_norm = (x_cpu - mid) / half
        y = self._compand(x_norm)

        # Uniformly quantize y in [-1, 1] into 256 indices.
        u = (y + 1.0) * 0.5 * float(levels - 1)
        idx = torch.round(u)
        idx = torch.clamp(idx, 0, levels - 1).to(dtype=torch.int64)
        idx_u8 = idx.to(dtype=torch.uint8)

        # Build explicit codebook by inverse companding uniformly-spaced y levels.
        y_levels = torch.linspace(-1.0, 1.0, steps=levels, dtype=torch.float32)
        x_levels_norm = self._expand(y_levels)
        codebook = (mid + half * x_levels_norm).to(dtype=torch.float32)

        if int(self.num_bits) == 4:
            q, original_numel = pack_uint4(idx_u8)
        else:
            q = idx_u8
            original_numel = int(idx_u8.numel())

        return {
            "codec": self._codec_name(),
            "q": q,
            "codebook": codebook,  # torch.float32 on CPU, shape (256,)
            "shape": tuple(x_cpu.shape),
            "range_mode": str(self.range_mode).strip().lower(),
            "mu": float(self.mu),
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
