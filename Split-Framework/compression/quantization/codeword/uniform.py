from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

Payload = Dict[str, Any]


@dataclass(frozen=True)
class UniformCodebookUInt8Codec:
    """Uniform 8-bit quantizer with explicit codewords (codebook).

    This is the "normal" uniform quantizer (not FP8 and not bit-truncation):
    - Build a uniform codebook of 256 real-valued codewords.
    - Quantize each value to the nearest codeword index (0..255) stored as uint8.
    - Decode by table lookup.

    Range modes:
    - `range_mode="minmax"` (default): codebook spans [min(x), max(x)] (asymmetric).
    - `range_mode="symmetric"`: codebook spans [-a, a] where a=max(abs(x)).

    Payload format:
    - codec: "uniform_codebook_uint8"
    - q: torch.uint8 tensor of indices on CPU
    - codebook: torch.float32 tensor of shape (256,) on CPU
    - shape: original shape
    - range_mode: stored for debugging/repro

    Notes:
    - Input is detached and encoded on CPU as float32 (like other codecs here).
    - If the range collapses (all values equal), q is all zeros and the codebook is constant.
    """

    range_mode: str = "minmax"

    def _codec_name(self) -> str:
        return "uniform_codebook_uint8"

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

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = self._as_cpu_f32(x)
        x_min, x_max = self._compute_endpoints(x_cpu)

        levels = 256
        if x_max == x_min:
            codebook = torch.full((levels,), x_min, dtype=torch.float32)
            q = torch.zeros_like(x_cpu, dtype=torch.uint8)
        else:
            codebook = torch.linspace(x_min, x_max, steps=levels, dtype=torch.float32)
            step = (x_max - x_min) / float(levels - 1)

            # Index to nearest codeword
            idx = torch.round((x_cpu - x_min) / step)
            idx = torch.clamp(idx, 0, levels - 1).to(dtype=torch.int64)
            q = idx.to(dtype=torch.uint8)

        return {
            "codec": self._codec_name(),
            "q": q,  # torch.uint8 on CPU
            "codebook": codebook,  # torch.float32 on CPU
            "shape": tuple(x_cpu.shape),
            "range_mode": str(self.range_mode).strip().lower(),
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
        if int(codebook.numel()) != 256:
            raise ValueError("payload['codebook'] must have 256 entries")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"

        q_dev = q.to(device=target_device)
        cb_dev = codebook.to(device=target_device)

        # Table lookup
        idx = q_dev.to(dtype=torch.int64)
        out = cb_dev[idx]

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)
