from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

Payload = Dict[str, Any]


@dataclass(frozen=True)
class TruncationInt8Codec:
    """Dynamic symmetric int8 quantization with per-tensor scaling.

    - Compute a *multiplier* scale per tensor: S = 127 / max(abs(x))
    - Quantize: q = round(x * S)
    - Clip: q in [-128, 127]
    - Store: q as int8
    - Send S (one FP32) alongside q
    - Dequantize: x_hat = q / S

    Optional override:
    - If `fixed_scale` is set (not None), that multiplier is used instead of
      computing S from the tensor.
    """

    fixed_scale: Optional[float] = None

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")
        if self.fixed_scale == 0:
            raise ValueError("fixed_scale must be non-zero")

        x_detached = x.detach().to(dtype=torch.float32)
        x_cpu = x_detached.to(device="cpu")

        if self.fixed_scale is None:
            max_abs = float(x_cpu.abs().max().item())
            scale = 1.0 if max_abs == 0.0 else (127.0 / max_abs)
        else:
            scale = float(self.fixed_scale)

        q = torch.round(x_cpu * scale)
        q = torch.clamp(q, -128, 127)
        q = q.to(dtype=torch.int8)

        return {
            "codec": "dynamic_symmetric_int8",
            "q": q,
            "scale": float(scale),
            "shape": tuple(x_cpu.shape),
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
        if not isinstance(q, torch.Tensor) or q.dtype != torch.int8:
            raise TypeError("payload['q'] must be a torch.int8 Tensor")

        scale = float(payload.get("scale", 0.0))
        if scale == 0.0:
            raise ValueError("payload scale must be non-zero")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"
        q_dev = q.to(device=target_device)

        x = q_dev.to(dtype=torch.float32) / scale

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            x = x.reshape(tuple(int(s) for s in shape))

        return x.to(dtype=dtype)
