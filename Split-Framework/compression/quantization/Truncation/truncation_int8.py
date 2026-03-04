from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

Payload = Dict[str, Any]


@dataclass(frozen=True)
class TruncationInt8Codec:
    """Bit-truncation codec: keep only the top 8 bits of FP32.

    This codec does *not* use scaling.

    Encoding:
    - Interpret FP32 values as their raw IEEE-754 32-bit patterns.
    - Keep only the most-significant byte (bits 31..24).
    - Store that byte as int8.

    Decoding:
    - Restore that byte to bits 31..24 and fill the remaining 24 bits with zeros.
    - Reinterpret back to float32.

    This is a very aggressive truncation (range/precision are severely reduced),
    but matches the requested "just cut off bits" behavior.
    """

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_detached = x.detach().to(dtype=torch.float32)
        x_cpu = x_detached.to(device="cpu")

        bits = x_cpu.contiguous().view(torch.int32)
        top_u8 = ((bits >> 24) & 0xFF).to(dtype=torch.uint8)
        q = top_u8.view(torch.int8)

        return {
            "codec": "trunc_bits_int8",
            "q": q,  # torch.int8 on CPU
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

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"

        q_dev = q.to(device=target_device)
        q_u8 = q_dev.view(torch.uint8)
        bits32 = (q_u8.to(dtype=torch.int32) << 24)
        out = bits32.view(torch.float32)

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)

