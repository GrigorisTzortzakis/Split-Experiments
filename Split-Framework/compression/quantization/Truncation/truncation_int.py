from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

Payload = Dict[str, Any]

SUPPORTED_NUM_BITS = (2, 3, 4, 6, 8)


@dataclass(frozen=True)
class TruncationIntCodec:
    """Arithmetic conversion to signed integers with rounding and no scaling.

    Encoding:
    - q = round(x)
    - clip q to the signed range for the selected bit width
    - store q as int8

    Decoding:
    - x_hat = q cast back to float32 (no scaling)
    """

    num_bits: int = 8
    granularity: str = "per_tensor"
    channel_axis: int = 1
    group_size: int = 32

    def _quant_bounds(self) -> tuple[int, int]:
        bits = int(self.num_bits)
        if bits not in SUPPORTED_NUM_BITS:
            raise ValueError(f"num_bits must be one of {SUPPORTED_NUM_BITS}")
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_detached = x.detach().to(dtype=torch.float32)
        x_cpu = x_detached.to(device="cpu")

        qmin, qmax = self._quant_bounds()
        q = torch.round(x_cpu)
        q = torch.clamp(q, qmin, qmax).to(dtype=torch.int8)

        return {
            "codec": "trunc_noscale_int",
            "q": q,
            "shape": tuple(x_cpu.shape),
            "num_bits": int(self.num_bits),
            "granularity": str(self.granularity),
            "channel_axis": int(self.channel_axis),
            "group_size": int(self.group_size),
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

        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits not in SUPPORTED_NUM_BITS:
            raise ValueError(f"payload num_bits must be one of {SUPPORTED_NUM_BITS}")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"
        q_dev = q.to(device=target_device)
        out = q_dev.to(dtype=torch.float32)

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)



TruncationInt8Codec = TruncationIntCodec

