from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from compression.quantization.bit_packing import pack_int4, shape_numel, unpack_int4

Payload = Dict[str, Any]


@dataclass(frozen=True)
class TruncationInt8Codec:
    """Arithmetic conversion to int8 with rounding and *no scaling*.

    Encoding:
    - q = round(x)
    - clip q to [-128, 127]
    - store q as int8

    Decoding:
    - x_hat = q cast back to float32 (no scaling)
    """

    num_bits: int = 8

    def _quant_bounds(self) -> tuple[int, int]:
        bits = int(self.num_bits)
        if bits == 8:
            return -128, 127
        if bits == 4:
            return -8, 7
        raise ValueError("num_bits must be either 8 or 4")

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_detached = x.detach().to(dtype=torch.float32)
        x_cpu = x_detached.to(device="cpu")

        qmin, qmax = self._quant_bounds()
        q = torch.round(x_cpu)
        q = torch.clamp(q, qmin, qmax).to(dtype=torch.int8)

        if int(self.num_bits) == 4:
            stored_q, original_numel = pack_int4(q)
        else:
            stored_q = q
            original_numel = int(q.numel())

        return {
            "codec": f"trunc_noscale_int{int(self.num_bits)}",
            "q": stored_q,
            "shape": tuple(x_cpu.shape),
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

        q = payload["q"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")

        num_bits = int(payload.get("num_bits", self.num_bits))
        if num_bits == 8:
            if q.dtype != torch.int8:
                raise TypeError("8-bit payload['q'] must be a torch.int8 Tensor")
        elif num_bits == 4:
            if q.dtype != torch.uint8:
                raise TypeError("4-bit payload['q'] must be a packed torch.uint8 Tensor")
        else:
            raise ValueError("payload num_bits must be either 8 or 4")

        shape = payload.get("shape", None)
        target_device = device if device is not None else "cpu"

        if num_bits == 4:
            original_numel = int(payload.get("num_values", shape_numel(shape) if shape is not None else 0))
            q_dev = unpack_int4(q, original_numel, device=target_device)
        else:
            q_dev = q.to(device=target_device)
        out = q_dev.to(dtype=torch.float32)

        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)

