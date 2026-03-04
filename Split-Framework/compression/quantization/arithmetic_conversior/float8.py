from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

Payload = Dict[str, Any]


@dataclass(frozen=True)
class TruncationFloat8Codec:
    """Software FP8 (1 byte) activation compression.

    This is a true floating-point *encoding* (not int quantization):
    - 1 sign bit
    - small exponent field
    - small mantissa field

    Supported layout:
    - E4M3: 4 exponent bits, 3 mantissa bits (bias=7)

    Notes:
    - Encodes to a torch.uint8 tensor (packed FP8 bits) on CPU.
    - Decodes back to float32 by default (or requested dtype/device).
    - Handles zeros, subnormals, inf, nan.

    Layout name must be E4M3 (aliases accepted: "e4m3", "fp8_e4m3", "e4m3fn").
    """

    layout: str = "e4m3"

    def _layout_params(self) -> tuple[int, int, int, int]:
        layout = str(self.layout).strip().lower()
        if layout in ("e4m3", "fp8_e4m3", "e4m3fn"):
            exp_bits, mant_bits, bias = 4, 3, 7
        else:
            raise ValueError(
                f"Unsupported FP8 layout: {self.layout!r}. Only E4M3 is supported in this codec."
            )

        max_exp = (1 << exp_bits) - 1
        return exp_bits, mant_bits, bias, max_exp

    def _codec_name(self) -> str:
        return "fp8_e4m3"

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        exp_bits, mant_bits, bias, max_exp = self._layout_params()

        x_detached = x.detach().to(dtype=torch.float32)
        x_cpu = x_detached.to(device="cpu")

        sign = (x_cpu < 0).to(dtype=torch.uint8)
        ax = x_cpu.abs()

        q = torch.zeros_like(ax, dtype=torch.uint8)

        is_nan = torch.isnan(ax)
        is_inf = torch.isinf(ax)
        is_zero = ax == 0

        # Reserve exp=max_exp for inf/nan
        if torch.any(is_inf):
            q_inf = (torch.tensor(max_exp, dtype=torch.uint8) << mant_bits)
            q = torch.where(is_inf, q_inf, q)

        if torch.any(is_nan):
            q_nan = (torch.tensor(max_exp, dtype=torch.uint8) << mant_bits) | torch.tensor(1, dtype=torch.uint8)
            q = torch.where(is_nan, q_nan, q)

        # Finite, non-zero values
        finite_nz = (~is_nan) & (~is_inf) & (~is_zero)
        if torch.any(finite_nz):
            ax_f = ax[finite_nz]

            # Use frexp to get exponent efficiently: ax = m * 2**e, m in [0.5, 1)
            m, e = torch.frexp(ax_f)
            exp_unbiased = e.to(dtype=torch.int32) - 1

            frac = m * 2.0 - 1.0  # in [0, 1)
            mant = torch.round(frac * float(1 << mant_bits)).to(dtype=torch.int32)

            # Handle mantissa rounding overflow (carry into exponent)
            mant_over = mant >= (1 << mant_bits)
            if torch.any(mant_over):
                mant = torch.where(mant_over, torch.zeros_like(mant), mant)
                exp_unbiased = torch.where(mant_over, exp_unbiased + 1, exp_unbiased)

            exp_biased = exp_unbiased + bias

            # Classify
            normal = exp_biased >= 1
            normal = normal & (exp_biased <= (max_exp - 1))
            overflow = exp_biased > (max_exp - 1)
            subnormal = exp_biased <= 0

            out_exp = torch.zeros_like(exp_biased, dtype=torch.int32)
            out_mant = torch.zeros_like(mant, dtype=torch.int32)

            # Normal numbers
            if torch.any(normal):
                out_exp = torch.where(normal, exp_biased, out_exp)
                out_mant = torch.where(normal, mant, out_mant)

            # Subnormals: exponent field = 0, mantissa encodes value
            # ax = (mant / 2**mant_bits) * 2**(1-bias)
            # => mant = ax * 2**(bias-1+mant_bits)
            if torch.any(subnormal):
                scale = float(2 ** (bias - 1 + mant_bits))
                mant_sub = torch.round(ax_f * scale).to(dtype=torch.int32)
                mant_sub = torch.clamp(mant_sub, 0, (1 << mant_bits) - 1)
                out_exp = torch.where(subnormal, torch.zeros_like(out_exp), out_exp)
                out_mant = torch.where(subnormal, mant_sub, out_mant)

            # Overflow -> inf
            if torch.any(overflow):
                out_exp = torch.where(overflow, torch.full_like(out_exp, max_exp), out_exp)
                out_mant = torch.where(overflow, torch.zeros_like(out_mant), out_mant)

            packed = ((out_exp.to(dtype=torch.uint8) & ((1 << exp_bits) - 1)) << mant_bits) | (
                out_mant.to(dtype=torch.uint8) & ((1 << mant_bits) - 1)
            )

            q = q.clone()
            q[finite_nz] = packed

        # Add sign bit (MSB)
        q = q | (sign << 7)

        return {
            "codec": self._codec_name(),
            "q": q,  # torch.uint8 on CPU; packed FP8 bits
            "shape": tuple(x_cpu.shape),
            "layout": str(self.layout).strip().lower(),
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
        if not isinstance(q, torch.Tensor) or q.dtype != torch.uint8:
            raise TypeError("payload['q'] must be a torch.uint8 Tensor")

        # Prefer payload layout if present, otherwise use the codec's default.
        layout = payload.get("layout", self.layout)
        exp_bits, mant_bits, bias, max_exp = TruncationFloat8Codec(layout=str(layout))._layout_params()

        target_device = device if device is not None else "cpu"
        q_dev = q.to(device=target_device)

        sign = (q_dev >> 7) & 0x1
        exp = (q_dev >> mant_bits) & ((1 << exp_bits) - 1)
        mant = q_dev & ((1 << mant_bits) - 1)

        exp_i = exp.to(dtype=torch.int32)
        mant_f = mant.to(dtype=torch.float32)

        is_zero = exp_i == 0
        is_special = exp_i == max_exp

        # Start with zeros
        out = torch.zeros_like(mant_f, dtype=torch.float32)

        # Subnormals (including zero): exp=0, mant!=0
        sub = is_zero & (mant != 0)
        if torch.any(sub):
            out = torch.where(
                sub,
                (mant_f / float(1 << mant_bits)) * float(2 ** (1 - bias)),
                out,
            )

        # Normals: 1 <= exp <= max_exp-1
        normal = (~is_zero) & (~is_special)
        if torch.any(normal):
            out = torch.where(
                normal,
                (1.0 + (mant_f / float(1 << mant_bits))) * torch.pow(2.0, (exp_i - bias).to(dtype=torch.float32)),
                out,
            )

        # Specials: exp=max_exp
        if torch.any(is_special):
            is_inf = is_special & (mant == 0)
            is_nan = is_special & (mant != 0)
            out = torch.where(is_inf, torch.full_like(out, float("inf")), out)
            out = torch.where(is_nan, torch.full_like(out, float("nan")), out)

        # Apply sign
        out = torch.where(sign != 0, -out, out)

        shape = payload.get("shape", None)
        if shape is not None:
            if not isinstance(shape, Sequence):
                raise TypeError("payload['shape'] must be a sequence of ints")
            out = out.reshape(tuple(int(s) for s in shape))

        return out.to(dtype=dtype)
