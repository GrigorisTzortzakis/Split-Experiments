from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

Payload = Dict[str, Any]
SUPPORTED_K_PERCENTS = (1, 5, 10, 25, 50)


def _storage_dtype(storage_bits: int) -> torch.dtype:
    bits = int(storage_bits)
    if bits <= 16:
        return torch.float16
    return torch.float32


def _normalize_k_percent(k_percent: float | int) -> int:
    value = float(k_percent)
    if 0.0 < value <= 1.0:
        value *= 100.0
    normalized = int(round(value))
    if normalized not in SUPPORTED_K_PERCENTS:
        raise ValueError(f"k_percent must be one of {SUPPORTED_K_PERCENTS}")
    return normalized


@dataclass(frozen=True)
class TopKSparsityCodec:
    k_percent: int = 1
    storage_bits: int = 32

    def _keep_count(self, numel: int) -> int:
        if numel <= 0:
            return 0
        ratio = float(_normalize_k_percent(self.k_percent)) / 100.0
        return max(1, min(numel, int(round(numel * ratio))))

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = x.detach().to(device="cpu")
        flat = x_cpu.reshape(-1)
        keep_count = self._keep_count(int(flat.numel()))
        if keep_count == 0:
            indices = torch.empty(0, dtype=torch.int64)
            values = flat.new_empty((0,))
        else:
            _, indices = torch.topk(flat.abs(), k=keep_count, largest=True, sorted=False)
            indices = indices.to(dtype=torch.int64)
            values = flat.index_select(0, indices).to(dtype=_storage_dtype(self.storage_bits))

        return {
            "codec": "top_k_sparsity",
            "q": values,
            "indices": indices,
            "shape": tuple(int(dim) for dim in x_cpu.shape),
            "k_percent": _normalize_k_percent(self.k_percent),
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
        if "q" not in payload or "indices" not in payload or "shape" not in payload:
            raise KeyError("Invalid payload, missing one of: 'q', 'indices', 'shape'")

        values = payload["q"]
        indices = payload["indices"]
        shape = tuple(int(dim) for dim in payload["shape"])
        if not isinstance(values, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64:
            raise TypeError("payload['indices'] must be a torch.int64 Tensor")

        target_device = device if device is not None else "cpu"
        out = torch.zeros(shape, dtype=dtype, device=target_device)
        if values.numel() == 0:
            return out

        flat = out.reshape(-1)
        flat.index_copy_(0, indices.to(device=target_device), values.to(device=target_device, dtype=dtype))
        return out