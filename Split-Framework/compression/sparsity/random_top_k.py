from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .top_k import SUPPORTED_K_PERCENTS, _normalize_k_percent, _storage_dtype

Payload = Dict[str, Any]


@dataclass(frozen=True)
class RandomTopKSparsityCodec:
    k_percent: int = 5
    seed: Optional[int] = None
    storage_bits: int = 32

    def _keep_count(self, numel: int) -> int:
        if numel <= 0:
            return 0
        ratio = float(_normalize_k_percent(self.k_percent)) / 100.0
        return max(1, min(numel, int(round(numel * ratio))))

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        normalized_k = _normalize_k_percent(self.k_percent)
        if normalized_k not in tuple(value for value in SUPPORTED_K_PERCENTS if value != 1):
            raise ValueError("random top-k supports k values of 5, 10, 25, or 50 percent")

        x_cpu = x.detach().to(device="cpu")
        flat = x_cpu.reshape(-1)
        keep_count = self._keep_count(int(flat.numel()))
        if keep_count == 0:
            indices = torch.empty(0, dtype=torch.int64)
            values = flat.new_empty((0,))
        else:
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(self.seed))
            indices = torch.randperm(int(flat.numel()), generator=generator)[:keep_count].to(dtype=torch.int64)
            values = flat.index_select(0, indices).to(dtype=_storage_dtype(self.storage_bits))

        return {
            "codec": "random_top_k_sparsity",
            "q": values,
            "indices": indices,
            "shape": tuple(int(dim) for dim in x_cpu.shape),
            "k_percent": normalized_k,
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