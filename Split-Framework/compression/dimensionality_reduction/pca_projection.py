from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

Payload = Dict[str, Any]


def _storage_dtype(storage_bits: int) -> torch.dtype:
    bits = int(storage_bits)
    if bits <= 16:
        return torch.float16
    return torch.float32


@dataclass
class LowRankPCAProjectionCodec:
    reduction_ratio: float = 0.25
    niter: int = 2
    storage_bits: int = 32
    _encode_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)
    _decode_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, init=False, repr=False)

    def _matrix_view(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, int]]:
        shape = tuple(int(dim) for dim in x.shape)
        if x.ndim <= 1:
            matrix = x.reshape(1, -1)
        else:
            matrix = x.reshape(-1, int(x.shape[-1]))
        return matrix, shape, (int(matrix.shape[0]), int(matrix.shape[1]))

    def _rank(self, row_count: int, feature_dim: int) -> int:
        ratio = float(self.reduction_ratio)
        if ratio <= 0.0:
            ratio = 0.25
        return max(1, min(feature_dim, row_count, int(round(feature_dim * ratio))))

    def _cache_id(self, *, matrix_shape: tuple[int, int], rank: int) -> str:
        ratio_pct = int(round(float(self.reduction_ratio) * 1000.0))
        return f"rows{matrix_shape[0]}_cols{matrix_shape[1]}_rank{rank}_ratio{ratio_pct}_niter{int(self.niter)}"

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        matrix, shape, matrix_shape = self._matrix_view(x_cpu)
        feature_dim = int(matrix.shape[1])
        rank = self._rank(int(matrix.shape[0]), feature_dim)
        cache_id = self._cache_id(matrix_shape=matrix_shape, rank=rank)

        if cache_id in self._encode_cache:
            basis, mean = self._encode_cache[cache_id]
            projected = torch.matmul(matrix - mean, basis)
            cache_write = False
        else:
            mean = matrix.mean(dim=0, keepdim=True)
            centered = matrix - mean

            if feature_dim == 1 or matrix.shape[0] == 1:
                basis = torch.eye(feature_dim, rank, dtype=torch.float32)
                projected = torch.matmul(centered, basis)
            else:
                _u, _s, v = torch.pca_lowrank(centered, q=rank, center=False, niter=max(1, int(self.niter)))
                basis = v[:, :rank].contiguous()
                projected = torch.matmul(centered, basis)

            self._encode_cache[cache_id] = (basis, mean)
            cache_write = True

        storage_dtype = _storage_dtype(self.storage_bits)
        payload = {
            "codec": "low_rank_pca_projection",
            "q": projected.to(dtype=storage_dtype),
            "shape": shape,
            "matrix_shape": matrix_shape,
            "reduction_ratio": float(self.reduction_ratio),
            "cache_id": cache_id,
            "cache_write": cache_write,
        }
        if cache_write:
            payload["basis"] = basis.to(dtype=storage_dtype)
            payload["mean"] = mean.to(dtype=storage_dtype)
        return payload

    def decode(
        self,
        payload: Payload,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not isinstance(payload, dict):
            raise TypeError(f"decode expects a dict payload, got {type(payload)!r}")
        if "q" not in payload or "shape" not in payload:
            raise KeyError("Invalid payload, missing one of: 'q', 'shape'")

        projected = payload["q"]
        shape = tuple(int(dim) for dim in payload["shape"])
        matrix_shape = tuple(int(dim) for dim in payload.get("matrix_shape", (1, int(torch.tensor(shape).prod().item()))))
        cache_id = str(payload.get("cache_id") or self._cache_id(matrix_shape=matrix_shape, rank=int(projected.shape[-1]) if projected.ndim >= 2 else 1))
        if not isinstance(projected, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")

        if "basis" in payload and "mean" in payload:
            basis = payload["basis"]
            mean = payload["mean"]
            if not isinstance(basis, torch.Tensor):
                raise TypeError("payload['basis'] must be a torch.Tensor")
            if not isinstance(mean, torch.Tensor):
                raise TypeError("payload['mean'] must be a torch.Tensor")
            self._decode_cache[cache_id] = (basis, mean)
        else:
            cached = self._decode_cache.get(cache_id)
            if cached is None:
                raise KeyError(f"PCA payload is missing basis/mean and cache_id {cache_id!r} was not initialized")
            basis, mean = cached

        target_device = device if device is not None else "cpu"
        matrix = torch.matmul(
            projected.to(device=target_device, dtype=dtype),
            basis.to(device=target_device, dtype=dtype).transpose(0, 1),
        ) + mean.to(device=target_device, dtype=dtype)
        return matrix.reshape(shape if len(shape) > 1 else (matrix_shape[1],)).reshape(shape)