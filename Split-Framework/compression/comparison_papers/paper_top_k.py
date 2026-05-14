"""Reducing Communication for Split Learning by Randomized Top-k Sparsification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from compression.sparsity.top_k import _normalize_k_percent, _storage_dtype, _store_indices

Payload = Dict[str, Any]
PAPER_TOP_K_CACHE_ID_ATTR = "_paper_top_k_cache_id"


def get_paper_top_k_cache_id(tensor: Any) -> Optional[str]:
    if not isinstance(tensor, torch.Tensor):
        return None
    cache_id = getattr(tensor, PAPER_TOP_K_CACHE_ID_ATTR, None)
    return None if cache_id is None else str(cache_id)


def set_paper_top_k_cache_id(tensor: Any, cache_id: Optional[str]) -> Any:
    if isinstance(tensor, torch.Tensor) and cache_id:
        setattr(tensor, PAPER_TOP_K_CACHE_ID_ATTR, str(cache_id))
    return tensor


def transfer_paper_top_k_cache_id(source: Any, target: Any) -> Any:
    return set_paper_top_k_cache_id(target, get_paper_top_k_cache_id(source))


@dataclass
class PaperTopKSparsityCodec:
    k_percent: int = 5
    alpha: float = 0.1
    seed: Optional[int] = None
    storage_bits: int = 32
    _cache_counter: int = field(default=0, init=False, repr=False)
    _selection_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _generator: Optional[torch.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.seed))
            self._generator = generator

    def _keep_count(self, flat_dim: int) -> int:
        if flat_dim <= 0:
            return 0
        ratio = float(_normalize_k_percent(self.k_percent)) / 100.0
        return max(1, min(flat_dim, int(round(flat_dim * ratio))))

    def _normalized_alpha(self) -> float:
        value = float(self.alpha)
        if value < 0.0 or value > 1.0:
            raise ValueError("alpha must be in [0, 1]")
        return value

    def _matrix_view(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
        x_cpu = x.detach().to(device="cpu")
        shape = tuple(int(dim) for dim in x_cpu.shape)
        if x_cpu.ndim <= 1:
            return x_cpu.reshape(1, -1), shape, int(x_cpu.numel())
        return x_cpu.reshape(int(x_cpu.shape[0]), -1), shape, int(torch.tensor(shape[1:]).prod().item())

    def _next_cache_id(self) -> str:
        cache_id = f"paper_top_k_{self._cache_counter}"
        self._cache_counter += 1
        return cache_id

    def _select_indices(self, row: torch.Tensor, *, keep_count: int, generator: Optional[torch.Generator]) -> torch.Tensor:
        if keep_count <= 0:
            return torch.empty(0, dtype=torch.int64)

        top_indices = torch.topk(row.abs(), k=keep_count, largest=True, sorted=False).indices.to(dtype=torch.int64)
        alpha = self._normalized_alpha()
        if alpha <= 0.0:
            return top_indices

        top_pool = [int(idx) for idx in top_indices.tolist()]
        top_lookup = set(top_pool)
        rest_pool = [idx for idx in range(int(row.numel())) if idx not in top_lookup]
        if not rest_pool:
            return top_indices

        selected: list[int] = []
        while len(selected) < keep_count:
            top_remaining = len(top_pool)
            rest_remaining = len(rest_pool)
            if top_remaining == 0 and rest_remaining == 0:
                break
            if top_remaining == 0:
                choose_top = False
            elif rest_remaining == 0:
                choose_top = True
            else:
                choose_top = bool(torch.rand(1, generator=generator).item() < (1.0 - alpha))

            pool = top_pool if choose_top else rest_pool
            if not pool:
                pool = rest_pool if choose_top else top_pool
            picked_position = int(torch.randint(len(pool), (1,), generator=generator).item())
            selected.append(pool.pop(picked_position))

        return torch.tensor(selected, dtype=torch.int64)

    def _cache_selection(self, *, cache_id: str, indices: torch.Tensor, shape: Tuple[int, ...], flat_dim: int) -> None:
        self._selection_cache[cache_id] = {
            "indices": indices.to(device="cpu"),
            "shape": shape,
            "flat_dim": int(flat_dim),
        }

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        cache_id = get_paper_top_k_cache_id(x)
        if cache_id is not None:
            cached = self._selection_cache.get(cache_id)
            if cached is None:
                raise KeyError(f"Missing cached forward selection for {cache_id!r}")

            matrix, shape, flat_dim = self._matrix_view(x)
            indices = cached["indices"]
            if tuple(shape) != tuple(cached["shape"]) or int(flat_dim) != int(cached["flat_dim"]):
                raise ValueError("Backward gradient shape does not match cached forward Paper Top-k selection")
            gather_indices = indices.to(dtype=torch.int64)
            values = torch.gather(matrix, 1, gather_indices).to(dtype=_storage_dtype(self.storage_bits))
            del self._selection_cache[cache_id]
            return {
                "codec": "paper_top_k_sparsity",
                "direction": "backward",
                "q": values,
                "shape": shape,
                "cache_id": cache_id,
            }

        matrix, shape, flat_dim = self._matrix_view(x)
        keep_count = self._keep_count(flat_dim)
        generator = self._generator

        index_rows = []
        value_rows = []
        for row in matrix:
            row_indices = self._select_indices(row.to(dtype=torch.float32), keep_count=keep_count, generator=generator)
            index_rows.append(row_indices)
            value_rows.append(row.index_select(0, row_indices).to(dtype=_storage_dtype(self.storage_bits)))

        indices = _store_indices(torch.stack(index_rows, dim=0), numel=flat_dim)
        values = torch.stack(value_rows, dim=0)
        payload = {
            "codec": "paper_top_k_sparsity",
            "direction": "forward",
            "q": values,
            "indices": indices,
            "shape": shape,
            "k_percent": _normalize_k_percent(self.k_percent),
            "alpha": self._normalized_alpha(),
            "training_randomized": True,
        }
        cache_id = self._next_cache_id()
        payload["cache_id"] = cache_id
        self._cache_selection(
            cache_id=cache_id,
            indices=indices,
            shape=shape,
            flat_dim=flat_dim,
        )
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

        values = payload["q"]
        shape = tuple(int(dim) for dim in payload["shape"])
        if not isinstance(values, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")
        target_device = device if device is not None else "cpu"
        direction = str(payload.get("direction") or "forward").strip().lower()

        if direction == "forward":
            indices = payload.get("indices")
            if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.uint8, torch.uint16, torch.int32, torch.int64):
                raise TypeError("forward Paper Top-k payload must include integral indices")
            matrix_shape = (1, int(torch.tensor(shape).prod().item())) if len(shape) <= 1 else (int(shape[0]), int(torch.tensor(shape[1:]).prod().item()))
            if indices.ndim != 2 or tuple(indices.shape) != tuple(values.shape):
                raise ValueError("forward Paper Top-k values and indices must share the same 2D shape")
            out = torch.zeros(matrix_shape, dtype=dtype, device=target_device)
            out.scatter_(1, indices.to(device=target_device, dtype=torch.int64), values.to(device=target_device, dtype=dtype))
            cache_id = payload.get("cache_id")
            if cache_id is not None:
                self._cache_selection(
                    cache_id=str(cache_id),
                    indices=indices,
                    shape=shape,
                    flat_dim=int(matrix_shape[1]),
                )
            return set_paper_top_k_cache_id(out.reshape(shape), str(cache_id) if cache_id is not None else None)

        cache_id = str(payload.get("cache_id") or "")
        if not cache_id:
            raise KeyError("backward Paper Top-k payload must include cache_id")
        cached = self._selection_cache.pop(cache_id, None)
        if cached is None:
            raise KeyError(f"Missing cached forward selection for backward Paper Top-k payload {cache_id!r}")
        matrix_shape = (1, int(torch.tensor(shape).prod().item())) if len(shape) <= 1 else (int(shape[0]), int(torch.tensor(shape[1:]).prod().item()))
        out = torch.zeros(matrix_shape, dtype=dtype, device=target_device)
        out.scatter_(1, cached["indices"].to(device=target_device, dtype=torch.int64), values.to(device=target_device, dtype=dtype))
        return out.reshape(shape)