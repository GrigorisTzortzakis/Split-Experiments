from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

Payload = Dict[str, Any]


def _storage_dtype(storage_bits: int) -> torch.dtype:
    bits = int(storage_bits)
    if bits <= 16:
        return torch.float16
    return torch.float32


def _orthogonal_projection(input_dim: int, latent_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    weights = torch.randn((input_dim, latent_dim), generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(weights, mode="reduced")
    return q[:, :latent_dim].contiguous()


@dataclass(frozen=True)
class AutoencoderCodec:
    reduction_ratio: float = 0.25
    block_size: int = 128
    seed: int = 17
    storage_bits: int = 32

    def _latent_dim(self, block_size: int) -> int:
        ratio = float(self.reduction_ratio)
        if ratio <= 0.0:
            ratio = 0.25
        return max(1, min(block_size, int(round(block_size * ratio))))

    def encode(self, x: torch.Tensor) -> Payload:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"encode expects a torch.Tensor, got {type(x)!r}")

        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        flat = x_cpu.reshape(-1)
        block_size = max(4, int(self.block_size))
        pad_length = (-int(flat.numel())) % block_size
        if pad_length:
            flat = F.pad(flat, (0, pad_length))

        blocks = flat.reshape(-1, block_size)
        latent_dim = self._latent_dim(block_size)
        encoder = _orthogonal_projection(block_size, latent_dim, int(self.seed) + block_size * 31 + latent_dim)
        latent = torch.matmul(blocks, encoder).to(dtype=_storage_dtype(self.storage_bits))

        return {
            "codec": "autoencoder",
            "q": latent,
            "shape": tuple(int(dim) for dim in x_cpu.shape),
            "block_size": block_size,
            "latent_dim": latent_dim,
            "pad_length": int(pad_length),
            "seed": int(self.seed),
            "reduction_ratio": float(self.reduction_ratio),
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
        if "q" not in payload or "shape" not in payload:
            raise KeyError("Invalid payload, missing one of: 'q', 'shape'")

        latent = payload["q"]
        shape = tuple(int(dim) for dim in payload["shape"])
        block_size = int(payload.get("block_size", self.block_size))
        latent_dim = int(payload.get("latent_dim", self._latent_dim(block_size)))
        pad_length = int(payload.get("pad_length", 0) or 0)
        seed = int(payload.get("seed", self.seed))
        if not isinstance(latent, torch.Tensor):
            raise TypeError("payload['q'] must be a torch.Tensor")

        target_device = device if device is not None else "cpu"
        encoder = _orthogonal_projection(block_size, latent_dim, seed + block_size * 31 + latent_dim).to(device=target_device, dtype=dtype)
        blocks = torch.matmul(latent.to(device=target_device, dtype=dtype), encoder.transpose(0, 1))
        flat = blocks.reshape(-1)
        if pad_length:
            flat = flat[:-pad_length]
        return flat.reshape(shape)