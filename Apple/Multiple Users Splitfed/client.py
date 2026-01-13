from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms
import websockets
from websockets.client import WebSocketClientProtocol

from split_models import SplitClientNet

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None  # type: ignore


TensorPayload = Dict[str, Any]

_MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


_DTYPE_TO_TORCH = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


def _tensor_to_payload(t: torch.Tensor) -> TensorPayload:
    t = t.detach().cpu().contiguous()
    raw = t.numpy().tobytes(order="C")
    return {
        "dtype": str(t.numpy().dtype),
        "shape": list(t.shape),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def _payload_to_tensor(p: TensorPayload) -> torch.Tensor:
    dtype = _DTYPE_TO_TORCH.get(str(p["dtype"]))
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {p['dtype']}")
    shape = tuple(int(x) for x in p["shape"])
    raw = base64.b64decode(p["data_b64"].encode("ascii"))
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)


def _mnist_raw_ok(mnist_root: Path) -> bool:
    raw_dir = mnist_root / "MNIST" / "raw"
    return all((raw_dir / name).is_file() for name in _MNIST_RAW_FILES)


def _normalize_mnist_root(path: Path) -> Path:
    # Torchvision expects root that *contains* MNIST/.
    path = path.expanduser().resolve()
    if path.name == "raw" and path.parent.name == "MNIST":
        return path.parent.parent
    if path.name == "MNIST":
        return path.parent
    return path


def _find_existing_mnist_root(preferred: Path) -> Path:
    preferred = _normalize_mnist_root(preferred)
    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve()
    # Search a handful of likely locations (fast, no full-disk walk).
    candidates = [
        preferred,
        cwd,
        cwd / "data",
        cwd.parent / "data",
        cwd.parent.parent / "data",
        here.parent / "data",
        here.parent.parent / "data",
        here.parent.parent.parent / "data",
    ]
    # Workspace layout in this repo: Split Experiments/data/MNIST/raw/...
    if len(here.parents) > 2:
        candidates.append(here.parents[2] / "data")
    for root in candidates:
        root = _normalize_mnist_root(root)
        if _mnist_raw_ok(root):
            return root
    raise FileNotFoundError(
        "MNIST dataset not found. Point --data-dir to a folder containing MNIST/raw "
        "(with train-images-idx3-ubyte etc). To allow downloading, pass --download."
    )


@dataclass
class ResourceTracker:
    cpu_samples: List[float]
    rss_samples: List[float]
    cpu_capacity: float = float(psutil.cpu_count(logical=True) or 1)

    def record(self, cpu_pct: float, rss_mb: float) -> None:
        self.cpu_samples.append(float(cpu_pct))
        self.rss_samples.append(float(rss_mb))

    def summary(self) -> Dict[str, float]:
        if not self.cpu_samples or not self.rss_samples:
            return {
                "samples": len(self.cpu_samples),
                "cpu_pct_avg": 0.0,
                "cpu_pct_max": 0.0,
                "cpu_pct_avg_per_core": 0.0,
                "cpu_pct_max_per_core": 0.0,
                "rss_mb_avg": 0.0,
                "rss_mb_max": 0.0,
            }
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples)
        rss_avg = sum(self.rss_samples) / len(self.rss_samples)
        capacity = max(self.cpu_capacity, 1.0)
        cpu_max = max(self.cpu_samples)
        return {
            "samples": len(self.cpu_samples),
            "cpu_pct_avg": cpu_avg,
            "cpu_pct_max": cpu_max,
            "cpu_pct_avg_per_core": cpu_avg / capacity,
            "cpu_pct_max_per_core": cpu_max / capacity,
            "rss_mb_avg": rss_avg,
            "rss_mb_max": max(self.rss_samples),
        }


def _pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enforce_limits(cpu_seconds: int, memory_mb: int, logger: logging.Logger) -> None:
    if resource is None:
        if cpu_seconds > 0 or memory_mb > 0:
            logger.warning("resource module unavailable; hard limits disabled on this platform")
        return
    if cpu_seconds > 0:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError) as exc:
            logger.warning("failed to apply RLIMIT_CPU: %s", exc)
    if memory_mb > 0:
        as_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError) as exc:
            logger.warning("failed to apply RLIMIT_AS: %s; will enforce via sampling", exc)


class SplitLearningClient:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.device = _pick_device(args.device)
        self._output_suffix = f"_client{int(args.client_id)}" if int(args.num_clients) > 1 else ""
        self.model = SplitClientNet(cut_layer=args.cut_layer).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.train_loader, self.test_loader = self._prepare_data(args)
        self.nk = int(len(self.train_loader.dataset))
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)

        # Clients should not emit per-client files by default in the multi-user setup.
        # (Server writes the combined metrics/artifacts.) Only materialize output paths
        # if explicitly requested.
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path: Optional[Path] = None
        if getattr(args, "client_checkpoint", "") and getattr(args, "write_checkpoints", False):
            self.checkpoint_path = self.output_dir / self._with_suffix(args.client_checkpoint)

        self.resource_report_path: Optional[Path] = None
        if getattr(args, "resource_report", ""):
            self.resource_report_path = self.output_dir / self._with_suffix(args.resource_report)
        self.resource_tracker = ResourceTracker([], [])
        self.client_limits = {
            "cpu_seconds": args.cpu_seconds,
            "memory_mb": args.memory_mb,
            "device": str(self.device),
        }
        self._msg_queue: Optional[asyncio.Queue[Dict[str, Any]]] = None
        self._msg_buffer: List[Dict[str, Any]] = []
        self._fed_msg_queue: Optional[asyncio.Queue[Dict[str, Any]]] = None
        self._fed_msg_buffer: List[Dict[str, Any]] = []
        if args.resume_client_checkpoint:
            self._maybe_resume_from_checkpoint(Path(args.resume_client_checkpoint))

        self._server_sfl_variant: Optional[str] = None

    def _with_suffix(self, filename: str) -> str:
        if not self._output_suffix:
            return filename
        name = Path(filename)
        if name.suffix:
            return f"{name.stem}{self._output_suffix}{name.suffix}"
        return f"{filename}{self._output_suffix}"

    def _prepare_data(self, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
        if getattr(args, "download", False):
            root = _normalize_mnist_root(Path(args.data_dir))
            root.mkdir(parents=True, exist_ok=True)
            download = True
        else:
            root = _find_existing_mnist_root(Path(args.data_dir))
            download = False
        mnist_norm = transforms.Normalize((0.1307,), (0.3081,))
        tfm = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), mnist_norm])
        train_ds = tv_datasets.MNIST(root=str(root), train=True, download=download, transform=tfm)
        test_ds = tv_datasets.MNIST(root=str(root), train=False, download=download, transform=tfm)
        num_clients = int(getattr(args, "num_clients", 1) or 1)
        client_id = int(getattr(args, "client_id", 0) or 0)
        if num_clients > 1:
            if client_id < 0 or client_id >= num_clients:
                raise ValueError(f"client_id must be in [0, {num_clients - 1}]")
            indices = list(range(client_id, len(train_ds), num_clients))
            train_ds = Subset(train_ds, indices)
            test_indices = list(range(client_id, len(test_ds), num_clients))
            test_ds = Subset(test_ds, test_indices)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=bool(getattr(args, "shuffle", False)),
            num_workers=args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        return train_loader, test_loader

    async def run(self) -> None:
        split_uri = f"ws://{self.args.host}:{self.args.port}"
        fed_uri = f"ws://{self.args.fed_host}:{self.args.fed_port}"
        max_size = self.args.max_message_mb * 1024 * 1024
        async with websockets.connect(split_uri, max_size=max_size) as ws, websockets.connect(
            fed_uri, max_size=max_size
        ) as fed_ws:
            self._msg_queue = asyncio.Queue()
            self._fed_msg_queue = asyncio.Queue()
            recv_task = asyncio.create_task(self._recv_loop(ws, is_fed=False))
            fed_recv_task = asyncio.create_task(self._recv_loop(fed_ws, is_fed=True))

            hello = await self._recv_type({"server_info"})
            if hello.get("type") == "server_info":
                self._server_sfl_variant = str(hello.get("sfl_variant")) if hello.get("sfl_variant") is not None else None
                self.logger.info(
                    "connected to server device=%s cut_layer=%s",
                    hello.get("device"),
                    hello.get("cut_layer"),
                )
            else:
                self.logger.warning("missing server info; continuing anyway")

            await self._send_hello(ws)
            await self._fed_send_hello(fed_ws)
            await self._train_splitfed(ws, fed_ws)
            metrics = await self._evaluate(ws)
            artifact_paths = self._save_artifacts(metrics)
            await self._finalize_server(ws, metrics, artifact_paths)
            self._log_metric_summary(metrics)

            recv_task.cancel()
            fed_recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task
            with contextlib.suppress(asyncio.CancelledError):
                await fed_recv_task

    async def _recv_loop(self, ws: WebSocketClientProtocol, *, is_fed: bool) -> None:
        try:
            async for message in ws:
                try:
                    obj = json.loads(message)
                except Exception:
                    continue
                if is_fed:
                    if self._fed_msg_queue is not None:
                        await self._fed_msg_queue.put(obj)
                else:
                    if self._msg_queue is not None:
                        await self._msg_queue.put(obj)
        except asyncio.CancelledError:
            return
        except Exception:
            return

    async def _recv_type(self, types: set[str]) -> Dict[str, Any]:
        if self._msg_queue is None:
            raise RuntimeError("message queue not initialized")
        for i, msg in enumerate(list(self._msg_buffer)):
            if msg.get("type") in types:
                return self._msg_buffer.pop(i)
        while True:
            msg = await self._msg_queue.get()
            if msg.get("type") in types:
                return msg
            self._msg_buffer.append(msg)

    async def _recv_fed_type(self, types: set[str]) -> Dict[str, Any]:
        if self._fed_msg_queue is None:
            raise RuntimeError("fed message queue not initialized")
        for i, msg in enumerate(list(self._fed_msg_buffer)):
            if msg.get("type") in types:
                return self._fed_msg_buffer.pop(i)
        while True:
            msg = await self._fed_msg_queue.get()
            if msg.get("type") in types:
                return msg
            self._fed_msg_buffer.append(msg)

    async def _send_hello(self, ws: WebSocketClientProtocol) -> None:
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "client_id": int(self.args.client_id),
                        "num_clients": int(self.args.num_clients),
                        "nk": int(self.nk),
                    }
                )
            )
            resp = await self._recv_type({"hello_ack", "error"})
            if resp.get("type") == "hello_ack":
                self.logger.info(
                    "server acknowledged client_id=%s expected_clients=%s",
                    resp.get("client_id"),
                    resp.get("expected_clients"),
                )
        except Exception as exc:  # pragma: no cover - best effort
            self.logger.warning("hello handshake failed: %s", exc)

    async def _fed_send_hello(self, fed_ws: WebSocketClientProtocol) -> None:
        await fed_ws.send(json.dumps({"type": "hello", "client_id": int(self.args.client_id)}))
        _ = await self._recv_fed_type({"hello_ack", "error"})

    def _save_artifacts(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        # Client no longer writes per-client report artifacts; the server writes a single
        # aggregated confusion matrix artifact once it has all client reports.
        return {}

    async def _train_splitfed(self, split_ws: WebSocketClientProtocol, fed_ws: WebSocketClientProtocol) -> None:
        num_rounds = int(self.args.num_rounds)
        local_epochs = int(self.args.local_epochs)
        if num_rounds <= 0:
            raise ValueError("--num-rounds must be > 0")
        if local_epochs <= 0:
            raise ValueError("--local-epochs must be > 0")

        steps_per_epoch = len(self.train_loader)
        if self.args.max_batches and int(self.args.max_batches) > 0:
            steps_per_epoch = min(int(steps_per_epoch), int(self.args.max_batches))

        # Paper-aligned: round 0 starts from WC_0 initialized by the fed_server.
        wc0 = await self._fed_get_global_weights(fed_ws, round_id=0)
        if wc0:
            self.model.load_state_dict(wc0)
            self.optimizer.state.clear()

        # Later rounds start from FedAvg weights published by the fed_server.
        for round_id in range(num_rounds):
            if round_id > 0:
                # Gate start of round r on BOTH barriers:
                # - fed_server publishes WC_global for round r
                # - main server broadcasts server_round_ready for round r
                wc, _ = await asyncio.gather(
                    self._fed_wait_global_weights(fed_ws, round_id),
                    self._wait_server_round_ready(round_id),
                )
                if wc:
                    self.model.load_state_dict(wc)
                    self.optimizer.state.clear()

            self.model.train()
            local_steps = 0
            running_correct = 0
            running_total = 0

            for local_epoch in range(local_epochs):
                train_iter = iter(self.train_loader)
                for batch_idx in range(steps_per_epoch):
                    images, labels = next(train_iter)
                    batch_t0 = time.perf_counter()

                    images = images.to(self.device, non_blocking=True)
                    labels_dev = labels.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad(set_to_none=True)
                    acts = self.model(images)
                    step_idx = int(local_epoch) * int(steps_per_epoch) + int(batch_idx)

                    # Algorithm 2: noise layer after cut layer (L).
                    acts_to_send = acts
                    act_noise_std = float(getattr(self.args, "activation_noise_std", 0.0) or 0.0)
                    if act_noise_std > 0.0:
                        gen = torch.Generator(device=self.device.type)
                        gen.manual_seed(int(self.args.seed) + int(round_id) * 1000003 + int(step_idx))
                        noise = torch.randn(acts_to_send.shape, device=acts_to_send.device, generator=gen) * act_noise_std
                        acts_to_send = acts_to_send + noise

                    await split_ws.send(
                        json.dumps(
                            {
                                "type": "train",
                                "client_id": int(self.args.client_id),
                                "round_id": int(round_id),
                                "step_idx": int(step_idx),
                                "activations": _tensor_to_payload(acts_to_send.detach().cpu()),
                                "labels": _tensor_to_payload(labels_dev.detach().cpu()),
                            }
                        )
                    )
                    resp = await self._recv_type({"train_ack", "error"})
                    if resp.get("type") != "train_ack":
                        raise RuntimeError(f"unexpected split server response: {resp}")

                    grads = _payload_to_tensor(resp["grads"]).to(self.device)

                    if bool(getattr(self.args, "dp_enable", False)):
                        self._dp_step(images, grads, round_id=int(round_id), step_idx=int(step_idx))
                    else:
                        acts.backward(grads)
                        self.optimizer.step()

                    running_correct += int(resp.get("correct", 0))
                    running_total += int(resp.get("total", 0))
                    local_steps += 1

                    if self.args.log_every > 0 and local_steps % self.args.log_every == 0:
                        cpu_pct = self.proc.cpu_percent(None)
                        rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                        acc = running_correct / max(running_total, 1)
                        dt_ms = (time.perf_counter() - batch_t0) * 1000.0
                        self._record_resource_sample(cpu_pct, rss_mb)
                        self.logger.info(
                            "[client] round=%d epoch_local=%d step=%d loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                            int(round_id),
                            int(local_epoch),
                            int(local_steps),
                            float(resp.get("loss", 0.0)),
                            acc,
                            cpu_pct,
                            rss_mb,
                            dt_ms,
                        )

            nk = int(len(getattr(self.train_loader, "dataset", [])))
            steps_in_round = int(steps_per_epoch) * int(local_epochs)

            # Inform main server that this client finished the round.
            await split_ws.send(
                json.dumps(
                    {
                        "type": "round_done",
                        "client_id": int(self.args.client_id),
                        "round_id": int(round_id),
                        "nk": int(max(nk, 1)),
                        "steps_in_round": int(steps_in_round),
                    }
                )
            )
            _ = await self._recv_type({"round_done_ack", "error"})

            # Submit client-front weights to fed_server for FedAvg.
            await self._fed_submit_update(fed_ws, round_id=round_id, nk=nk)

        self._record_resource_sample()

    def _dp_step(self, images: torch.Tensor, upstream_grads: torch.Tensor, *, round_id: int, step_idx: int) -> None:
        """DP-SGD style client update driven by upstream dL/dA from the server.

        Uses per-example gradient clipping with norm C and adds Gaussian noise
        with std = σ * C / batch_size.
        """

        clip_norm = float(getattr(self.args, "dp_clip_norm", 1.0) or 1.0)
        noise_multiplier = float(getattr(self.args, "dp_noise_multiplier", 0.0) or 0.0)
        if clip_norm <= 0:
            raise ValueError("--dp-clip-norm must be > 0")

        model = self.model
        model.train()

        try:
            from torch.func import functional_call, grad, vmap
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch.func is required for per-example DP gradients") from exc

        params = {name: p for name, p in model.named_parameters()}
        buffers = {name: b for name, b in model.named_buffers()}

        def _single_sample_objective(p, b, x, g):
            a = functional_call(model, (p, b), (x.unsqueeze(0),))
            a = a.squeeze(0)
            # Scalar surrogate whose gradient matches backprop with upstream gradient g.
            return (a * g).sum()

        per_sample_grads = vmap(grad(_single_sample_objective), in_dims=(None, None, 0, 0))(
            params,
            buffers,
            images,
            upstream_grads,
        )

        batch_size = int(images.shape[0])
        sq = None
        for g in per_sample_grads.values():
            flat = g.reshape(batch_size, -1).to(dtype=torch.float32)
            part = (flat * flat).sum(dim=1)
            sq = part if sq is None else (sq + part)
        assert sq is not None
        norms = torch.sqrt(sq + 1e-12)
        factors = torch.clamp(clip_norm / norms, max=1.0)

        self.optimizer.zero_grad(set_to_none=True)
        noise_std = noise_multiplier * clip_norm / float(max(batch_size, 1))
        gen = torch.Generator(device=self.device.type)
        gen.manual_seed(int(self.args.seed) + int(round_id) * 1000003 + int(step_idx) * 97 + 12345)

        for name, p in model.named_parameters():
            g = per_sample_grads[name]
            view_shape = (batch_size,) + (1,) * (g.ndim - 1)
            g_clipped = g * factors.view(view_shape)
            g_avg = g_clipped.mean(dim=0)
            if noise_std > 0:
                g_avg = g_avg + torch.randn(g_avg.shape, device=g_avg.device, generator=gen) * noise_std
            p.grad = g_avg.to(dtype=p.dtype)

        self.optimizer.step()

    async def _wait_server_round_ready(self, round_id: int) -> None:
        # Wait until main server broadcasts server_round_ready for the given round.
        for i, msg in enumerate(list(self._msg_buffer)):
            if msg.get("type") == "server_round_ready" and int(msg.get("round_id", -1)) == int(round_id):
                self._msg_buffer.pop(i)
                return
        if self._msg_queue is None:
            raise RuntimeError("message queue not initialized")
        while True:
            msg = await self._msg_queue.get()
            if msg.get("type") != "server_round_ready":
                self._msg_buffer.append(msg)
                continue
            if int(msg.get("round_id", -1)) != int(round_id):
                self._msg_buffer.append(msg)
                continue
            return

    async def _evaluate(self, ws: WebSocketClientProtocol) -> Dict[str, Any]:
        cm = torch.zeros((10, 10), dtype=torch.int64)
        total = 0
        correct = 0
        self.model.eval()
        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            acts = self.model(images)
            await ws.send(
                json.dumps(
                    {
                        "type": "infer",
                        "activations": _tensor_to_payload(acts.detach().cpu()),
                    }
                )
            )
            resp = await self._recv_type({"logits", "error"})
            if resp.get("type") != "logits":
                raise RuntimeError(f"unexpected server response: {resp}")
            logits = _payload_to_tensor(resp["logits"])
            preds = logits.argmax(dim=1)
            labels_cpu = labels.cpu()
            total += int(labels_cpu.numel())
            correct += int((preds == labels_cpu).sum().item())
            idx = (labels_cpu.to(torch.int64) * 10 + preds.to(torch.int64)).view(-1)
            cm += torch.bincount(idx, minlength=100).view(10, 10)
        metrics = compute_metrics(cm)
        metrics["accuracy"] = correct / max(total, 1)
        metrics["confusion_matrix"] = cm
        self._record_resource_sample()
        return metrics

    def _state_dict_to_b64(self) -> str:
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _b64_to_state_dict(self, payload_b64: str) -> Dict[str, Any]:
        raw = base64.b64decode(payload_b64.encode("ascii"))
        buf = io.BytesIO(raw)
        obj = torch.load(buf, map_location=self.device)
        if not isinstance(obj, dict):
            raise ValueError("invalid state_dict payload")
        return obj

    async def _fed_wait_global_weights(self, fed_ws: WebSocketClientProtocol, round_id: int) -> Optional[Dict[str, Any]]:
        # Wait until fed_server broadcasts global weights for the given round.
        # First, check buffered out-of-order messages.
        for i, msg in enumerate(list(self._fed_msg_buffer)):
            if msg.get("type") == "global_client_weights" and int(msg.get("round_id", -1)) == int(round_id):
                self._fed_msg_buffer.pop(i)
                state_b64 = msg.get("state_b64")
                if isinstance(state_b64, str) and state_b64:
                    return self._b64_to_state_dict(state_b64)
                return None

        if self._fed_msg_queue is None:
            raise RuntimeError("fed message queue not initialized")
        while True:
            msg = await self._fed_msg_queue.get()
            if msg.get("type") != "global_client_weights":
                self._fed_msg_buffer.append(msg)
                continue
            if int(msg.get("round_id", -1)) != int(round_id):
                self._fed_msg_buffer.append(msg)
                continue
            state_b64 = msg.get("state_b64")
            if isinstance(state_b64, str) and state_b64:
                return self._b64_to_state_dict(state_b64)
            return None

    async def _fed_get_global_weights(self, fed_ws: WebSocketClientProtocol, *, round_id: int) -> Optional[Dict[str, Any]]:
        # Request global client weights for a round (used for WC_0 at startup).
        await fed_ws.send(json.dumps({"type": "get_global_client_weights", "round_id": int(round_id)}))
        msg = await self._recv_fed_type({"global_client_weights", "error"})
        if msg.get("type") != "global_client_weights":
            return None
        if int(msg.get("round_id", -1)) != int(round_id):
            # Unexpected; fall back to waiting.
            return await self._fed_wait_global_weights(fed_ws, round_id)
        state_b64 = msg.get("state_b64")
        if isinstance(state_b64, str) and state_b64:
            return self._b64_to_state_dict(state_b64)
        return None

    async def _fed_submit_update(self, fed_ws: WebSocketClientProtocol, *, round_id: int, nk: int) -> None:
        await fed_ws.send(
            json.dumps(
                {
                    "type": "submit_client_update",
                    "client_id": int(self.args.client_id),
                    "round_id": int(round_id),
                    "nk": int(max(nk, 1)),
                    "state_b64": self._state_dict_to_b64(),
                }
            )
        )
        _ = await self._recv_fed_type({"submit_client_update_ack", "error"})

    def _build_client_report(self, metrics: Dict[str, Any], artifact_paths: Dict[str, str]) -> Dict[str, Any]:
        report_metrics = dict(metrics)
        # Don't embed the full confusion matrix in JSON; we save it as an artifact.
        cm = report_metrics.pop("confusion_matrix", None)
        report: Dict[str, Any] = {
            "client_id": int(self.args.client_id),
            "num_clients": int(self.args.num_clients),
            "metrics": report_metrics,
            "client_resources": self._client_resource_summary(),
            "artifacts": artifact_paths,
        }
        if isinstance(cm, torch.Tensor):
            report["confusion_matrix"] = _tensor_to_payload(cm.to(torch.int64).cpu())
        return report

    def _log_metric_summary(self, metrics: Dict[str, Any]) -> None:
        self.logger.info(
            "evaluation accuracy=%.4f micro_f1=%.4f macro_f1=%.4f",
            metrics.get("accuracy", 0.0),
            metrics.get("micro", {}).get("f1", 0.0),
            metrics.get("macro", {}).get("f1", 0.0),
        )
        for entry in metrics.get("per_class", []):
            self.logger.info(
                "class=%s precision=%.4f recall=%.4f f1=%.4f support=%d",
                entry.get("class"),
                entry.get("precision", 0.0),
                entry.get("recall", 0.0),
                entry.get("f1", 0.0),
                entry.get("support", 0),
            )

    def _record_resource_sample(
        self, cpu_pct: Optional[float] = None, rss_mb: Optional[float] = None
    ) -> tuple[float, float]:
        if cpu_pct is None:
            cpu_pct = self.proc.cpu_percent(None)
        if rss_mb is None:
            rss_mb = self.proc.memory_info().rss / (1024 * 1024)
        self.resource_tracker.record(cpu_pct, rss_mb)
        self._enforce_memory_budget(rss_mb)
        return float(cpu_pct), float(rss_mb)

    def _enforce_memory_budget(self, rss_mb: float) -> None:
        limit = self.args.memory_mb
        if limit > 0 and rss_mb > limit:
            msg = (
                f"client RSS {rss_mb:.1f}MB exceeded limit {limit}MB; "
                "stopping to honor memory constraint"
            )
            self.logger.error(msg)
            raise MemoryError(msg)

    def _client_resource_summary(self) -> Dict[str, Any]:
        summary = self.resource_tracker.summary()
        summary["limits"] = self.client_limits
        summary["device"] = str(self.device)
        if getattr(self.args, "write_checkpoints", False) and self.checkpoint_path is not None:
            summary["checkpoint_path"] = str(self.checkpoint_path)
        if self.resource_report_path is not None:
            summary["resource_report_path"] = str(self.resource_report_path)
        return summary

    def _maybe_resume_from_checkpoint(self, ckpt_path: Path) -> None:
        ckpt = ckpt_path.expanduser().resolve()
        if not ckpt.is_file():
            self.logger.warning("resume checkpoint not found: %s", ckpt)
            return
        payload = torch.load(ckpt, map_location=self.device)
        model_state = payload.get("model_state", payload)
        self.model.load_state_dict(model_state)
        opt_state = payload.get("optimizer_state")
        if opt_state:
            self.optimizer.load_state_dict(opt_state)
        self.logger.info("resumed client weights from %s", ckpt)

    async def _finalize_server(
        self,
        ws: WebSocketClientProtocol,
        metrics: Dict[str, Any],
        artifact_paths: Dict[str, str],
    ) -> Dict[str, Any]:
        try:
            client_report = self._build_client_report(metrics, artifact_paths)
            await ws.send(json.dumps({"type": "finalize", "client_report": client_report}))
            resp = await self._recv_type({"finalized", "error"})
            if resp.get("type") == "finalized":
                return resp.get("resource_summary", {})
            self.logger.warning("unexpected server finalize response: %s", resp)
            return {}
        except Exception as exc:  # pragma: no cover - network best-effort
            self.logger.warning("server finalize failed: %s", exc)
            return {}


def compute_metrics(cm: torch.Tensor) -> Dict[str, Any]:
    num_classes = cm.shape[0]
    per_class: List[Dict[str, float]] = []
    diag = cm.diag().to(torch.float64)
    preds_per_class = cm.sum(dim=0).to(torch.float64)
    labels_per_class = cm.sum(dim=1).to(torch.float64)
    total_samples = float(cm.sum().item())
    total_tp = float(diag.sum().item())
    for cls in range(num_classes):
        tp = float(cm[cls, cls].item())
        fp = float(preds_per_class[cls].item() - tp)
        fn = float(labels_per_class[cls].item() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        per_class.append(
            {
                "class": cls,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(labels_per_class[cls].item()),
            }
        )
    macro_precision = sum(item["precision"] for item in per_class) / num_classes
    macro_recall = sum(item["recall"] for item in per_class) / num_classes
    macro_f1 = sum(item["f1"] for item in per_class) / num_classes
    micro_precision = total_tp / max(total_samples, 1.0)
    micro_recall = total_tp / max(total_samples, 1.0)
    micro_f1 = micro_precision  # equal for single-label classification
    return {
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    }


def save_confusion_matrix_counts(cm: torch.Tensor, counts_path: Path, logger: logging.Logger) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("matplotlib unavailable, skipping confusion matrix plots: %s", exc)
        return False
    cm_np = cm.numpy()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(cm_np, cmap="viridis")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm_np[i, j]), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(counts_path, dpi=200)
    plt.close(fig)
    return True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split learning client for MNIST")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fed-host", default="127.0.0.1")
    p.add_argument("--fed-port", type=int, default=8766)
    p.add_argument("--cut-layer", type=int, default=1, choices=(0, 1))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of SplitFed rounds (defaults to --num-epochs for backward compatibility)",
    )
    p.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs per round (E in SplitFed)",
    )
    # Backward compatibility with earlier scripts: treat --num-epochs as num_rounds.
    p.add_argument("--num-epochs", type=int, default=5, help="(deprecated) alias for --num-rounds")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training data (default: off for deterministic paper-aligned order)",
    )
    p.add_argument("--max-batches", type=int, default=0, help="Limit batches per epoch (0 disables)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--data-dir", default="./data")
    p.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading MNIST if not found locally (default: off)",
    )
    p.add_argument("--output-dir", default=".")
    # Default: don't write client-side files in multi-user runs.
    p.add_argument("--client-checkpoint", default="")
    p.add_argument("--resume-client-checkpoint", default="", help="Path to client checkpoint to resume")
    p.add_argument(
        "--write-checkpoints",
        action="store_true",
        help="Write checkpoints to disk (default: off)",
    )
    p.add_argument("--resource-report", default="")
    p.add_argument("--cpu-seconds", type=int, default=0, help="Hard CPU seconds limit (0 disables)")
    p.add_argument("--memory-mb", type=int, default=0, help="Hard address-space limit in MB (0 disables)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num-clients", type=int, default=1, help="Total clients participating (use 2)")
    p.add_argument("--client-id", type=int, default=0, help="Client index in [0, num_clients-1]")
    p.add_argument(
        "--max-message-mb",
        type=int,
        default=128,
        help="Maximum websocket frame size in MB",
    )

    # Algorithm 2 (DP + noise layer)
    p.add_argument(
        "--dp-enable",
        action="store_true",
        help="Enable DP client update: per-example clipping + Gaussian noise (Algorithm 2)",
    )
    p.add_argument(
        "--dp-clip-norm",
        type=float,
        default=1.0,
        help="Clipping norm C for per-example gradients",
    )
    p.add_argument(
        "--dp-noise-multiplier",
        type=float,
        default=0.0,
        help="Noise multiplier σ (noise std = σ*C/batch_size)",
    )
    p.add_argument(
        "--activation-noise-std",
        type=float,
        default=0.0,
        help="Stddev of the activation noise layer after the cut",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.num_rounds is None:
        args.num_rounds = int(args.num_epochs)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("split-client")
    # Paper-aligned determinism: same initialization across clients.
    _set_seed(int(args.seed))
    _enforce_limits(args.cpu_seconds, args.memory_mb, logger)
    client = SplitLearningClient(args, logger)
    logger.info(
        "client id=%d/%d device=%s cut_layer=%d rounds=%d local_epochs=%d batch=%d",
        int(args.client_id),
        int(args.num_clients),
        client.device,
        args.cut_layer,
        int(args.num_rounds),
        int(args.local_epochs),
        args.batch_size,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt; stopping early")


if __name__ == "__main__":
    main()
