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
        if args.resume_client_checkpoint:
            self._maybe_resume_from_checkpoint(Path(args.resume_client_checkpoint))

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
        uri = f"ws://{self.args.host}:{self.args.port}"
        max_size = self.args.max_message_mb * 1024 * 1024
        async with websockets.connect(uri, max_size=max_size) as ws:
            self._msg_queue = asyncio.Queue()
            recv_task = asyncio.create_task(self._recv_loop(ws))

            hello = await self._recv_type({"server_info"})
            if hello.get("type") == "server_info":
                self.logger.info(
                    "connected to server device=%s cut_layer=%s",
                    hello.get("device"),
                    hello.get("cut_layer"),
                )
            else:
                self.logger.warning("missing server info; continuing anyway")

            await self._send_hello(ws)
            await self._train(ws)
            # Ensure evaluation uses the final canonical client-front weights.
            await self._sync_client_weights(ws)
            metrics = await self._evaluate(ws)
            artifact_paths = self._save_artifacts(metrics)
            await self._finalize_server(ws, metrics, artifact_paths)
            self._log_metric_summary(metrics)

            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task

    async def _recv_loop(self, ws: WebSocketClientProtocol) -> None:
        try:
            async for message in ws:
                try:
                    obj = json.loads(message)
                except Exception:
                    continue
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

    async def _send_hello(self, ws: WebSocketClientProtocol) -> None:
        try:
            total_steps = int(self.args.total_steps or 0)
            if total_steps <= 0:
                # Interpret --num-epochs as GLOBAL epochs: one global epoch means each client's
                # shard is consumed once. With round-robin K=turn_batches, a "step" is 1 batch.
                batches_per_epoch = len(self.train_loader)
                if self.args.max_batches:
                    batches_per_epoch = min(int(batches_per_epoch), int(self.args.max_batches))
                total_steps = int(self.args.num_epochs) * int(batches_per_epoch) * int(self.args.num_clients)
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "client_id": int(self.args.client_id),
                        "num_clients": int(self.args.num_clients),
                        "turn_batches": int(self.args.turn_batches),
                        "total_steps": int(total_steps),
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

    def _save_artifacts(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        # Client no longer writes per-client report artifacts; the server writes a single
        # aggregated confusion matrix artifact once it has all client reports.
        return {}

    async def _train(self, ws: WebSocketClientProtocol) -> None:
        # Turn-based training: wait for server "your_turn", pull latest client-front weights,
        # train K batches, push updated weights, repeat until server broadcasts "training_done".
        if int(self.args.turn_batches) <= 0:
            raise ValueError("--turn-batches must be > 0")

        train_iter = iter(self.train_loader)
        local_epoch = 0
        global_batches_seen = 0
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        while True:
            msg = await self._recv_type({"your_turn", "training_done"})
            if msg.get("type") == "training_done":
                break
            if int(msg.get("client_id", -1)) != int(self.args.client_id):
                # Shouldn't happen, but ignore.
                continue

            await self._sync_client_weights(ws)
            self.model.train()
            for _ in range(int(self.args.turn_batches)):
                try:
                    images, labels = next(train_iter)
                except StopIteration:
                    local_epoch += 1
                    train_iter = iter(self.train_loader)
                    images, labels = next(train_iter)

                batch_t0 = time.perf_counter()
                images = images.to(self.device, non_blocking=True)
                labels_dev = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                acts = self.model(images)
                await ws.send(
                    json.dumps(
                        {
                            "type": "train",
                            "activations": _tensor_to_payload(acts.detach().cpu()),
                            "labels": _tensor_to_payload(labels_dev.detach().cpu()),
                        }
                    )
                )
                resp = await self._recv_type({"train_ack", "turn_exhausted", "not_your_turn", "training_done", "error"})
                if resp.get("type") == "training_done":
                    # Server ended training while we were in-flight. Push our updated
                    # weights so everyone evaluates the same final client-front.
                    with contextlib.suppress(Exception):
                        await self._push_client_weights(ws)
                    return
                if resp.get("type") != "train_ack":
                    raise RuntimeError(f"unexpected server response: {resp}")

                grads = _payload_to_tensor(resp["grads"]).to(self.device)
                acts.backward(grads)
                self.optimizer.step()

                running_loss += float(resp.get("loss", 0.0))
                running_correct += int(resp.get("correct", 0))
                running_total += int(resp.get("total", 0))
                global_batches_seen += 1

                if self.args.log_every > 0 and global_batches_seen % self.args.log_every == 0:
                    cpu_pct = self.proc.cpu_percent(None)
                    rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                    acc = running_correct / max(running_total, 1)
                    dt_ms = (time.perf_counter() - batch_t0) * 1000.0
                    self._record_resource_sample(cpu_pct, rss_mb)
                    self.logger.info(
                        "[client] turns epoch_local=%d steps=%d loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                        local_epoch,
                        global_batches_seen,
                        float(resp.get("loss", 0.0)),
                        acc,
                        cpu_pct,
                        rss_mb,
                        dt_ms,
                    )

                if self.args.max_batches and (global_batches_seen % max(int(self.args.max_batches), 1) == 0):
                    # Debug-only limiter: this does NOT end training; it only yields shorter local epochs.
                    pass

            await self._push_client_weights(ws)
            self._record_resource_sample()

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

    async def _sync_client_weights(self, ws: WebSocketClientProtocol) -> None:
        await ws.send(json.dumps({"type": "get_client_weights"}))
        resp = await self._recv_type({"client_weights", "error"})
        if resp.get("type") != "client_weights":
            return
        payload_b64 = resp.get("state_b64")
        if isinstance(payload_b64, str) and payload_b64:
            state = self._b64_to_state_dict(payload_b64)
            self.model.load_state_dict(state)
            # Weights came from a different client (baton passing). Drop local momentum/history,
            # otherwise stale optimizer buffers can destabilize training.
            self.optimizer.state.clear()

    async def _push_client_weights(self, ws: WebSocketClientProtocol) -> None:
        await ws.send(json.dumps({"type": "put_client_weights", "state_b64": self._state_dict_to_b64()}))
        resp = await self._recv_type({"put_client_weights_ack", "not_your_turn", "error", "training_done"})
        if resp.get("type") == "training_done":
            return
        if resp.get("type") != "put_client_weights_ack":
            raise RuntimeError(f"unexpected server response: {resp}")

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
    p.add_argument("--cut-layer", type=int, default=1, choices=(0, 1))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Total GLOBAL training steps (batches) across all clients (0 derives from --num-epochs)",
    )
    p.add_argument(
        "--turn-batches",
        type=int,
        default=1,
        help="How many batches the current client trains per turn before passing weights",
    )
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
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
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
        "client id=%d/%d device=%s cut_layer=%d epochs=%d batch=%d",
        int(args.client_id),
        int(args.num_clients),
        client.device,
        args.cut_layer,
        args.num_epochs,
        args.batch_size,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt; stopping early")


if __name__ == "__main__":
    main()
