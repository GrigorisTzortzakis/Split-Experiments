from __future__ import annotations

import argparse
import asyncio
import base64
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
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms
import websockets
from websockets.client import WebSocketClientProtocol

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX fallback
    resource = None  # type: ignore


TensorPayload = Dict[str, Any]


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


class SplitClientNet(nn.Module):
    def __init__(self, cut_layer: int) -> None:
        super().__init__()
        if cut_layer not in (0, 1):
            raise ValueError("cut_layer must be 0 or 1")
        self.cut_layer = cut_layer
        self.act = nn.Tanh()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        if self.cut_layer == 0:
            return x
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        return x


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
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / args.client_checkpoint
        self.resource_report_path = self.output_dir / args.resource_report
        self.resource_tracker = ResourceTracker([], [])
        self.client_limits = {
            "cpu_seconds": args.cpu_seconds,
            "memory_mb": args.memory_mb,
            "device": str(self.device),
        }
        if args.resume_client_checkpoint:
            self._maybe_resume_from_checkpoint(Path(args.resume_client_checkpoint))

    def _prepare_data(self, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
        root = Path(args.data_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        mnist_norm = transforms.Normalize((0.1307,), (0.3081,))
        tfm = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), mnist_norm])
        train_ds = tv_datasets.MNIST(root=str(root), train=True, download=True, transform=tfm)
        test_ds = tv_datasets.MNIST(root=str(root), train=False, download=True, transform=tfm)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
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
            hello = json.loads(await ws.recv())
            if hello.get("type") == "server_info":
                self.logger.info(
                    "connected to server device=%s cut_layer=%s",
                    hello.get("device"),
                    hello.get("cut_layer"),
                )
            else:
                self.logger.warning("missing server info; continuing anyway")
            await self._train(ws)
            self._save_client_checkpoint()
            metrics = await self._evaluate(ws)
            server_summary = await self._finalize_server(ws)
            self._write_metrics(metrics, server_summary)

    async def _train(self, ws: WebSocketClientProtocol) -> None:
        batches_per_epoch = math.ceil(len(self.train_loader.dataset) / self.args.batch_size)
        if self.args.max_batches:
            batches_per_epoch = min(batches_per_epoch, self.args.max_batches)
        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            seen_batches = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
                batch_t0 = time.perf_counter()
                images = images.to(self.device, non_blocking=True)
                labels_dev = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                acts = self.model(images)
                await ws.send(
                    json.dumps(
                        {
                            "type": "train",
                            "epoch": epoch,
                            "batch": batch_idx,
                            "activations": _tensor_to_payload(acts.detach().cpu()),
                            "labels": _tensor_to_payload(labels_dev.detach().cpu()),
                        }
                    )
                )
                resp = json.loads(await ws.recv())
                if resp.get("type") != "train_ack":
                    raise RuntimeError(f"unexpected server response: {resp}")
                grads = _payload_to_tensor(resp["grads"]).to(self.device)
                acts.backward(grads)
                self.optimizer.step()
                running_loss += float(resp.get("loss", 0.0))
                running_correct += int(resp.get("correct", 0))
                running_total += int(resp.get("total", 0))
                seen_batches += 1
                if self.args.log_every > 0 and batch_idx % self.args.log_every == 0:
                    cpu_pct = self.proc.cpu_percent(None)
                    rss_mb = self.proc.memory_info().rss / (1024 * 1024)
                    acc = running_correct / max(running_total, 1)
                    dt_ms = (time.perf_counter() - batch_t0) * 1000.0
                    self._record_resource_sample(cpu_pct, rss_mb)
                    self.logger.info(
                        "[client] epoch=%d/%d batch=%d/%d loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                        epoch,
                        self.args.num_epochs,
                        batch_idx,
                        batches_per_epoch,
                        float(resp.get("loss", 0.0)),
                        acc,
                        cpu_pct,
                        rss_mb,
                        dt_ms,
                    )
                if self.args.max_batches and batch_idx >= self.args.max_batches:
                    break
            epoch_loss = running_loss / max(seen_batches, 1)
            epoch_acc = running_correct / max(running_total, 1)
            self.logger.info(
                "[client] epoch %d summary: loss=%.4f acc=%.4f",
                epoch,
                epoch_loss,
                epoch_acc,
            )
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
            resp = json.loads(await ws.recv())
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

    def _write_metrics(self, metrics: Dict[str, Any], server_summary: Dict[str, Any]) -> None:
        cm = metrics.pop("confusion_matrix")
        resource_report = {
            "client": self._client_resource_summary(),
            "server": server_summary,
        }
        metrics["resource_report"] = resource_report
        metrics_path = self.output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        torch.save(cm, self.output_dir / "confusion_matrix.pt")
        counts_png = self.output_dir / "confusion_matrix_counts.png"
        plotted = save_confusion_matrix_counts(cm, counts_png, self.logger)
        if plotted:
            self.logger.info("confusion matrix PNG saved to %s", counts_png)
        with self.resource_report_path.open("w", encoding="utf-8") as fh:
            json.dump(resource_report, fh, indent=2)
        self.logger.info("wrote metrics to %s", metrics_path)
        self.logger.info("wrote resource report to %s", self.resource_report_path)
        self._log_metric_summary(metrics)

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
        summary["checkpoint_path"] = str(self.checkpoint_path)
        return summary

    def _save_client_checkpoint(self) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cut_layer": self.args.cut_layer,
            "epoch": self.args.num_epochs,
        }
        torch.save(payload, self.checkpoint_path)
        self.logger.info("saved client checkpoint to %s", self.checkpoint_path)

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

    async def _finalize_server(self, ws: WebSocketClientProtocol) -> Dict[str, Any]:
        try:
            await ws.send(json.dumps({"type": "finalize"}))
            resp = json.loads(await ws.recv())
            if resp.get("type") == "finalized":
                return resp.get("resource_summary", {})
            self.logger.warning("unexpected server finalize response: %s", resp)
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
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0, help="Limit batches per epoch (0 disables)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--client-checkpoint", default="client_checkpoint.pt")
    p.add_argument("--resume-client-checkpoint", default="", help="Path to client checkpoint to resume")
    p.add_argument("--resource-report", default="resource_report.json")
    p.add_argument("--cpu-seconds", type=int, default=0, help="Hard CPU seconds limit (0 disables)")
    p.add_argument("--memory-mb", type=int, default=0, help="Hard address-space limit in MB (0 disables)")
    p.add_argument("--seed", type=int, default=17)
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
    _set_seed(args.seed)
    _enforce_limits(args.cpu_seconds, args.memory_mb, logger)
    client = SplitLearningClient(args, logger)
    logger.info(
        "client device=%s cut_layer=%d epochs=%d batch=%d", client.device, args.cut_layer, args.num_epochs, args.batch_size
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt; stopping early")


if __name__ == "__main__":
    main()
