from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch
from torch import nn
import websockets
from websockets.server import WebSocketServerProtocol

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
class ServerConfig:
    host: str
    port: int
    cut_layer: int
    learning_rate: float
    momentum: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    log_every: int
    cpu_seconds: int
    memory_mb: int
    max_message_mb: int
    output_dir: str
    checkpoint_name: str
    resource_report_name: str
    resume_checkpoint: Optional[str]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _enforce_limits(cfg: ServerConfig, logger: logging.Logger) -> None:
    if resource is None:
        if cfg.cpu_seconds > 0 or cfg.memory_mb > 0:
            logger.warning("resource module unavailable; hard limits disabled on this platform")
        return
    if cfg.cpu_seconds > 0:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
        except (ValueError, OSError) as exc:
            logger.warning("failed to apply RLIMIT_CPU: %s", exc)
    if cfg.memory_mb > 0:
        as_bytes = cfg.memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError) as exc:
            logger.warning("failed to apply RLIMIT_AS: %s; will monitor RSS instead", exc)


class SplitServerNet(nn.Module):
    def __init__(self, cut_layer: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if cut_layer not in (0, 1):
            raise ValueError("cut_layer must be 0 or 1")
        self.cut_layer = cut_layer
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        if cut_layer == 0:
            self.net = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 2),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ResourceTracker:
    cpu_samples: list[float]
    rss_samples: list[float]
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


class SplitLearningServer:
    def __init__(self, cfg: ServerConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.device = _pick_device()
        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / cfg.checkpoint_name
        self.resource_report_path = self.output_dir / cfg.resource_report_name
        self.model = SplitServerNet(
            cut_layer=cfg.cut_layer,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.batch_count = 0
        self.resource_tracker = ResourceTracker([], [])
        self._finalized = False
        self._last_summary: Optional[Dict[str, Any]] = None
        if cfg.resume_checkpoint:
            self._maybe_resume_from_checkpoint(Path(cfg.resume_checkpoint))

    async def handle(self, websocket: WebSocketServerProtocol) -> None:
        client = websocket.remote_address
        await websocket.send(
            json.dumps(
                {
                    "type": "server_info",
                    "device": str(self.device),
                    "cut_layer": self.cfg.cut_layer,
                }
            )
        )
        self.logger.info(f"client connected from {client}")
        try:
            async for message in websocket:
                req = json.loads(message)
                kind = req.get("type")
                if kind == "train":
                    await self._handle_train(req, websocket)
                elif kind == "infer":
                    await self._handle_infer(req, websocket)
                elif kind == "finalize":
                    await self._handle_finalize(websocket)
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "unknown request"}))
        except websockets.ConnectionClosedOK:
            self.logger.info(f"client {client} disconnected")
        except websockets.ConnectionClosedError:
            self.logger.warning(f"client {client} closed unexpectedly")
        except Exception:
            self.logger.exception("server error during client session")
            await websocket.close(code=1011, reason="internal error")

    async def _handle_train(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        batch_t0 = time.perf_counter()
        acts = _payload_to_tensor(req["activations"]).to(self.device)
        labels = _payload_to_tensor(req["labels"]).to(self.device)
        self.model.train()
        acts.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(acts)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        grads = acts.grad.detach().cpu()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = int((preds == labels).sum().item())
            total = int(labels.numel())
        self.batch_count += 1
        cpu_pct, rss_mb = self._record_current_process_stats()
        if self.cfg.log_every > 0 and self.batch_count % self.cfg.log_every == 0:
            dt_ms = (time.perf_counter() - batch_t0) * 1000.0
            self.logger.info(
                "[server] batches=%d loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                self.batch_count,
                float(loss.item()),
                correct / max(total, 1),
                cpu_pct,
                rss_mb,
                dt_ms,
            )
        await websocket.send(
            json.dumps(
                {
                    "type": "train_ack",
                    "loss": float(loss.item()),
                    "correct": correct,
                    "total": total,
                    "grads": _tensor_to_payload(grads),
                }
            )
        )

    async def _handle_infer(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        acts = _payload_to_tensor(req["activations"]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(acts).detach().cpu()
        await websocket.send(json.dumps({"type": "logits", "logits": _tensor_to_payload(logits)}))

    async def _handle_finalize(self, websocket: WebSocketServerProtocol) -> None:
        self._record_current_process_stats()
        summary = self._resource_summary()
        if not self._finalized:
            checkpoint_path = self._write_checkpoint()
            summary["checkpoint_path"] = checkpoint_path
            summary["resource_report_path"] = str(self.resource_report_path)
            self._write_resource_report(summary)
            self._last_summary = summary
            self._finalized = True
        else:
            summary = self._last_summary or summary
        await websocket.send(
            json.dumps(
                {
                    "type": "finalized",
                    "resource_summary": summary,
                }
            )
        )

    def _record_current_process_stats(self) -> tuple[float, float]:
        cpu_pct = self.proc.cpu_percent(None)
        rss_mb = self.proc.memory_info().rss / (1024 * 1024)
        self.resource_tracker.record(cpu_pct, rss_mb)
        self._check_memory_budget(rss_mb)
        return float(cpu_pct), float(rss_mb)

    def _check_memory_budget(self, rss_mb: float) -> None:
        if self.cfg.memory_mb > 0 and rss_mb > self.cfg.memory_mb:
            msg = (
                f"server RSS {rss_mb:.1f}MB exceeded limit {self.cfg.memory_mb}MB; "
                "aborting to respect memory constraint"
            )
            self.logger.error(msg)
            raise MemoryError(msg)

    def _resource_summary(self) -> Dict[str, Any]:
        stats = self.resource_tracker.summary()
        stats.update(
            {
                "limits": {
                    "cpu_seconds": self.cfg.cpu_seconds,
                    "memory_mb": self.cfg.memory_mb,
                },
                "device": str(self.device),
            }
        )
        return stats

    def _write_checkpoint(self) -> str:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "cut_layer": self.cfg.cut_layer,
        }
        torch.save(payload, self.checkpoint_path)
        self.logger.info("saved server checkpoint to %s", self.checkpoint_path)
        return str(self.checkpoint_path)

    def _write_resource_report(self, summary: Dict[str, Any]) -> None:
        with self.resource_report_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        self.logger.info("wrote server resource report to %s", self.resource_report_path)

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
        self.logger.info("resumed server weights from %s", ckpt)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split learning server for Alibaba SLA violation prediction")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cut-layer", type=int, default=1, choices=(0, 1))
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--cpu-seconds", type=int, default=0, help="Hard CPU seconds limit (0 disables)")
    p.add_argument("--memory-mb", type=int, default=0, help="Hard address-space limit in MB (0 disables)")
    p.add_argument(
        "--max-message-mb",
        type=int,
        default=128,
        help="Maximum accepted websocket frame size in MB",
    )
    p.add_argument("--output-dir", default=".", help="Directory for checkpoints/metrics")
    p.add_argument("--checkpoint-name", default="server_checkpoint.pt")
    p.add_argument("--resource-report-name", default="server_resources.json")
    p.add_argument("--resume-checkpoint", default="", help="Optional server checkpoint to resume")
    return p.parse_args(argv)


async def _run_server(cfg: ServerConfig, logger: logging.Logger) -> None:
    server = SplitLearningServer(cfg, logger)
    max_size = cfg.max_message_mb * 1024 * 1024
    async with websockets.serve(server.handle, cfg.host, cfg.port, max_size=max_size):
        logger.info(
            "listening on ws://%s:%d cut_layer=%d device=%s",
            cfg.host,
            cfg.port,
            cfg.cut_layer,
            server.device,
        )
        await asyncio.Future()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("split-server")
    cfg = ServerConfig(
        host=args.host,
        port=args.port,
        cut_layer=args.cut_layer,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        log_every=args.log_every,
        cpu_seconds=args.cpu_seconds,
        memory_mb=args.memory_mb,
        max_message_mb=args.max_message_mb,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        resource_report_name=args.resource_report_name,
        resume_checkpoint=args.resume_checkpoint or None,
    )
    _enforce_limits(cfg, logger)
    try:
        asyncio.run(_run_server(cfg, logger))
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt; shutting down")


if __name__ == "__main__":
    main()
