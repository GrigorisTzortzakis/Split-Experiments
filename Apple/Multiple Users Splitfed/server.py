from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import random
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import psutil
import torch
from torch import nn

from split_models import SplitServerNet

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    sfl_variant: str
    expected_clients: int
    learning_rate: float
    momentum: float
    weight_decay: float
    log_every: int
    device: str
    seed: int
    cpu_seconds: int
    memory_mb: int
    max_message_mb: int
    output_dir: str
    checkpoint_name: str
    resource_report_name: str
    resume_checkpoint: Optional[str]
    write_checkpoints: bool
    mlflow_enabled: bool
    mlflow_uri: str
    mlflow_experiment: str
    mlflow_run_name: str
    mlflow_tags: list[str]


def _pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
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
        torch.backends.cudnn.benchmark = False


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
        _set_seed(int(cfg.seed))
        self.device = _pick_device(cfg.device)
        self.output_dir = Path(cfg.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / cfg.checkpoint_name
        self.resource_report_path = self.output_dir / cfg.resource_report_name
        self.model = SplitServerNet(cut_layer=cfg.cut_layer).to(self.device)
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
        self._step_lock = asyncio.Lock()
        self._client_id_counter = 0
        self._client_ids: dict[WebSocketServerProtocol, int] = {}
        self._ws_by_client_id: dict[int, WebSocketServerProtocol] = {}
        self._clients_helloed: set[int] = set()
        self._clients_done_training: set[int] = set()
        self._clients_finalized: set[int] = set()
        self._finalize_waiters: set[WebSocketServerProtocol] = set()
        self._expected_clients_override: Optional[int] = None
        self._client_reports: dict[int, Dict[str, Any]] = {}
        self._confusion_matrix_sum: Optional[torch.Tensor] = None
        self._client_conns: set[WebSocketServerProtocol] = set()

        # SplitFed behavior (Thapa et al. SplitFed)
        self._sfl_variant = str(cfg.sfl_variant)
        self._global_step: int = 0

        # Client sample counts for weighted aggregation (nk).
        self._nk_by_client_id: Dict[int, int] = {}

        # Global server-side weights by round (W^S_t). Round 0 is the deterministic init.
        self._ws_global_by_round: Dict[int, Dict[str, torch.Tensor]] = {}
        self._ws_global_hash_by_round: Dict[int, str] = {}
        ws0 = self._copy_state_dict_cpu(self.model.state_dict())
        self._ws_global_by_round[0] = ws0
        self._ws_global_hash_by_round[0] = self._hash_state_dict(ws0)

        # SFLV1: single shared server model; aggregated-gradient update.
        # round_id -> client_id -> deque[(step_idx, activations, labels, websocket)]
        self._pending_v1: Dict[int, Dict[int, Deque[tuple[int, TensorPayload, TensorPayload, WebSocketServerProtocol]]]] = {}
        self._v1_processing: Dict[int, bool] = {}

        # SFLV2: single shared server model; sequential updates in randomized order.
        # round_id -> deque[(client_id, step_idx, activations, labels, websocket)]
        self._pending_v2: Dict[int, Deque[tuple[int, int, TensorPayload, TensorPayload, WebSocketServerProtocol]]] = {}
        self._v2_processing: Dict[int, bool] = {}

        # Round completion tracking (for server_round_ready barrier).
        self._round_done: Dict[int, Dict[int, int]] = {}

        # Optional MLflow tracking (server-side only).
        self._mlflow_enabled = bool(getattr(cfg, "mlflow_enabled", False))
        self._mlflow_logged = False
        self._mlflow = None
        if self._mlflow_enabled:
            try:
                import mlflow  # type: ignore

                self._mlflow = mlflow
                if cfg.mlflow_uri:
                    mlflow.set_tracking_uri(cfg.mlflow_uri)
                if cfg.mlflow_experiment:
                    mlflow.set_experiment(cfg.mlflow_experiment)
                mlflow.start_run(run_name=(cfg.mlflow_run_name or None))
                for item in (cfg.mlflow_tags or []):
                    if isinstance(item, str) and "=" in item:
                        k, v = item.split("=", 1)
                        mlflow.set_tag(k.strip(), v.strip())
                mlflow.log_params(
                    {
                        "cut_layer": int(cfg.cut_layer),
                        "learning_rate": float(cfg.learning_rate),
                        "momentum": float(cfg.momentum),
                        "weight_decay": float(cfg.weight_decay),
                        "expected_clients": int(self._expected_clients()),
                        "seed": int(cfg.seed),
                        "device": str(self.device),
                    }
                )
            except Exception as exc:
                # Don't crash training if MLflow isn't available/misconfigured.
                self.logger.warning("MLflow disabled due to error: %s", exc)
                self._mlflow_enabled = False

        if cfg.resume_checkpoint:
            self._maybe_resume_from_checkpoint(Path(cfg.resume_checkpoint))

    async def handle(self, websocket: WebSocketServerProtocol) -> None:
        client = websocket.remote_address
        self._client_conns.add(websocket)
        await websocket.send(
            json.dumps(
                {
                    "type": "server_info",
                    "device": str(self.device),
                    "cut_layer": self.cfg.cut_layer,
                    "expected_clients": self._expected_clients(),
                    "sfl_variant": str(self._sfl_variant),
                }
            )
        )
        self.logger.info(f"client connected from {client}")
        try:
            async for message in websocket:
                req = json.loads(message)
                kind = req.get("type")
                if kind == "hello":
                    await self._handle_hello(req, websocket)
                elif kind == "train":
                    await self._handle_train(req, websocket)
                elif kind == "infer":
                    await self._handle_infer(req, websocket)
                elif kind == "round_done":
                    await self._handle_round_done(req, websocket)
                elif kind == "done_training":
                    await self._handle_done_training(websocket)
                elif kind == "finalize":
                    await self._handle_finalize(req, websocket)
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "unknown request"}))
        except websockets.ConnectionClosedOK:
            self.logger.info(f"client {client} disconnected")
        except websockets.ConnectionClosedError:
            self.logger.warning(f"client {client} closed unexpectedly")
        except Exception:
            self.logger.exception("server error during client session")
            await websocket.close(code=1011, reason="internal error")
        finally:
            cid = self._client_ids.pop(websocket, None)
            self._finalize_waiters.discard(websocket)
            self._client_conns.discard(websocket)
            if cid is not None:
                self._ws_by_client_id.pop(cid, None)
                self._clients_helloed.discard(cid)
                self._nk_by_client_id.pop(int(cid), None)
                # Best-effort cleanup: drop any pending entries for this client.
                try:
                    for _round, per_client in list(self._pending_v1.items()):
                        q = per_client.get(int(cid))
                        if q is not None:
                            q.clear()
                    for _round, q2 in list(self._pending_v2.items()):
                        self._pending_v2[_round] = deque([item for item in q2 if int(item[0]) != int(cid)])
                except Exception:
                    pass
                self.logger.info("client_id=%s cleaned up", cid)

    def _copy_state_dict_cpu(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {str(k): v.detach().cpu().clone() for k, v in state_dict.items()}

    def _hash_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    async def _broadcast(self, msg: Dict[str, Any]) -> None:
        payload = json.dumps(msg)
        for ws in list(self._ws_by_client_id.values()):
            try:
                await ws.send(payload)
            except Exception:
                self.logger.debug("broadcast failed", exc_info=True)

    def _expected_clients(self) -> int:
        # Default comes from CLI/config, but allow the clients to tell us
        # how many participants are in the run so we can aggregate reports.
        return int(self._expected_clients_override or int(self.cfg.expected_clients) or 2)

    def _get_or_assign_client_id(self, websocket: WebSocketServerProtocol, requested: Optional[int] = None) -> int:
        if websocket in self._client_ids:
            return self._client_ids[websocket]
        if requested is not None:
            cid = int(requested)
        else:
            cid = self._client_id_counter
            self._client_id_counter += 1
        self._client_ids[websocket] = cid
        self._ws_by_client_id[cid] = websocket
        return cid

    def _state_dict_to_b64(self, state_dict: Dict[str, Any]) -> str:
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _b64_to_state_dict(self, payload_b64: str) -> Dict[str, Any]:
        raw = base64.b64decode(payload_b64.encode("ascii"))
        buf = io.BytesIO(raw)
        obj = torch.load(buf, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("invalid state_dict payload")
        return obj

    async def _handle_hello(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        requested = req.get("client_id")
        try:
            requested_expected = int(req.get("num_clients")) if req.get("num_clients") is not None else None
        except (TypeError, ValueError):
            requested_expected = None
        if requested_expected and requested_expected > 0:
            if self._expected_clients_override is None:
                self._expected_clients_override = requested_expected
            else:
                # Be permissive: keep the maximum to avoid under-counting.
                self._expected_clients_override = max(self._expected_clients_override, requested_expected)
        cid = self._get_or_assign_client_id(websocket, requested)

        # Save nk now (paper Algorithm-style server update uses nk/n during training).
        try:
            nk = int(req.get("nk")) if req.get("nk") is not None else None
        except (TypeError, ValueError):
            nk = None
        if nk is not None and nk > 0:
            self._nk_by_client_id[int(cid)] = int(nk)

        if len(set(self._client_ids.values())) > self._expected_clients():
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"too many clients; expected {self._expected_clients()}",
                    }
                )
            )
            await websocket.close(code=1008, reason="too many clients")
            return
        await websocket.send(
            json.dumps(
                {
                    "type": "hello_ack",
                    "client_id": cid,
                    "expected_clients": self._expected_clients(),
                }
            )
        )

        self._clients_helloed.add(int(cid))
    # No turn-based scheduling in SplitFed.

    async def _handle_train(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        cid = self._get_or_assign_client_id(websocket)
        round_id = int(req.get("round_id", 0) or 0)
        step_idx = int(req.get("step_idx", 0) or 0)

        if self._sfl_variant == "v1":
            per_round = self._pending_v1.setdefault(round_id, {})
            q = per_round.setdefault(int(cid), deque())
            q.append((step_idx, req["activations"], req["labels"], websocket))
            if not self._v1_processing.get(round_id, False):
                self._v1_processing[round_id] = True
                asyncio.create_task(self._process_v1_round(round_id))
            return

        q2 = self._pending_v2.setdefault(round_id, deque())
        q2.append((int(cid), step_idx, req["activations"], req["labels"], websocket))
        if not self._v2_processing.get(round_id, False):
            self._v2_processing[round_id] = True
            asyncio.create_task(self._process_v2_round(round_id))
        return

    async def _process_v1_round(self, round_id: int) -> None:
        try:
            expected = self._expected_clients()
            while True:
                per_round = self._pending_v1.get(round_id) or {}
                if expected <= 0:
                    break

                # Need at least one pending batch per client.
                if not all((cid in per_round and len(per_round[cid]) > 0) for cid in range(expected)):
                    break

                popped: Dict[int, tuple[int, TensorPayload, TensorPayload, WebSocketServerProtocol]] = {}
                for cid in range(expected):
                    popped[cid] = per_round[cid].popleft()

                nks = {cid: int(self._nk_by_client_id.get(cid, 1)) for cid in range(expected)}
                total_n = sum(max(v, 0) for v in nks.values())
                if total_n <= 0:
                    total_n = expected
                    nks = {cid: 1 for cid in range(expected)}
                weights = {cid: float(nks[cid]) / float(total_n) for cid in range(expected)}

                async with self._step_lock:
                    batch_t0 = time.perf_counter()
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)

                    acts_t: Dict[int, torch.Tensor] = {}
                    logits_t: Dict[int, torch.Tensor] = {}
                    labels_t: Dict[int, torch.Tensor] = {}
                    losses_t: Dict[int, torch.Tensor] = {}

                    for cid, (step_idx, acts_payload, labels_payload, _ws) in popped.items():
                        acts = _payload_to_tensor(acts_payload).to(self.device)
                        labels = _payload_to_tensor(labels_payload).to(self.device)
                        acts.requires_grad_(True)
                        logits = self.model(acts)
                        loss = self.criterion(logits, labels)
                        acts_t[cid] = acts
                        logits_t[cid] = logits
                        labels_t[cid] = labels
                        losses_t[cid] = loss

                    total_loss = None
                    for cid in range(expected):
                        w = float(weights.get(cid, 0.0))
                        part = losses_t[cid] * w
                        total_loss = part if total_loss is None else (total_loss + part)
                    assert total_loss is not None
                    total_loss.backward()
                    self.optimizer.step()

                    losses: list[float] = []
                    correct_sum = 0
                    total_sum = 0
                    acks: list[tuple[WebSocketServerProtocol, Dict[str, Any]]] = []
                    next_step = int(self._global_step + 1)
                    for cid, (step_idx, _acts_payload, _labels_payload, ws) in popped.items():
                        acts = acts_t[cid]
                        logits = logits_t[cid]
                        labels = labels_t[cid]
                        loss_val = float(losses_t[cid].item())
                        grads_acts_cpu = acts.grad.detach().cpu()
                        with torch.no_grad():
                            preds = logits.argmax(dim=1)
                            correct = int((preds == labels).sum().item())
                            total = int(labels.numel())
                        losses.append(loss_val)
                        correct_sum += correct
                        total_sum += total
                        acks.append(
                            (
                                ws,
                                {
                                    "type": "train_ack",
                                    "loss": loss_val,
                                    "correct": correct,
                                    "total": total,
                                    "grads": _tensor_to_payload(grads_acts_cpu),
                                    "round_id": int(round_id),
                                    "step_idx": int(step_idx),
                                    "global_step": next_step,
                                    "sfl_variant": "v1",
                                },
                            )
                        )

                    self._global_step += 1
                    self.batch_count += expected
                    cpu_pct, rss_mb = self._record_current_process_stats()
                    if self.cfg.log_every > 0 and self.batch_count % self.cfg.log_every == 0:
                        dt_ms = (time.perf_counter() - batch_t0) * 1000.0
                        avg_loss = sum(losses) / max(len(losses), 1)
                        acc = correct_sum / max(total_sum, 1)
                        self.logger.info(
                            "[server:v1] steps=%d round=%d agg_update clients=%d avg_loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                            self._global_step,
                            round_id,
                            expected,
                            float(avg_loss),
                            float(acc),
                            cpu_pct,
                            rss_mb,
                            dt_ms,
                        )
                        if self._mlflow_enabled and (self._mlflow is not None):
                            try:
                                self._mlflow.log_metric("train_loss", float(avg_loss), step=int(self._global_step))
                                self._mlflow.log_metric("train_acc", float(acc), step=int(self._global_step))
                            except Exception:
                                pass

                for ws, payload in acks:
                    try:
                        await ws.send(json.dumps(payload))
                    except Exception:
                        self.logger.debug("failed to send train_ack (v1)", exc_info=True)
        finally:
            self._v1_processing[round_id] = False

    async def _process_v2_round(self, round_id: int) -> None:
        try:
            while True:
                q = self._pending_v2.get(round_id)
                if not q:
                    break
                seed = int(self.cfg.seed) + int(round_id) * 1000003 + int(self._global_step)
                rng = random.Random(seed)
                idx = rng.randrange(len(q))
                client_id, step_idx, acts_payload, labels_payload, ws = q[idx]
                del q[idx]
                if len(q) == 0:
                    self._pending_v2.pop(round_id, None)

                async with self._step_lock:
                    batch_t0 = time.perf_counter()
                    acts = _payload_to_tensor(acts_payload).to(self.device)
                    labels = _payload_to_tensor(labels_payload).to(self.device)
                    self.model.train()
                    acts.requires_grad_(True)
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(acts)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                    grads_acts_cpu = acts.grad.detach().cpu()
                    with torch.no_grad():
                        preds = logits.argmax(dim=1)
                        correct = int((preds == labels).sum().item())
                        total = int(labels.numel())

                    self.batch_count += 1
                    self._global_step += 1
                    cpu_pct, rss_mb = self._record_current_process_stats()
                    if self.cfg.log_every > 0 and self.batch_count % self.cfg.log_every == 0:
                        dt_ms = (time.perf_counter() - batch_t0) * 1000.0
                        self.logger.info(
                            "[server:v2] steps=%d round=%d client_id=%d loss=%.4f acc=%.4f cpu=%.1f rss=%.1fMB dt=%.1fms",
                            self._global_step,
                            round_id,
                            int(client_id),
                            float(loss.item()),
                            correct / max(total, 1),
                            cpu_pct,
                            rss_mb,
                            dt_ms,
                        )
                        if self._mlflow_enabled and (self._mlflow is not None):
                            try:
                                self._mlflow.log_metric("train_loss", float(loss.item()), step=int(self._global_step))
                                self._mlflow.log_metric("train_acc", correct / max(total, 1), step=int(self._global_step))
                            except Exception:
                                pass

                try:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "train_ack",
                                "loss": float(loss.item()),
                                "correct": correct,
                                "total": total,
                                "grads": _tensor_to_payload(grads_acts_cpu),
                                "round_id": round_id,
                                "step_idx": step_idx,
                                "sfl_variant": "v2",
                            }
                        )
                    )
                except Exception:
                    self.logger.debug("failed to send train_ack (v2)", exc_info=True)
        finally:
            self._v2_processing[round_id] = False

    async def _handle_infer(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        async with self._step_lock:
            acts = _payload_to_tensor(req["activations"]).to(self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(acts).detach().cpu()
            await websocket.send(json.dumps({"type": "logits", "logits": _tensor_to_payload(logits)}))

    async def _handle_round_done(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        cid = int(req.get("client_id"))
        round_id = int(req.get("round_id"))
        nk = int(req.get("nk"))
        steps_in_round = req.get("steps_in_round")

        await websocket.send(json.dumps({"type": "round_done_ack", "round_id": round_id}))

        expected = self._expected_clients()
        done = self._round_done.setdefault(round_id, {})
        done[cid] = nk

        if expected > 0 and len(done) < expected:
            return

        # End-of-round barrier reached: snapshot server weights and broadcast readiness.
        next_round = int(round_id) + 1

        self._ws_global_by_round[next_round] = self._copy_state_dict_cpu(self.model.state_dict())
        self._ws_global_hash_by_round[next_round] = self._hash_state_dict(self._ws_global_by_round[next_round])
        self._round_done.pop(round_id, None)

        # Best-effort cleanup.
        self._pending_v1.pop(round_id, None)
        self._pending_v2.pop(round_id, None)
        self._v1_processing.pop(round_id, None)
        self._v2_processing.pop(round_id, None)

        await self._broadcast({"type": "server_round_ready", "round_id": next_round})
        self.logger.info("[server] server_round_ready round=%d", next_round)

    async def _handle_done_training(self, websocket: WebSocketServerProtocol) -> None:
        # Paper-aligned behavior: no global barrier. A client may proceed independently.
        cid = self._get_or_assign_client_id(websocket)
        self._clients_done_training.add(cid)
        await websocket.send(
            json.dumps(
                {
                    "type": "train_complete",
                    "done": len(self._clients_done_training),
                    "expected": self._expected_clients(),
                }
            )
        )

    async def _handle_finalize(self, req: Dict[str, Any], websocket: WebSocketServerProtocol) -> None:
        # Paper-aligned behavior: no finalize quorum / no waiting.
        # Each client can finalize independently. We still aggregate all client reports
        # into a single JSON once the expected number have reported.
        cid = self._get_or_assign_client_id(websocket)
        self._clients_finalized.add(cid)

        # Store client-side report if provided.
        # Expected payload (best-effort): {"client_report": {...}}
        try:
            report = req.get("client_report")
            if isinstance(report, dict):
                cm_payload = report.get("confusion_matrix")
                if isinstance(cm_payload, dict):
                    cm = _payload_to_tensor(cm_payload).to(torch.int64)
                    if cm.shape == (10, 10):
                        if self._confusion_matrix_sum is None:
                            self._confusion_matrix_sum = cm.clone()
                        else:
                            self._confusion_matrix_sum += cm
                    # Don't store full confusion matrices per-client in JSON.
                    report = dict(report)
                    report.pop("confusion_matrix", None)
                self._client_reports[int(cid)] = report
        except Exception:
            self.logger.debug("failed to parse client report", exc_info=True)

        async with self._step_lock:
            self._record_current_process_stats()
            summary = self._resource_summary()
            if not self._finalized:
                if self.cfg.write_checkpoints:
                    checkpoint_path = self._write_checkpoint()
                    summary["checkpoint_path"] = checkpoint_path
                self._last_summary = summary
                self._finalized = True
            else:
                summary = self._last_summary or summary

        await websocket.send(json.dumps({"type": "finalized", "resource_summary": summary}))

        # If we have all client reports, write one combined metrics.json.
        try:
            expected = self._expected_clients()
            if expected > 0 and len(self._client_reports) >= expected:
                aggregate: Dict[str, Any] = {}
                if self._confusion_matrix_sum is not None:
                    cm_path = self.output_dir / "confusion_matrix.pt"
                    torch.save(self._confusion_matrix_sum, cm_path)
                    png_path = self.output_dir / "confusion_matrix_counts.png"
                    self._save_confusion_matrix_counts(self._confusion_matrix_sum, png_path)
                    aggregate_metrics = compute_metrics(self._confusion_matrix_sum)
                    total = float(self._confusion_matrix_sum.sum().item())
                    acc = float(self._confusion_matrix_sum.diag().sum().item()) / max(total, 1.0)
                    aggregate_metrics["accuracy"] = acc
                    aggregate = {
                        "metrics": aggregate_metrics,
                        "confusion_matrix_pt": str(cm_path),
                        "confusion_matrix_counts_png": str(png_path),
                    }
                combined = {
                    "type": "split_learning_report",
                    "expected_clients": expected,
                    "received_clients": len(self._client_reports),
                    "server": summary,
                    "clients": {str(k): v for k, v in sorted(self._client_reports.items())},
                    "aggregate": aggregate,
                }
                out_path = self.output_dir / "metrics.json"
                with out_path.open("w", encoding="utf-8") as fh:
                    json.dump(combined, fh, indent=2)
                self.logger.info("wrote combined metrics report to %s", out_path)

                # MLflow logging: do it once when the combined report is written.
                if self._mlflow_enabled and (self._mlflow is not None) and not self._mlflow_logged:
                    try:
                        mlflow = self._mlflow
                        mlflow.log_artifact(str(out_path))
                        if self._confusion_matrix_sum is not None:
                            mlflow.log_artifact(str(cm_path))
                            mlflow.log_artifact(str(png_path))

                        # Log aggregate metrics.
                        if aggregate.get("metrics"):
                            m = aggregate["metrics"]
                            metrics_to_log: Dict[str, float] = {}
                            if isinstance(m, dict):
                                acc = m.get("accuracy")
                                if isinstance(acc, (int, float)):
                                    metrics_to_log["aggregate_accuracy"] = float(acc)
                                macro = m.get("macro")
                                micro = m.get("micro")
                                if isinstance(macro, dict):
                                    for key in ("precision", "recall", "f1"):
                                        if isinstance(macro.get(key), (int, float)):
                                            metrics_to_log[f"aggregate_macro_{key}"] = float(macro[key])
                                if isinstance(micro, dict):
                                    for key in ("precision", "recall", "f1"):
                                        if isinstance(micro.get(key), (int, float)):
                                            metrics_to_log[f"aggregate_micro_{key}"] = float(micro[key])
                            if metrics_to_log:
                                mlflow.log_metrics(metrics_to_log)

                        # Log per-client accuracies (lightweight and useful).
                        for cid_str, report in combined.get("clients", {}).items():
                            if isinstance(report, dict):
                                mm = report.get("metrics")
                                if isinstance(mm, dict) and isinstance(mm.get("accuracy"), (int, float)):
                                    mlflow.log_metric(f"client_{cid_str}_accuracy", float(mm["accuracy"]))

                        # Log model weights as artifacts (temp files; no extra clutter in repo).
                        with tempfile.NamedTemporaryFile(suffix="_server_state.pt", delete=False) as tf:
                            torch.save(self.model.state_dict(), tf.name)
                            mlflow.log_artifact(tf.name, artifact_path="models")
                        try:
                            Path(tf.name).unlink(missing_ok=True)
                        except Exception:
                            pass

                        # Close the run so MLflow UI shows it as finished.
                        try:
                            mlflow.end_run()
                        except Exception:
                            pass
                        self._mlflow_logged = True
                    except Exception:
                        self.logger.debug("MLflow logging failed", exc_info=True)

                # Cleanup legacy per-client report artifacts to avoid clutter when
                # running many clients.
                for pattern in (
                    "metrics_client*.json",
                    "resource_report_client*.json",
                    "confusion_matrix_client*.pt",
                    "confusion_matrix_counts_client*.png",
                ):
                    for path in self.output_dir.glob(pattern):
                        try:
                            path.unlink(missing_ok=True)
                        except Exception:
                            self.logger.debug("failed to remove %s", path, exc_info=True)
        except Exception:
            self.logger.debug("failed to write combined metrics.json", exc_info=True)

    def _save_confusion_matrix_counts(self, cm: torch.Tensor, out_path: Path) -> None:
        cm_np = cm.detach().cpu().numpy()
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
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

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


def compute_metrics(cm: torch.Tensor) -> Dict[str, Any]:
    num_classes = int(cm.shape[0])
    per_class: list[Dict[str, float]] = []
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
    macro_precision = sum(item["precision"] for item in per_class) / max(num_classes, 1)
    macro_recall = sum(item["recall"] for item in per_class) / max(num_classes, 1)
    macro_f1 = sum(item["f1"] for item in per_class) / max(num_classes, 1)
    micro_precision = total_tp / max(total_samples, 1.0)
    micro_recall = total_tp / max(total_samples, 1.0)
    micro_f1 = micro_precision
    return {
        "per_class": per_class,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split learning server for MNIST")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--cut-layer", type=int, default=1, choices=(0, 1))
    p.add_argument(
        "--sfl-variant",
        default="v1",
        choices=("v1", "v2"),
        help=(
            "SplitFed server-side variant: "
            "v1=per-client server models + FedAvg server weights per round; "
            "v2=sync receive per step + random sequential updates on shared server model"
        ),
    )
    p.add_argument(
        "--expected-clients",
        type=int,
        default=2,
        help="Expected number of clients (used for SFLV1 barriers and report aggregation)",
    )
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Device for server model (default: auto)",
    )
    p.add_argument("--seed", type=int, default=17, help="Random seed for deterministic init")
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
    p.add_argument(
        "--write-checkpoints",
        action="store_true",
        help="Write server checkpoint to disk (default: off)",
    )
    p.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking (server-side). Logs metrics.json + confusion artifacts.",
    )
    p.add_argument(
        "--mlflow-uri",
        default="file:./mlruns",
        help="MLflow tracking URI (default: file:./mlruns)",
    )
    p.add_argument(
        "--mlflow-experiment",
        default="split-learning",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--mlflow-run-name",
        default="",
        help="Optional MLflow run name",
    )
    p.add_argument(
        "--mlflow-tag",
        action="append",
        default=[],
        help="MLflow tag in key=value form (repeatable)",
    )
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
        sfl_variant=str(args.sfl_variant),
        expected_clients=int(args.expected_clients),
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        device=args.device,
        seed=int(args.seed),
        cpu_seconds=args.cpu_seconds,
        memory_mb=args.memory_mb,
        max_message_mb=args.max_message_mb,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        resource_report_name=args.resource_report_name,
        resume_checkpoint=args.resume_checkpoint or None,
        write_checkpoints=bool(args.write_checkpoints),
        mlflow_enabled=bool(args.mlflow),
        mlflow_uri=str(args.mlflow_uri or ""),
        mlflow_experiment=str(args.mlflow_experiment or ""),
        mlflow_run_name=str(args.mlflow_run_name or ""),
        mlflow_tags=list(args.mlflow_tag or []),
    )
    _enforce_limits(cfg, logger)
    try:
        asyncio.run(_run_server(cfg, logger))
    except KeyboardInterrupt:
        logger.info("received KeyboardInterrupt; shutting down")


if __name__ == "__main__":
    main()
