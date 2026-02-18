#!/usr/bin/env python3
"""
Plot utilities for Split-Framework log files (GLOBAL-EPOCH FIX).

Key fix:
- In multi-client logs, `epoch=` in metric lines increments once per **client**
  completion, so you see (global_epochs * num_clients) "epochs".
- This plots **global epochs / rounds** instead: 1 global epoch = ALL clients
  finished once (one full round).
- We infer intended global epoch count from filename `_<N>epochs_` and number of
  clients from filename prefix `<K>client_` (fallback: max worker_num in comm).
  If the metric epoch span equals `N*K`, we compress by `K` and keep the last
  validation point in each round.

Per-run plots:
1) test-validation accuracy vs global epoch
2) total communication vs global epoch (cumulative bytes)
3) send vs receive vs global epoch (cumulative bytes)
4) test-validation accuracy vs total communication

Optional summary:
5) final tradeoff scatter: final accuracy vs final total communication
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Parsing
# -----------------------------

@dataclass(frozen=True)
class MetricPoint:
    phase: str
    epoch: int
    step: int
    acc: float
    loss: float
    t_seconds: Optional[float] = None


@dataclass(frozen=True)
class CommPoint:
    worker_num: int
    # NOTE: despite the names, these fields are BYTES PER EPOCH as logged by SplitNNClient
    epoch_send: int
    epoch_receive: int
    # cumulative totals (bytes)
    total_send: int
    total_receive: int
    t_seconds: Optional[float] = None


@dataclass(frozen=True)
class RunSeries:
    log_path: Path
    title: str
    label: str
    n_clients: Optional[int]
    epochs_plot: List[int]
    test_acc: List[float]
    test_loss: List[float]
    train_loss: List[float]
    comm_cumulative_total_bytes: List[int]
    comm_cumulative_send_bytes: List[int]
    comm_cumulative_recv_bytes: List[int]


_METRIC_RE = re.compile(
    r"phase=(?P<phase>train|validation|test)\s+"
    r"acc=(?P<acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"loss=(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"epoch=(?P<epoch>\d+)\s+and\s+step=(?P<step>\d+)",
    re.IGNORECASE,
)

_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

_COMM_RE = re.compile(
    r"worker_num=(?P<worker>\d+)\s+"
    r"epoch_send=(?P<epoch_send>\d+)\s+"
    r"epoch_receive=(?P<epoch_receive>\d+)\s+"
    r"total_send=(?P<total_send>\d+)\s+"
    r"total_receive=(?P<total_receive>\d+)",
    re.IGNORECASE,
)

_CLIENTS_PREFIX_RE = re.compile(r"^(?P<n>\d+)\s*client(?:s)?_", re.IGNORECASE)
_EPOCHS_TAG_RE = re.compile(r"_(?P<n>\d+)epochs_", re.IGNORECASE)

# Matches both: '_hetero_a0.5' and '_hetero_something_a0.5'
_HETERO_A_RE = re.compile(r"_hetero(?:_[^_]*)?_a(?P<a>\d+(?:\.\d+)?)", re.IGNORECASE)
_ALPHA_RE = re.compile(r"_alpha(?P<a>\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_timestamp_seconds(line: str, t0: Optional[datetime]) -> Tuple[Optional[float], Optional[datetime]]:
    """Parse log timestamp at start of line and return (elapsed_seconds, t0)."""
    m = _TS_RE.search(line)
    if not m:
        return None, t0
    try:
        t = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    except Exception:
        return None, t0
    if t0 is None:
        t0 = t
    return (t - t0).total_seconds(), t0


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_metrics(log_text: str) -> List[MetricPoint]:
    points: List[MetricPoint] = []
    t0: Optional[datetime] = None
    for line in log_text.splitlines():
        m = _METRIC_RE.search(line)
        if not m:
            continue
        t_seconds, t0 = _parse_timestamp_seconds(line, t0)
        points.append(
            MetricPoint(
                phase=m.group("phase").lower(),
                epoch=int(m.group("epoch")),
                step=int(m.group("step")),
                acc=_safe_float(m.group("acc")),
                loss=_safe_float(m.group("loss")),
                t_seconds=t_seconds,
            )
        )
    return points


def parse_comm(log_text: str) -> List[CommPoint]:
    """Parse SplitNNClient comm counters (send/receive totals)."""
    points: List[CommPoint] = []
    t0: Optional[datetime] = None
    for line in log_text.splitlines():
        if "SplitNNClient" not in line or "total_send" not in line:
            continue
        m = _COMM_RE.search(line)
        if not m:
            continue
        t_seconds, t0 = _parse_timestamp_seconds(line, t0)
        points.append(
            CommPoint(
                worker_num=int(m.group("worker")),
                epoch_send=int(m.group("epoch_send")),
                epoch_receive=int(m.group("epoch_receive")),
                total_send=int(m.group("total_send")),
                total_receive=int(m.group("total_receive")),
                t_seconds=t_seconds,
            )
        )
    points.sort(key=lambda p: (p.t_seconds is None, p.t_seconds or 0.0))
    return points


def _split_by_phase(points: List[MetricPoint]) -> Dict[str, List[MetricPoint]]:
    out: Dict[str, List[MetricPoint]] = {}
    for p in points:
        out.setdefault(p.phase, []).append(p)
    for k in list(out.keys()):
        out[k].sort(key=lambda x: (x.epoch, x.step))
    return out


def _extract_client_count_from_stem(stem: str) -> Optional[int]:
    """Infer number of clients from filename stem, e.g. '10client_lenet_50epochs_...'."""
    m = _CLIENTS_PREFIX_RE.match(stem)
    if not m:
        return None
    try:
        n = int(m.group("n"))
    except Exception:
        return None
    return n if n > 0 else None


def _infer_client_count_from_comm(comm: List[CommPoint]) -> Optional[int]:
    if not comm:
        return None
    n = max((p.worker_num for p in comm), default=0)
    return n if n > 0 else None


def _extract_target_epochs_from_stem(stem: str) -> Optional[int]:
    """Infer intended *global* epoch count from filename stem, e.g. '_50epochs_' -> 50."""
    m = _EPOCHS_TAG_RE.search(stem)
    if not m:
        return None
    try:
        n = int(m.group("n"))
    except Exception:
        return None
    return n if n > 0 else None


def _choose_comm_unit(max_bytes: float) -> Tuple[str, float]:
    """Return (unit, divisor). Use MB unless values reach >= 1 GB."""
    try:
        m = float(max_bytes)
    except Exception:
        m = 0.0
    # Decimal units for readability.
    if m >= 1e9:
        return "GB", 1e9
    return "MB", 1e6


def _scale_bytes(values: List[float], *, unit: str, divisor: float) -> List[float]:
    if divisor <= 0:
        return [0.0 for _ in values]
    return [float(v) / divisor for v in values]


def _scale_bytes_auto(values: List[float]) -> Tuple[List[float], str, float]:
    unit, div = _choose_comm_unit(max(values) if values else 0.0)
    return _scale_bytes(values, unit=unit, divisor=div), unit, div


def _finite_or_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return x if math.isfinite(float(x)) else None
    except Exception:
        return None


def _run_label_from_stem(stem: str, *, n_clients: Optional[int]) -> str:
    # Examples we want:
    # - 1c
    # - 5c-hetero a=0.5
    # - 10c-homo
    base = f"{int(n_clients)}c" if n_clients is not None else stem

    s = stem.lower()
    parts: List[str] = [base]
    if "_homo_" in s:
        parts.append("homo")
    elif "_hetero_" in s:
        parts.append("hetero")

    a_val: Optional[str] = None
    m = _HETERO_A_RE.search(stem)
    if m:
        a_val = m.group("a")
    else:
        m2 = _ALPHA_RE.search(stem)
        if m2:
            a_val = m2.group("a")
    if a_val is not None:
        parts.append(f"a={a_val}")

    if len(parts) == 1:
        return parts[0]
    # 5c-hetero a=0.5 (join first two with '-', rest with spaces)
    head = "-".join(parts[:2])
    tail = " ".join(parts[2:])
    return f"{head} {tail}".strip()


def _scenario_key_from_stem(stem: str) -> str:
    """Group runs for mixed-client comparisons.

    - IID: 'iid'
    - Non-IID: group by the same alpha/a value so comparisons are fair.
      Examples: 'non_iid_a0.5', 'non_iid_a0.1', 'non_iid_alpha0'
    """
    s = stem.lower()

    # Non-IID buckets first
    m = _HETERO_A_RE.search(stem)
    if m:
        return f"non_iid_a{m.group('a')}"
    m2 = _ALPHA_RE.search(stem)
    if m2:
        return f"non_iid_alpha{m2.group('a')}"

    # IID: explicit homo tag OR single-client baseline OR anything that doesn't declare non-IID
    if "_homo_" in s:
        return "iid"
    if s.startswith("1client_") or s.startswith("1clients_"):
        return "iid"
    return "iid"


# -----------------------------
# Global-epoch (round) handling
# -----------------------------

def _choose_epoch_scale(
    val_pts: List[MetricPoint],
    *,
    log_stem: str,
    n_clients: Optional[int],
) -> Tuple[int, Optional[int]]:
    """Return (scale, target_global_epochs).

    scale=1 means metric `epoch=` is already global.
    scale>1 means metric `epoch=` is an inner counter we need to compress.
    """
    target_epochs = _extract_target_epochs_from_stem(log_stem)
    if not val_pts:
        return 1, target_epochs

    min_epoch = min(p.epoch for p in val_pts)
    max_epoch = max(p.epoch for p in val_pts)
    logged_span = (max_epoch - min_epoch) + 1

    # If filename says "50epochs" and we actually logged 0..49 => already global.
    if target_epochs is not None and logged_span == target_epochs:
        return 1, target_epochs

    # Preferred: if we know number of clients and the span is exactly expanded by K clients.
    if (
        target_epochs is not None
        and n_clients is not None
        and logged_span == target_epochs * n_clients
    ):
        return int(n_clients), target_epochs

    # Fallback: if span is an integer multiple of the target, compress by that multiple.
    if target_epochs is not None and logged_span > target_epochs and logged_span % target_epochs == 0:
        return logged_span // target_epochs, target_epochs

    # Otherwise: don't guess; treat logged epoch as global.
    return 1, target_epochs


def validation_last_by_global_epoch(
    phases: Dict[str, List[MetricPoint]],
    *,
    log_stem: str,
    n_clients: Optional[int],
) -> Dict[int, MetricPoint]:
    """Return last validation point per *global epoch* (round)."""
    val_pts = phases.get("validation") or []
    if not val_pts:
        return {}

    scale, target_epochs = _choose_epoch_scale(val_pts, log_stem=log_stem, n_clients=n_clients)

    min_epoch = min(p.epoch for p in val_pts)
    max_epoch = max(p.epoch for p in val_pts)
    logged_span = (max_epoch - min_epoch) + 1

    # Keep only complete rounds (drop a partial last round if present).
    max_full_global = (logged_span // scale) - 1
    if target_epochs is not None:
        max_full_global = min(max_full_global, target_epochs - 1)

    last_by_global: Dict[int, MetricPoint] = {}
    for p in val_pts:
        g = (p.epoch - min_epoch) // scale
        if g < 0 or g > max_full_global:
            continue
        last_by_global[g] = p  # overwrite => keep last point within the round
    return last_by_global


def _global_epoch_params_from_eval_points(
    eval_pts: List[MetricPoint],
    *,
    log_stem: str,
    n_clients: Optional[int],
) -> Optional[Tuple[int, int, int]]:
    """Return (scale, min_epoch_ref, max_full_global) derived from eval points."""
    if not eval_pts:
        return None
    scale, target_epochs = _choose_epoch_scale(eval_pts, log_stem=log_stem, n_clients=n_clients)

    min_epoch = min(p.epoch for p in eval_pts)
    max_epoch = max(p.epoch for p in eval_pts)
    logged_span = (max_epoch - min_epoch) + 1

    max_full_global = (logged_span // scale) - 1
    if target_epochs is not None:
        max_full_global = min(max_full_global, target_epochs - 1)
    return int(scale), int(min_epoch), int(max_full_global)


def _last_points_by_global_epoch(
    pts: List[MetricPoint],
    *,
    min_epoch_ref: int,
    scale: int,
    max_full_global: int,
) -> Dict[int, MetricPoint]:
    if not pts:
        return {}
    last_by_global: Dict[int, MetricPoint] = {}
    for p in sorted(pts, key=lambda x: (x.epoch, x.step)):
        g = (p.epoch - min_epoch_ref) // int(scale)
        if g < 0 or g > max_full_global:
            continue
        last_by_global[int(g)] = p
    return last_by_global


def _match_comm_to_validation_epochs(
    *,
    comm: List[CommPoint],
    val_last_by_epoch: Dict[int, MetricPoint],
) -> Dict[int, Dict[int, CommPoint]]:
    """Match per-worker comm snapshots to each validation epoch by timestamp."""
    if not comm:
        return {}

    by_worker: Dict[int, List[CommPoint]] = {}
    for p in comm:
        by_worker.setdefault(p.worker_num, []).append(p)
    for w in by_worker:
        by_worker[w].sort(key=lambda x: (x.t_seconds is None, x.t_seconds or 0.0))

    out: Dict[int, Dict[int, CommPoint]] = {}
    for epoch, mp in val_last_by_epoch.items():
        if mp.t_seconds is None:
            continue
        epoch_map: Dict[int, CommPoint] = {}
        for w, pts in by_worker.items():
            best: Optional[CommPoint] = None
            for cp in pts:
                if cp.t_seconds is None:
                    continue
                if cp.t_seconds <= mp.t_seconds:
                    best = cp
                else:
                    break
            if best is not None:
                epoch_map[w] = best
        if epoch_map:
            out[epoch] = epoch_map
    return out


def _sum_comm(epoch_map: Dict[int, CommPoint]) -> Tuple[int, int, int, int]:
    """Return (epoch_send_bytes, epoch_recv_bytes, total_send_bytes, total_recv_bytes) summed over workers."""
    epoch_send = sum(p.epoch_send for p in epoch_map.values())
    epoch_recv = sum(p.epoch_receive for p in epoch_map.values())
    total_send = sum(p.total_send for p in epoch_map.values())
    total_recv = sum(p.total_receive for p in epoch_map.values())
    return epoch_send, epoch_recv, total_send, total_recv


def compute_run_series(
    *,
    log_path: Path,
    title: str,
    clients_override: Optional[int] = None,
) -> Optional[RunSeries]:
    text = log_path.read_text(errors="replace")
    metrics = parse_metrics(text)
    comm = parse_comm(text)
    phases = _split_by_phase(metrics)

    n_clients = (
        clients_override
        if clients_override is not None
        else _extract_client_count_from_stem(log_path.stem) or _infer_client_count_from_comm(comm)
    )

    # Prefer explicit test phase if present, otherwise use validation.
    eval_phase = "test" if (phases.get("test") or []) else "validation"
    eval_pts = phases.get(eval_phase) or []
    params = _global_epoch_params_from_eval_points(eval_pts, log_stem=log_path.stem, n_clients=n_clients)
    if params is None:
        return None

    scale, min_epoch_ref, max_full_global = params
    eval_last = _last_points_by_global_epoch(
        eval_pts,
        min_epoch_ref=min_epoch_ref,
        scale=scale,
        max_full_global=max_full_global,
    )
    if not eval_last:
        return None

    epochs = sorted(eval_last.keys())
    epochs_plot = [e + 1 for e in epochs]  # 1-based global epoch (round)
    test_acc = [eval_last[e].acc for e in epochs]
    test_loss = [eval_last[e].loss for e in epochs]

    # Train loss: align to the same global epochs (use same min_epoch_ref/scale)
    train_last = _last_points_by_global_epoch(
        phases.get("train") or [],
        min_epoch_ref=min_epoch_ref,
        scale=scale,
        max_full_global=max_full_global,
    )
    train_loss = [train_last[e].loss if e in train_last else float("nan") for e in epochs]

    # NOTE: comm matching expects a dict[int, MetricPoint] keyed by global epoch.
    # We use the eval_last points here so comm aligns with the plotted test curve.
    comm_by_epoch = _match_comm_to_validation_epochs(comm=comm, val_last_by_epoch=eval_last)

    comm_cumulative_total: List[int] = []
    comm_cumulative_send: List[int] = []
    comm_cumulative_recv: List[int] = []
    for e in epochs:
        epoch_map = comm_by_epoch.get(e, {})
        _es, _er, ts, tr = _sum_comm(epoch_map) if epoch_map else (0, 0, 0, 0)
        comm_cumulative_send.append(int(ts))
        comm_cumulative_recv.append(int(tr))
        comm_cumulative_total.append(int(ts + tr))

    label = _run_label_from_stem(log_path.stem, n_clients=n_clients)
    return RunSeries(
        log_path=log_path,
        title=title,
        label=label,
        n_clients=n_clients,
        epochs_plot=epochs_plot,
        test_acc=test_acc,
        test_loss=test_loss,
        train_loss=train_loss,
        comm_cumulative_total_bytes=comm_cumulative_total,
        comm_cumulative_send_bytes=comm_cumulative_send,
        comm_cumulative_recv_bytes=comm_cumulative_recv,
    )


# -----------------------------
# Plotting
# -----------------------------

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_required(
    *,
    log_path: Path,
    out_dir: Path,
    title: str,
    clients_override: Optional[int] = None,
) -> Tuple[List[Path], Optional[Tuple[float, int]]]:
    """Generate the 4 per-run required plots. Returns (written_files, final_point)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[Path] = []
    series = compute_run_series(log_path=log_path, title=title, clients_override=clients_override)
    if series is None:
        return written, None

    n_clients = series.n_clients
    epochs_plot = series.epochs_plot
    test_acc = series.test_acc
    test_loss = series.test_loss
    train_loss = series.train_loss

    comm_cumulative_total = series.comm_cumulative_total_bytes
    comm_cumulative_send = series.comm_cumulative_send_bytes
    comm_cumulative_recv = series.comm_cumulative_recv_bytes

    # Scale comm to MB by default; switch to GB if values reach >= 1GB.
    comm_scale_unit, comm_scale_div = _choose_comm_unit(
        max([
            max(comm_cumulative_total) if comm_cumulative_total else 0,
            max(comm_cumulative_send) if comm_cumulative_send else 0,
            max(comm_cumulative_recv) if comm_cumulative_recv else 0,
        ])
    )
    comm_cumulative_total_scaled = _scale_bytes([float(b) for b in comm_cumulative_total], unit=comm_scale_unit, divisor=comm_scale_div)
    comm_cumulative_send_scaled = _scale_bytes([float(b) for b in comm_cumulative_send], unit=comm_scale_unit, divisor=comm_scale_div)
    comm_cumulative_recv_scaled = _scale_bytes([float(b) for b in comm_cumulative_recv], unit=comm_scale_unit, divisor=comm_scale_div)

    # Per-round (delta cumulative) total comm
    comm_per_round_total_bytes: List[int] = []
    for i, b in enumerate(comm_cumulative_total):
        if i == 0:
            comm_per_round_total_bytes.append(int(max(0, b)))
        else:
            comm_per_round_total_bytes.append(int(max(0, b - comm_cumulative_total[i - 1])))
    comm_per_round_scaled, comm_per_round_unit, comm_per_round_div = _scale_bytes_auto([float(b) for b in comm_per_round_total_bytes])

    comm_cumulative_total_scaled_per_client: Optional[List[float]] = None
    comm_per_round_scaled_per_client: Optional[List[float]] = None
    comm_per_round_unit_per_client: Optional[str] = None
    if n_clients is not None and int(n_clients) > 0:
        per_client_bytes = [float(b) / int(n_clients) for b in comm_cumulative_total]
        comm_cumulative_total_scaled_per_client, comm_pc_unit, comm_pc_div = _scale_bytes_auto(per_client_bytes)

        per_round_per_client_bytes = [float(b) / int(n_clients) for b in comm_per_round_total_bytes]
        comm_per_round_scaled_per_client, comm_per_round_unit_per_client, _ = _scale_bytes_auto(per_round_per_client_bytes)

    x_label = "global epoch (round)"

    # 1) test accuracy vs global epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, test_acc, label="test acc", linewidth=2)
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — test accuracy vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_acc_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 1b) test loss vs global epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, test_loss, label="test loss", linewidth=2)
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — test loss vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_loss_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 1c) training + test loss vs global epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, train_loss, label="train loss", linewidth=2)
    ax.plot(epochs_plot, test_loss, label="test loss", linewidth=2, linestyle="--")
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — train & test loss vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_train_loss_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 2) Total communication vs global epoch (cumulative)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epochs_plot,
        comm_cumulative_total_scaled,
        label=f"cumulative_total_{comm_scale_unit} (send+recv, sum over clients)",
        linewidth=2,
    )
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — total communication vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(comm_scale_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_total_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 2b) Total communication per client vs global epoch (cumulative / client)
    if comm_cumulative_total_scaled_per_client is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            epochs_plot,
            comm_cumulative_total_scaled_per_client,
            label=f"cumulative_total_{comm_pc_unit}_per_client (send+recv)",
            linewidth=2,
        )
        if epochs_plot:
            ax.set_xlim(min(epochs_plot), max(epochs_plot))
        ax.set_title(f"{title} — total communication per client vs global epoch")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{comm_pc_unit} / client")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_comm_total_per_client_vs_global_epoch.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

    # 2c) Communication per round vs global epoch (delta cumulative)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epochs_plot,
        comm_per_round_scaled,
        label=f"per_round_total_{comm_per_round_unit} (delta cumulative, sum over clients)",
        linewidth=2,
    )
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — communication per round vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{comm_per_round_unit} / round")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_per_round_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 2d) Communication per round per client vs global epoch
    if comm_per_round_scaled_per_client is not None and comm_per_round_unit_per_client is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            epochs_plot,
            comm_per_round_scaled_per_client,
            label=f"per_round_total_{comm_per_round_unit_per_client}_per_client (delta cumulative)",
            linewidth=2,
        )
        if epochs_plot:
            ax.set_xlim(min(epochs_plot), max(epochs_plot))
        ax.set_title(f"{title} — communication per round per client vs global epoch")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{comm_per_round_unit_per_client} / (round·client)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_comm_per_round_per_client_vs_global_epoch.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

    # 3) Send vs receive vs global epoch (cumulative)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, comm_cumulative_send_scaled, label=f"cumulative_send_{comm_scale_unit}", linewidth=2)
    ax.plot(epochs_plot, comm_cumulative_recv_scaled, label=f"cumulative_receive_{comm_scale_unit}", linewidth=2)
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} — send vs receive vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(comm_scale_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_send_recv_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 4) test accuracy vs total communication (scaled)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(comm_cumulative_total_scaled, test_acc, label="test acc", linewidth=2)
    ax.set_title(f"{title} — test accuracy vs total communication")
    ax.set_xlabel(f"cumulative communication ({comm_scale_unit}, send+recv, sum over clients)")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_acc_vs_total_comm.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # 4b) test accuracy vs total communication per client (scaled / client)
    if comm_cumulative_total_scaled_per_client is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(comm_cumulative_total_scaled_per_client, test_acc, label="test acc", linewidth=2)
        ax.set_title(f"{title} — test accuracy vs communication per client")
        ax.set_xlabel(f"cumulative communication ({comm_pc_unit}/client, send+recv)")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_test_acc_vs_total_comm_per_client.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

    final_acc = float(test_acc[-1]) if test_acc else float("nan")
    final_comm = int(comm_cumulative_total[-1]) if comm_cumulative_total else 0
    return written, (final_acc, final_comm)


def plot_mixed_clients_overlays(*, runs: List[RunSeries], out_dir: Path, title_prefix: str) -> List[Path]:
    """Write overlay plots (all runs in same figure) into mixed_clients folder."""
    if not runs:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    # A) Test accuracy vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ax.plot(r.epochs_plot, r.test_acc, linewidth=2, label=r.label)
    ax.set_title(f"{title_prefix} — test accuracy vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_test_acc_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    # B) Test loss vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ax.plot(r.epochs_plot, r.test_loss, linewidth=2, label=r.label)
    ax.set_title(f"{title_prefix} — test loss vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_test_loss_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    # B2) Training + test loss vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ax.plot(r.epochs_plot, r.train_loss, linewidth=2, label=f"{r.label} (train)")
        ax.plot(r.epochs_plot, r.test_loss, linewidth=2, linestyle="--", label=f"{r.label} (test)")
    ax.set_title(f"{title_prefix} — train & test loss vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_train_loss_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    # C) Total comm (scaled) vs epoch
    max_bytes = max((max(r.comm_cumulative_total_bytes) for r in runs if r.comm_cumulative_total_bytes), default=0)
    unit, div = _choose_comm_unit(float(max_bytes))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ys = _scale_bytes([float(b) for b in r.comm_cumulative_total_bytes], unit=unit, divisor=div)
        ax.plot(r.epochs_plot, ys, linewidth=2, label=r.label)
    ax.set_title(f"{title_prefix} — total communication vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel(f"{unit} (cumulative, send+recv)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_comm_total_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    # D) Total comm per client (scaled / client) vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted_any = False
    max_per_client_bytes = 0.0
    for r in runs:
        if r.n_clients is None or int(r.n_clients) <= 0 or not r.comm_cumulative_total_bytes:
            continue
        max_per_client_bytes = max(max_per_client_bytes, float(max(r.comm_cumulative_total_bytes)) / int(r.n_clients))
    unit_pc, div_pc = _choose_comm_unit(max_per_client_bytes)
    for r in runs:
        if r.n_clients is None or int(r.n_clients) <= 0:
            continue
        ys = _scale_bytes([float(b) / int(r.n_clients) for b in r.comm_cumulative_total_bytes], unit=unit_pc, divisor=div_pc)
        ax.plot(r.epochs_plot, ys, linewidth=2, label=r.label)
        plotted_any = True
    if plotted_any:
        ax.set_title(f"{title_prefix} — total communication per client vs global epoch")
        ax.set_xlabel("global epoch (round)")
        ax.set_ylabel(f"{unit_pc} / client (cumulative)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        p = out_dir / "overlay_comm_total_per_client_vs_global_epoch.png"
        fig.tight_layout()
        fig.savefig(p, dpi=180)
        written.append(p)
    plt.close(fig)

    # E) Comm per round (scaled / round) vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    max_delta_bytes = 0.0
    deltas_by_run: Dict[str, List[int]] = {}
    for r in runs:
        cum = r.comm_cumulative_total_bytes
        deltas: List[int] = []
        for i, b in enumerate(cum):
            if i == 0:
                deltas.append(int(max(0, b)))
            else:
                deltas.append(int(max(0, b - cum[i - 1])))
        deltas_by_run[r.label] = deltas
        if deltas:
            max_delta_bytes = max(max_delta_bytes, float(max(deltas)))
    unit_r, div_r = _choose_comm_unit(max_delta_bytes)
    for r in runs:
        deltas = deltas_by_run.get(r.label, [])
        ys = _scale_bytes([float(b) for b in deltas], unit=unit_r, divisor=div_r)
        ax.plot(r.epochs_plot, ys, linewidth=2, label=r.label)
    ax.set_title(f"{title_prefix} — communication per round vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel(f"{unit_r} / round")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_comm_per_round_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    return written


# -----------------------------
# CLI
# -----------------------------

def _default_group_for_file(p: Path) -> str:
    # Expected structure: results/logs/<group>/<file>.log
    parts = list(p.parts)
    try:
        idx = parts.index("logs")
        return parts[idx + 1]
    except Exception:
        return "unknown"


def _clients_subdir(n_clients: int) -> str:
    return "1_client" if int(n_clients) == 1 else f"{int(n_clients)}_clients"


def _individual_clients_dir() -> str:
    return "individual_clients"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Split-Framework .log files (global epochs)")
    parser.add_argument("--logs-root", default="results/logs", help="Root folder containing .log files")
    parser.add_argument("--plots-root", default="results/plots", help="Output root for plots")
    parser.add_argument(
        "--variant",
        default=None,
        help="Logs group folder under logs-root (e.g., vanilla-lenet). Kept name --variant for backwards compatibility.",
    )
    parser.add_argument("--log-file", default=None, help="Plot a single .log file")
    parser.add_argument("--all", action="store_true", help="Plot all logs under logs-root")
    parser.add_argument(
        "--clients",
        default=None,
        help="Override number of clients (otherwise inferred from filename or comm logs)",
    )
    parser.add_argument(
        "--summary-scatter",
        action="store_true",
        help="Also write final tradeoff scatter for the selected logs.",
    )
    args = parser.parse_args(argv)

    logs_root = Path(args.logs_root)
    plots_root = Path(args.plots_root)

    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.all:
        log_files = sorted(logs_root.rglob("*.log"))
    elif args.variant:
        log_files = sorted((logs_root / args.variant).glob("*.log"))
    else:
        parser.error("Provide one of: --log-file, --variant, or --all")

    if not log_files:
        print("No log files found.")
        return 2

    clients_override: Optional[int] = None
    if args.clients is not None:
        try:
            clients_override = int(args.clients)
        except Exception:
            raise SystemExit(f"--clients must be an int, got: {args.clients}")

    any_written = 0
    # Keep series and scatter points (for a single global final tradeoff plot).
    group_runs: Dict[str, List[RunSeries]] = {}
    all_points: List[Tuple[str, RunSeries, float, int]] = []  # (group, run, final_acc, final_comm_bytes)
    group_client_counts: Dict[str, List[int]] = {}

    for lp in log_files:
        if not lp.exists():
            print(f"SKIP (missing): {lp}")
            continue

        group = args.variant or _default_group_for_file(lp)
        inferred = _extract_client_count_from_stem(lp.stem)
        n_clients = clients_override if clients_override is not None else inferred

        if n_clients is not None:
            out_dir = plots_root / group / _individual_clients_dir() / _clients_subdir(int(n_clients))
        else:
            out_dir = plots_root / group / _individual_clients_dir() / "unknown_clients"

        title = lp.stem
        series = compute_run_series(log_path=lp, title=title, clients_override=clients_override)
        if series is not None:
            group_runs.setdefault(group, []).append(series)
        written, final_point = plot_required(
            log_path=lp,
            out_dir=out_dir,
            title=title,
            clients_override=clients_override,
        )
        any_written += len(written)
        print(f"PLOTTED: {lp} -> {len(written)} files")
        if final_point is not None:
            final_acc, final_comm = final_point
            if series is not None:
                all_points.append((group, series, float(final_acc), int(final_comm)))
            if n_clients is not None:
                group_client_counts.setdefault(group, []).append(int(n_clients))

    # Mixed-clients overlays (comparison plots for multiple client counts)
    for group, runs in group_runs.items():
        if not runs:
            continue
        uniq = sorted(set(int(r.n_clients) for r in runs if r.n_clients is not None))
        if clients_override is None and len(uniq) > 1:
            # Split into: IID vs Non-IID grouped by same alpha/a.
            by_scenario: Dict[str, List[RunSeries]] = {}
            for r in runs:
                by_scenario.setdefault(_scenario_key_from_stem(r.log_path.stem), []).append(r)

            for scenario, subruns in sorted(by_scenario.items()):
                # Write overlays even if there's only one run/client-count in the scenario.
                # This ensures scenarios like alpha0 still get their mixed_clients plots.
                out_dir = plots_root / group / "mixed_clients" / scenario
                written = plot_mixed_clients_overlays(
                    runs=subruns,
                    out_dir=out_dir,
                    title_prefix=f"{group} ({scenario})",
                )
                any_written += len(written)
                if written:
                    print(f"WROTE: {len(written)} overlay plots -> {out_dir}")

    if args.summary_scatter and all_points:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Single global final tradeoff (write exactly once under plots_root)
        xs_bytes = [float(comm_bytes) for (_group, _run, _acc, comm_bytes) in all_points]
        xs_scaled, unit, _ = _scale_bytes_auto(xs_bytes)
        ys = [acc for (_group, _run, acc, _comm) in all_points]
        labels = [f"{group}: {run.label}" for (group, run, _acc, _comm) in all_points]

        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        # Plot each point separately so the legend is readable and unambiguous.
        for x, y, lab in zip(xs_scaled, ys, labels):
            ax.scatter([x], [y], s=55, label=lab)

        # Add padding so points and tick labels don't feel cramped.
        ax.margins(x=0.08, y=0.10)
        ax.set_title("Final tradeoff — accuracy vs total communication")
        ax.set_xlabel(f"final cumulative communication ({unit}, send+recv, sum over clients)")
        ax.set_ylabel("final test accuracy")
        ax.grid(True, alpha=0.25)

        # Legend below the plot (no overlaps with points).
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            framealpha=0.9,
        )

        # Write inside the selected variant folder when --variant is used.
        out_path = plots_root / "final_tradeoff_scatter.png"
        if args.variant:
            out_path = plots_root / args.variant / "final_tradeoff_scatter.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.28)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        any_written += 1
        print(f"WROTE: {out_path}")

    print(f"Done. Wrote {any_written} plot files under {plots_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())