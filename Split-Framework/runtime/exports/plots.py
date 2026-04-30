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
import ast
import os
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


@dataclass(frozen=True)
class BasicRunSeries:
    log_path: Path
    model_group: str
    model_name: str
    label: str
    scenario: str
    n_clients: int
    epochs: List[int]
    val_acc: List[float]
    val_loss: List[float]
    comm_total_mib: List[float]
    comm_cumulative_mib: List[float]
    epoch_time_seconds: List[float]


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

_EPOCH_SUMMARY_LINE_RE = re.compile(
    r"epoch_summary\s+.*?epoch=(?P<epoch>\d+)\s+"
    r"train_acc=(?P<train_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"train_loss=(?P<train_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"val_acc=(?P<val_acc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"val_loss=(?P<val_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"raw_acts_bytes=(?P<raw_acts_bytes>\d+)\s+"
    r"quantized_acts_bytes=(?P<quantized_acts_bytes>\d+)\s+"
    r"acts_metadata_bytes=(?P<acts_metadata_bytes>\d+)\s+"
    r"raw_grads_bytes=(?P<raw_grads_bytes>\d+)\s+"
    r"quantized_grads_bytes=(?P<quantized_grads_bytes>\d+)\s+"
    r"grads_metadata_bytes=(?P<grads_metadata_bytes>\d+)"
    r".*?epoch_time=(?P<epoch_time>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
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


# -----------------------------
# Data split (from logs): samples per client
# -----------------------------


_LOCAL_SAMPLES_RE = re.compile(r"rank\s*=\s*(?P<rank>\d+)\s*,\s*local_sample_number\s*=\s*(?P<n>\d+)")
_DT_TAG_RE = re.compile(r"_(?P<d>\d{2}-\d{2}-\d{4})_(?P<t>\d{2}-\d{2})")
_PC_TAG_RE = re.compile(r"_pc(?P<n>\d+)", re.IGNORECASE)
_SEED_RE = re.compile(r"\bseed\b\s*[:=]\s*(?P<seed>-?\d+)", re.IGNORECASE)
_TRAINDATA_CLS_COUNTS_RE = re.compile(r"traindata_cls_counts:\s*(?P<d>\{.*\})")
_CLASS_PER_CLIENT_RE = re.compile(r"class_per_client:\s*(?P<l>\[\[.*\]\])")


def _extract_run_datetime_from_stem(stem: str) -> Optional[datetime]:
    m = _DT_TAG_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group('d')} {m.group('t')}", "%d-%m-%Y %H-%M")
    except Exception:
        return None


def _parse_local_samples_from_log(log_path: Path) -> Dict[int, int]:
    """Return mapping of MPI rank -> local sample number (clients are ranks 1..N)."""
    out: Dict[int, int] = {}
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return out

    for line in txt.splitlines():
        m = _LOCAL_SAMPLES_RE.search(line)
        if not m:
            continue
        r = int(m.group("rank"))
        n = int(m.group("n"))
        # Keep the last occurrence per rank.
        out[r] = n
    return out


def _try_parse_traindata_cls_counts_from_log(log_path: Path) -> Optional[Dict[int, Dict[int, int]]]:
    """Parse DataPartitioner `traindata_cls_counts` dict from log if present.

    Expected shape:
      {partition_id: {class_label: count, ...}, ...}
    """
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return None

    last_dict: Optional[str] = None
    for line in txt.splitlines():
        m = _TRAINDATA_CLS_COUNTS_RE.search(line)
        if not m:
            continue
        last_dict = m.group("d")

    if not last_dict:
        return None

    try:
        raw = ast.literal_eval(last_dict)
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    out: Dict[int, Dict[int, int]] = {}
    for k, v in raw.items():
        try:
            pid = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue
        inner: Dict[int, int] = {}
        for kk, vv in v.items():
            try:
                inner[int(kk)] = int(vv)
            except Exception:
                continue
        out[pid] = inner
    return out


def _try_parse_class_per_client_from_log(log_path: Path) -> Optional[List[List[int]]]:
    """Parse DataPartitioner `class_per_client` list from log if present."""
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return None

    last_list: Optional[str] = None
    for line in txt.splitlines():
        m = _CLASS_PER_CLIENT_RE.search(line)
        if not m:
            continue
        last_list = m.group("l")

    if not last_list:
        return None

    try:
        raw = ast.literal_eval(last_list)
    except Exception:
        return None

    if not isinstance(raw, list):
        return None

    out: List[List[int]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)):
            return None
        labels: List[int] = []
        for x in item:
            try:
                labels.append(int(x))
            except Exception:
                return None
        out.append(labels)
    return out


def _extract_partition_client_number_from_stem(stem: str) -> Optional[int]:
    m = _PC_TAG_RE.search(stem)
    if not m:
        return None
    try:
        n = int(m.group("n"))
    except Exception:
        return None
    return n if n > 0 else None


def _try_parse_seed_from_log(log_path: Path) -> Optional[int]:
    try:
        txt = log_path.read_text(errors="ignore")
    except Exception:
        return None
    m = _SEED_RE.search(txt)
    if not m:
        return None
    try:
        return int(m.group("seed"))
    except Exception:
        return None


def _partition_settings_from_scenario(scenario: str) -> Tuple[str, Optional[float]]:
    """Return (partition_method, partition_alpha)."""
    if scenario == "iid":
        return "homo", None
    # IMPORTANT: handle 'non_iid_alpha*' before 'non_iid_a*' because
    # 'non_iid_alpha0' also starts with 'non_iid_a'.
    if scenario.startswith("non_iid_alpha"):
        suffix = scenario.split("non_iid_alpha", 1)[1]
        try:
            a = float(suffix)
        except Exception:
            # If we can't parse it, assume extreme non-IID.
            return "alpha0", 0.0
        if a <= 0.0:
            return "alpha0", 0.0
        # If the repo ever logs non-zero `_alphaX` in filenames, treat it like Dirichlet.
        return "hetero", float(a)
    if scenario.startswith("non_iid_a"):
        a = scenario.split("non_iid_a", 1)[1]
        try:
            return "hetero", float(a)
        except Exception:
            return "hetero", None
    # Fallback to IID
    return "homo", None


def _scenario_title(scenario: str) -> str:
    if scenario == "iid":
        return "homogeneous partition"
    if scenario.startswith("non_iid_a"):
        a = scenario.split("non_iid_a", 1)[1]
        return f"Dirichlet non-IID (a={a})"
    if scenario == "non_iid_alpha0":
        return "alpha0 / extreme non-IID"
    return scenario


def plot_data_split_samples_per_client_from_logs(
    *,
    log_files: List[Path],
    plots_root: Path,
    group: str,
    clients_wanted: List[int],
    cfg: "object",
    dataset: str,
    partition_client_number_fallback: int,
    seed_fallback: Optional[int],
) -> List[Path]:
    """Write one figure per (scenario, client-count) under mixed_clients/<scenario>/.

    Figure layout matches the requested style:
    - one figure per client-count (e.g. data_split_10c.png)
    - blue bars
    - value labels on top of bars
    """
    if not log_files:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    _ensure_data_dir_for_cfg(cfg, project_root=project_root)

    # Load labels once (used to compute per-client label counts).
    import numpy as np

    y_train = _load_train_labels_for_split(cfg, dataset=dataset)
    y_arr = np.asarray(y_train)
    classes = np.unique(y_arr)
    # In this repo, labels are typically 0..K-1.
    # If not, we still plot by these actual class ids.
    class_labels = [int(x) for x in classes.tolist()]
    class_to_col = {int(c): i for i, c in enumerate(class_labels)}
    n_classes = int(len(class_labels))

    # Group logs by scenario and by client count.
    by_scenario: Dict[str, Dict[int, List[Path]]] = {}
    for lp in log_files:
        stem = lp.stem
        n_clients = _extract_client_count_from_stem(stem)
        if n_clients is None:
            continue
        scenario = _scenario_key_from_stem(stem)
        by_scenario.setdefault(scenario, {}).setdefault(int(n_clients), []).append(lp)

    written: List[Path] = []
    clients_wanted_sorted = [int(x) for x in clients_wanted]

    for scenario, by_clients in sorted(by_scenario.items()):
        # Pick one log per requested client count (prefer newest timestamp in filename).
        selected: List[Tuple[int, Path]] = []
        for n in clients_wanted_sorted:
            cands = by_clients.get(int(n), [])
            if not cands:
                continue
            # newest by embedded timestamp, fallback to mtime.
            def _key(p: Path):
                dt = _extract_run_datetime_from_stem(p.stem)
                if dt is not None:
                    return (1, dt)
                try:
                    return (0, datetime.fromtimestamp(p.stat().st_mtime))
                except Exception:
                    return (0, datetime.min)

            chosen = sorted(cands, key=_key)[-1]
            selected.append((int(n), chosen))

        if not selected:
            continue

        # Each requested client-count becomes its own separate picture.
        cmap_name = "tab10" if n_classes <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(n_classes)]

        out_dir = plots_root / group / "mixed_clients" / scenario
        out_dir.mkdir(parents=True, exist_ok=True)

        for n_clients, lp in selected:
            # Top plot values from logs (with fallback).
            # Some logs may miss ranks (e.g. if a client didn't print the line);
            # falling back to reconstructed totals avoids misleading zero bars.
            rank_to_n = _parse_local_samples_from_log(lp)

            # Bottom plot values: prefer *log-truth* if available, otherwise reconstruct.
            label_counts_source = "reconstructed"
            label_counts: "np.ndarray"

            part_method, part_alpha = _partition_settings_from_scenario(scenario)
            traindata_cls_counts = _try_parse_traindata_cls_counts_from_log(lp)
            class_per_client = _try_parse_class_per_client_from_log(lp)

            can_use_log_label_counts = False
            if traindata_cls_counts is not None:
                # This is the most reliable: explicit per-partition per-class counts.
                label_counts = np.zeros((int(n_clients), int(n_classes)), dtype=np.int64)
                for rank in range(1, int(n_clients) + 1):
                    pid = int(rank) - 1
                    per_class = traindata_cls_counts.get(pid, {})
                    for cls_id, cnt in per_class.items():
                        col = class_to_col.get(int(cls_id))
                        if col is None:
                            continue
                        label_counts[pid, int(col)] = int(cnt)
                can_use_log_label_counts = True
                label_counts_source = "log"
            elif part_method in {"alpha0", "extreme_noniid", "disjoint_labels"} and class_per_client is not None:
                # For alpha0/extreme non-IID, logs include label assignment (`class_per_client`).
                # If each active partition maps to exactly 1 label, we can derive per-label counts
                # from `local_sample_number` exactly (no reconstruction / seeds).
                ok_single = True
                for rank in range(1, int(n_clients) + 1):
                    pid = int(rank) - 1
                    if pid < 0 or pid >= len(class_per_client):
                        ok_single = False
                        break
                    labels_for_pid = class_per_client[pid]
                    if len(labels_for_pid) != 1:
                        ok_single = False
                        break
                if ok_single:
                    label_counts = np.zeros((int(n_clients), int(n_classes)), dtype=np.int64)
                    for rank in range(1, int(n_clients) + 1):
                        pid = int(rank) - 1
                        lbl = int(class_per_client[pid][0])
                        col = class_to_col.get(lbl)
                        if col is None:
                            continue
                        # Prefer explicit per-rank sample count from log.
                        # If the log is missing the line for a rank, alpha0 still lets us infer
                        # the exact count deterministically: it's just the total number of samples
                        # of that label in the dataset.
                        n_samples = rank_to_n.get(rank)
                        if n_samples is None:
                            try:
                                n_samples = int((y_arr == int(lbl)).sum())
                            except Exception:
                                n_samples = 0
                        label_counts[pid, int(col)] = int(n_samples)
                    can_use_log_label_counts = True
                    label_counts_source = "log"

            if not can_use_log_label_counts:
                pc = _extract_partition_client_number_from_stem(lp.stem)
                if pc is None:
                    pc = int(partition_client_number_fallback)

                seed_run = _try_parse_seed_from_log(lp)
                seed_use = seed_run if seed_run is not None else seed_fallback

                _seed_all(seed_use)
                cfg["client_number"] = int(n_clients)
                cfg["partition_client_number"] = int(pc)
                cfg["partition_method"] = str(part_method)
                if part_alpha is not None:
                    cfg["partition_alpha"] = float(part_alpha)

                net_dataidx_map = _partition_indices_for_split(cfg, dataset=dataset)
                label_counts = np.zeros((int(n_clients), int(n_classes)), dtype=np.int64)
                for cid in range(int(n_clients)):
                    idxs = net_dataidx_map.get(cid, [])
                    if not idxs:
                        continue
                    ys = y_arr[np.asarray(idxs, dtype=np.int64)]
                    vals, freqs = np.unique(ys, return_counts=True)
                    for v, f in zip(vals.tolist(), freqs.tolist()):
                        col = class_to_col.get(int(v))
                        if col is None:
                            continue
                        label_counts[cid, int(col)] = int(f)

            max_y_labels = int(label_counts.sum(axis=1).max()) if label_counts.size else 0

            # Reconstructed totals (always available).
            recon_totals = [int(x) for x in label_counts.sum(axis=1).tolist()] if label_counts.size else [0] * int(n_clients)
            missing_ranks = [r for r in range(1, int(n_clients) + 1) if int(r) not in rank_to_n]
            # Prefer log counts where present, otherwise use reconstructed totals.
            sample_counts = [int(rank_to_n.get(r, recon_totals[i])) for i, r in enumerate(range(1, int(n_clients) + 1))]
            max_y_samples = max(sample_counts) if sample_counts else 0

            # Build a per-case figure: 2 rows (samples + label composition).
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9.6, 6.2), squeeze=True)
            supt = f"{group} dataset split ({_scenario_title(scenario)}) â€” {int(n_clients)} clients"
            fig.suptitle(supt, fontsize=12)

            xs = list(range(1, int(n_clients) + 1))

            # Top: samples per client.
            bars = ax_top.bar(xs, sample_counts, color="#1f77b4")
            if missing_ranks:
                miss = ",".join(str(x) for x in missing_ranks)
                ax_top.set_title(f"{lp.name} (missing ranks in log: {miss})", fontsize=10)
            else:
                ax_top.set_title(lp.name, fontsize=10)
            ax_top.set_xlabel("Client")
            ax_top.set_ylabel("samples")
            ax_top.set_xticks(xs)
            ax_top.set_ylim(0, int(max_y_samples * 1.12) if max_y_samples > 0 else 1)
            ax_top.grid(axis="y", alpha=0.25)
            for b in bars:
                h = int(b.get_height())
                ax_top.text(
                    b.get_x() + b.get_width() / 2,
                    h + max(10, int(max_y_samples * 0.01)),
                    f"{h}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Bottom: stacked label counts.
            ax_bot.set_title(f"label composition ({label_counts_source})", fontsize=10)
            bottoms = np.zeros(int(n_clients), dtype=np.int64)
            handles = []
            labels = []
            for j, lbl in enumerate(class_labels):
                vals = label_counts[:, j]
                hbars = ax_bot.bar(
                    xs,
                    vals,
                    bottom=bottoms,
                    color=colors[j],
                    edgecolor="white",
                    linewidth=0.3,
                    label=str(lbl),
                )

                for x, v, btm in zip(xs, vals.tolist(), bottoms.tolist()):
                    if int(v) <= 0:
                        continue
                    thresh = max(12, int(max_y_labels * 0.08)) if max_y_labels > 0 else 12
                    if int(v) < thresh:
                        continue
                    ax_bot.text(
                        x,
                        int(btm) + int(v) / 2,
                        f"{lbl}:{int(v)}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

                bottoms = bottoms + vals
                handles.append(hbars[0])
                labels.append(str(lbl))

            ax_bot.set_xlabel("Client")
            ax_bot.set_ylabel("samples by label")
            ax_bot.set_xticks(xs)
            ax_bot.set_ylim(0, int(max_y_labels * 1.12) if max_y_labels > 0 else 1)
            ax_bot.grid(axis="y", alpha=0.20)

            ax_bot.legend(
                handles,
                labels,
                title="Label",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=min(len(labels), 10),
                framealpha=0.95,
                fontsize=8,
            )

            fig.tight_layout(rect=[0, 0.06, 1, 0.94])

            out_path = out_dir / f"data_split_{int(n_clients)}c.png"
            fig.savefig(_as_extended_path_str(out_path), dpi=180)
            plt.close(fig)
            written.append(out_path)

    return written


# -----------------------------
# Data split (label distribution) plots
# -----------------------------


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    seed_i = int(seed)
    if seed_i < 0:
        return

    import random

    random.seed(seed_i)
    try:
        import numpy as np

        np.random.seed(seed_i)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
    except Exception:
        # Torch isn't strictly required for producing the split indices.
        pass


def _ensure_data_dir_for_cfg(cfg: "object", *, project_root: Path) -> None:
    """Mirror setup/main.py: keep dataset root under Split-Framework/datasets/downloads by default."""
    data_dir = None
    try:
        data_dir = cfg["dataDir"]
    except Exception:
        data_dir = None
    if not data_dir:
        data_dir = str(project_root / "datasets" / "downloads")
    p = Path(str(data_dir))
    if not p.is_absolute():
        p = (project_root / p).resolve()
    try:
        cfg["dataDir"] = str(p)
    except Exception:
        pass


def _load_train_labels_for_split(cfg: "object", *, dataset: str):
    from datasets.Dataset_Loader import TorchvisionDatasetController

    ds = TorchvisionDatasetController(parse=cfg, dataset_name=str(dataset).lower())
    _x_train, y_train, _x_test, _y_test = ds.loadData()

    # y_train is np.ndarray for MNIST and torch.Tensor for CIFAR in this repo.
    if hasattr(y_train, "cpu"):
        y_train = y_train.cpu().numpy()
    return y_train


def _partition_indices_for_split(cfg: "object", *, dataset: str) -> Dict[int, List[int]]:
    from datasets.Dataset_Loader import TorchvisionDatasetController

    ds = TorchvisionDatasetController(parse=cfg, dataset_name=str(dataset).lower())
    _x_train, _y_train, _x_test, _y_test, net_dataidx_map, _cls_counts = ds.partition_data()

    out: Dict[int, List[int]] = {}
    for k, v in net_dataidx_map.items():
        # v can be np.ndarray or list
        if hasattr(v, "tolist"):
            out[int(k)] = [int(x) for x in v.tolist()]
        else:
            out[int(k)] = [int(x) for x in list(v)]
    return out


def _client_class_counts_for_split(*, y_train, net_dataidx_map: Dict[int, List[int]], active_clients: int, n_classes: int):
    import numpy as np

    m = np.zeros((int(active_clients), int(n_classes)), dtype=np.int64)
    y_arr = np.asarray(y_train)
    for cid in range(int(active_clients)):
        idxs = net_dataidx_map.get(cid, [])
        if not idxs:
            continue
        ys = y_arr[np.asarray(idxs, dtype=np.int64)]
        for c in range(int(n_classes)):
            m[cid, c] = int(np.sum(ys == c))
    return m


def _row_normalize_for_split(counts):
    import numpy as np

    denom = counts.sum(axis=1, keepdims=True).astype(np.float64)
    denom[denom == 0] = 1.0
    return counts.astype(np.float64) / denom


def plot_mixed_clients_data_splits(
    *,
    cfg: "object",
    dataset: str,
    group: str,
    plots_root: Path,
    partition_client_number: int,
    active_clients_list: List[int],
    alphas: List[float],
    include_alpha0: bool,
    seed: Optional[int],
) -> List[Path]:
    """Write one data split plot per scenario under results/plots/<group>/mixed_clients/<scenario>/data_split.png."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    project_root = Path(__file__).resolve().parents[2]
    _ensure_data_dir_for_cfg(cfg, project_root=project_root)
    _seed_all(seed)

    cfg["partition_client_number"] = int(partition_client_number)

    y_train = _load_train_labels_for_split(cfg, dataset=dataset)
    n_classes = int(len(np.unique(np.asarray(y_train))))

    # Scenario specs
    cases: List[Tuple[str, Dict[str, object]]] = [("iid", {"partition_method": "homo"})]
    for a in alphas:
        a_f = float(a)
        cases.append((f"non_iid_a{a_f:g}", {"partition_method": "hetero", "partition_alpha": a_f}))
    if include_alpha0:
        cases.append(("non_iid_alpha0", {"partition_method": "alpha0", "partition_alpha": 0.0}))

    written: List[Path] = []

    for scenario, override in cases:
        # Generate one figure PER active-client-count (matches the style you shared).
        active_list_sorted = sorted(set(int(x) for x in active_clients_list))
        largest_active = max(active_list_sorted) if active_list_sorted else None

        # Use Matplotlib tab colors like your example.
        cmap_name = "tab10" if n_classes <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i % cmap.N) for i in range(n_classes)]

        for active_clients in active_list_sorted:
            cfg["client_number"] = int(active_clients)
            for k, v in override.items():
                cfg[str(k)] = v

            net_dataidx_map = _partition_indices_for_split(cfg, dataset=dataset)
            counts = _client_class_counts_for_split(
                y_train=y_train,
                net_dataidx_map=net_dataidx_map,
                active_clients=int(active_clients),
                n_classes=n_classes,
            )

            x = np.arange(1, int(active_clients) + 1)
            bottom = np.zeros(int(active_clients), dtype=np.int64)

            fig, ax = plt.subplots(figsize=(12.5, 5.3))
            handles = []
            labels = []
            for cls in range(n_classes):
                vals = counts[:, cls]
                h = ax.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=colors[cls],
                    edgecolor="white",
                    linewidth=0.4,
                    label=str(cls),
                )
                bottom = bottom + vals
                handles.append(h[0])
                labels.append(str(cls))

            ax.set_title(f"Label distribution (stacked bars) â€” {active_clients}c {scenario}")
            ax.set_xlabel("Client")
            ax.set_ylabel("Samples (count)")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.25)

            # Legend below plot, like the example.
            leg = ax.legend(
                handles,
                labels,
                title="Label",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=min(n_classes, 10),
                framealpha=0.95,
                fontsize=9,
            )
            plt.setp(leg.get_title(), fontsize=10)

            fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.28)

            out_dir = plots_root / group / "mixed_clients" / scenario
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"data_split_{int(active_clients)}c.png"
            fig.savefig(_as_extended_path_str(out_path), dpi=180)
            plt.close(fig)
            written.append(out_path)

            # Also write/overwrite data_split.png for the largest active client count.
            if largest_active is not None and int(active_clients) == int(largest_active):
                out_path2 = out_dir / "data_split.png"
                # Re-save the same figure bytes by copying from disk (keeps this simple).
                try:
                    out_path2.write_bytes(out_path.read_bytes())
                    written.append(out_path2)
                except Exception:
                    pass

    return written


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

def _as_extended_path_str(p: Path) -> str:
    """Convert to an extended-length absolute path on Windows.

    Needed because the new results folder structure increases path depth, and
    many plot filenames are long enough to exceed MAX_PATH.
    """

    if os.name != "nt":
        return str(p)

    abs_path = p.resolve(strict=False)
    s = str(abs_path)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s

def _ensure_parent(p: Path) -> None:
    os.makedirs(_as_extended_path_str(p.parent), exist_ok=True)


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        return

    for child in sorted(root.iterdir(), key=lambda item: len(item.parts), reverse=True):
        if child.is_dir():
            _prune_empty_dirs(child)

    try:
        next(root.iterdir())
    except StopIteration:
        try:
            root.rmdir()
        except OSError:
            pass
    except OSError:
        pass


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
    ax.set_title(f"{title} â€” test accuracy vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_acc_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
    plt.close(fig)
    written.append(p)

    # 1b) test loss vs global epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, test_loss, label="test loss", linewidth=2)
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} â€” test loss vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_loss_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
    plt.close(fig)
    written.append(p)

    # 1c) training + test loss vs global epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, train_loss, label="train loss", linewidth=2)
    ax.plot(epochs_plot, test_loss, label="test loss", linewidth=2, linestyle="--")
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} â€” train & test loss vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_train_loss_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
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
    ax.set_title(f"{title} â€” total communication vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(comm_scale_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_total_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
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
        ax.set_title(f"{title} â€” total communication per client vs global epoch")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{comm_pc_unit} / client")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_comm_total_per_client_vs_global_epoch.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(_as_extended_path_str(p), dpi=160)
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
    ax.set_title(f"{title} â€” communication per round vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{comm_per_round_unit} / round")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_per_round_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
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
        ax.set_title(f"{title} â€” communication per round per client vs global epoch")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{comm_per_round_unit_per_client} / (roundÂ·client)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_comm_per_round_per_client_vs_global_epoch.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(_as_extended_path_str(p), dpi=160)
        plt.close(fig)
        written.append(p)

    # 3) Send vs receive vs global epoch (cumulative)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_plot, comm_cumulative_send_scaled, label=f"cumulative_send_{comm_scale_unit}", linewidth=2)
    ax.plot(epochs_plot, comm_cumulative_recv_scaled, label=f"cumulative_receive_{comm_scale_unit}", linewidth=2)
    if epochs_plot:
        ax.set_xlim(min(epochs_plot), max(epochs_plot))
    ax.set_title(f"{title} â€” send vs receive vs global epoch")
    ax.set_xlabel(x_label)
    ax.set_ylabel(comm_scale_unit)
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_comm_send_recv_vs_global_epoch.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
    plt.close(fig)
    written.append(p)

    # 4) test accuracy vs total communication (scaled)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(comm_cumulative_total_scaled, test_acc, label="test acc", linewidth=2)
    ax.set_title(f"{title} â€” test accuracy vs total communication")
    ax.set_xlabel(f"cumulative communication ({comm_scale_unit}, send+recv, sum over clients)")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    p = out_dir / f"{log_path.stem}_test_acc_vs_total_comm.png"
    _ensure_parent(p)
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=160)
    plt.close(fig)
    written.append(p)

    # 4b) test accuracy vs total communication per client (scaled / client)
    if comm_cumulative_total_scaled_per_client is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(comm_cumulative_total_scaled_per_client, test_acc, label="test acc", linewidth=2)
        ax.set_title(f"{title} â€” test accuracy vs communication per client")
        ax.set_xlabel(f"cumulative communication ({comm_pc_unit}/client, send+recv)")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend()
        p = out_dir / f"{log_path.stem}_test_acc_vs_total_comm_per_client.png"
        _ensure_parent(p)
        fig.tight_layout()
        fig.savefig(_as_extended_path_str(p), dpi=160)
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
    from matplotlib import colors as mcolors

    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    base_colors = list(plt.rcParams.get("axes.prop_cycle", plt.cycler(color=list(mcolors.TABLEAU_COLORS.values()))).by_key().get("color", list(mcolors.TABLEAU_COLORS.values())))
    sorted_client_counts = sorted({int(r.n_clients) for r in runs if r.n_clients is not None})
    color_by_clients: Dict[int, str] = {
        client_count: base_colors[idx % len(base_colors)]
        for idx, client_count in enumerate(sorted_client_counts)
    }

    def _run_color(run: RunSeries) -> Optional[str]:
        if run.n_clients is None:
            return None
        return color_by_clients.get(int(run.n_clients))

    # A) Test accuracy vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ax.plot(r.epochs_plot, r.test_acc, linewidth=2, color=_run_color(r), label=r.label)
    ax.set_title(f"{title_prefix} â€” test accuracy vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_test_acc_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=180)
    plt.close(fig)
    written.append(p)

    # B) Test loss vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ax.plot(r.epochs_plot, r.test_loss, linewidth=2, color=_run_color(r), label=r.label)
    ax.set_title(f"{title_prefix} â€” test loss vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_test_loss_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=180)
    plt.close(fig)
    written.append(p)

    # B2) Training + test loss vs epoch
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        run_color = _run_color(r)
        ax.plot(r.epochs_plot, r.train_loss, linewidth=2, color=run_color, label=f"{r.label} (train)")
        ax.plot(r.epochs_plot, r.test_loss, linewidth=2, linestyle="--", color=run_color, label=f"{r.label} (test)")
    ax.set_title(f"{title_prefix} â€” train & test loss vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_train_loss_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=180)
    plt.close(fig)
    written.append(p)

    # C) Total comm (scaled) vs epoch
    max_bytes = max((max(r.comm_cumulative_total_bytes) for r in runs if r.comm_cumulative_total_bytes), default=0)
    unit, div = _choose_comm_unit(float(max_bytes))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in runs:
        ys = _scale_bytes([float(b) for b in r.comm_cumulative_total_bytes], unit=unit, divisor=div)
        ax.plot(r.epochs_plot, ys, linewidth=2, label=r.label)
    ax.set_title(f"{title_prefix} â€” total communication vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel(f"{unit} (cumulative, send+recv)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_comm_total_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=180)
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
        ax.set_title(f"{title_prefix} â€” total communication per client vs global epoch")
        ax.set_xlabel("global epoch (round)")
        ax.set_ylabel(f"{unit_pc} / client (cumulative)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        p = out_dir / "overlay_comm_total_per_client_vs_global_epoch.png"
        fig.tight_layout()
        fig.savefig(_as_extended_path_str(p), dpi=180)
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
    ax.set_title(f"{title_prefix} â€” communication per round vs global epoch")
    ax.set_xlabel("global epoch (round)")
    ax.set_ylabel(f"{unit_r} / round")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    p = out_dir / "overlay_comm_per_round_vs_global_epoch.png"
    fig.tight_layout()
    fig.savefig(_as_extended_path_str(p), dpi=180)
    plt.close(fig)
    written.append(p)

    return written


def _basic_label_from_path(log_path: Path) -> str:
    rel = log_path.as_posix().lower()
    if "/baseline/" in rel:
        return "baseline (no quant)"
    if "/arithmetic_conversion/int8/" in rel:
        return "int8"
    if "/arithmetic_conversion/fp8/" in rel:
        return "fp8"
    if "/codeword/uniform/" in rel:
        return "uniform"
    if "/codeword/uniform_per_channel/" in rel:
        return "uniform per-channel"
    if "/codeword/non_uniform/mu_law/" in rel:
        return "mu-law"
    if "/codeword/non_uniform/mu_law_per_channel/" in rel:
        return "mu-law per-channel"
    if "/codeword/non_uniform/loyd/" in rel:
        return "loyd"
    if "/codeword/non_uniform/loyd_per_channel/" in rel:
        return "loyd per-channel"
    return log_path.stem


def _scenario_display_name(scenario: str) -> str:
    if scenario == "iid":
        return "IID"
    if scenario.startswith("non_iid_a"):
        return f"Non-IID a={scenario.split('non_iid_a', 1)[1]}"
    if scenario == "non_iid_alpha0":
        return "Alpha0"
    return scenario


def _group_name_from_log_path(log_path: Path) -> str:
    parent = log_path.parent
    if parent.name.lower() == "baseline":
        model_dir = parent.parent
        algorithm_dir = model_dir.parent
        if algorithm_dir.name and algorithm_dir.name.lower() != "logs":
            return f"{algorithm_dir.name}/{model_dir.name}"
        return model_dir.name
    return parent.name


def _model_name_from_group(group: str) -> str:
    if "/" in group:
        return group.rsplit("/", 1)[1]
    if "-" in group:
        return group.split("-", 1)[1]
    return group


def _epoch_scale_from_logged_epochs(epochs: List[int], *, log_stem: str, n_clients: int) -> int:
    if not epochs:
        return 1
    target_epochs = _extract_target_epochs_from_stem(log_stem)
    if target_epochs is None:
        return 1

    logged_span = (max(epochs) - min(epochs)) + 1
    if logged_span == target_epochs:
        return 1
    if n_clients > 0 and logged_span == target_epochs * n_clients:
        return int(n_clients)
    if logged_span > target_epochs and logged_span % target_epochs == 0:
        return logged_span // target_epochs
    return 1


def _compress_epoch_summary_series(
    *,
    log_stem: str,
    n_clients: int,
    epochs: List[int],
    val_acc: List[float],
    val_loss: List[float],
    comm_total_mib: List[float],
    epoch_time_seconds: List[float],
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    if not epochs:
        return [], [], [], [], []

    scale = _epoch_scale_from_logged_epochs(epochs, log_stem=log_stem, n_clients=n_clients)
    min_epoch = min(epochs)
    by_global_epoch: Dict[int, List[float]] = {}

    for epoch, acc, loss, comm, epoch_time in zip(
        epochs,
        val_acc,
        val_loss,
        comm_total_mib,
        epoch_time_seconds,
    ):
        global_epoch = (int(epoch) - int(min_epoch)) // int(scale)

        # Keep the final validation metrics within the round, but aggregate
        # communication and time across all sub-epochs that belong to it.
        bucket = by_global_epoch.setdefault(int(global_epoch), [float(acc), float(loss), 0.0, 0.0])
        bucket[0] = float(acc)
        bucket[1] = float(loss)
        bucket[2] += float(comm)
        bucket[3] += float(epoch_time)

    epochs_out = sorted(by_global_epoch.keys())
    return (
        [epoch + 1 for epoch in epochs_out],
        [by_global_epoch[epoch][0] for epoch in epochs_out],
        [by_global_epoch[epoch][1] for epoch in epochs_out],
        [by_global_epoch[epoch][2] for epoch in epochs_out],
        [by_global_epoch[epoch][3] for epoch in epochs_out],
    )


def _parse_epoch_summary_series(log_path: Path) -> Optional[BasicRunSeries]:
    text = log_path.read_text(errors="replace")
    epochs: List[int] = []
    val_acc: List[float] = []
    val_loss: List[float] = []
    comm_total_mib: List[float] = []
    epoch_time_seconds: List[float] = []

    for line in text.splitlines():
        match = _EPOCH_SUMMARY_LINE_RE.search(line)
        if not match:
            continue

        total_bytes = (
            int(match.group("quantized_acts_bytes"))
            + int(match.group("acts_metadata_bytes"))
            + int(match.group("quantized_grads_bytes"))
            + int(match.group("grads_metadata_bytes"))
        )
        epochs.append(int(match.group("epoch")))
        val_acc.append(float(match.group("val_acc")))
        val_loss.append(float(match.group("val_loss")))
        comm_total_mib.append(float(total_bytes) / (1024.0 * 1024.0))
        epoch_time_seconds.append(float(match.group("epoch_time")))

    if not epochs:
        run_series = compute_run_series(log_path=log_path, title=log_path.stem)
        if run_series is None or run_series.n_clients is None:
            return None

        comm_total_mib_plot: List[float] = []
        prev_total = 0
        for total_bytes in run_series.comm_cumulative_total_bytes:
            delta_bytes = max(0, int(total_bytes) - int(prev_total))
            prev_total = int(total_bytes)
            comm_total_mib_plot.append(float(delta_bytes) / (1024.0 * 1024.0))

        comm_cumulative_mib_plot = [
            float(total_bytes) / (1024.0 * 1024.0)
            for total_bytes in run_series.comm_cumulative_total_bytes
        ]

        group = _group_name_from_log_path(log_path)
        return BasicRunSeries(
            log_path=log_path,
            model_group=group,
            model_name=_model_name_from_group(group),
            label=_basic_label_from_path(log_path),
            scenario=_scenario_key_from_stem(log_path.stem),
            n_clients=int(run_series.n_clients),
            epochs=[int(x) for x in run_series.epochs_plot],
            val_acc=[float(x) for x in run_series.test_acc],
            val_loss=[float(x) for x in run_series.test_loss],
            comm_total_mib=comm_total_mib_plot,
            comm_cumulative_mib=comm_cumulative_mib_plot,
            epoch_time_seconds=[float("nan") for _ in run_series.epochs_plot],
        )

    n_clients = _extract_client_count_from_stem(log_path.stem)
    if n_clients is None:
        return None

    epochs_plot, val_acc_plot, val_loss_plot, comm_total_mib_plot, epoch_time_plot = _compress_epoch_summary_series(
        log_stem=log_path.stem,
        n_clients=int(n_clients),
        epochs=epochs,
        val_acc=val_acc,
        val_loss=val_loss,
        comm_total_mib=comm_total_mib,
        epoch_time_seconds=epoch_time_seconds,
    )
    comm_cumulative_mib_plot: List[float] = []
    cumulative_total = 0.0
    for value in comm_total_mib_plot:
        cumulative_total += float(value)
        comm_cumulative_mib_plot.append(cumulative_total)

    group = _group_name_from_log_path(log_path)

    return BasicRunSeries(
        log_path=log_path,
        model_group=group,
        model_name=_model_name_from_group(group),
        label=_basic_label_from_path(log_path),
        scenario=_scenario_key_from_stem(log_path.stem),
        n_clients=int(n_clients),
        epochs=epochs_plot,
        val_acc=val_acc_plot,
        val_loss=val_loss_plot,
        comm_total_mib=comm_total_mib_plot,
        comm_cumulative_mib=comm_cumulative_mib_plot,
        epoch_time_seconds=epoch_time_plot,
    )


def plot_basic_scenario_comparisons(*, log_files: List[Path], out_dir: Path) -> List[Path]:
    if not log_files:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [item for item in (_parse_epoch_summary_series(path) for path in log_files) if item is not None]
    if not series:
        return []

    wanted_order = [
        "baseline (no quant)",
        "int8",
        "fp8",
        "uniform",
        "uniform per-channel",
        "mu-law",
        "mu-law per-channel",
        "loyd",
        "loyd per-channel",
    ]
    label_order = {label: idx for idx, label in enumerate(wanted_order)}
    title_map = {
        "iid": "IID",
        "non_iid_a0.5": "Non-IID a=0.5",
        "non_iid_a0.1": "Non-IID a=0.1",
        "non_iid_alpha0": "Alpha0",
    }
    colors = {
        "baseline (no quant)": "#111111",
        "int8": "#1b6ef3",
        "fp8": "#d94841",
        "uniform": "#1f8f55",
        "uniform per-channel": "#2aa76a",
        "mu-law": "#d98e04",
        "mu-law per-channel": "#f0aa2b",
        "loyd": "#7a52cc",
        "loyd per-channel": "#9b78e6",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    by_scenario: Dict[str, List[BasicRunSeries]] = {}
    for item in series:
        by_scenario.setdefault(item.scenario, []).append(item)

    for scenario in ["iid", "non_iid_a0.5", "non_iid_a0.1", "non_iid_alpha0"]:
        runs = by_scenario.get(scenario, [])
        if not runs:
            continue

        runs = sorted(runs, key=lambda item: (label_order.get(item.label, 999), item.label))
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=True)
        fig.suptitle(f"ResNet18 3-client baseline vs quantized â€” {title_map.get(scenario, scenario)}", fontsize=14)

        ax_acc, ax_loss, ax_comm = axes
        for run in runs:
            color = colors.get(run.label)
            line_width = 3.2 if run.label == "baseline (no quant)" else 2.2
            line_style = "-" if run.label == "baseline (no quant)" else "-"
            z_order = 5 if run.label == "baseline (no quant)" else 3
            ax_acc.plot(run.epochs, run.val_acc, linewidth=line_width, linestyle=line_style, color=color, label=run.label, zorder=z_order)
            ax_loss.plot(run.epochs, run.val_loss, linewidth=line_width, linestyle=line_style, color=color, label=run.label, zorder=z_order)
            ax_comm.plot(run.epochs, run.comm_total_mib, linewidth=line_width, linestyle=line_style, color=color, label=run.label, zorder=z_order)

        ax_acc.set_ylabel("Val accuracy")
        ax_acc.grid(True, alpha=0.25)
        ax_acc.legend(loc="best", fontsize=9)

        ax_loss.set_ylabel("Val loss")
        ax_loss.grid(True, alpha=0.25)

        ax_comm.set_ylabel("Comm / epoch (MiB)")
        ax_comm.set_xlabel("Epoch")
        ax_comm.grid(True, alpha=0.25)

        out_path = out_dir / f"resnet18_3client_{scenario}_basic_comparison.png"
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(_as_extended_path_str(out_path), dpi=180)
        plt.close(fig)
        written.append(out_path)

    return written


def plot_baseline_only_model_client_groups(*, log_files: List[Path], plots_root: Path) -> List[Path]:
    if not log_files:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_order = ["iid", "non_iid_a0.5", "non_iid_a0.1", "non_iid_alpha0"]
    scenario_colors = {
        "iid": "#111111",
        "non_iid_a0.5": "#1b6ef3",
        "non_iid_a0.1": "#d94841",
        "non_iid_alpha0": "#1f8f55",
    }

    series = [item for item in (_parse_epoch_summary_series(path) for path in log_files) if item is not None and item.label == "baseline (no quant)"]
    if not series:
        return []

    by_group: Dict[Tuple[str, int], List[BasicRunSeries]] = {}
    for item in series:
        by_group.setdefault((item.model_group, int(item.n_clients)), []).append(item)

    written: List[Path] = []
    for (model_group, n_clients), runs in sorted(by_group.items(), key=lambda item: (item[0][0], item[0][1])):
        runs_sorted = sorted(runs, key=lambda item: scenario_order.index(item.scenario) if item.scenario in scenario_order else 999)
        if not runs_sorted:
            continue

        epoch_count = max(runs_sorted[0].epochs) if runs_sorted[0].epochs else 0
        epoch_tag = f"{epoch_count}epochs"

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.9), sharex=False)
        ax_acc, ax_loss, ax_comm, ax_tradeoff = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        fig.suptitle(f"{runs_sorted[0].model_name} baseline only â€” {epoch_count} epochs â€” {n_clients}-client", fontsize=15)

        out_dir = plots_root / model_group / f"baseline_only_{epoch_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        legend_handles = []
        legend_labels = []
        for run in runs_sorted:
            color = scenario_colors.get(run.scenario, None)
            label = _scenario_display_name(run.scenario)
            acc_line, = ax_acc.plot(run.epochs, run.val_acc, linewidth=2.8, color=color, label=label)
            ax_loss.plot(run.epochs, run.val_loss, linewidth=2.8, color=color, label=label)
            ax_comm.plot(run.epochs, run.comm_total_mib, linewidth=2.8, color=color, label=label)
            ax_tradeoff.plot(run.comm_cumulative_mib, run.val_acc, linewidth=2.8, color=color, label=label)
            legend_handles.append(acc_line)
            legend_labels.append(label)

        ax_acc.set_title("Validation accuracy vs epoch")
        ax_acc.set_xlabel("Global epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(True, alpha=0.25)

        ax_loss.set_title("Validation loss vs epoch")
        ax_loss.set_xlabel("Global epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.25)

        ax_comm.set_title("Communication vs epoch")
        ax_comm.set_xlabel("Global epoch")
        ax_comm.set_ylabel("MiB / epoch")
        ax_comm.grid(True, alpha=0.25)

        ax_tradeoff.set_title("Total communication")
        ax_tradeoff.set_xlabel("Total communication (MiB)")
        ax_tradeoff.set_ylabel("Validation accuracy")
        ax_tradeoff.grid(True, alpha=0.25)

        ax_acc.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=min(4, len(legend_labels)),
            framealpha=0.95,
        )
        ax_loss.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.10),
            ncol=min(4, len(legend_labels)),
            framealpha=0.95,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])

        out_path = out_dir / f"baseline_only_{epoch_tag}_{n_clients}client_metrics.png"
        fig.savefig(_as_extended_path_str(out_path), dpi=180)
        plt.close(fig)
        written.append(out_path)

    return written


def plot_baseline_client_count_comparisons(*, log_files: List[Path], plots_root: Path) -> List[Path]:
    if not log_files:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        item
        for item in (_parse_epoch_summary_series(path) for path in log_files)
        if item is not None and item.label == "baseline (no quant)"
    ]
    if not series:
        return []

    by_model: Dict[str, List[BasicRunSeries]] = {}
    for item in series:
        by_model.setdefault(item.model_group, []).append(item)

    client_order = [1, 3, 5]
    client_colors = {
        1: "#111111",
        3: "#1b6ef3",
        5: "#d94841",
    }

    written: List[Path] = []
    for model_group, runs in sorted(by_model.items()):
        for scenario in ["iid", "non_iid_a0.5", "non_iid_a0.1", "non_iid_alpha0"]:
            picked: List[BasicRunSeries] = []
            for n_clients in client_order:
                matches = [run for run in runs if run.scenario == scenario and int(run.n_clients) == int(n_clients)]
                if not matches:
                    continue
                picked.append(matches[-1])

            if len(picked) < 2:
                continue

            epoch_count = max(picked[0].epochs) if picked[0].epochs else 0
            epoch_tag = f"{epoch_count}epochs"

            fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.9), sharex=False)
            ax_acc, ax_loss, ax_comm, ax_tradeoff = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            fig.suptitle(
                f"{picked[0].model_name} baseline only client comparison â€” {epoch_count} epochs â€” {_scenario_display_name(scenario)}",
                fontsize=15,
            )

            out_dir = plots_root / model_group / f"baseline_only_{epoch_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            legend_handles = []
            legend_labels = []
            for run in picked:
                color = client_colors.get(int(run.n_clients))
                label = f"{int(run.n_clients)} client" if int(run.n_clients) == 1 else f"{int(run.n_clients)} clients"
                acc_line, = ax_acc.plot(run.epochs, run.val_acc, linewidth=2.8, color=color, label=label)
                ax_loss.plot(run.epochs, run.val_loss, linewidth=2.8, color=color, label=label)
                ax_comm.plot(run.epochs, run.comm_total_mib, linewidth=2.8, color=color, label=label)
                ax_tradeoff.plot(run.comm_cumulative_mib, run.val_acc, linewidth=2.8, color=color, label=label)
                legend_handles.append(acc_line)
                legend_labels.append(label)

            ax_acc.set_title("Validation accuracy vs epoch")
            ax_acc.set_xlabel("Global epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.grid(True, alpha=0.25)

            ax_loss.set_title("Validation loss vs epoch")
            ax_loss.set_xlabel("Global epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.25)

            ax_comm.set_title("Communication vs epoch")
            ax_comm.set_xlabel("Global epoch")
            ax_comm.set_ylabel("MiB / epoch")
            ax_comm.grid(True, alpha=0.25)

            ax_tradeoff.set_title("Total communication")
            ax_tradeoff.set_xlabel("Total communication (MiB)")
            ax_tradeoff.set_ylabel("Validation accuracy")
            ax_tradeoff.grid(True, alpha=0.25)

            ax_acc.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=min(3, len(legend_labels)),
                framealpha=0.95,
            )
            ax_loss.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=min(3, len(legend_labels)),
                framealpha=0.95,
            )
            fig.tight_layout(rect=[0, 0.02, 1, 0.93])

            out_path = out_dir / f"baseline_only_{epoch_tag}_{scenario}_client_comparison.png"
            fig.savefig(_as_extended_path_str(out_path), dpi=180)
            plt.close(fig)
            written.append(out_path)

    return written


# -----------------------------
# CLI
# -----------------------------

def _default_group_for_file(p: Path, *, logs_root: Optional[Path] = None) -> str:
    """Infer the group (subfolder) for a log file.

        Legacy expectation was: results/logs/<group>/<file>.log
    We now support nested groups, e.g.:
            results/logs/<algorithm>/<model>/baseline/<file>.log

    If logs_root is provided and p is under it, we mirror the *relative parent*
    folder as the group string.
    """

    try:
        if logs_root is not None:
            logs_root = Path(logs_root)
            if p.is_relative_to(logs_root):
                rel = p.relative_to(logs_root)
                parent = rel.parent
                if str(parent) in (".", ""):
                    return "unknown"
                parts = rel.parts
                if len(parts) >= 3:
                    return f"{parts[0]}/{parts[1]}"
                return str(parent).replace("\\", "/")
    except Exception:
        pass

    # Fallback: try to infer using the old ".../logs/<group>/..." convention.
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


def _group_and_subpath_for_log(log_path: Path, logs_root: Path) -> Tuple[str, Path]:
    """Return (group, subpath) for a log file.

        New logs layout is:
            results/logs/<algorithm>/<model>/<subpath>/<file>.log

        Where <group> is typically '<algorithm>/<model>' (e.g., 'vanilla/resnet18'), and
    <subpath> may be 'baseline' or 'reduce_comm_cost/quantization/<technique>/<direction>/<variant>'.

    For backwards compatibility with older layouts, we fall back to the previous
    default group inference when the path is not under logs_root.
    """
    try:
        rel = log_path.resolve().relative_to(logs_root.resolve())
    except Exception:
        group = _default_group_for_file(log_path, logs_root=logs_root)
        return group, Path()

    if not rel.parts:
        group = _default_group_for_file(log_path, logs_root=logs_root)
        return group, Path()

    if len(rel.parts) >= 3 and rel.parts[1] not in {"baseline", "reduce_comm_cost"}:
        group = f"{rel.parts[0]}/{rel.parts[1]}"
        subpath = Path(*rel.parts[2:-1]) if len(rel.parts) > 3 else Path()
        return group, subpath

    group = rel.parts[0]
    subpath = Path(*rel.parts[1:-1]) if len(rel.parts) > 2 else Path()
    return group, subpath


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Split-Framework .log files (global epochs)")
    parser.add_argument(
        "--basic-comparison",
        action="store_true",
        help="Write one 3-panel comparison figure per scenario (accuracy, loss, communication).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Write 4-panel baseline-only figures per model and client count.",
    )
    parser.add_argument(
        "--data-splits",
        action="store_true",
        help="Write mixed-clients data split plots (from logs) under results/plots/<variant>/mixed_clients/<scenario>/data_split_<Nc>.png",
    )
    parser.add_argument(
        "--config",
        default=str(Path("setup/config/config.yaml")),
        help="(Unused for --data-splits now; kept for backwards compatibility)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="(Unused for --data-splits now; kept for backwards compatibility)",
    )
    parser.add_argument(
        "--partition-client-number",
        type=int,
        default=10,
        help="(Unused for --data-splits now; kept for backwards compatibility)",
    )
    parser.add_argument(
        "--split-clients",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Client counts to include in each data split figure (default: 1 5 10).",
    )
    parser.add_argument(
        "--split-alphas",
        type=float,
        nargs="*",
        default=[0.1, 0.5],
        help="(Unused for --data-splits now; kept for backwards compatibility)",
    )
    parser.add_argument(
        "--include-alpha0",
        action="store_true",
        help="(Unused for --data-splits now; alpha0 is inferred from filenames)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="(Unused for --data-splits now; kept for backwards compatibility)",
    )
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

    # --data-splits mode: generate label-distribution plots without touching logs.
    if args.data_splits:
        # We need a variant to locate the right log files.
        if not args.variant:
            parser.error("--data-splits requires --variant (e.g., vanilla-lenet)")

        # New layout nests logs under baseline/ or reduce_comm_cost/...; use rglob.
        log_files = sorted((logs_root / str(args.variant)).rglob("*.log"))

        # Config is used here to reconstruct the per-label split (for the stacked bars).
        from runtime.exports.config import yaml_config

        cfg_path = Path(str(args.config)).expanduser()
        cfg = yaml_config.load(str(cfg_path))
        dataset = str(args.dataset or cfg.get("dataset") or "mnist").lower() if hasattr(cfg, "get") else str(args.dataset or cfg["dataset"] or "mnist").lower()

        # Avoid accidental logging / file writes during plotting.
        try:
            cfg["log_save_path"] = "/dev/null"
        except Exception:
            pass
        try:
            cfg["download"] = False
        except Exception:
            pass

        # Seed fallback: prefer CLI, else config.
        seed_fallback: Optional[int] = None
        if args.split_seed is not None:
            seed_fallback = int(args.split_seed)
            try:
                cfg["seed"] = int(args.split_seed)
            except Exception:
                pass
        else:
            try:
                seed_fallback = int(cfg["seed"])
            except Exception:
                seed_fallback = None

        written = plot_data_split_samples_per_client_from_logs(
            log_files=log_files,
            plots_root=plots_root,
            group=str(args.variant),
            clients_wanted=[int(x) for x in args.split_clients],
            cfg=cfg,
            dataset=dataset,
            partition_client_number_fallback=int(args.partition_client_number),
            seed_fallback=seed_fallback,
        )
        for p in written:
            print(f"WROTE: {p}")
        return 0

    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.all:
        log_files = sorted(logs_root.rglob("*.log"))
    elif args.variant:
        # New layout nests logs under baseline/ or reduce_comm_cost/...; use rglob.
        log_files = sorted((logs_root / args.variant).rglob("*.log"))
    else:
        parser.error("Provide one of: --log-file, --variant, or --all")

    if not log_files:
        print("No log files found.")
        return 2

    if args.basic_comparison:
        selected_logs: List[Path] = []
        for path in log_files:
            stem = path.stem.lower()
            rel = path.as_posix().lower()
            if "resnet18" not in stem or not stem.startswith("3client_"):
                continue
            if "/baseline/" in rel:
                selected_logs.append(path)
                continue
            if "/reduce_comm_cost/quantization/" not in rel:
                continue
            if "/arithmetic_conversion/int8/" in rel or "/arithmetic_conversion/fp8/" in rel:
                selected_logs.append(path)

        out_dir = plots_root / (str(args.variant) if args.variant else "comparisons")
        written = plot_basic_scenario_comparisons(log_files=selected_logs, out_dir=out_dir)
        for p in written:
            print(f"WROTE: {p}")
        print(f"Done. Wrote {len(written)} basic comparison plot files under {out_dir}")
        return 0

    if args.baseline_only:
        selected_logs = [path for path in log_files if "/baseline/" in path.as_posix().lower()]
        written = plot_baseline_only_model_client_groups(log_files=selected_logs, plots_root=plots_root)
        written.extend(plot_baseline_client_count_comparisons(log_files=selected_logs, plots_root=plots_root))
        _prune_empty_dirs(plots_root)
        for p in written:
            print(f"WROTE: {p}")
        print(f"Done. Wrote {len(written)} baseline-only plot files under {plots_root}")
        return 0

    if not args.log_file:
        selected_logs = [path for path in log_files if "/baseline/" in path.as_posix().lower()]
        written = plot_baseline_only_model_client_groups(log_files=selected_logs, plots_root=plots_root)
        written.extend(plot_baseline_client_count_comparisons(log_files=selected_logs, plots_root=plots_root))
        _prune_empty_dirs(plots_root)
        for p in written:
            print(f"WROTE: {p}")
        print(f"Done. Wrote {len(written)} baseline-only plot files under {plots_root}")
        return 0

    clients_override: Optional[int] = None
    if args.clients is not None:
        try:
            clients_override = int(args.clients)
        except Exception:
            raise SystemExit(f"--clients must be an int, got: {args.clients}")

    any_written = 0
    # Keep series and scatter points.
    # Keyed by (group, subpath) so baseline and quantized runs don't get mixed overlays.
    group_runs: Dict[Tuple[str, str], List[RunSeries]] = {}
    all_points: List[Tuple[str, str, RunSeries, float, int]] = []  # (group, subpath, run, final_acc, final_comm_bytes)
    group_client_counts: Dict[Tuple[str, str], List[int]] = {}

    for lp in log_files:
        if not lp.exists():
            print(f"SKIP (missing): {lp}")
            continue

        group, subpath = _group_and_subpath_for_log(lp, logs_root=logs_root)
        # If the user selected logs via --variant (not --log-file), keep legacy behavior and
        # avoid extra nesting. For --log-file, preserve inferred subpath so plots mirror logs
        # layout (e.g., plots/<group>/baseline/...).
        if args.variant and not args.log_file:
            group = str(args.variant)
            subpath = Path()
        subpath_str = subpath.as_posix() if str(subpath) else ""
        inferred = _extract_client_count_from_stem(lp.stem)
        n_clients = clients_override if clients_override is not None else inferred

        if n_clients is not None:
            out_dir = plots_root / group / subpath / _individual_clients_dir() / _clients_subdir(int(n_clients))
        else:
            out_dir = plots_root / group / subpath / _individual_clients_dir() / "unknown_clients"

        title = lp.stem
        series = compute_run_series(log_path=lp, title=title, clients_override=clients_override)
        if series is not None:
            group_runs.setdefault((group, subpath_str), []).append(series)
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
                all_points.append((group, subpath_str, series, float(final_acc), int(final_comm)))
            if n_clients is not None:
                group_client_counts.setdefault((group, subpath_str), []).append(int(n_clients))

    # Mixed-clients overlays (comparison plots for multiple client counts)
    for (group, subpath_str), runs in group_runs.items():
        if not runs:
            continue
        uniq = sorted(set(int(r.n_clients) for r in runs if r.n_clients is not None))
        if clients_override is None and len(uniq) > 1:
            # Split into: IID vs Non-IID grouped by same alpha/a and target epoch count.
            by_scenario_and_epochs: Dict[Tuple[str, str], List[RunSeries]] = {}
            for r in runs:
                scenario = _scenario_key_from_stem(r.log_path.stem)
                target_epochs = _extract_target_epochs_from_stem(r.log_path.stem)
                epoch_tag = f"{int(target_epochs)}epochs" if target_epochs is not None else "unknown_epochs"
                by_scenario_and_epochs.setdefault((scenario, epoch_tag), []).append(r)

            for (scenario, epoch_tag), subruns in sorted(by_scenario_and_epochs.items()):
                # Write overlays even if there's only one run/client-count in the scenario.
                # This ensures scenarios like alpha0 still get their mixed_clients plots.
                out_dir = plots_root / group / Path(subpath_str) / "mixed_clients" / scenario / epoch_tag
                written = plot_mixed_clients_overlays(
                    runs=subruns,
                    out_dir=out_dir,
                    title_prefix=f"{group}{('/' + subpath_str) if subpath_str else ''} ({scenario}, {epoch_tag})",
                )
                any_written += len(written)
                if written:
                    print(f"WROTE: {len(written)} overlay plots -> {out_dir}")

    if args.summary_scatter and all_points:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Single global final tradeoff (write exactly once under plots_root)
        xs_bytes = [float(comm_bytes) for (_group, _sub, _run, _acc, comm_bytes) in all_points]
        xs_scaled, unit, _ = _scale_bytes_auto(xs_bytes)
        ys = [acc for (_group, _sub, _run, acc, _comm) in all_points]
        labels = [f"{group}{(' / ' + sub) if sub else ''}: {run.label}" for (group, sub, run, _acc, _comm) in all_points]

        fig, ax = plt.subplots(figsize=(12.0, 7.0))
        # Plot each point separately so the legend is readable and unambiguous.
        for x, y, lab in zip(xs_scaled, ys, labels):
            ax.scatter([x], [y], s=55, label=lab)

        # Add padding so points and tick labels don't feel cramped.
        ax.margins(x=0.08, y=0.10)
        ax.set_title("Final tradeoff â€” accuracy vs total communication")
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
        # Prefer writing under a single inferred group when possible.
        out_path = plots_root / "final_tradeoff_scatter.png"
        if args.variant:
            out_path = plots_root / args.variant / "final_tradeoff_scatter.png"
        else:
            uniq_groups = sorted(set(g for (g, _sub, _run, _acc, _comm) in all_points))
            if len(uniq_groups) == 1:
                out_path = plots_root / uniq_groups[0] / "final_tradeoff_scatter.png"
        _ensure_parent(out_path)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.28)
        fig.savefig(_as_extended_path_str(out_path), dpi=180)
        plt.close(fig)
        any_written += 1
        print(f"WROTE: {out_path}")

    _prune_empty_dirs(plots_root)
    print(f"Done. Wrote {any_written} plot files under {plots_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
