import argparse
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape


EPOCH_SUMMARY_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"epoch_summary\s+(?P<body>.+)$"
)

TENSOR_DISTRIBUTION_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"tensor_distribution\s+(?P<body>.+)$"
)

COMM_BREAKDOWN_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"rank=(?P<rank>\d+)\s+node_type=(?P<node_type>\w+)\s+"
    r"total_send=(?P<total_send>\d+)\s+total_receive=(?P<total_receive>\d+)"
)

KEY_VALUE_RE = re.compile(r"(?P<key>\w+)=([^\s]+)")

FILENAME_RE = re.compile(
    r"(?P<clients>\d+)client_"
    r"(?P<model>.+?)_"
    r"(?P<epochs>\d+)epochs_"
    r"(?P<split>(?:homo|hetero_a[0-9.]+|alpha0))"
    r"(?:_pc(?P<pc>\d+))?"
    r"_(?P<date>\d{2}-\d{2}-\d{4})_(?P<time>\d{2}-\d{2})\.log$"
)

NUMERIC_FIELDS = {
    "epoch",
    "train_acc",
    "train_loss",
    "val_acc",
    "val_loss",
    "raw_acts_bytes",
    "quantized_acts_bytes",
    "acts_metadata_bytes",
    "raw_grads_bytes",
    "quantized_grads_bytes",
    "grads_metadata_bytes",
    "acts_compression",
    "grads_compression",
    "total_compression",
    "acts_quant_time",
    "grads_quant_time",
    "send_time",
    "recv_time",
    "epoch_time",
    "rank",
    "mean",
    "std",
    "min",
    "max",
    "mean_abs",
    "max_abs",
    "p95_abs",
    "p99_abs",
    "skewness",
    "kurtosis",
    "mean_channel_std",
    "std_channel_std",
    "max_channel_scale_over_min_channel_scale",
    "spread_score",
    "outlier_score",
    "near_zero_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export result logs into an Excel-compatible workbook."
    )
    parser.add_argument(
        "--logs-dir",
        default="results/logs",
        help="Directory containing experiment log files.",
    )
    parser.add_argument(
        "--output",
        default="results/xml/early_results_workbook.xml",
        help="Base SpreadsheetML workbook path used to derive four output files.",
    )
    parser.add_argument(
        "--target-epochs",
        type=int,
        default=None,
        help="If set, keep only runs whose filename target epoch count matches this value.",
    )
    parser.add_argument(
        "--method-families",
        nargs="*",
        default=None,
        help="Optional list of top-level method families to include, e.g. vanilla-resnet18 splitfed-resnet18.",
    )
    parser.add_argument(
        "--single-workbook",
        action="store_true",
        help="Write a single workbook XML instead of the default four-part export.",
    )
    return parser.parse_args()


def _to_number(value: str):
    if value in {"nan", "NaN"}:
        return float("nan")
    try:
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def _clean_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return value


def parse_filename_metadata(path: Path) -> Dict[str, Optional[object]]:
    match = FILENAME_RE.match(path.name)
    metadata: Dict[str, Optional[object]] = {
        "clients": None,
        "model": None,
        "target_epochs": None,
        "partition": None,
        "alpha": None,
        "pc": None,
        "run_timestamp": None,
    }
    if not match:
        return metadata

    split = match.group("split")
    alpha = None
    partition = split
    if split.startswith("hetero_a"):
        partition = "hetero"
        alpha = float(split.split("a", 1)[1])
    elif split == "alpha0":
        partition = "alpha0"
        alpha = 0.0

    metadata.update(
        {
            "clients": int(match.group("clients")),
            "model": match.group("model"),
            "target_epochs": int(match.group("epochs")),
            "partition": partition,
            "alpha": alpha,
            "pc": int(match.group("pc")) if match.group("pc") else None,
            "run_timestamp": f"{match.group('date')} {match.group('time').replace('-', ':')}",
        }
    )
    return metadata


def parse_epoch_summary_line(line: str) -> Optional[Dict[str, object]]:
    match = EPOCH_SUMMARY_RE.match(line.strip())
    if not match:
        return None

    parsed: Dict[str, object] = {"timestamp": match.group("timestamp")}
    for kv_match in KEY_VALUE_RE.finditer(match.group("body")):
        key = kv_match.group("key")
        value = kv_match.group(2)
        parsed[key] = _to_number(value) if key in NUMERIC_FIELDS else value

    if parsed.get("node_type") != "server" or parsed.get("rank") != 0:
        return None

    return parsed


def parse_tensor_distribution_line(line: str) -> Optional[Dict[str, object]]:
    match = TENSOR_DISTRIBUTION_RE.match(line.strip())
    if not match:
        return None

    parsed: Dict[str, object] = {"timestamp": match.group("timestamp")}
    for kv_match in KEY_VALUE_RE.finditer(match.group("body")):
        key = kv_match.group("key")
        value = kv_match.group(2)
        parsed[key] = _to_number(value) if key in NUMERIC_FIELDS else value

    required_fields = {"rank", "node_type", "phase", "epoch", "sample", "tensor"}
    if not required_fields.issubset(parsed):
        return None

    return parsed


def safe_rel_path(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def parse_log_path_metadata(rel_path: str) -> Tuple[Optional[str], str, str]:
    path_parts = rel_path.split("/")
    if len(path_parts) >= 4 and path_parts[1] not in {"baseline", "reduce_comm_cost"}:
        method_family = f"{path_parts[0]}-{path_parts[1]}"
        method_variant = "/".join(path_parts[2:-1]) if len(path_parts) > 3 else ""
        experiment_label = "/".join(path_parts[:-1]) if len(path_parts) > 1 else rel_path
        return method_family, method_variant, experiment_label

    if "reduce_comm_cost" in path_parts and "combined" in path_parts:
        combined_index = path_parts.index("combined")
        method_parts = path_parts[combined_index + 2 : -1]
        method_family = "combined"
        method_variant = "+".join(part for part in method_parts if part)
        experiment_label = "/".join(path_parts[:-1]) if len(path_parts) > 1 else rel_path
        return method_family, method_variant, experiment_label

    method_family = path_parts[0] if path_parts else None
    method_variant = "/".join(path_parts[1:-1]) if len(path_parts) > 2 else ""
    experiment_label = "/".join(path_parts[:-1]) if len(path_parts) > 1 else rel_path
    return method_family, method_variant, experiment_label


def load_fallback_run_series(log_path: Path):
    try:
        from plots import compute_run_series
    except ImportError:
        from runtime.exports.plots import compute_run_series

    return compute_run_series(log_path=log_path, title=log_path.stem)


def build_fallback_epoch_rows(
    *,
    run_id: str,
    log_path: str,
    method_family: Optional[str],
    method_variant: str,
    experiment_label: str,
    filename_meta: Dict[str, Optional[object]],
    fallback_series,
) -> List[Dict[str, object]]:
    epoch_rows: List[Dict[str, object]] = []
    prev_total = 0
    prev_send = 0
    prev_recv = 0

    for epoch, val_acc, val_loss, total_bytes, send_bytes, recv_bytes in zip(
        fallback_series.epochs_plot,
        fallback_series.test_acc,
        fallback_series.test_loss,
        fallback_series.comm_cumulative_total_bytes,
        fallback_series.comm_cumulative_send_bytes,
        fallback_series.comm_cumulative_recv_bytes,
    ):
        total_bytes = int(total_bytes)
        send_bytes = int(send_bytes)
        recv_bytes = int(recv_bytes)
        delta_total = max(0, total_bytes - prev_total)
        delta_send = max(0, send_bytes - prev_send)
        delta_recv = max(0, recv_bytes - prev_recv)
        prev_total = total_bytes
        prev_send = send_bytes
        prev_recv = recv_bytes

        epoch_rows.append(
            {
                "run_id": run_id,
                "log_path": log_path,
                "method_family": method_family,
                "method_variant": method_variant,
                "experiment_label": experiment_label,
                "clients": filename_meta["clients"],
                "model": filename_meta["model"],
                "target_epochs": filename_meta["target_epochs"],
                "partition": filename_meta["partition"],
                "alpha": filename_meta["alpha"],
                "pc": filename_meta["pc"],
                "run_timestamp": filename_meta["run_timestamp"],
                "epoch": int(epoch),
                "train_acc": None,
                "train_loss": None,
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "raw_acts_bytes": None,
                "quantized_acts_bytes": delta_send,
                "acts_metadata_bytes": None,
                "raw_grads_bytes": None,
                "quantized_grads_bytes": delta_recv,
                "grads_metadata_bytes": None,
                "total_compression": None,
                "send_time": None,
                "recv_time": None,
                "epoch_time": None,
                "timestamp": None,
                "fallback_total_comm_bytes": delta_total,
            }
        )

    return epoch_rows


def build_run_summary_from_epochs(
    *,
    rel_path: str,
    method_family: Optional[str],
    method_variant: str,
    experiment_label: str,
    filename_meta: Dict[str, Optional[object]],
    epochs: List[Dict[str, object]],
    is_fallback: bool,
    fallback_comm_totals: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    epochs.sort(key=lambda row: int(row["epoch"]))
    best_epoch = max(
        epochs,
        key=lambda row: (
            float(row.get("val_acc") or float("-inf")),
            -float(row.get("val_loss") or float("inf")),
        ),
    )
    final_epoch = epochs[-1]
    avg_epoch_time = average([row.get("epoch_time") for row in epochs])
    avg_send_time = average([row.get("send_time") for row in epochs])
    avg_recv_time = average([row.get("recv_time") for row in epochs])
    avg_total_compression = average([row.get("total_compression") for row in epochs])

    if is_fallback:
        final_quantized_megabytes = float(final_epoch.get("fallback_total_comm_bytes") or 0) / (1024 * 1024)
        total_quantized_megabytes = sum(
            float(row.get("fallback_total_comm_bytes") or 0) / (1024 * 1024)
            for row in epochs
        )
        avg_quantized_megabytes = average(
            [
                float(row.get("fallback_total_comm_bytes") or 0) / (1024 * 1024)
                for row in epochs
            ]
        )
        avg_raw_megabytes = None
        final_raw_megabytes = None
        total_raw_megabytes = None
        if fallback_comm_totals and final_quantized_megabytes == 0 and total_quantized_megabytes == 0:
            cumulative_quantized_megabytes = (
                float(fallback_comm_totals.get("total_send") or 0)
                + float(fallback_comm_totals.get("total_receive") or 0)
            ) / (1024 * 1024)
            if cumulative_quantized_megabytes > 0:
                final_quantized_megabytes = cumulative_quantized_megabytes
                total_quantized_megabytes = cumulative_quantized_megabytes
                avg_quantized_megabytes = cumulative_quantized_megabytes / max(len(epochs), 1)
    else:
        final_raw_megabytes = (
            (
                float(final_epoch.get("raw_acts_bytes") or 0)
                + float(final_epoch.get("raw_grads_bytes") or 0)
                + float(final_epoch.get("acts_metadata_bytes") or 0)
                + float(final_epoch.get("grads_metadata_bytes") or 0)
            )
            / (1024 * 1024)
        )
        final_quantized_megabytes = (
            (
                float(final_epoch.get("quantized_acts_bytes") or 0)
                + float(final_epoch.get("quantized_grads_bytes") or 0)
                + float(final_epoch.get("acts_metadata_bytes") or 0)
                + float(final_epoch.get("grads_metadata_bytes") or 0)
            )
            / (1024 * 1024)
        )
        total_raw_megabytes = sum(
            (
                float(row.get("raw_acts_bytes") or 0)
                + float(row.get("raw_grads_bytes") or 0)
                + float(row.get("acts_metadata_bytes") or 0)
                + float(row.get("grads_metadata_bytes") or 0)
            )
            / (1024 * 1024)
            for row in epochs
        )
        total_quantized_megabytes = sum(
            (
                float(row.get("quantized_acts_bytes") or 0)
                + float(row.get("quantized_grads_bytes") or 0)
                + float(row.get("acts_metadata_bytes") or 0)
                + float(row.get("grads_metadata_bytes") or 0)
            )
            / (1024 * 1024)
            for row in epochs
        )
        avg_raw_megabytes = average(
            [
                (
                    float(row.get("raw_acts_bytes") or 0)
                    + float(row.get("raw_grads_bytes") or 0)
                    + float(row.get("acts_metadata_bytes") or 0)
                    + float(row.get("grads_metadata_bytes") or 0)
                )
                / (1024 * 1024)
                for row in epochs
            ]
        )
        avg_quantized_megabytes = average(
            [
                (
                    float(row.get("quantized_acts_bytes") or 0)
                    + float(row.get("quantized_grads_bytes") or 0)
                    + float(row.get("acts_metadata_bytes") or 0)
                    + float(row.get("grads_metadata_bytes") or 0)
                )
                / (1024 * 1024)
                for row in epochs
            ]
        )

    return {
        "run_id": rel_path,
        "log_path": rel_path,
        "method_family": method_family,
        "method_variant": method_variant,
        "experiment_label": experiment_label,
        "clients": filename_meta["clients"],
        "model": filename_meta["model"],
        "target_epochs": filename_meta["target_epochs"],
        "observed_epochs": len(epochs),
        "partition": filename_meta["partition"],
        "alpha": filename_meta["alpha"],
        "pc": filename_meta["pc"],
        "run_timestamp": filename_meta["run_timestamp"],
        "first_log_timestamp": epochs[0].get("timestamp"),
        "last_log_timestamp": final_epoch.get("timestamp"),
        "final_epoch": final_epoch["epoch"],
        "final_train_acc": final_epoch.get("train_acc"),
        "final_train_loss": final_epoch.get("train_loss"),
        "final_val_acc": final_epoch.get("val_acc"),
        "final_val_loss": final_epoch.get("val_loss"),
        "best_epoch": best_epoch.get("epoch"),
        "best_val_acc": best_epoch.get("val_acc"),
        "best_val_loss": best_epoch.get("val_loss"),
        "best_train_acc_at_best_val": best_epoch.get("train_acc"),
        "generalization_gap_final": subtract_safe(
            final_epoch.get("train_acc"), final_epoch.get("val_acc")
        ),
        "generalization_gap_best": subtract_safe(
            best_epoch.get("train_acc"), best_epoch.get("val_acc")
        ),
        "avg_epoch_time_s": avg_epoch_time,
        "avg_send_time_s": avg_send_time,
        "avg_recv_time_s": avg_recv_time,
        "avg_total_compression": avg_total_compression,
        "avg_raw_total_mb": avg_raw_megabytes,
        "avg_quantized_total_mb": avg_quantized_megabytes,
        "final_raw_total_mb": final_raw_megabytes,
        "final_quantized_total_mb": final_quantized_megabytes,
        "total_raw_run_mb": total_raw_megabytes,
        "total_quantized_run_mb": total_quantized_megabytes,
    }


def parse_latest_comm_breakdown_totals(log_path: Path) -> Optional[Dict[str, object]]:
    latest: Optional[Dict[str, object]] = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            match = COMM_BREAKDOWN_RE.match(raw_line.strip())
            if not match:
                continue
            if match.group("node_type") != "server" or int(match.group("rank")) != 0:
                continue
            latest = {
                "timestamp": match.group("timestamp"),
                "total_send": int(match.group("total_send")),
                "total_receive": int(match.group("total_receive")),
            }
    return latest


def load_runs(
    logs_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    run_summaries: List[Dict[str, object]] = []
    epoch_rows: List[Dict[str, object]] = []
    tensor_rows: List[Dict[str, object]] = []

    for log_path in sorted(logs_dir.rglob("*.log")):
        filename_meta = parse_filename_metadata(log_path)
        rel_path = safe_rel_path(log_path, logs_dir)
        method_family, method_variant, experiment_label = parse_log_path_metadata(rel_path)
        fallback_comm_totals = parse_latest_comm_breakdown_totals(log_path)

        epochs: List[Dict[str, object]] = []
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                parsed = parse_epoch_summary_line(raw_line)
                if parsed:
                    epoch_row = {
                        "run_id": rel_path,
                        "log_path": rel_path,
                        "method_family": method_family,
                        "method_variant": method_variant,
                        "experiment_label": experiment_label,
                        "clients": filename_meta["clients"],
                        "model": filename_meta["model"],
                        "target_epochs": filename_meta["target_epochs"],
                        "partition": filename_meta["partition"],
                        "alpha": filename_meta["alpha"],
                        "pc": filename_meta["pc"],
                        "run_timestamp": filename_meta["run_timestamp"],
                        **parsed,
                    }
                    epochs.append(epoch_row)
                    epoch_rows.append(epoch_row)

                tensor_parsed = parse_tensor_distribution_line(raw_line)
                if tensor_parsed:
                    tensor_rows.append(
                        {
                            "run_id": rel_path,
                            "log_path": rel_path,
                            "method_family": method_family,
                            "method_variant": method_variant,
                            "experiment_label": experiment_label,
                            "clients": filename_meta["clients"],
                            "model": filename_meta["model"],
                            "target_epochs": filename_meta["target_epochs"],
                            "partition": filename_meta["partition"],
                            "alpha": filename_meta["alpha"],
                            "pc": filename_meta["pc"],
                            "run_timestamp": filename_meta["run_timestamp"],
                            **tensor_parsed,
                        }
                    )

        is_fallback = False
        if not epochs:
            fallback_series = load_fallback_run_series(log_path)
            if fallback_series is None:
                continue
            epochs = build_fallback_epoch_rows(
                run_id=rel_path,
                log_path=rel_path,
                method_family=method_family,
                method_variant=method_variant,
                experiment_label=experiment_label,
                filename_meta=filename_meta,
                fallback_series=fallback_series,
            )
            epoch_rows.extend(epochs)
            is_fallback = True

        run_summaries.append(
            build_run_summary_from_epochs(
                rel_path=rel_path,
                method_family=method_family,
                method_variant=method_variant,
                experiment_label=experiment_label,
                filename_meta=filename_meta,
                epochs=epochs,
                is_fallback=is_fallback,
                fallback_comm_totals=fallback_comm_totals,
            )
        )

    run_summaries.sort(
        key=lambda row: (
            none_last_sort_key(row.get("best_val_acc"), descending=True),
            none_last_sort_key(row.get("final_val_acc"), descending=True),
            str(row.get("run_id")),
        )
    )
    epoch_rows.sort(key=lambda row: (str(row.get("run_id")), int(row.get("epoch") or 0)))
    tensor_rows.sort(
        key=lambda row: (
            str(row.get("run_id")),
            int(row.get("epoch") or 0),
            str(row.get("sample") or ""),
            str(row.get("node_type") or ""),
            int(row.get("rank") or 0),
            str(row.get("tensor") or ""),
        )
    )
    return run_summaries, epoch_rows, tensor_rows


def average(values: Iterable[Optional[object]]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def subtract_safe(left, right) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def none_last_sort_key(value, descending: bool = False):
    if value is None:
        return (1, 0)
    numeric_value = float(value)
    return (0, -numeric_value if descending else numeric_value)


def build_conclusions(run_summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not run_summaries:
        return rows

    top_runs = sorted(
        run_summaries,
        key=lambda row: (
            none_last_sort_key(row.get("best_val_acc"), descending=True),
            none_last_sort_key(row.get("final_val_acc"), descending=True),
        ),
    )[:10]

    for index, run in enumerate(top_runs, start=1):
        rows.append(
            {
                "section": "Top Runs",
                "item": index,
                "summary": (
                    f"{run.get('run_id')} reached best val_acc={format_float(run.get('best_val_acc'))} "
                    f"at epoch {run.get('best_epoch')} and finished at val_acc={format_float(run.get('final_val_acc'))}."
                ),
            }
        )

    by_group: Dict[Tuple[object, object, object], List[Dict[str, object]]] = defaultdict(list)
    for run in run_summaries:
        key = (run.get("method_variant"), run.get("model"), run.get("clients"))
        by_group[key].append(run)

    for (method_variant, model, clients), runs in sorted(by_group.items(), key=lambda item: str(item[0])):
        if len(runs) < 2:
            continue
        ranked = sorted(
            runs,
            key=lambda row: none_last_sort_key(row.get("best_val_acc"), descending=True),
        )
        best_run = ranked[0]
        worst_run = ranked[-1]
        if best_run.get("run_id") == worst_run.get("run_id"):
            continue
        improvement = subtract_safe(best_run.get("best_val_acc"), worst_run.get("best_val_acc"))
        rows.append(
            {
                "section": "Group Comparison",
                "item": f"{method_variant or 'baseline'} | {model} | {clients} clients",
                "summary": (
                    f"Best partition setting was {best_run.get('partition')}"
                    f"{format_alpha(best_run.get('alpha'))} with best val_acc={format_float(best_run.get('best_val_acc'))}; "
                    f"worst was {worst_run.get('partition')}{format_alpha(worst_run.get('alpha'))} at "
                    f"best val_acc={format_float(worst_run.get('best_val_acc'))}; delta={format_float(improvement)}."
                ),
            }
        )

    compression_runs = [run for run in run_summaries if run.get("avg_total_compression") not in (None, 0)]
    if compression_runs:
        best_compression = sorted(
            compression_runs,
            key=lambda row: none_last_sort_key(row.get("avg_total_compression"), descending=True),
        )[0]
        rows.append(
            {
                "section": "Communication",
                "item": 1,
                "summary": (
                    f"Highest observed average compression was {format_float(best_compression.get('avg_total_compression'))} "
                    f"for {best_compression.get('run_id')}, with average payload "
                    f"{format_float(best_compression.get('avg_quantized_total_mb'))} MB per epoch."
                ),
            }
        )

    overfit_runs = [
        run for run in run_summaries if run.get("generalization_gap_final") is not None
    ]
    if overfit_runs:
        worst_gap = sorted(
            overfit_runs,
            key=lambda row: none_last_sort_key(row.get("generalization_gap_final"), descending=True),
        )[0]
        rows.append(
            {
                "section": "Generalization",
                "item": 1,
                "summary": (
                    f"Largest final train/validation accuracy gap was {format_float(worst_gap.get('generalization_gap_final'))} "
                    f"for {worst_gap.get('run_id')}, suggesting the strongest overfitting among current runs."
                ),
            }
        )

    return rows


def summarize_tensor_distributions(
    tensor_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in tensor_rows:
        key = (
            row.get("run_id"),
            row.get("method_family"),
            row.get("method_variant"),
            row.get("clients"),
            row.get("model"),
            row.get("partition"),
            row.get("alpha"),
            row.get("pc"),
            row.get("phase"),
            row.get("sample"),
            row.get("node_type"),
            row.get("tensor"),
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, object]] = []
    for key, rows in grouped.items():
        (
            run_id,
            method_family,
            method_variant,
            clients,
            model,
            partition,
            alpha,
            pc,
            phase,
            sample,
            node_type,
            tensor,
        ) = key
        ranks = sorted({int(row.get("rank") or 0) for row in rows})
        summary_rows.append(
            {
                "run_id": run_id,
                "method_family": method_family,
                "method_variant": method_variant,
                "clients": clients,
                "model": model,
                "partition": partition,
                "alpha": alpha,
                "pc": pc,
                "phase": phase,
                "sample": sample,
                "node_type": node_type,
                "tensor": tensor,
                "records": len(rows),
                "ranks": ",".join(str(rank) for rank in ranks),
                "avg_mean": average([row.get("mean") for row in rows]),
                "avg_std": average([row.get("std") for row in rows]),
                "avg_mean_abs": average([row.get("mean_abs") for row in rows]),
                "avg_max_abs": average([row.get("max_abs") for row in rows]),
                "avg_p95_abs": average([row.get("p95_abs") for row in rows]),
                "avg_p99_abs": average([row.get("p99_abs") for row in rows]),
                "avg_skewness": average([row.get("skewness") for row in rows]),
                "avg_kurtosis": average([row.get("kurtosis") for row in rows]),
                "avg_mean_channel_std": average([row.get("mean_channel_std") for row in rows]),
                "avg_std_channel_std": average([row.get("std_channel_std") for row in rows]),
                "avg_scale_ratio": average(
                    [row.get("max_channel_scale_over_min_channel_scale") for row in rows]
                ),
                "avg_spread_score": average([row.get("spread_score") for row in rows]),
                "avg_outlier_score": average([row.get("outlier_score") for row in rows]),
                "avg_near_zero_fraction": average(
                    [row.get("near_zero_fraction") for row in rows]
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            str(row.get("run_id")),
            int(row.get("records") or 0),
            str(row.get("phase") or ""),
            str(row.get("sample") or ""),
            str(row.get("node_type") or ""),
            str(row.get("tensor") or ""),
        )
    )
    return summary_rows


def format_alpha(alpha) -> str:
    if alpha is None:
        return ""
    return f" (alpha={alpha:g})"


def format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def xml_cell(value) -> str:
    value = _clean_value(value)
    if value is None:
        return "<Cell/>"
    if isinstance(value, bool):
        cell_type = "Boolean"
        cell_value = "1" if value else "0"
    elif isinstance(value, int):
        cell_type = "Number"
        cell_value = str(value)
    elif isinstance(value, float):
        cell_type = "Number"
        cell_value = format(value, ".12g")
    else:
        cell_type = "String"
        cell_value = escape(str(value))
    return f'<Cell><Data ss:Type="{cell_type}">{cell_value}</Data></Cell>'


def build_worksheet(name: str, rows: List[Dict[str, object]], columns: List[str]) -> str:
    worksheet_rows = [
        "<Row>" + "".join(xml_cell(column) for column in columns) + "</Row>"
    ]
    for row in rows:
        worksheet_rows.append(
            "<Row>" + "".join(xml_cell(row.get(column)) for column in columns) + "</Row>"
        )
    table = "<Table>" + "".join(worksheet_rows) + "</Table>"
    return f'<Worksheet ss:Name="{escape(name)}">{table}</Worksheet>'


def build_workbook_xml(worksheets: List[Tuple[str, List[Dict[str, object]], List[str]]]) -> str:
    return (
        '<?xml version="1.0"?>'
        '<?mso-application progid="Excel.Sheet"?>'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:html="http://www.w3.org/TR/REC-html40">'
        + "".join(build_worksheet(name, rows, columns) for name, rows, columns in worksheets)
        + "</Workbook>"
    )


def workbook_part_output_paths(output_path: Path) -> Tuple[Path, Path, Path, Path]:
    base_name = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    return (
        parent / f"{base_name}_summary{suffix}",
        parent / f"{base_name}_epochs_part1{suffix}",
        parent / f"{base_name}_epochs_part2{suffix}",
        parent / f"{base_name}_tensor_details{suffix}",
    )


def split_rows_evenly(rows: List[Dict[str, object]], parts: int) -> List[List[Dict[str, object]]]:
    if parts <= 1:
        return [rows]
    total = len(rows)
    base_size, remainder = divmod(total, parts)
    chunks: List[List[Dict[str, object]]] = []
    start = 0
    for idx in range(parts):
        chunk_size = base_size + (1 if idx < remainder else 0)
        end = start + chunk_size
        chunks.append(rows[start:end])
        start = end
    return chunks


def write_workbooks(
    output_path: Path,
    run_summaries: List[Dict[str, object]],
    epoch_rows: List[Dict[str, object]],
    tensor_summary_rows: List[Dict[str, object]],
    tensor_rows: List[Dict[str, object]],
    conclusion_rows: List[Dict[str, object]],
) -> Tuple[Path, Path, Path, Path]:
    summary_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "target_epochs",
        "observed_epochs",
        "best_epoch",
        "best_val_acc",
        "best_val_loss",
        "final_epoch",
        "final_train_acc",
        "final_val_acc",
        "final_train_loss",
        "final_val_loss",
        "generalization_gap_final",
        "avg_epoch_time_s",
        "avg_send_time_s",
        "avg_recv_time_s",
        "avg_total_compression",
        "avg_raw_total_mb",
        "avg_quantized_total_mb",
        "final_raw_total_mb",
        "final_quantized_total_mb",
        "total_raw_run_mb",
        "total_quantized_run_mb",
        "run_timestamp",
        "first_log_timestamp",
        "last_log_timestamp",
        "log_path",
    ]
    epoch_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "epoch",
        "train_acc",
        "train_loss",
        "val_acc",
        "val_loss",
        "raw_acts_bytes",
        "quantized_acts_bytes",
        "acts_metadata_bytes",
        "raw_grads_bytes",
        "quantized_grads_bytes",
        "grads_metadata_bytes",
        "total_compression",
        "send_time",
        "recv_time",
        "epoch_time",
        "timestamp",
    ]
    tensor_summary_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "phase",
        "sample",
        "node_type",
        "tensor",
        "records",
        "ranks",
        "avg_mean",
        "avg_std",
        "avg_mean_abs",
        "avg_max_abs",
        "avg_p95_abs",
        "avg_p99_abs",
        "avg_skewness",
        "avg_kurtosis",
        "avg_mean_channel_std",
        "avg_std_channel_std",
        "avg_scale_ratio",
        "avg_spread_score",
        "avg_outlier_score",
        "avg_near_zero_fraction",
    ]
    tensor_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "rank",
        "node_type",
        "phase",
        "epoch",
        "sample",
        "tensor",
        "shape",
        "mean",
        "std",
        "min",
        "max",
        "mean_abs",
        "max_abs",
        "p95_abs",
        "p99_abs",
        "skewness",
        "kurtosis",
        "mean_channel_std",
        "std_channel_std",
        "max_channel_scale_over_min_channel_scale",
        "spread",
        "spread_score",
        "outliers",
        "outlier_score",
        "near_zero",
        "near_zero_fraction",
        "timestamp",
    ]
    conclusion_columns = ["section", "item", "summary"]

    summary_workbook = build_workbook_xml(
        [
            ("Run Summary", run_summaries, summary_columns),
            ("Early Conclusions", conclusion_rows, conclusion_columns),
        ]
    )
    epoch_workbook = build_workbook_xml(
        [("Epoch Details", epoch_rows, epoch_columns)]
    )
    epoch_part1_rows, epoch_part2_rows = split_rows_evenly(epoch_rows, parts=2)
    tensor_details_workbook = build_workbook_xml(
        [("Tensor Dist Details", tensor_rows, tensor_columns)]
    )
    summary_workbook = build_workbook_xml(
        [
            ("Run Summary", run_summaries, summary_columns),
            ("Early Conclusions", conclusion_rows, conclusion_columns),
            ("Tensor Dist Summary", tensor_summary_rows, tensor_summary_columns),
        ]
    )
    epoch_part1_workbook = build_workbook_xml(
        [("Epoch Details Part 1", epoch_part1_rows, epoch_columns)]
    )
    epoch_part2_workbook = build_workbook_xml(
        [("Epoch Details Part 2", epoch_part2_rows, epoch_columns)]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path, epoch_part1_output_path, epoch_part2_output_path, tensor_details_output_path = workbook_part_output_paths(output_path)
    summary_output_path.write_text(summary_workbook, encoding="utf-8")
    epoch_part1_output_path.write_text(epoch_part1_workbook, encoding="utf-8")
    epoch_part2_output_path.write_text(epoch_part2_workbook, encoding="utf-8")
    tensor_details_output_path.write_text(tensor_details_workbook, encoding="utf-8")
    return summary_output_path, epoch_part1_output_path, epoch_part2_output_path, tensor_details_output_path


def write_single_workbook(
    output_path: Path,
    run_summaries: List[Dict[str, object]],
    epoch_rows: List[Dict[str, object]],
    tensor_summary_rows: List[Dict[str, object]],
    conclusion_rows: List[Dict[str, object]],
) -> Path:
    summary_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "target_epochs",
        "observed_epochs",
        "best_epoch",
        "best_val_acc",
        "best_val_loss",
        "final_epoch",
        "final_train_acc",
        "final_val_acc",
        "final_train_loss",
        "final_val_loss",
        "generalization_gap_final",
        "avg_epoch_time_s",
        "avg_send_time_s",
        "avg_recv_time_s",
        "avg_total_compression",
        "avg_raw_total_mb",
        "avg_quantized_total_mb",
        "final_raw_total_mb",
        "final_quantized_total_mb",
        "total_raw_run_mb",
        "total_quantized_run_mb",
        "run_timestamp",
        "log_path",
    ]
    epoch_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "epoch",
        "train_acc",
        "train_loss",
        "val_acc",
        "val_loss",
        "total_compression",
        "send_time",
        "recv_time",
        "epoch_time",
        "timestamp",
    ]
    tensor_summary_columns = [
        "run_id",
        "method_family",
        "method_variant",
        "clients",
        "model",
        "partition",
        "alpha",
        "pc",
        "phase",
        "sample",
        "node_type",
        "tensor",
        "records",
        "ranks",
        "avg_mean",
        "avg_std",
        "avg_mean_abs",
        "avg_max_abs",
        "avg_p95_abs",
        "avg_p99_abs",
        "avg_spread_score",
        "avg_outlier_score",
        "avg_near_zero_fraction",
    ]
    conclusion_columns = ["section", "item", "summary"]

    workbook = build_workbook_xml(
        [
            ("Run Summary", run_summaries, summary_columns),
            ("Epoch Details", epoch_rows, epoch_columns),
            ("Tensor Dist Summary", tensor_summary_rows, tensor_summary_columns),
            ("Conclusions", conclusion_rows, conclusion_columns),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(workbook, encoding="utf-8")
    return output_path


def filter_rows(
    run_summaries: List[Dict[str, object]],
    epoch_rows: List[Dict[str, object]],
    tensor_rows: List[Dict[str, object]],
    *,
    target_epochs: Optional[int],
    method_families: Optional[Iterable[str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    families = {str(item) for item in method_families} if method_families else None

    def _keep(row: Dict[str, object]) -> bool:
        if target_epochs is not None and int(row.get("target_epochs") or -1) != int(target_epochs):
            return False
        if families is not None and str(row.get("method_family")) not in families:
            return False
        return True

    filtered_runs = [row for row in run_summaries if _keep(row)]
    allowed_run_ids = {str(row.get("run_id")) for row in filtered_runs}
    filtered_epochs = [row for row in epoch_rows if str(row.get("run_id")) in allowed_run_ids]
    filtered_tensors = [row for row in tensor_rows if str(row.get("run_id")) in allowed_run_ids]
    return filtered_runs, filtered_epochs, filtered_tensors


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[2]
    logs_dir = (base_dir / args.logs_dir).resolve()
    output_arg = Path(args.output)
    if output_arg.as_posix() == "results/early_results_workbook.xml":
        output_arg = Path("results/xml/early_results_workbook.xml")
    output_path = (base_dir / output_arg).resolve()

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory does not exist: {logs_dir}")

    run_summaries, epoch_rows, tensor_rows = load_runs(logs_dir)
    run_summaries, epoch_rows, tensor_rows = filter_rows(
        run_summaries,
        epoch_rows,
        tensor_rows,
        target_epochs=args.target_epochs,
        method_families=args.method_families,
    )
    tensor_summary_rows = summarize_tensor_distributions(tensor_rows)
    conclusion_rows = build_conclusions(run_summaries)

    if args.single_workbook:
        output_single = write_single_workbook(
            output_path,
            run_summaries,
            epoch_rows,
            tensor_summary_rows,
            conclusion_rows,
        )
        print(f"Wrote single workbook to {output_single}")
        print(
            f"Included {len(run_summaries)} runs, {len(epoch_rows)} epoch rows, and {len(tensor_summary_rows)} tensor summary rows"
        )
        return

    summary_output_path, epoch_part1_output_path, epoch_part2_output_path, tensor_details_output_path = write_workbooks(
        output_path,
        run_summaries,
        epoch_rows,
        tensor_summary_rows,
        tensor_rows,
        conclusion_rows,
    )

    print(f"Wrote summary workbook to {summary_output_path}")
    print(f"Wrote epoch workbook part 1 to {epoch_part1_output_path}")
    print(f"Wrote epoch workbook part 2 to {epoch_part2_output_path}")
    print(f"Wrote tensor details workbook to {tensor_details_output_path}")
    print(
        f"Included {len(run_summaries)} runs, {len(epoch_rows)} epoch rows, and {len(tensor_rows)} tensor distribution rows"
    )


if __name__ == "__main__":
    main()
