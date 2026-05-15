from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from runtime.exports.export_results_workbook import (
    build_fallback_epoch_rows,
    build_run_summary_from_epochs,
    load_fallback_run_series,
    parse_epoch_summary_line,
    parse_filename_metadata,
    parse_latest_comm_breakdown_totals,
    parse_log_path_metadata,
    safe_rel_path,
)


DIRECTION_TOKENS = {"forward", "backward", "forward_backward"}


def _path_parts_lower(path: Path) -> Tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def _is_quantized_log(log_path: Path) -> bool:
    parts = _path_parts_lower(log_path)
    return "quantization" in parts or "reduce_comm_cost" in parts


def _is_baseline_log(log_path: Path) -> bool:
    return "baseline" in _path_parts_lower(log_path)


def _format_partition_label(summary: Dict[str, object]) -> str:
    partition = str(summary.get("partition") or "unknown")
    alpha = summary.get("alpha")
    if partition == "homo":
        return "IID"
    if alpha in (None, "", 0, "0"):
        return partition.upper()
    return f"Non-IID a={alpha}"


def _display_algorithm_name(name: str) -> str:
    mapping = {
        "vanilla": "Vanilla",
        "splitfed": "SplitFed",
        "splitfed2": "SplitFed2",
        "fedavg": "FedAvg",
        "parallel": "Parallel",
        "central": "Central",
    }
    return mapping.get(name.lower(), name)


def _log_relative_parts(rel_log: str) -> Tuple[str, ...]:
    parts = Path(rel_log).parts
    try:
        logs_index = parts.index("logs")
    except ValueError:
        return parts
    return parts[logs_index + 1 :]


def _quantization_context(rel_log: str) -> Dict[str, str]:
    parts = _log_relative_parts(rel_log)
    if "quantization" not in parts:
        if "combined" not in parts:
            return {}
        combined_index = parts.index("combined")
        tail = list(parts[combined_index + 1 : -1])
        if not tail:
            return {}
        profile = tail[0] if len(tail) >= 1 else ""
        family = "combined"
        variant = "+".join(part for part in tail[1:] if part and not part.endswith("bit"))
        method = variant or family
        return {
            "profile": profile,
            "family": family,
            "variant": variant,
            "method": method,
            "direction": "",
            "algorithm_leaf": "",
            "label": " | ".join(part for part in (profile, method) if part),
        }

    quantization_index = parts.index("quantization")
    tail = list(parts[quantization_index + 1 : -1])
    if not tail:
        return {}

    profile = tail[0] if len(tail) >= 1 else ""
    family = tail[1] if len(tail) >= 2 else ""
    direction = ""
    algorithm_leaf = ""
    variant_parts: List[str] = []

    for index, part in enumerate(tail[2:], start=2):
        if part in DIRECTION_TOKENS:
            direction = part
            if index + 1 < len(tail):
                algorithm_leaf = tail[index + 1]
            break
        variant_parts.append(part)

    if not algorithm_leaf and tail:
        algorithm_leaf = tail[-1]

    variant = "/".join(variant_parts)
    method = "/".join(part for part in (family, variant) if part)
    label_parts = [profile, method or family, direction]
    label = " | ".join(part for part in label_parts if part)

    return {
        "profile": profile,
        "family": family,
        "variant": variant,
        "method": method or family,
        "direction": direction,
        "algorithm_leaf": algorithm_leaf,
        "label": label,
    }


def _method_experiment_name(summary: Dict[str, object], rel_log: str, *, is_quantized: bool) -> str:
    algorithm = _algorithm_name(summary, rel_log)
    suffix = "Acc"
    if is_quantized:
        suffix = "Quantized Acc"
    return f"{_display_algorithm_name(algorithm)} {suffix}"


def _algorithm_name(summary: Dict[str, object], rel_log: str) -> str:
    parts = _log_relative_parts(rel_log)
    if parts:
        return parts[0]
    return str(summary.get("method_family") or summary.get("method_variant") or "")


def _readable_run_name(summary: Dict[str, object], rel_log: str, *, is_quantized: bool) -> str:
    clients = summary.get("clients")
    partition_label = _format_partition_label(summary)
    components = [
        f"{clients} clients" if clients is not None else None,
        partition_label,
        str(summary.get("model") or "model"),
    ]
    if is_quantized:
        quantization = _quantization_context(rel_log)
        components.append(quantization.get("label") or "quantized")
    return " | ".join(part for part in components if part)


def _comparison_tags(summary: Dict[str, object], *, is_quantized: bool, rel_log: str) -> Dict[str, str]:
    clients = summary.get("clients")
    partition = str(summary.get("partition") or "")
    alpha = summary.get("alpha")
    display_split = _format_partition_label(summary)
    quantization = _quantization_context(rel_log) if is_quantized else {}

    tags = {
        "clients": str(clients) if clients is not None else "",
        "data_split": display_split,
        "partition": partition,
        "alpha": "" if alpha in (None, "") else str(alpha),
        "quantization_mode": "quantized" if is_quantized else "baseline",
        "quantization_profile": quantization.get("profile", ""),
        "quantization_family": quantization.get("family", ""),
        "quantization_variant": quantization.get("variant", ""),
        "quantization_method": quantization.get("method", ""),
        "quantization_direction": quantization.get("direction", ""),
        "quantization_label": quantization.get("label", ""),
    }
    return tags


def _args_quantization_tags(args) -> Dict[str, str]:
    if not bool(args.get("quantize_forward") or args.get("quantize_backward")):
        return {
            "quantization_profile": "",
            "quantization_family": "",
            "quantization_variant": "",
            "quantization_method": "",
            "quantization_direction": "",
            "quantization_label": "",
        }

    forward_method = str(args.get("forward_quantization") or "").strip()
    backward_method = str(args.get("backward_quantization") or "").strip()
    forward_parts = [part.strip() for part in forward_method.split("+") if part.strip()]
    backward_parts = [part.strip() for part in backward_method.split("+") if part.strip()]
    has_combined = len(forward_parts) > 1 or len(backward_parts) > 1
    sparse_methods = {"top_k", "topk", "top_k_sparsity", "random_top_k", "random_topk", "random_top_k_sparsity"}
    comparison_paper_methods = {"paper_top_k", "paper_topk", "paper_top_k_sparsity", "split_fc", "splitfc", "entropy", "paper_entropy"}
    dimensionality_methods = {"random_projection", "autoencoder", "low_rank_pca", "low_rank_projection", "pca_projection", "pca", "low_rank"}
    all_methods = set(forward_parts + backward_parts)
    if has_combined:
        profile = f"{int(args.get('quantization_bits') or 8)}bit"
        family = "combined"
    elif forward_method in sparse_methods or backward_method in sparse_methods:
        try:
            sparsity_value = float(args.get("sparsity_k") or 1)
            if 0.0 < sparsity_value < 1.0:
                sparsity_value *= 100.0
            profile = f"{int(round(sparsity_value))}pct"
        except Exception:
            profile = "1pct"
        family = "sparsity"
    elif forward_method in dimensionality_methods or backward_method in dimensionality_methods:
        if forward_method in {"random_projection", "autoencoder", "low_rank_pca", "low_rank_projection", "pca_projection", "pca", "low_rank"} or backward_method in {"random_projection", "autoencoder", "low_rank_pca", "low_rank_projection", "pca_projection", "pca", "low_rank"}:
            try:
                reduction_ratio = float(args.get("dimensionality_reduction_ratio") or 0.25)
                if reduction_ratio > 1.0:
                    reduction_ratio /= 100.0
                profile = f"{int(round(reduction_ratio * 100.0))}pct"
            except Exception:
                profile = "25pct"
        else:
            profile = "chunk4_codebook64"
        family = "dimensionality_reduction"
    elif forward_method in comparison_paper_methods or backward_method in comparison_paper_methods:
        if forward_method in {"entropy", "paper_entropy"} or backward_method in {"entropy", "paper_entropy"}:
            profile = "entropy"
        else:
            try:
                ratio = float(args.get("split_fc_reduction_ratio") or 16.0)
                profile = f"{int(round(100.0 / max(ratio, 1.0)))}pct"
            except Exception:
                profile = "6pct"
        family = "comparison_papers"
    else:
        profile = f"{int(args.get('quantization_bits') or 8)}bit"
        family = "quantized"

    if bool(args.get("quantize_forward")) and bool(args.get("quantize_backward")):
        direction = "forward_backward"
    elif bool(args.get("quantize_forward")):
        direction = "forward"
    elif bool(args.get("quantize_backward")):
        direction = "backward"
    else:
        direction = ""

    methods = []
    if forward_method:
        methods.append(f"fwd:{forward_method}")
    if backward_method and backward_method != forward_method:
        methods.append(f"bwd:{backward_method}")
    if backward_method and backward_method == forward_method and not methods:
        methods.append(backward_method)
    if backward_method and backward_method == forward_method and methods == [f"fwd:{forward_method}"]:
        methods = [backward_method]
    variant = " | ".join(methods)
    method = variant or family
    label_parts = [profile, method, direction]

    return {
        "quantization_profile": profile,
        "quantization_family": family,
        "quantization_variant": variant,
        "quantization_method": method,
        "quantization_direction": direction,
        "quantization_label": " | ".join(part for part in label_parts if part),
    }


def _args_experiment_name(args, *, is_quantized: bool) -> str:
    algorithm_name = str(args.get("variants_type") or "Split-Framework")
    if not is_quantized:
        return f"{_display_algorithm_name(algorithm_name)} Acc"

    return f"{_display_algorithm_name(algorithm_name)} Quantized Acc"


COMMON_SUMMARY_METRIC_MAP = {
    "final_train_acc": "acc_train_final",
    "final_val_acc": "acc_val_final",
    "best_val_acc": "acc_val_best",
    "best_train_acc_at_best_val": "acc_train_at_best_val",
    "final_train_loss": "loss_train_final",
    "final_val_loss": "loss_val_final",
    "best_val_loss": "loss_val_best",
    "best_epoch": "epoch_best",
    "generalization_gap_final": "gap_final",
    "generalization_gap_best": "gap_best",
    "avg_epoch_time_s": "time_epoch_avg_s",
    "avg_send_time_s": "time_send_avg_s",
    "avg_recv_time_s": "time_recv_avg_s",
    "avg_total_compression": "comm_compression_avg",
}

BASELINE_SUMMARY_METRIC_MAP = {
    "avg_raw_total_mb": "comm_total_avg_mb",
    "final_raw_total_mb": "comm_total_final_mb",
    "total_raw_run_mb": "comm_total_run_mb",
}

QUANTIZED_SUMMARY_METRIC_MAP = {
    "avg_raw_total_mb": "comm_raw_avg_mb",
    "avg_quantized_total_mb": "comm_quantized_avg_mb",
    "final_raw_total_mb": "comm_raw_final_mb",
    "final_quantized_total_mb": "comm_quantized_final_mb",
    "total_raw_run_mb": "comm_raw_run_mb",
    "total_quantized_run_mb": "comm_quantized_run_mb",
}

EPOCH_METRIC_FIELDS = {
    "train_acc": "acc_train_epoch",
    "train_loss": "loss_train_epoch",
    "val_acc": "acc_val_epoch",
    "val_loss": "loss_val_epoch",
    "total_compression": "comm_compression_epoch",
    "send_time": "time_send_epoch_s",
    "recv_time": "time_recv_epoch_s",
    "epoch_time": "time_epoch_s",
}

PARAM_KEYS = (
    "variants_type",
    "dataset",
    "model",
    "partition_method",
    "partition_alpha",
    "partition_client_number",
    "client_number",
    "max_rank",
    "epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "split_layer",
    "seed",
    "device",
    "quantize_forward",
    "quantize_backward",
    "forward_quantization",
    "backward_quantization",
    "quantization_bits",
    "sparsity_k",
    "dimensionality_reduction_ratio",
    "log_step",
)


def _import_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError:
        return None
    return mlflow


def mlflow_enabled(args) -> bool:
    return bool(args["mlflow_enabled"])


def _default_tracking_uri(project_root: Path) -> str:
    return (project_root / "runtime" / "mlflow" / "mlruns").resolve().as_uri()


def _clean_param_value(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _clean_metric_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _per_epoch_comm_mb(epoch_row: Dict[str, object], *, is_quantized: bool) -> Optional[float]:
    fallback_bytes = epoch_row.get("fallback_total_comm_bytes")
    if fallback_bytes is not None:
        return float(fallback_bytes) / (1024 * 1024)

    if is_quantized:
        total_bytes = (
            float(epoch_row.get("quantized_acts_bytes") or 0)
            + float(epoch_row.get("quantized_grads_bytes") or 0)
            + float(epoch_row.get("acts_metadata_bytes") or 0)
            + float(epoch_row.get("grads_metadata_bytes") or 0)
        )
    else:
        total_bytes = (
            float(epoch_row.get("raw_acts_bytes") or 0)
            + float(epoch_row.get("raw_grads_bytes") or 0)
            + float(epoch_row.get("acts_metadata_bytes") or 0)
            + float(epoch_row.get("grads_metadata_bytes") or 0)
        )
    return total_bytes / (1024 * 1024)


def _minimal_summary_for_log(log_path: Path, logs_root: Path) -> Dict[str, object]:
    filename_meta = parse_filename_metadata(log_path)
    rel_path = safe_rel_path(log_path, logs_root)
    method_family, method_variant, experiment_label = parse_log_path_metadata(rel_path)
    fallback_comm_totals = parse_latest_comm_breakdown_totals(log_path)

    summary: Dict[str, object] = {
        "run_id": rel_path,
        "log_path": rel_path,
        "method_family": method_family,
        "method_variant": method_variant,
        "experiment_label": experiment_label,
        "clients": filename_meta["clients"],
        "model": filename_meta["model"],
        "target_epochs": filename_meta["target_epochs"],
        "observed_epochs": 0,
        "partition": filename_meta["partition"],
        "alpha": filename_meta["alpha"],
        "pc": filename_meta["pc"],
        "run_timestamp": filename_meta["run_timestamp"],
        "import_note": "metadata_only_log",
    }

    if fallback_comm_totals:
        total_bytes = float(fallback_comm_totals.get("total_comm_bytes") or 0)
        if total_bytes > 0:
            summary["total_raw_run_mb"] = total_bytes / (1024 * 1024)

    return summary


def _load_single_run(log_path: Path, logs_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    filename_meta = parse_filename_metadata(log_path)
    rel_path = safe_rel_path(log_path, logs_root)
    method_family, method_variant, experiment_label = parse_log_path_metadata(rel_path)
    fallback_comm_totals = parse_latest_comm_breakdown_totals(log_path)

    epochs: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parsed = parse_epoch_summary_line(raw_line)
            if not parsed:
                continue
            epochs.append(
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
                    **parsed,
                }
            )

    is_fallback = False
    if not epochs:
        fallback_series = load_fallback_run_series(log_path)
        if fallback_series is None:
            minimal_summary = _minimal_summary_for_log(log_path, logs_root)
            if minimal_summary.get("clients") is None or minimal_summary.get("model") is None:
                return [], {}
            return [], minimal_summary
        epochs = build_fallback_epoch_rows(
            run_id=rel_path,
            log_path=rel_path,
            method_family=method_family,
            method_variant=method_variant,
            experiment_label=experiment_label,
            filename_meta=filename_meta,
            fallback_series=fallback_series,
        )
        is_fallback = True

    summary = build_run_summary_from_epochs(
        rel_path=rel_path,
        method_family=method_family,
        method_variant=method_variant,
        experiment_label=experiment_label,
        filename_meta=filename_meta,
        epochs=epochs,
        is_fallback=is_fallback,
        fallback_comm_totals=fallback_comm_totals,
    )
    return epochs, summary


def _log_params(mlflow, args) -> None:
    params: Dict[str, str] = {}
    for key in PARAM_KEYS:
        cleaned = _clean_param_value(args[key])
        if cleaned is None:
            continue
        params[key] = cleaned
    if params:
        mlflow.log_params(params)


def _log_summary_metrics(mlflow, summary: Dict[str, object], *, is_quantized: bool) -> None:
    metrics: Dict[str, float] = {}
    for key, target_key in COMMON_SUMMARY_METRIC_MAP.items():
        cleaned = _clean_metric_value(summary.get(key))
        if cleaned is None:
            continue
        metrics[target_key] = cleaned

    metric_map = QUANTIZED_SUMMARY_METRIC_MAP if is_quantized else BASELINE_SUMMARY_METRIC_MAP
    for source_key, target_key in metric_map.items():
        cleaned = _clean_metric_value(summary.get(source_key))
        if cleaned is None:
            continue
        metrics[target_key] = cleaned

    if metrics:
        mlflow.log_metrics(metrics)


def _log_epoch_metrics(mlflow, epochs: Iterable[Dict[str, object]], *, is_quantized: bool) -> None:
    for row in epochs:
        step = int(row.get("epoch") or 0)
        for source_key, metric_name in EPOCH_METRIC_FIELDS.items():
            cleaned = _clean_metric_value(row.get(source_key))
            if cleaned is not None:
                mlflow.log_metric(metric_name, cleaned, step=step)
        comm_metric_name = "comm_quantized_epoch_mb" if is_quantized else "comm_total_epoch_mb"
        comm_mb = _clean_metric_value(_per_epoch_comm_mb(row, is_quantized=is_quantized))
        if comm_mb is not None:
            mlflow.log_metric(comm_metric_name, comm_mb, step=step)


def _get_experiment_id(mlflow, experiment_name: str) -> Optional[str]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    return str(experiment.experiment_id)


def _existing_run_logs(mlflow, experiment_id: str) -> Set[str]:
    existing: Set[str] = set()
    for run in mlflow.search_runs(experiment_ids=[experiment_id], output_format="list", max_results=50000):
        run_log = run.data.tags.get("run_log")
        if run_log:
            existing.add(run_log)
    return existing


def _delete_experiment_dir_by_name(mlflow, tracking_uri: str, experiment_name: str) -> bool:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return False

    if not tracking_uri.startswith("file:///"):
        return False

    store_root = Path(tracking_uri.removeprefix("file:///"))
    experiment_dir = store_root / str(experiment.experiment_id)
    if not experiment_dir.exists():
        return False

    import shutil

    shutil.rmtree(experiment_dir)
    return True


def _log_backfill_params(mlflow, summary: Dict[str, object]) -> None:
    params = {
        "variants_type": summary.get("method_family") or summary.get("method_variant"),
        "method_variant": summary.get("method_variant"),
        "experiment_label": summary.get("experiment_label"),
        "model": summary.get("model"),
        "partition_client_number": summary.get("clients"),
        "epochs": summary.get("target_epochs"),
        "partition_method": summary.get("partition"),
        "partition_alpha": summary.get("alpha"),
        "run_timestamp": summary.get("run_timestamp"),
        "algorithm": summary.get("method_family"),
    }
    cleaned_params = {
        key: value
        for key, raw_value in params.items()
        if (value := _clean_param_value(raw_value)) is not None
    }
    if cleaned_params:
        mlflow.log_params(cleaned_params)


def backfill_logs_to_mlflow(
    *,
    project_root: Path,
    experiment_name: str = "Split-Framework-History",
    tracking_uri: Optional[str] = None,
    include_quantized: bool = False,
    baseline_only: bool = True,
    group_by_method: bool = True,
    reset_matching_experiments: bool = False,
) -> Dict[str, object]:
    mlflow = _import_mlflow()
    if mlflow is None:
        return {"imported": 0, "skipped": 0, "failed": 0, "total_logs": 0, "error": "mlflow_missing"}

    logs_root = (project_root / "results" / "logs").resolve()
    log_paths = sorted(logs_root.rglob("*.log"))
    resolved_tracking_uri = str(tracking_uri or "").strip() or _default_tracking_uri(project_root)

    mlflow.set_tracking_uri(resolved_tracking_uri)

    filtered_logs: List[Path] = []
    for log_path in log_paths:
        if baseline_only and not _is_baseline_log(log_path):
            continue
        if not include_quantized and _is_quantized_log(log_path):
            continue
        filtered_logs.append(log_path)

    existing_run_logs_by_experiment: Dict[str, Set[str]] = {}
    if reset_matching_experiments:
        if group_by_method:
            seen_experiments: Set[str] = set()
            for log_path in filtered_logs:
                rel_log = safe_rel_path(log_path, project_root)
                _, summary = _load_single_run(log_path, logs_root)
                if not summary:
                    continue
                is_quantized = _is_quantized_log(log_path)
                seen_experiments.add(_method_experiment_name(summary, rel_log, is_quantized=is_quantized))
            for experiment_name_to_delete in seen_experiments:
                _delete_experiment_dir_by_name(mlflow, resolved_tracking_uri, experiment_name_to_delete)
        else:
            _delete_experiment_dir_by_name(mlflow, resolved_tracking_uri, experiment_name)

    imported = 0
    skipped = 0
    failed = 0

    for log_path in filtered_logs:
        rel_log = safe_rel_path(log_path, project_root)
        try:
            epochs, summary = _load_single_run(log_path, logs_root)
            if not summary:
                skipped += 1
                continue

            is_quantized = _is_quantized_log(log_path)

            target_experiment_name = (
                _method_experiment_name(summary, rel_log, is_quantized=is_quantized)
                if group_by_method
                else experiment_name
            )
            mlflow.set_experiment(target_experiment_name)
            experiment_id = _get_experiment_id(mlflow, target_experiment_name)
            if experiment_id is None:
                failed += 1
                continue

            existing_run_logs = existing_run_logs_by_experiment.get(target_experiment_name)
            if existing_run_logs is None:
                existing_run_logs = _existing_run_logs(mlflow, experiment_id)
                existing_run_logs_by_experiment[target_experiment_name] = existing_run_logs

            if rel_log in existing_run_logs:
                skipped += 1
                continue

            run_name = _readable_run_name(summary, rel_log, is_quantized=is_quantized)
            algorithm_name = _algorithm_name(summary, rel_log)
            with mlflow.start_run(run_name=run_name):
                comparison_tags = _comparison_tags(summary, is_quantized=is_quantized, rel_log=rel_log)
                mlflow.set_tags(
                    {
                        "status": "finished",
                        "algorithm": algorithm_name,
                        "algorithm_family": algorithm_name,
                        "method": str(summary.get("method_variant") or ""),
                        "model": str(summary.get("model") or ""),
                        "dataset": str(summary.get("dataset") or ""),
                        "method_variant": str(summary.get("method_variant") or ""),
                        "run_log": rel_log,
                        "import_source": "historical_backfill",
                        **comparison_tags,
                    }
                )
                _log_backfill_params(mlflow, summary)
                _log_summary_metrics(mlflow, summary, is_quantized=is_quantized)
                _log_epoch_metrics(mlflow, epochs, is_quantized=is_quantized)
                mlflow.log_artifact(str(log_path), artifact_path="logs")
            imported += 1
            existing_run_logs.add(rel_log)
        except Exception:
            failed += 1

    return {
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "total_logs": len(filtered_logs),
        "experiment_name": experiment_name,
        "tracking_uri": resolved_tracking_uri,
    }


def export_run_to_mlflow(*, args, project_root: Path, success: bool, error_message: Optional[str] = None) -> bool:
    if not mlflow_enabled(args):
        return False

    mlflow = _import_mlflow()
    if mlflow is None:
        return False

    tracking_uri = str(args["mlflow_tracking_uri"] or "").strip() or _default_tracking_uri(project_root)
    run_name = str(args["mlflow_run_name"] or "").strip()
    log_path = Path(str(args["log_save_path"])).resolve()
    rel_log = safe_rel_path(log_path, project_root)
    if not run_name:
        run_name = log_path.stem

    logs_root = (project_root / "results" / "logs").resolve()
    epochs: List[Dict[str, object]] = []
    summary: Dict[str, object] = {}
    if log_path.exists():
        epochs, summary = _load_single_run(log_path, logs_root)

    is_quantized = _is_quantized_log(log_path)

    experiment_name = str(args["mlflow_experiment_name"] or "").strip()
    if not experiment_name:
        if summary:
            experiment_name = _method_experiment_name(summary, rel_log, is_quantized=is_quantized)
        else:
            experiment_name = _args_experiment_name(args, is_quantized=is_quantized)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        summary_tags = _comparison_tags(summary, is_quantized=is_quantized, rel_log=rel_log) if summary else {
            "clients": str(args["client_number"] or args["partition_client_number"] or ""),
            "data_split": str(args["partition_method"] or ""),
            "partition": str(args["partition_method"] or ""),
            "alpha": str(args["partition_alpha"] or ""),
            "quantization_mode": "quantized" if is_quantized else "baseline",
            **_args_quantization_tags(args),
        }
        mlflow.set_tags(
            {
                "status": "finished" if success else "failed",
                "algorithm": str(args["variants_type"] or ""),
                "model": str(args["model"] or ""),
                "dataset": str(args["dataset"] or ""),
                "run_log": rel_log,
                **summary_tags,
            }
        )
        if error_message:
            mlflow.set_tag("error_message", _clean_param_value(error_message) or "unknown")

        _log_params(mlflow, args)
        if summary:
            _log_summary_metrics(mlflow, summary, is_quantized=is_quantized)
        if epochs:
            _log_epoch_metrics(mlflow, epochs, is_quantized=is_quantized)

        config_path = args["config_path"]
        if config_path:
            config_file = Path(str(config_path)).resolve()
            if config_file.exists():
                mlflow.log_artifact(str(config_file), artifact_path="config")
        if log_path.exists():
            mlflow.log_artifact(str(log_path), artifact_path="logs")

    return True
