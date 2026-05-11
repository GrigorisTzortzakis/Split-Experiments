"""Minimal GUI launcher for Split-Framework experiments.

This stays intentionally self-contained so the setup folder does not grow into
another subsystem. It builds commands for setup/main.py and can run them
sequentially from a queue.
"""

from __future__ import annotations

import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import ast
import importlib.util
import json
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


SETUP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SETUP_DIR.parent


def _find_python_executable() -> str:
    candidates = [
        PROJECT_ROOT.parents[1] / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


ALGORITHM_OPTIONS: List[Tuple[str, str]] = [
    ("vanilla", "vanilla"),
    ("central", "central"),
    ("gnn", "gnn"),
    ("asyVanilla", "asyVanilla"),
    ("asyVanilla2", "asyVanilla2"),
    ("Asynchronous", "Asynchronous"),
    ("vertical", "vertical"),
    ("SGLR", "SGLR"),
    ("SplitFed", "SplitFed"),
    ("SplitFed2", "SplitFed2"),
    ("Parallel", "parallel"),
    ("comp_model", "comp_model"),
    ("fedavg", "fedavg"),
    ("fedprox", "fedprox"),
]

MODEL_OPTIONS: List[Tuple[str, str]] = [
    ("resnet18", "resnet18"),
    ("densenet121", "densenet121"),
    ("efficientnet_b0", "efficientnet_b0"),
    ("bigru", "bilstm"),
    ("bert_tiny", "bert_tiny"),
]
DATASET_OPTIONS: List[str] = ["cifar10", "cifar100", "ag_news"]
DEVICE_OPTIONS: List[Tuple[str, str]] = [("gpu", "gpu"), ("cpu", "cpu")]
SUPPORTED_DATASETS_BY_MODEL: Dict[str, Tuple[str, ...]] = {
    "resnet18": ("cifar10",),
    "densenet121": ("cifar10",),
    "efficientnet_b0": ("cifar100",),
    "bilstm": ("ag_news",),
    "bert_tiny": ("ag_news",),
}
FIXED_SPLIT_BY_MODEL: Dict[str, str] = {
    "resnet18": "default",
    "densenet121": "default",
    "efficientnet_b0": "default",
    "bilstm": "default",
    "bert_tiny": "default",
}
PARTITION_OPTIONS: List[Tuple[str, str, Optional[float]]] = [
    ("iid", "homo", None),
    ("non iid a=0.5", "hetero", 0.5),
    ("non iid a=0.1", "hetero", 0.1),
    ("non iid a=0", "alpha0", 0.0),
]
COMM_REDUCTION_OPTIONS: List[Tuple[str, str]] = [
    ("none", "none"),
    ("arithmetic_conversion", "arithmetic_conversion"),
    ("codeword", "codeword"),
    ("Truncation", "Truncation"),
    ("sparsity", "sparsity"),
    ("dimensionality_reduction", "dimensionality_reduction"),
]
COMM_DIRECTION_OPTIONS: List[Tuple[str, str]] = [
    ("forward_backward", "both"),
    ("forward", "forward"),
    ("backward", "backward"),
]
QUANTIZATION_BITS_OPTIONS: List[Tuple[str, object]] = [
    ("2bit", 2),
    ("3bit", 3),
    ("4bit", 4),
    ("6bit", 6),
    ("8bit", 8),
    ("16bit", 16),
    ("32bit", 32),
]
SPARSITY_K_OPTIONS: List[Tuple[str, int]] = [
    ("1%", 1),
    ("5%", 5),
    ("10%", 10),
    ("25%", 25),
    ("50%", 50),
]
DIMENSIONALITY_REDUCTION_RATIO_OPTIONS: List[Tuple[str, float]] = [
    ("12.5%", 0.125),
    ("25%", 0.25),
    ("50%", 0.5),
]
QUANTIZATION_OPTIONS: List[Tuple[str, str]] = [
    ("int", "int"),
    ("float", "float"),
    ("uniform", "uniform"),
    ("non_uniform_loyd", "non_uniform_loyd"),
    ("non_uniform_mlaw", "non_uniform_mlaw"),
    ("truncation_int", "truncation_int"),
    ("top_k", "top_k"),
    ("random_top_k", "random_top_k"),
    ("paper_top_k", "paper_top_k"),
    ("random_projection", "random_projection"),
    ("low_rank_pca", "low_rank_pca"),
]
PIPELINE_ADDON_OPTIONS: List[Tuple[str, str]] = [("none", "none"), *QUANTIZATION_OPTIONS]
QUANTIZATION_GRANULARITY_OPTIONS: List[Tuple[str, str]] = [
    ("per_tensor", "per_tensor"),
    ("per_channel", "per_channel"),
    ("per_group", "per_group"),
]

COMM_METHODS_BY_MODE: Dict[str, Tuple[str, ...]] = {
    "arithmetic_conversion": ("int", "float"),
    "codeword": (
        "uniform",
        "non_uniform_loyd",
        "non_uniform_mlaw",
    ),
    "Truncation": ("truncation_int",),
    "sparsity": ("top_k", "random_top_k", "paper_top_k"),
    "dimensionality_reduction": ("random_projection", "low_rank_pca"),
}


def _parse_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def _load_simple_yaml(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            data[key] = _parse_scalar(value) if value else ""
    return data


def _load_defaults(path: Path) -> Dict[str, object]:
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from runtime.exports.config import yaml_config

        cfg = yaml_config.load(str(path))
        if hasattr(cfg, "as_dict"):
            return cfg.as_dict()
        return dict(cfg)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        return _load_simple_yaml(path)


def _find_mpiexec() -> str:
    candidates = ["mpiexec", "mpiexec.exe", "mpirun", "mpirun.exe"]
    for candidate in candidates:
        resolved = shutil_which(candidate)
        if resolved:
            return resolved
    return "mpiexec"


def shutil_which(name: str) -> Optional[str]:
    path_env = os.environ.get("PATH", "")
    for directory in path_env.split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.exists():
            return str(candidate)
    return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class Job:
    name: str
    command: List[str]
    summary: str
    status: str = "Waiting"


class ExperimentMenu:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Split-Framework Experiment Menu")
        self.root.geometry("1180x780")
        self.root.minsize(1080, 700)

        self.config_path = SETUP_DIR / "config" / "config.yaml"
        self.main_path = SETUP_DIR / "main.py"
        self.menu_state_path = SETUP_DIR / ".experiment_menu_state.json"
        self.defaults = _load_defaults(self.config_path)
        self.saved_state = self._load_menu_state()
        self.mpiexec_path = _find_mpiexec()
        self.python_path = _find_python_executable()

        self.jobs: List[Job] = []
        self.jobs_lock = threading.Lock()
        self.process: Optional[subprocess.Popen[str]] = None
        self.mlflow_process: Optional[subprocess.Popen[str]] = None
        self.runner_thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.running = False
        self.ui_queue: queue.Queue[Tuple[str, str]] = queue.Queue()

        self.algorithm_map: Dict[str, str] = {label: value for label, value in ALGORITHM_OPTIONS}
        self.model_map: Dict[str, str] = {label: value for label, value in MODEL_OPTIONS}
        self.device_map: Dict[str, str] = {label: value for label, value in DEVICE_OPTIONS}
        self.partition_map: Dict[str, str] = {label: value for label, value, _alpha in PARTITION_OPTIONS}
        self.partition_alpha_map: Dict[str, Optional[float]] = {label: alpha for label, _value, alpha in PARTITION_OPTIONS}
        self.comm_reduction_map: Dict[str, str] = {label: value for label, value in COMM_REDUCTION_OPTIONS}
        self.mode_selection_map: Dict[str, Tuple[str, Optional[str]]] = self._build_mode_selection_map()
        self.comm_direction_map: Dict[str, str] = {label: value for label, value in COMM_DIRECTION_OPTIONS}
        self.quantization_bits_map: Dict[str, object] = {label: value for label, value in QUANTIZATION_BITS_OPTIONS}
        self.sparsity_k_map: Dict[str, int] = {label: value for label, value in SPARSITY_K_OPTIONS}
        self.dimensionality_reduction_ratio_map: Dict[str, float] = {label: value for label, value in DIMENSIONALITY_REDUCTION_RATIO_OPTIONS}
        self.quantization_granularity_map: Dict[str, str] = {label: value for label, value in QUANTIZATION_GRANULARITY_OPTIONS}
        self.quantization_map: Dict[str, str] = {label: value for label, value in QUANTIZATION_OPTIONS}
        self.pipeline_addon_map: Dict[str, str] = {label: value for label, value in PIPELINE_ADDON_OPTIONS}

        self.algorithm_var = tk.StringVar(value=self._default_algorithm_label())
        self.model_var = tk.StringVar(value=self._default_model_label())
        self.dataset_var = tk.StringVar(value=str(self.defaults.get("dataset") or DATASET_OPTIONS[0]))
        self.device_var = tk.StringVar(value=self._default_device_label())
        self.partition_var = tk.StringVar(value=self._default_partition_label())
        self.partition_alpha_var = tk.StringVar(value=self._default_alpha_value())
        self.clients_var = tk.IntVar(value=max(1, _safe_int(self.defaults.get("max_rank"), 3)))
        self.epochs_var = tk.IntVar(value=max(1, _safe_int(self.defaults.get("epochs"), 30)))
        self.batch_size_var = tk.IntVar(value=max(1, _safe_int(self.defaults.get("batch_size"), 64)))
        default_split_layer = _safe_int(self.defaults.get("split_layer"), 1)
        if bool(self.defaults.get("split_before_relu", False)):
            default_split_layer = 0
        self.split_layer_var = tk.IntVar(value=max(0, default_split_layer))
        self.split_point_var = tk.StringVar(value=self._current_split_description())
        self.lr_var = tk.StringVar(value=str(self.defaults.get("lr") or 0.01))
        self.seed_var = tk.IntVar(value=_safe_int(self.defaults.get("seed"), 0))
        self.comm_reduction_var = tk.StringVar(value=self._default_comm_reduction_label())
        self.comm_direction_var = tk.StringVar(value=self._default_comm_direction_label())
        self.quantization_bits_var = tk.StringVar(value=self._default_quantization_bits_label())
        self.sparsity_k_var = tk.StringVar(value=self._default_sparsity_k_label())
        self.dimensionality_reduction_ratio_var = tk.StringVar(value=self._default_dimensionality_reduction_ratio_label())
        self.quantization_granularity_var = tk.StringVar(value=self._default_quantization_granularity_label())
        self.forward_quantization_var = tk.StringVar(value=self._default_quantization_label("forward_quantization"))
        self.backward_quantization_var = tk.StringVar(value=self._default_quantization_label("backward_quantization"))
        self.forward_quantization_addon_var = tk.StringVar(value=self._default_quantization_addon_label("forward_quantization"))
        self.backward_quantization_addon_var = tk.StringVar(value=self._default_quantization_addon_label("backward_quantization"))
        self._syncing_quantization_fields = False
        self._syncing_mode_selection = False

        self.status_var = tk.StringVar(value="Idle")
        self.command_preview_var = tk.StringVar(value="")

        self._apply_saved_state()
        self._sync_fixed_pair_state()
        self._build_ui()
        self._update_partition_state()
        self._update_reduction_state()
        self._refresh_command_preview()
        self.root.after(150, self._drain_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _python_has_module(self, module_name: str) -> bool:
        try:
            probe = subprocess.run(
                [
                    self.python_path,
                    "-c",
                    (
                        "import importlib.util, sys; "
                        f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception:
            return _module_available(module_name)
        return probe.returncode == 0

    def _load_menu_state(self) -> Dict[str, object]:
        if not self.menu_state_path.exists():
            return {}
        try:
            with self.menu_state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, ValueError, TypeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _save_menu_state(self) -> None:
        state = self._collect_menu_state()
        try:
            with self.menu_state_path.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
        except OSError:
            pass

    def _collect_menu_state(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm_var.get(),
            "model": self.model_var.get(),
            "dataset": self.dataset_var.get(),
            "device": self.device_var.get(),
            "partition": self.partition_var.get(),
            "partition_alpha": self.partition_alpha_var.get(),
            "clients": self.clients_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get(),
            "split_layer": self.split_layer_var.get(),
            "lr": self.lr_var.get(),
            "seed": self.seed_var.get(),
            "comm_reduction": self.comm_reduction_var.get(),
            "comm_direction": self.comm_direction_var.get(),
            "quantization_bits": self.quantization_bits_var.get(),
            "sparsity_k": self.sparsity_k_var.get(),
            "dimensionality_reduction_ratio": self.dimensionality_reduction_ratio_var.get(),
            "quantization_granularity": self.quantization_granularity_var.get(),
            "forward_quantization": self.forward_quantization_var.get(),
            "backward_quantization": self.backward_quantization_var.get(),
            "forward_quantization_addon": self.forward_quantization_addon_var.get(),
            "backward_quantization_addon": self.backward_quantization_addon_var.get(),
        }

    def _split_quantization_pipeline(self, value: object) -> Tuple[str, str]:
        raw = str(value or "").strip().lower()
        if not raw:
            return "int", "none"
        parts = [part.strip() for part in raw.split("+") if part.strip()]
        primary = parts[0] if parts else "int"
        addon = parts[1] if len(parts) > 1 else "none"
        return primary, addon

    def _compose_quantization_pipeline(self, primary_label: str, addon_label: str) -> str:
        primary = self.quantization_map.get(primary_label, primary_label)
        addon = self.pipeline_addon_map.get(addon_label, addon_label)
        if addon in {"", "none", primary}:
            return str(primary)
        return f"{primary}+{addon}"

    def _selected_pipeline_methods(self) -> Set[str]:
        return {
            self.forward_quantization_var.get(),
            self.backward_quantization_var.get(),
            self.forward_quantization_addon_var.get(),
            self.backward_quantization_addon_var.get(),
        }

    def _build_mode_selection_map(self) -> Dict[str, Tuple[str, Optional[str]]]:
        options: Dict[str, Tuple[str, Optional[str]]] = {"none": ("none", None)}
        for label, value in COMM_REDUCTION_OPTIONS:
            if value == "none":
                continue
            methods = COMM_METHODS_BY_MODE.get(value, ())
            if not methods:
                options[label] = (value, None)
                continue
            for method in methods:
                options[self._mode_selection_label(value, method)] = (value, method)
        return options

    def _mode_selection_label(self, reduction_mode: str, method: Optional[str]) -> str:
        if reduction_mode == "none" or not method:
            return reduction_mode
        return f"{reduction_mode} / {method}"

    def _resolve_mode_selection(self) -> Tuple[str, Optional[str]]:
        label = self.comm_reduction_var.get()
        if label in self.mode_selection_map:
            return self.mode_selection_map[label]
        if label in self.comm_reduction_map:
            reduction_mode = self.comm_reduction_map[label]
            methods = COMM_METHODS_BY_MODE.get(reduction_mode, ())
            return reduction_mode, (methods[0] if methods else None)
        return "none", None

    def _sync_mode_selection_from_codecs(self) -> None:
        if self._syncing_mode_selection:
            return
        reduction_mode, _selected_method = self._resolve_mode_selection()
        if reduction_mode == "none":
            target_label = "none"
        else:
            target_method = self.forward_quantization_var.get()
            if target_method not in COMM_METHODS_BY_MODE.get(reduction_mode, ()): 
                methods = COMM_METHODS_BY_MODE.get(reduction_mode, ())
                target_method = methods[0] if methods else None
            target_label = self._mode_selection_label(reduction_mode, target_method)
        if self.comm_reduction_var.get() != target_label:
            self._syncing_mode_selection = True
            try:
                self.comm_reduction_var.set(target_label)
            finally:
                self._syncing_mode_selection = False

    def _apply_saved_state(self) -> None:
        if not self.saved_state:
            return

        if self.saved_state.get("algorithm") in self.algorithm_map:
            self.algorithm_var.set(str(self.saved_state["algorithm"]))
        saved_model = str(self.saved_state.get("model") or "").strip().lower().replace("-", "_")
        saved_model_aliases = {
            "agnews_bilstm": "bigru",
            "bilstm": "bigru",
            "bigru": "bigru",
            "bi_gru": "bigru",
            "agnews_bert_tiny": "bert_tiny",
            "berttiny": "bert_tiny",
        }
        saved_model = saved_model_aliases.get(saved_model, saved_model)
        if saved_model in self.model_map:
            self.model_var.set(saved_model)
        if self.saved_state.get("dataset") in DATASET_OPTIONS:
            self.dataset_var.set(str(self.saved_state["dataset"]))
        if self.saved_state.get("device") in self.device_map:
            self.device_var.set(str(self.saved_state["device"]))
        if self.saved_state.get("partition") in self.partition_map:
            self.partition_var.set(str(self.saved_state["partition"]))
        if self.saved_state.get("partition_alpha") is not None:
            self.partition_alpha_var.set(str(self.saved_state["partition_alpha"]))

        self.clients_var.set(max(1, _safe_int(self.saved_state.get("clients"), self.clients_var.get())))
        self.epochs_var.set(max(1, _safe_int(self.saved_state.get("epochs"), self.epochs_var.get())))
        self.batch_size_var.set(max(1, _safe_int(self.saved_state.get("batch_size"), self.batch_size_var.get())))
        saved_split_layer = _safe_int(self.saved_state.get("split_layer"), self.split_layer_var.get())
        if bool(self.saved_state.get("split_before_relu", False)):
            saved_split_layer = 0
        self.split_layer_var.set(max(0, saved_split_layer))
        self.lr_var.set(str(self.saved_state.get("lr", self.lr_var.get())))
        self.seed_var.set(_safe_int(self.saved_state.get("seed"), self.seed_var.get()))

        if self.saved_state.get("comm_direction") in self.comm_direction_map:
            self.comm_direction_var.set(str(self.saved_state["comm_direction"]))
        saved_quantization_bits = self.saved_state.get("quantization_bits")
        quantization_bit_aliases = {
            "8bit": "8bit-static",
            "4bit": "4bit-static",
        }
        if saved_quantization_bits in quantization_bit_aliases:
            saved_quantization_bits = quantization_bit_aliases[str(saved_quantization_bits)]
        valid_quantization_bits = {label for label, _value in QUANTIZATION_BITS_OPTIONS}
        if saved_quantization_bits in valid_quantization_bits:
            self.quantization_bits_var.set(str(saved_quantization_bits))

        saved_sparsity_k = self.saved_state.get("sparsity_k")
        valid_sparsity_k = {label for label, _value in SPARSITY_K_OPTIONS}
        if saved_sparsity_k in valid_sparsity_k:
            self.sparsity_k_var.set(str(saved_sparsity_k))

        saved_dimensionality_ratio = self.saved_state.get("dimensionality_reduction_ratio")
        valid_dimensionality_ratios = {label for label, _value in DIMENSIONALITY_REDUCTION_RATIO_OPTIONS}
        if saved_dimensionality_ratio in valid_dimensionality_ratios:
            self.dimensionality_reduction_ratio_var.set(str(saved_dimensionality_ratio))

        saved_quantization_granularity = str(self.saved_state.get("quantization_granularity") or "").strip().lower()
        valid_quantization_granularities = {label for label, _value in QUANTIZATION_GRANULARITY_OPTIONS}
        if saved_quantization_granularity in valid_quantization_granularities:
            self.quantization_granularity_var.set(saved_quantization_granularity)

        valid_quantizations = {label for label, _value in QUANTIZATION_OPTIONS}
        if self.saved_state.get("forward_quantization") in valid_quantizations:
            self.forward_quantization_var.set(str(self.saved_state["forward_quantization"]))
        if self.saved_state.get("backward_quantization") in valid_quantizations:
            self.backward_quantization_var.set(str(self.saved_state["backward_quantization"]))
        else:
            saved_forward_primary, saved_forward_addon = self._split_quantization_pipeline(self.saved_state.get("forward_quantization"))
            saved_backward_primary, saved_backward_addon = self._split_quantization_pipeline(self.saved_state.get("backward_quantization"))
            if saved_forward_primary in valid_quantizations:
                self.forward_quantization_var.set(saved_forward_primary)
                self.forward_quantization_addon_var.set(saved_forward_addon if saved_forward_addon in self.pipeline_addon_map else "none")
            if saved_backward_primary in valid_quantizations:
                self.backward_quantization_var.set(saved_backward_primary)
                self.backward_quantization_addon_var.set(saved_backward_addon if saved_backward_addon in self.pipeline_addon_map else "none")

        saved_comm_reduction = self.saved_state.get("comm_reduction")
        if saved_comm_reduction in self.mode_selection_map:
            self.comm_reduction_var.set(str(saved_comm_reduction))
        elif saved_comm_reduction in self.comm_reduction_map:
            reduction_mode = self.comm_reduction_map[str(saved_comm_reduction)]
            selected_method = self.forward_quantization_var.get()
            if selected_method not in COMM_METHODS_BY_MODE.get(reduction_mode, ()): 
                methods = COMM_METHODS_BY_MODE.get(reduction_mode, ())
                selected_method = methods[0] if methods else None
            self.comm_reduction_var.set(self._mode_selection_label(reduction_mode, selected_method))
        else:
            self._sync_mode_selection_from_codecs()

    def _job_display(self, job: Job) -> str:
        return f"[{job.status}] {job.name} - {job.summary}"

    def _current_model_value(self) -> str:
        return self.model_map.get(self.model_var.get(), MODEL_OPTIONS[0][1])

    def _current_split_description(self) -> str:
        return FIXED_SPLIT_BY_MODEL.get(self._current_model_value(), "Fixed imported split")

    def _current_supported_datasets(self) -> Tuple[str, ...]:
        return SUPPORTED_DATASETS_BY_MODEL.get(self._current_model_value(), tuple(DATASET_OPTIONS))

    def _sync_fixed_pair_state(self) -> None:
        supported_datasets = self._current_supported_datasets()
        if supported_datasets:
            self.dataset_var.set(supported_datasets[0])
        if hasattr(self, "dataset_combo"):
            self.dataset_combo.configure(values=list(supported_datasets), state="disabled")
        self.split_point_var.set(self._current_split_description())

    def _refresh_queue_list(self) -> None:
        selection = self.queue_list.curselection()
        selected_index = selection[0] if selection else None

        self.queue_list.delete(0, "end")
        with self.jobs_lock:
            jobs_snapshot = list(self.jobs)

        for job in jobs_snapshot:
            self.queue_list.insert("end", self._job_display(job))

        if selected_index is not None and 0 <= selected_index < self.queue_list.size():
            self.queue_list.selection_set(selected_index)

    def _default_algorithm_label(self) -> str:
        current = str(self.defaults.get("variants_type") or "vanilla")
        for label, value in ALGORITHM_OPTIONS:
            if value == current:
                return label
        return ALGORITHM_OPTIONS[0][0]

    def _default_model_label(self) -> str:
        current = str(self.defaults.get("model") or MODEL_OPTIONS[0][1]).strip().lower()
        for label, value in MODEL_OPTIONS:
            if value.strip().lower() == current:
                return label
        alias_map = {
            "mobilenetv3small": "densenet121",
            "mobilenet_v3_small": "densenet121",
            "densenet_121": "densenet121",
            "efficientnetb0": "efficientnet_b0",
            "agnews_bilstm": "bigru",
            "bilstm": "bigru",
            "bigru": "bigru",
            "bi_gru": "bigru",
            "agnews_bert_tiny": "bert_tiny",
            "berttiny": "bert_tiny",
        }
        if current in alias_map:
            return alias_map[current]
        return MODEL_OPTIONS[0][0]

    def _default_partition_label(self) -> str:
        current = str(self.defaults.get("partition_method") or "homo")
        alpha = self.defaults.get("partition_alpha")
        for label, value, preset_alpha in PARTITION_OPTIONS:
            if value != current:
                continue
            if value == "hetero" and alpha is not None and preset_alpha is not None and float(alpha) != float(preset_alpha):
                continue
            return label
        return PARTITION_OPTIONS[0][0]

    def _default_device_label(self) -> str:
        current = str(self.defaults.get("device") or DEVICE_OPTIONS[0][1]).strip().lower()
        for label, value in DEVICE_OPTIONS:
            if value == current:
                return label
        return DEVICE_OPTIONS[0][0]

    def _default_alpha_value(self) -> str:
        preset_alpha = self.partition_alpha_map.get(self._default_partition_label())
        if preset_alpha is None:
            return ""
        return str(preset_alpha)

    def _default_comm_reduction_label(self) -> str:
        quantize_forward = self.defaults.get("quantize_forward")
        quantize_backward = self.defaults.get("quantize_backward")
        quantize_legacy = self.defaults.get("quantize_activations")
        enabled = bool(quantize_forward) or bool(quantize_backward) or bool(quantize_legacy)
        if not enabled:
            return "none"
        forward_kind, _forward_addon = self._split_quantization_pipeline(self.defaults.get("forward_quantization"))
        backward_kind, _backward_addon = self._split_quantization_pipeline(self.defaults.get("backward_quantization"))
        kind = self._default_quantization_label("forward_quantization") if forward_kind else self._default_quantization_label("backward_quantization")
        reduction_mode = "arithmetic_conversion"
        if kind in {"top_k", "random_top_k"}:
            reduction_mode = "sparsity"
        elif kind in {"random_projection", "autoencoder", "low_rank_pca"}:
            reduction_mode = "dimensionality_reduction"
        elif kind in {"uniform", "non_uniform_loyd", "non_uniform_mlaw"}:
            reduction_mode = "codeword"
        elif kind in {"truncation_int"}:
            reduction_mode = "Truncation"
        return self._mode_selection_label(reduction_mode, kind)

    def _default_comm_direction_label(self) -> str:
        quantize_forward = bool(self.defaults.get("quantize_forward")) if self.defaults.get("quantize_forward") is not None else bool(self.defaults.get("quantize_activations")) if self.defaults.get("quantize_activations") is not None else False
        quantize_backward = bool(self.defaults.get("quantize_backward"))
        if quantize_forward and quantize_backward:
            return "forward_backward"
        if quantize_backward:
            return "backward"
        return "forward"

    def _default_quantization_bits_label(self) -> str:
        forward_kind, _forward_addon = self._split_quantization_pipeline(self.defaults.get("forward_quantization"))
        backward_kind, _backward_addon = self._split_quantization_pipeline(self.defaults.get("backward_quantization"))
        kind = self._default_quantization_label("forward_quantization") if forward_kind else self._default_quantization_label("backward_quantization")
        current = int(_safe_int(self.defaults.get("quantization_bits"), 32 if kind in {"top_k", "random_top_k", "paper_top_k", "random_projection", "autoencoder", "low_rank_pca"} else 8))
        if kind in {"top_k", "random_top_k", "paper_top_k", "random_projection", "autoencoder", "low_rank_pca"}:
            if current not in {8, 16, 32}:
                current = 32
        elif current not in {2, 3, 4, 6, 8}:
            current = 8
        return f"{current}bit"

    def _default_sparsity_k_label(self) -> str:
        current = self.defaults.get("sparsity_k")
        try:
            numeric = float(current)
            if 0.0 < numeric < 1.0:
                numeric *= 100.0
            normalized = int(round(numeric))
        except Exception:
            normalized = 0

        forward_kind, _forward_addon = self._split_quantization_pipeline(self.defaults.get("forward_quantization"))
        backward_kind, _backward_addon = self._split_quantization_pipeline(self.defaults.get("backward_quantization"))
        kind = forward_kind or backward_kind
        if kind in {"random_top_k", "random_topk", "random_top_k_sparsity", "paper_top_k", "paper_topk", "paper_top_k_sparsity"}:
            if normalized not in {5, 10, 25, 50}:
                normalized = 5
        else:
            if normalized not in {1, 5, 10, 25, 50}:
                normalized = 1
        return f"{normalized}%"

    def _default_dimensionality_reduction_ratio_label(self) -> str:
        current = self.defaults.get("dimensionality_reduction_ratio")
        try:
            numeric = float(current)
            if numeric > 1.0:
                numeric /= 100.0
        except Exception:
            numeric = 0.25
        if numeric not in {0.125, 0.25, 0.5}:
            numeric = 0.25
        return f"{numeric * 100.0:g}%"

    def _default_quantization_granularity_label(self) -> str:
        current = str(self.defaults.get("quantization_granularity") or "").strip().lower().replace("-", "_")
        if current in {"per_tensor", "per_channel", "per_group"}:
            return current

        forward_kind, _forward_addon = self._split_quantization_pipeline(self.defaults.get("forward_quantization"))
        backward_kind, _backward_addon = self._split_quantization_pipeline(self.defaults.get("backward_quantization"))
        kind = forward_kind or backward_kind
        if kind in {
            "dynamic_symmetric_int8_per_channel",
            "dynamic_int8_per_channel",
            "int8_per_channel",
            "per_channel_int8",
            "uniform_per_channel",
            "uniform_per_channel_codebook_uint8",
            "non_uniform_loyd_per_channel",
            "non_uniform_loyd_per_channel_codebook_uint8",
            "non_uniform_mlaw_per_channel",
            "mulaw_per_channel_codebook_uint8",
            "mu_law_per_channel",
            "mlaw_per_channel",
        }:
            return "per_channel"
        return "per_tensor"

    def _default_quantization_label(self, key: str) -> str:
        current, _addon = self._split_quantization_pipeline(self.defaults.get(key) or "int")
        alias_map = {
            "dynamic_symmetric_int8": "int",
            "dynamic_int8": "int",
            "int8": "int",
            "int": "int",
            "dynamic_symmetric_int8_per_channel": "int",
            "dynamic_int8_per_channel": "int",
            "int8_per_channel": "int",
            "per_channel_int8": "int",
            "fixed_scale_int8": "int",
            "fixed_int8": "int",
            "fp8_e4m3": "float",
            "float8_e4m3": "float",
            "e4m3": "float",
            "float8": "float",
            "float": "float",
            "uniform_codebook_uint8": "uniform",
            "uniform": "uniform",
            "uniform_per_channel_codebook_uint8": "uniform",
            "uniform_per_channel": "uniform",
            "non_uniform_loyd_codebook_uint8": "non_uniform_loyd",
            "non_uniform_loyd": "non_uniform_loyd",
            "non_uniform_loyd_per_channel_codebook_uint8": "non_uniform_loyd",
            "non_uniform_loyd_per_channel": "non_uniform_loyd",
            "loyd_per_channel": "non_uniform_loyd",
            "mulaw_codebook_uint8": "non_uniform_mlaw",
            "non_uniform_mlaw": "non_uniform_mlaw",
            "mulaw_per_channel_codebook_uint8": "non_uniform_mlaw",
            "non_uniform_mlaw_per_channel": "non_uniform_mlaw",
            "mu_law_per_channel": "non_uniform_mlaw",
            "mlaw_per_channel": "non_uniform_mlaw",
            "trunc_noscale_int": "truncation_int",
            "trunc_noscale_int8": "truncation_int",
            "trunc_bits_int8": "truncation_int",
            "trunc_scale_int8": "truncation_int",
            "truncation_int": "truncation_int",
            "truncation_int8": "truncation_int",
            "top_k": "top_k",
            "topk": "top_k",
            "top_k_sparsity": "top_k",
            "random_top_k": "random_top_k",
            "random_topk": "random_top_k",
            "random_top_k_sparsity": "random_top_k",
            "paper_top_k": "paper_top_k",
            "paper_topk": "paper_top_k",
            "paper_top_k_sparsity": "paper_top_k",
            "random_projection": "random_projection",
            "autoencoder": "random_projection",
            "low_rank_pca": "low_rank_pca",
            "low_rank_projection": "low_rank_pca",
            "pca_projection": "low_rank_pca",
            "pca": "low_rank_pca",
            "low_rank": "low_rank_pca",
        }
        if current in alias_map:
            return alias_map[current]
        return QUANTIZATION_OPTIONS[0][0]

    def _default_quantization_addon_label(self, key: str) -> str:
        _primary, addon = self._split_quantization_pipeline(self.defaults.get(key) or "")
        return addon if addon in self.pipeline_addon_map else "none"

    def _build_ui(self) -> None:
        self.root.configure(bg="#eef2f6")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#f8fafc")
        style.configure("Root.TFrame", background="#eef2f6")
        style.configure("Title.TLabel", background="#eef2f6", foreground="#132238", font=("Segoe UI Semibold", 20))
        style.configure("Hint.TLabel", background="#eef2f6", foreground="#516173", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#f8fafc", foreground="#0e1b2a", font=("Segoe UI Semibold", 11))
        style.configure("Status.TLabel", background="#eef2f6", foreground="#243447", font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI Semibold", 10))

        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=16)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=5)
        root_frame.columnconfigure(1, weight=4)
        root_frame.rowconfigure(1, weight=1)

        header = ttk.Frame(root_frame, style="Root.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ttk.Label(header, text="Split-Framework Launcher", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Pick an experiment, queue as many runs as you want, then leave the menu running.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        form_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=16)
        form_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        form_panel.columnconfigure(1, weight=1)
        form_panel.columnconfigure(3, weight=1)

        queue_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=16)
        queue_panel.grid(row=1, column=1, sticky="nsew")
        queue_panel.columnconfigure(0, weight=1)
        queue_panel.rowconfigure(3, weight=1)

        self._add_section_label(form_panel, 0, "Experiment")
        self._add_combo(form_panel, "Algorithm", self.algorithm_var, [label for label, _ in ALGORITHM_OPTIONS], 1, 0)
        self._add_combo(form_panel, "Model", self.model_var, [label for label, _ in MODEL_OPTIONS], 1, 2)
        self.dataset_combo = self._add_combo(
            form_panel,
            "Dataset",
            self.dataset_var,
            list(self._current_supported_datasets()),
            2,
            0,
            state="disabled",
        )
        self._add_combo(form_panel, "Data Partition", self.partition_var, [label for label, _value, _alpha in PARTITION_OPTIONS], 2, 2)

        self._add_section_label(form_panel, 3, "Training")
        self._add_spin(form_panel, "Clients", self.clients_var, 1, 128, 4, 0)
        self._add_spin(form_panel, "Epochs", self.epochs_var, 1, 5000, 5, 0)
        self._add_spin(form_panel, "Batch Size", self.batch_size_var, 1, 4096, 5, 2)
        self._add_entry(form_panel, "Split Point", self.split_point_var, 6, 0, state="readonly", columnspan=3)
        self._add_entry(form_panel, "Learning Rate", self.lr_var, 7, 0)
        self._add_spin(form_panel, "Seed", self.seed_var, 0, 999999, 7, 2)
        self._add_combo(form_panel, "Device", self.device_var, [label for label, _ in DEVICE_OPTIONS], 8, 0)

        self._add_section_label(form_panel, 9, "Communication Reduction")
        self.mode_combo = self._add_combo(form_panel, "Mode", self.comm_reduction_var, list(self.mode_selection_map.keys()), 10, 0)
        self.mode_combo.configure(width=32)
        self._add_combo(form_panel, "Direction", self.comm_direction_var, [label for label, _ in COMM_DIRECTION_OPTIONS], 10, 2)
        self.quantization_bits_combo = self._add_combo(form_panel, "Bit Width", self.quantization_bits_var, [label for label, _ in QUANTIZATION_BITS_OPTIONS], 11, 0)
        self.sparsity_k_combo = self._add_combo(form_panel, "K", self.sparsity_k_var, [label for label, _ in SPARSITY_K_OPTIONS], 11, 2)
        self.quantization_granularity_combo = self._add_combo(form_panel, "Granularity", self.quantization_granularity_var, [label for label, _ in QUANTIZATION_GRANULARITY_OPTIONS], 12, 0)
        self.forward_quantization_combo = self._add_combo(
            form_panel,
            "Forward Codec",
            self.forward_quantization_var,
            [label for label, _ in QUANTIZATION_OPTIONS],
            12,
            2,
        )
        self.forward_quantization_addon_combo = self._add_combo(
            form_panel,
            "Forward Add-on",
            self.forward_quantization_addon_var,
            [label for label, _ in PIPELINE_ADDON_OPTIONS],
            13,
            0,
        )
        self.backward_quantization_combo = self._add_combo(
            form_panel,
            "Backward Codec",
            self.backward_quantization_var,
            [label for label, _ in QUANTIZATION_OPTIONS],
            13,
            2,
        )
        self.backward_quantization_addon_combo = self._add_combo(
            form_panel,
            "Backward Add-on",
            self.backward_quantization_addon_var,
            [label for label, _ in PIPELINE_ADDON_OPTIONS],
            14,
            0,
        )
        self.dimensionality_reduction_ratio_combo = self._add_combo(
            form_panel,
            "Reduced Dim",
            self.dimensionality_reduction_ratio_var,
            [label for label, _ in DIMENSIONALITY_REDUCTION_RATIO_OPTIONS],
            14,
            2,
        )
        self.sparsity_k_widgets = [
            *form_panel.grid_slaves(row=11, column=2),
            *form_panel.grid_slaves(row=11, column=3),
        ]
        self.dimensionality_reduction_ratio_widgets = [
            *form_panel.grid_slaves(row=14, column=2),
            *form_panel.grid_slaves(row=14, column=3),
        ]

        self.reduction_widgets = [
            *form_panel.grid_slaves(row=11, column=0),
            *form_panel.grid_slaves(row=11, column=1),
            *form_panel.grid_slaves(row=11, column=2),
            *form_panel.grid_slaves(row=11, column=3),
            *form_panel.grid_slaves(row=12, column=0),
            *form_panel.grid_slaves(row=12, column=1),
            *form_panel.grid_slaves(row=12, column=2),
            *form_panel.grid_slaves(row=12, column=3),
            *form_panel.grid_slaves(row=13, column=0),
            *form_panel.grid_slaves(row=13, column=1),
            *form_panel.grid_slaves(row=13, column=2),
            *form_panel.grid_slaves(row=13, column=3),
            *form_panel.grid_slaves(row=14, column=0),
            *form_panel.grid_slaves(row=14, column=1),
            *form_panel.grid_slaves(row=14, column=2),
            *form_panel.grid_slaves(row=14, column=3),
        ]

        preview_label = ttk.Label(form_panel, text="Command Preview", style="Section.TLabel")
        preview_label.grid(row=15, column=0, columnspan=4, sticky="w", pady=(18, 8))
        preview_box = tk.Text(
            form_panel,
            height=5,
            wrap="word",
            bg="#0f1720",
            fg="#d8e3ef",
            relief="flat",
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        preview_box.grid(row=16, column=0, columnspan=4, sticky="nsew")
        preview_box.configure(state="disabled")
        self.preview_box = preview_box

        buttons = ttk.Frame(form_panel, style="Panel.TFrame")
        buttons.grid(row=17, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        buttons.columnconfigure((0, 1, 2, 3), weight=1)
        ttk.Button(buttons, text="Add To Queue", command=self._add_job, style="Action.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(buttons, text="Run Queue", command=self._start_queue, style="Action.TButton").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Run Now", command=self._run_now, style="Action.TButton").grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(buttons, text="Start MLflow UI", command=self._start_mlflow_server, style="Action.TButton").grid(row=0, column=3, sticky="ew", padx=(8, 0))

        ttk.Label(queue_panel, text="Queue", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(queue_panel, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 10))

        self.queue_list = tk.Listbox(
            queue_panel,
            height=10,
            bg="#ffffff",
            fg="#132238",
            font=("Segoe UI", 10),
            relief="flat",
            selectbackground="#d4e7ff",
            selectforeground="#132238",
        )
        self.queue_list.grid(row=2, column=0, sticky="ew")

        queue_buttons = ttk.Frame(queue_panel, style="Panel.TFrame")
        queue_buttons.grid(row=3, column=0, sticky="new", pady=(10, 14))
        queue_buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(queue_buttons, text="Remove Selected", command=self._remove_selected).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(queue_buttons, text="Clear Queue", command=self._clear_queue).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(queue_buttons, text="Stop Current Run", command=self._stop_current_run).grid(row=0, column=2, sticky="ew", padx=(8, 0))

        ttk.Label(queue_panel, text="Live Output", style="Section.TLabel").grid(row=4, column=0, sticky="w", pady=(0, 8))
        output = tk.Text(
            queue_panel,
            wrap="word",
            bg="#0f1720",
            fg="#d8e3ef",
            relief="flat",
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        output.grid(row=5, column=0, sticky="nsew")
        queue_panel.rowconfigure(5, weight=1)
        self.output_box = output

        for variable in (
            self.algorithm_var,
            self.model_var,
            self.dataset_var,
            self.device_var,
            self.partition_var,
            self.partition_alpha_var,
            self.lr_var,
            self.comm_reduction_var,
            self.comm_direction_var,
            self.quantization_bits_var,
            self.sparsity_k_var,
            self.dimensionality_reduction_ratio_var,
            self.quantization_granularity_var,
            self.forward_quantization_var,
            self.backward_quantization_var,
            self.forward_quantization_addon_var,
            self.backward_quantization_addon_var,
        ):
            variable.trace_add("write", self._handle_form_change)

        for variable in (
            self.clients_var,
            self.epochs_var,
            self.batch_size_var,
            self.seed_var,
        ):
            variable.trace_add("write", self._handle_form_change)

    def _add_section_label(self, parent: ttk.Frame, row: int, text: str) -> None:
        ttk.Label(parent, text=text, style="Section.TLabel").grid(row=row, column=0, columnspan=4, sticky="w", pady=(0 if row == 0 else 18, 10))

    def _add_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: Sequence[str],
        row: int,
        column: int,
        state: str = "readonly",
    ):
        ttk.Label(parent, text=label, style="Section.TLabel").grid(row=row, column=column, sticky="w", pady=(0, 6))
        combo = ttk.Combobox(parent, textvariable=variable, values=list(values), state=state)
        combo.grid(row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=(0, 6))
        return combo

    def _add_spin(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        minimum: int,
        maximum: int,
        row: int,
        column: int,
    ) -> None:
        ttk.Label(parent, text=label, style="Section.TLabel").grid(row=row, column=column, sticky="w", pady=(0, 6))
        spin = ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum)
        spin.grid(row=row, column=column + 1, sticky="ew", padx=(0, 18), pady=(0, 6))

    def _add_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        row: int,
        column: int,
        state: str = "normal",
        columnspan: int = 1,
    ) -> None:
        ttk.Label(parent, text=label, style="Section.TLabel").grid(row=row, column=column, sticky="w", pady=(0, 6))
        entry = ttk.Entry(parent, textvariable=variable, state=state)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=(0, 18), pady=(0, 6))

    def _handle_form_change(self, *_args: object) -> None:
        self._sync_fixed_pair_state()
        self._update_partition_state()
        self._update_reduction_state()
        self._sync_mode_selection_from_codecs()
        self._refresh_command_preview()

    def _sync_quantization_direction_state(self) -> None:
        if self._syncing_quantization_fields:
            return

        direction = self.comm_direction_map.get(self.comm_direction_var.get(), "forward")
        backward_state = "readonly"

        if direction == "both":
            backward_state = "disabled"
            if self.backward_quantization_var.get() != self.forward_quantization_var.get():
                self._syncing_quantization_fields = True
                try:
                    self.backward_quantization_var.set(self.forward_quantization_var.get())
                finally:
                    self._syncing_quantization_fields = False
            if self.backward_quantization_addon_var.get() != self.forward_quantization_addon_var.get():
                self._syncing_quantization_fields = True
                try:
                    self.backward_quantization_addon_var.set(self.forward_quantization_addon_var.get())
                finally:
                    self._syncing_quantization_fields = False

        self.backward_quantization_combo.configure(state=backward_state)
        self.backward_quantization_addon_combo.configure(state=backward_state)

    def _update_partition_state(self) -> None:
        partition_value = self.partition_map.get(self.partition_var.get(), "homo")
        alpha_value = self.partition_alpha_map.get(self.partition_var.get())
        if partition_value == "hetero" and alpha_value is not None:
            self.partition_alpha_var.set(str(alpha_value))
        elif partition_value == "alpha0":
            self.partition_alpha_var.set("0.0")
        else:
            self.partition_alpha_var.set("")

    def _update_reduction_state(self) -> None:
        reduction_mode, selected_method = self._resolve_mode_selection()
        show_reduction = reduction_mode != "none"
        for widget in self.reduction_widgets:
            if show_reduction:
                widget.grid()
            else:
                widget.grid_remove()

        allowed_methods = COMM_METHODS_BY_MODE.get(reduction_mode, tuple(label for label, _ in QUANTIZATION_OPTIONS))
        self.forward_quantization_combo.configure(values=list(allowed_methods))
        self.backward_quantization_combo.configure(values=list(allowed_methods))
        addon_values = [label for label, _value in PIPELINE_ADDON_OPTIONS]
        self.forward_quantization_addon_combo.configure(values=addon_values)
        self.backward_quantization_addon_combo.configure(values=addon_values)
        if allowed_methods:
            if selected_method in allowed_methods:
                if self.forward_quantization_var.get() != selected_method:
                    self.forward_quantization_var.set(selected_method)
                if self.backward_quantization_var.get() not in allowed_methods:
                    self.backward_quantization_var.set(selected_method)
            if self.forward_quantization_var.get() not in allowed_methods:
                self.forward_quantization_var.set(allowed_methods[0])
            if self.backward_quantization_var.get() not in allowed_methods:
                self.backward_quantization_var.set(allowed_methods[0])

        selected_methods = self._selected_pipeline_methods()
        uses_sparsity = bool(selected_methods & {"top_k", "random_top_k", "paper_top_k"})
        uses_dimensionality_ratio = bool(selected_methods & {"random_projection", "autoencoder", "low_rank_pca"})
        is_sparsity = reduction_mode == "sparsity"
        is_dimensionality_reduction = reduction_mode == "dimensionality_reduction"
        if is_sparsity or is_dimensionality_reduction:
            bit_values = [label for label, value in QUANTIZATION_BITS_OPTIONS if int(value) in {8, 16, 32}]
            self.quantization_bits_combo.configure(values=bit_values, state="readonly")
            if self.quantization_bits_var.get() not in bit_values:
                self.quantization_bits_var.set("32bit")
        else:
            bit_values = [label for label, value in QUANTIZATION_BITS_OPTIONS if int(value) in {2, 3, 4, 6, 8}]
            self.quantization_bits_combo.configure(values=bit_values, state="readonly")
            if self.quantization_bits_var.get() not in bit_values:
                self.quantization_bits_var.set("8bit")
        self.quantization_granularity_combo.configure(state=("disabled" if (is_sparsity or is_dimensionality_reduction) else "readonly"))
        sparsity_values = [label for label, _value in SPARSITY_K_OPTIONS if not ({"random_top_k", "paper_top_k"} & selected_methods) or label != "1%"]
        self.sparsity_k_combo.configure(values=sparsity_values)
        self.sparsity_k_combo.configure(state=("readonly" if uses_sparsity else "disabled"))
        if self.sparsity_k_var.get() not in sparsity_values:
            self.sparsity_k_var.set(sparsity_values[0])
        self.dimensionality_reduction_ratio_combo.configure(state=("readonly" if uses_dimensionality_ratio else "disabled"))
        for widget in self.sparsity_k_widgets:
            if uses_sparsity:
                widget.grid()
            else:
                widget.grid_remove()
        for widget in self.dimensionality_reduction_ratio_widgets:
            if uses_dimensionality_ratio:
                widget.grid()
            else:
                widget.grid_remove()

        self._sync_quantization_direction_state()

    def _quoted_command(self, command: Sequence[str]) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline(list(command))
        return shlex.join(command)

    def _selected_algorithm_uses_fed_server(self) -> bool:
        algorithm = self.algorithm_map.get(self.algorithm_var.get(), "")
        return algorithm in {"SplitFed", "SplitFed2"}

    def _build_command(self) -> List[str]:
        algorithm = self.algorithm_map[self.algorithm_var.get()]
        clients = max(1, int(self.clients_var.get()))
        partition_value = self.partition_map[self.partition_var.get()]
        if algorithm == "central":
            active_clients = clients
            total_processes = 1
        else:
            active_clients = clients
            total_processes = clients + (2 if self._selected_algorithm_uses_fed_server() else 1)

        command = [
            self.mpiexec_path,
            "-np",
            str(total_processes),
            self.python_path,
            str(self.main_path),
            "--config",
            str(self.config_path),
            "--variants-type",
            algorithm,
            "--model",
            self.model_map[self.model_var.get()],
            "--dataset",
            self.dataset_var.get(),
            "--device",
            self.device_map[self.device_var.get()],
            "--partition-method",
            partition_value,
            "--partition-client-number",
            str(active_clients),
            "--batch-size",
            str(int(self.batch_size_var.get())),
            "--lr",
            str(float(self.lr_var.get())),
            "--epochs",
            str(int(self.epochs_var.get())),
            "--max-rank",
            str(active_clients),
            "--seed",
            str(int(self.seed_var.get())),
        ]

        if partition_value == "hetero":
            command.extend(["--partition-alpha", str(float(self.partition_alpha_var.get()))])
        elif partition_value == "alpha0":
            command.extend(["--partition-alpha", "0.0"])

        reduction_mode, _selected_method = self._resolve_mode_selection()
        if reduction_mode != "none":
            direction = self.comm_direction_map[self.comm_direction_var.get()]
            selected_methods = self._selected_pipeline_methods()
            uses_sparsity = bool(selected_methods & {"top_k", "random_top_k", "paper_top_k"})
            uses_dimensionality_ratio = bool(selected_methods & {"random_projection", "autoencoder", "low_rank_pca"})
            bits_value = self.quantization_bits_map[self.quantization_bits_var.get()]
            command.extend(["--quantization-bits", str(bits_value)])
            if uses_sparsity:
                sparsity_value = self.sparsity_k_map[self.sparsity_k_var.get()]
                command.extend(["--sparsity-k", str(sparsity_value)])
            if uses_dimensionality_ratio:
                ratio_value = self.dimensionality_reduction_ratio_map[self.dimensionality_reduction_ratio_var.get()]
                command.extend(["--dimensionality-reduction-ratio", str(ratio_value)])
            if reduction_mode not in {"sparsity", "dimensionality_reduction"}:
                granularity_value = self.quantization_granularity_map[self.quantization_granularity_var.get()]
                command.extend(["--quantization-granularity", granularity_value])
            if direction in {"forward", "both"}:
                command.append("--quantize-forward")
                forward_quantization = self._compose_quantization_pipeline(
                    self.forward_quantization_var.get(),
                    self.forward_quantization_addon_var.get(),
                )
                command.extend(["--forward-quantization", forward_quantization])
            else:
                command.append("--no-quantize-forward")

            if direction in {"backward", "both"}:
                command.append("--quantize-backward")
                backward_quantization = self._compose_quantization_pipeline(
                    self.backward_quantization_var.get(),
                    self.backward_quantization_addon_var.get(),
                )
                command.extend(["--backward-quantization", backward_quantization])
            else:
                command.append("--no-quantize-backward")
        else:
            command.extend(["--no-quantize-forward", "--no-quantize-backward"])

        return command

    def _build_summary(self) -> str:
        clients = int(self.clients_var.get())
        partition_value = self.partition_map[self.partition_var.get()]
        summary = [
            self.algorithm_var.get(),
            self.model_var.get(),
            self.dataset_var.get(),
            self.device_var.get(),
            f"{clients} clients",
            self.partition_var.get(),
            f"{int(self.epochs_var.get())} epochs",
        ]
        reduction_mode, _selected_method = self._resolve_mode_selection()
        if reduction_mode != "none":
            direction = self.comm_direction_map[self.comm_direction_var.get()]
            forward_pipeline = self._compose_quantization_pipeline(self.forward_quantization_var.get(), self.forward_quantization_addon_var.get())
            backward_pipeline = self._compose_quantization_pipeline(self.backward_quantization_var.get(), self.backward_quantization_addon_var.get())
            selected_methods = self._selected_pipeline_methods()
            uses_sparsity = bool(selected_methods & {"top_k", "random_top_k", "paper_top_k"})
            uses_dimensionality_ratio = bool(selected_methods & {"random_projection", "autoencoder", "low_rank_pca"})
            if reduction_mode == "sparsity" or uses_sparsity:
                summary.append(f"{self.quantization_bits_var.get()} {self.sparsity_k_var.get()} {forward_pipeline} {reduction_mode}:{direction}")
            elif reduction_mode == "dimensionality_reduction" or uses_dimensionality_ratio:
                if uses_dimensionality_ratio:
                    summary.append(f"{self.quantization_bits_var.get()} {self.dimensionality_reduction_ratio_var.get()} {forward_pipeline} {reduction_mode}:{direction}")
                else:
                    summary.append(f"{self.quantization_bits_var.get()} {forward_pipeline} {reduction_mode}:{direction}")
            else:
                pipeline_text = forward_pipeline if direction != "backward" else backward_pipeline
                summary.append(f"{self.quantization_bits_var.get()} {self.quantization_granularity_var.get()} {pipeline_text} {reduction_mode}:{direction}")
        return " | ".join(summary)

    def _validate_form(self) -> List[str]:
        errors: List[str] = []
        try:
            clients = int(self.clients_var.get())
            if clients <= 0:
                errors.append("Clients must be at least 1.")
        except (TypeError, ValueError):
            errors.append("Clients must be a whole number.")
            clients = 1

        try:
            if float(self.lr_var.get()) <= 0:
                errors.append("Learning rate must be greater than 0.")
        except (TypeError, ValueError):
            errors.append("Learning rate must be numeric.")

        if self.device_var.get() not in self.device_map:
            errors.append("Device must be cpu or gpu.")

        selected_model = self.model_var.get()
        selected_dataset = self.dataset_var.get()
        supported_datasets = SUPPORTED_DATASETS_BY_MODEL.get(selected_model, ())
        if supported_datasets and selected_dataset not in supported_datasets:
            supported_text = ", ".join(supported_datasets)
            errors.append(f"Model {selected_model} only supports dataset(s): {supported_text}.")
        elif selected_model in SUPPORTED_DATASETS_BY_MODEL and not supported_datasets:
            errors.append(f"Model {selected_model} is not supported for the available datasets in this launcher.")

        if selected_dataset == "ag_news" and selected_model == "bert_tiny":
            if not self._python_has_module("transformers"):
                    errors.append("BERT-tiny runs require transformers in the active virtual environment.")

        partition_value = self.partition_map[self.partition_var.get()]
        if partition_value == "hetero":
            try:
                if float(self.partition_alpha_var.get()) <= 0:
                    errors.append("Dirichlet alpha must be greater than 0 for hetero partitioning.")
            except (TypeError, ValueError):
                errors.append("Dirichlet alpha must be numeric.")

        reduction_mode, _selected_method = self._resolve_mode_selection()
        if reduction_mode != "none":
            allowed_methods = COMM_METHODS_BY_MODE.get(reduction_mode, ())
            if self.forward_quantization_var.get() not in allowed_methods:
                errors.append(f"Forward method must match {reduction_mode}.")
            if self.backward_quantization_var.get() not in allowed_methods:
                errors.append(f"Backward method must match {reduction_mode}.")
            if self.forward_quantization_addon_var.get() not in self.pipeline_addon_map:
                errors.append("Forward add-on method is invalid.")
            if self.backward_quantization_addon_var.get() not in self.pipeline_addon_map:
                errors.append("Backward add-on method is invalid.")
            selected_methods = self._selected_pipeline_methods()
            uses_sparsity = bool(selected_methods & {"top_k", "random_top_k", "paper_top_k"})
            uses_dimensionality_ratio = bool(selected_methods & {"random_projection", "autoencoder", "low_rank_pca"})
            if uses_sparsity:
                if self.sparsity_k_var.get() not in self.sparsity_k_map:
                    errors.append("Sparsity K must be one of the supported percentages.")
                if (
                    bool({"random_top_k", "paper_top_k"} & selected_methods)
                    and self.sparsity_k_var.get() == "1%"
                ):
                    errors.append("Random Top-k and Paper Top-k support 5%, 10%, 25%, or 50% only.")
            if uses_dimensionality_ratio:
                if self.dimensionality_reduction_ratio_var.get() not in self.dimensionality_reduction_ratio_map:
                    errors.append("Reduced dimension must be one of the supported percentages.")

        if not self.main_path.exists():
            errors.append(f"Missing runner: {self.main_path}")
        if not self.config_path.exists():
            errors.append(f"Missing config: {self.config_path}")

        return errors

    def _refresh_command_preview(self) -> None:
        try:
            preview = self._quoted_command(self._build_command())
        except Exception as exc:
            preview = f"Invalid command: {exc}"
        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        self.preview_box.insert("1.0", preview)
        self.preview_box.configure(state="disabled")

    def _add_job(self) -> None:
        errors = self._validate_form()
        if errors:
            messagebox.showerror("Invalid Settings", "\n".join(errors))
            return

        command = self._build_command()
        summary = self._build_summary()
        with self.jobs_lock:
            job = Job(name=f"Run {len(self.jobs) + 1}", command=command, summary=summary)
            self.jobs.append(job)
            queued_count = len(self.jobs)
        self._refresh_queue_list()
        self.status_var.set(f"Queued {queued_count} run(s)")

    def _run_now(self) -> None:
        if self.running:
            messagebox.showinfo("Queue Running", "A run is already in progress. Stop it first or add another job to the queue.")
            return
        self._add_job()
        with self.jobs_lock:
            has_jobs = bool(self.jobs)
        if has_jobs:
            self._start_queue()

    def _mlflow_tracking_uri(self) -> str:
        return (PROJECT_ROOT / "runtime" / "mlflow" / "mlruns").resolve().as_uri()

    def _mlflow_ui_url(self) -> str:
        return "http://127.0.0.1:5000"

    def _is_port_open(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.3)
            return sock.connect_ex((host, port)) == 0

    def _start_mlflow_server(self) -> None:
        if self._is_port_open("127.0.0.1", 5000):
            self.status_var.set("MLflow UI already running")
            self.output_box.insert("end", f"MLflow UI already running at {self._mlflow_ui_url()}\n")
            self.output_box.see("end")
            webbrowser.open(self._mlflow_ui_url())
            return

        command = [
            self.python_path,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            self._mlflow_tracking_uri(),
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
        ]
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        try:
            self.mlflow_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        except FileNotFoundError:
            messagebox.showerror("MLflow Missing", "Could not start MLflow. Python executable was not found.")
            return
        except Exception as exc:
            messagebox.showerror("MLflow Failed", f"Failed to start MLflow UI: {exc}")
            return

        self.status_var.set("Starting MLflow UI...")
        self.output_box.insert("end", f"Starting MLflow UI with: {self._quoted_command(command)}\n")
        self.output_box.see("end")
        self.root.after(1200, self._finalize_mlflow_launch)

    def _finalize_mlflow_launch(self) -> None:
        if self._is_port_open("127.0.0.1", 5000):
            self.status_var.set("MLflow UI running")
            self.output_box.insert("end", f"MLflow UI running at {self._mlflow_ui_url()}\n")
            self.output_box.see("end")
            webbrowser.open(self._mlflow_ui_url())
            return

        self.status_var.set("MLflow UI failed to start")
        self.output_box.insert("end", "MLflow UI did not start on port 5000. Check the spawned console for details.\n")
        self.output_box.see("end")

    def _remove_selected(self) -> None:
        if self.running:
            messagebox.showinfo("Queue Running", "Stop the current run before editing the queue.")
            return
        selection = self.queue_list.curselection()
        if not selection:
            return
        index = selection[0]
        with self.jobs_lock:
            del self.jobs[index]
            queued_count = len(self.jobs)
        self._refresh_queue_list()
        self.status_var.set(f"Queued {queued_count} run(s)")

    def _clear_queue(self) -> None:
        if self.running:
            messagebox.showinfo("Queue Running", "Stop the current run before clearing the queue.")
            return
        with self.jobs_lock:
            self.jobs.clear()
        self._refresh_queue_list()
        self.status_var.set("Queue cleared")

    def _start_queue(self) -> None:
        if self.running:
            messagebox.showinfo("Queue Running", "The queue is already running.")
            return
        with self.jobs_lock:
            waiting_jobs = sum(1 for job in self.jobs if job.status == "Waiting")
        if waiting_jobs == 0:
            self._add_job()
            with self.jobs_lock:
                waiting_jobs = sum(1 for job in self.jobs if job.status == "Waiting")
            if waiting_jobs == 0:
                return

        self.stop_requested = False
        self.running = True
        self.runner_thread = threading.Thread(target=self._run_queue_worker, daemon=True)
        self.runner_thread.start()
        with self.jobs_lock:
            waiting_jobs = sum(1 for job in self.jobs if job.status == "Waiting")
        self.status_var.set(f"Running queue ({waiting_jobs} waiting)")

    def _run_queue_worker(self) -> None:
        while not self.stop_requested:
            with self.jobs_lock:
                job = next((candidate for candidate in self.jobs if candidate.status == "Waiting"), None)
                if job is None:
                    break
                job.status = "Started"
            self.ui_queue.put(("refresh_queue", ""))
            self.ui_queue.put(("status", f"Running: {job.summary}"))
            self.ui_queue.put(("log", f"\n$ {self._quoted_command(job.command)}\n"))
            try:
                self.process = subprocess.Popen(
                    job.command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                self.ui_queue.put(("log", "Failed to start run. mpiexec was not found on PATH.\n"))
                self.ui_queue.put(("status", "mpiexec not found"))
                break
            except Exception as exc:
                self.ui_queue.put(("log", f"Failed to start run: {exc}\n"))
                self.ui_queue.put(("status", "Run failed to start"))
                break

            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.ui_queue.put(("log", line))
                if self.stop_requested:
                    break

            return_code = self.process.wait()
            if self.stop_requested:
                with self.jobs_lock:
                    job.status = "Stopped"
                self.ui_queue.put(("refresh_queue", ""))
                self.ui_queue.put(("log", f"Run stopped with exit code {return_code}.\n"))
                break

            final_job_status = "Finished" if return_code == 0 else "Failed"
            with self.jobs_lock:
                job.status = final_job_status
            self.ui_queue.put(("refresh_queue", ""))
            if return_code == 0:
                self.ui_queue.put(("log", f"Run finished successfully: {job.summary}\n"))
            else:
                self.ui_queue.put(("log", f"Run failed with exit code {return_code}: {job.summary}\n"))

        self.process = None
        self.running = False
        with self.jobs_lock:
            remaining_jobs = any(job.status == "Waiting" for job in self.jobs)
        final_status = "Queue stopped" if self.stop_requested else ("Queue finished" if not remaining_jobs else self.status_var.get())
        self.ui_queue.put(("status", final_status))

    def _stop_current_run(self) -> None:
        self.stop_requested = True
        if self.process is None:
            self.status_var.set("Stop requested")
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.status_var.set("Stopping current run...")

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                event, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if event == "log":
                self.output_box.insert("end", payload)
                self.output_box.see("end")
            elif event == "status":
                self.status_var.set(payload)
            elif event == "refresh_queue":
                self._refresh_queue_list()

        self.root.after(150, self._drain_ui_queue)

    def _on_close(self) -> None:
        if self.running and not messagebox.askyesno("Exit Launcher", "A run is still active. Stop it and close the launcher?"):
            return
        self._save_menu_state()
        self._stop_current_run()
        if self.mlflow_process is not None and self.mlflow_process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.mlflow_process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ExperimentMenu(root)
    root.mainloop()


if __name__ == "__main__":
    main()
