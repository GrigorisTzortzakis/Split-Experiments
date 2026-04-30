"""
SLFrame - Split Learning Framework
===================================
Main entry point for running split learning experiments.

Usage:
    mpiexec -np <num_processes> python main.py

    Where <num_processes> = 1 server + N clients
    Example: mpiexec -np 3 python main.py  (1 server + 2 clients)

"""

import os
import torch
import logging
import sys
import random
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np

# Ensure project root (Split-Framework/) is on sys.path so imports work reliably.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure all relative paths (results/, saved_progress/, data/) are under Split-Framework/.
os.chdir(PROJECT_ROOT)

from runtime.exports.log import Log
from runtime.mlflow.tracker import export_run_to_mlflow, mlflow_enabled
from datasets.Dataset_Picker import datasetFactory
from runtime.exports.config import yaml_config
from models.model_factory import model_factory
from runtime.MPI.start_MPI import SplitNN_distributed, SplitNN_init


def init_training_device(requested_device, process_id, num_workers, gpu_num_per_machine, model_name=None):
    """
    Initialize the GPU/CPU device for this process.
    Server (process 0) gets GPU 0, clients are distributed across available GPUs.
    """
    requested = str(requested_device or "gpu").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")

    available_gpu_count = torch.cuda.device_count()
    normalized_model_name = str(model_name or "").strip().lower().replace("-", "_")
    if normalized_model_name in {"bilstm", "bigru", "bi_gru"} and torch.cuda.is_available() and available_gpu_count >= 1:
        # For the text GRU path, CPU fallback is disproportionately expensive.
        # Keep all ranks on the same GPU when only one GPU is available.
        return torch.device("cuda:0")

    if available_gpu_count <= 1 and num_workers > 1:
        if process_id == 0 and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if process_id == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    
    # Map client processes to GPUs
    gpu_slots = max(1, min(int(gpu_num_per_machine), available_gpu_count if available_gpu_count > 0 else 1))
    gpu_index = (process_id - 1) % gpu_slots
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    return device


def _move_model_obj_to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_move_model_obj_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_model_obj_to_device(v, device) for v in obj)
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _configure_process_runtime(args, process_id, worker_number):
    requested_device = args["device"]
    device = init_training_device(
        requested_device,
        process_id,
        worker_number - 1,
        args.gpu_num_per_server,
        args.get("model") if isinstance(args, dict) else getattr(args, "model", None),
    )

    return device


def main():
    """Main entry point for split learning experiments."""
    
    # Optional CLI overrides (safe with MPI because every rank receives the same argv).
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        dest="config",
        default=None,
        help="Path to a YAML config (defaults to setup/config/config.yaml).",
    )
    parser.add_argument("--variants-type", dest="variants_type", default=None)
    parser.add_argument("--dataset", dest="dataset", default=None)
    parser.add_argument("--model", dest="model", default=None)
    parser.add_argument("--partition-method", dest="partition_method", default=None)
    parser.add_argument("--partition-alpha", dest="partition_alpha", type=float, default=None)
    parser.add_argument("--device", dest="device", default=None)
    parser.add_argument("--split-layer", dest="split_layer", type=int, default=None)
    parser.add_argument("--split-before-relu", dest="split_before_relu", action="store_true", default=None)
    parser.add_argument("--no-split-before-relu", dest="split_before_relu", action="store_false", default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="lr", type=float, default=None)
    parser.add_argument("--momentum", dest="momentum", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--log-step", dest="log_step", type=int, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--max-rank", dest="max_rank", type=int, default=None)
    parser.add_argument(
        "--partition-client-number",
        dest="partition_client_number",
        type=int,
        default=None,
        help="Deprecated compatibility flag. Partition count now always matches the selected client count.",
    )

    # Communication compression / quantization (forward/backward) overrides.
    # Note: these are read by runtime.MPI.Messaging_MPI.MessageManager.
    parser.add_argument("--quantize-forward", dest="quantize_forward", action="store_true", default=None)
    parser.add_argument("--no-quantize-forward", dest="quantize_forward", action="store_false", default=None)
    parser.add_argument("--quantize-backward", dest="quantize_backward", action="store_true", default=None)
    parser.add_argument("--no-quantize-backward", dest="quantize_backward", action="store_false", default=None)
    parser.add_argument("--forward-quantization", dest="forward_quantization", type=str, default=None)
    parser.add_argument("--backward-quantization", dest="backward_quantization", type=str, default=None)
    parser.add_argument("--quantization-bits", dest="quantization_bits", type=int, default=None)
    parser.add_argument("--dynamic-quantization", dest="dynamic_quantization", action="store_true", default=None)
    parser.add_argument("--no-dynamic-quantization", dest="dynamic_quantization", action="store_false", default=None)
    parser.add_argument("--dynamic-quantization-mode", dest="dynamic_quantization_mode", type=str, default=None)
    parser.add_argument("--dynamic-quantization-baseline-low-scale", dest="dynamic_quantization_baseline_low_scale", type=float, default=None)
    parser.add_argument("--dynamic-quantization-baseline-high-scale", dest="dynamic_quantization_baseline_high_scale", type=float, default=None)
    parser.add_argument("--dynamic-quantization-forward-low-scale", dest="dynamic_quantization_forward_low_scale", type=float, default=None)
    parser.add_argument("--dynamic-quantization-forward-high-scale", dest="dynamic_quantization_forward_high_scale", type=float, default=None)
    parser.add_argument("--dynamic-quantization-backward-low-scale", dest="dynamic_quantization_backward_low_scale", type=float, default=None)
    parser.add_argument("--dynamic-quantization-backward-high-scale", dest="dynamic_quantization_backward_high_scale", type=float, default=None)
    parser.add_argument("--truncation-scale", dest="truncation_scale", type=float, default=None)
    parser.add_argument("--forward-truncation-scale", dest="forward_truncation_scale", type=float, default=None)
    parser.add_argument("--backward-truncation-scale", dest="backward_truncation_scale", type=float, default=None)
    parser.add_argument("--mlflow", dest="mlflow_enabled", action="store_true", default=None)
    parser.add_argument("--no-mlflow", dest="mlflow_enabled", action="store_false", default=None)
    parser.add_argument("--mlflow-uri", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--mlflow-experiment", dest="mlflow_experiment_name", default=None)
    parser.add_argument("--mlflow-run-name", dest="mlflow_run_name", default=None)
    cli_args, _unknown = parser.parse_known_args()

    # 1. Load configuration from YAML
    cfg_path = (
        Path(cli_args.config).expanduser().resolve()
        if cli_args.config
        else (Path(__file__).resolve().parent / "config" / "config.yaml")
    )
    args = yaml_config.load(str(cfg_path))
    if cli_args.variants_type is not None:
        args["variants_type"] = cli_args.variants_type
    if cli_args.dataset is not None:
        args["dataset"] = cli_args.dataset
    if cli_args.model is not None:
        args["model"] = cli_args.model
    if cli_args.partition_method is not None:
        args["partition_method"] = cli_args.partition_method
    if cli_args.partition_alpha is not None:
        args["partition_alpha"] = cli_args.partition_alpha
    if cli_args.device is not None:
        args["device"] = cli_args.device
    if cli_args.batch_size is not None:
        args["batch_size"] = cli_args.batch_size
    if cli_args.lr is not None:
        args["lr"] = cli_args.lr
    if cli_args.momentum is not None:
        args["momentum"] = cli_args.momentum
    if cli_args.weight_decay is not None:
        args["weight_decay"] = cli_args.weight_decay
    if cli_args.epochs is not None:
        args["epochs"] = cli_args.epochs
    if cli_args.log_step is not None:
        args["log_step"] = cli_args.log_step
    if cli_args.seed is not None:
        args["seed"] = cli_args.seed
    if cli_args.max_rank is not None:
        args["max_rank"] = cli_args.max_rank
    if cli_args.partition_client_number is not None:
        args["partition_client_number"] = cli_args.partition_client_number

    # Apply quantization overrides (stored as extra config keys when not part of the base Config dataclass).
    if cli_args.quantize_forward is not None:
        args["quantize_forward"] = bool(cli_args.quantize_forward)
    if cli_args.quantize_backward is not None:
        args["quantize_backward"] = bool(cli_args.quantize_backward)
    if cli_args.forward_quantization is not None:
        args["forward_quantization"] = str(cli_args.forward_quantization)
    if cli_args.backward_quantization is not None:
        args["backward_quantization"] = str(cli_args.backward_quantization)
    if cli_args.quantization_bits is not None:
        args["quantization_bits"] = int(cli_args.quantization_bits)
    if cli_args.dynamic_quantization is not None:
        args["dynamic_quantization"] = bool(cli_args.dynamic_quantization)
    if cli_args.dynamic_quantization_mode is not None:
        args["dynamic_quantization_mode"] = str(cli_args.dynamic_quantization_mode)
        args["dynamic_quantization"] = True
    if cli_args.dynamic_quantization_baseline_low_scale is not None:
        args["dynamic_quantization_baseline_low_scale"] = float(cli_args.dynamic_quantization_baseline_low_scale)
    if cli_args.dynamic_quantization_baseline_high_scale is not None:
        args["dynamic_quantization_baseline_high_scale"] = float(cli_args.dynamic_quantization_baseline_high_scale)
    if cli_args.dynamic_quantization_forward_low_scale is not None:
        args["dynamic_quantization_forward_low_scale"] = float(cli_args.dynamic_quantization_forward_low_scale)
    if cli_args.dynamic_quantization_forward_high_scale is not None:
        args["dynamic_quantization_forward_high_scale"] = float(cli_args.dynamic_quantization_forward_high_scale)
    if cli_args.dynamic_quantization_backward_low_scale is not None:
        args["dynamic_quantization_backward_low_scale"] = float(cli_args.dynamic_quantization_backward_low_scale)
    if cli_args.dynamic_quantization_backward_high_scale is not None:
        args["dynamic_quantization_backward_high_scale"] = float(cli_args.dynamic_quantization_backward_high_scale)
    if cli_args.truncation_scale is not None:
        args["truncation_scale"] = float(cli_args.truncation_scale)
    if cli_args.forward_truncation_scale is not None:
        args["forward_truncation_scale"] = float(cli_args.forward_truncation_scale)
    if cli_args.backward_truncation_scale is not None:
        args["backward_truncation_scale"] = float(cli_args.backward_truncation_scale)
    if cli_args.mlflow_enabled is not None:
        args["mlflow_enabled"] = bool(cli_args.mlflow_enabled)
    if cli_args.mlflow_tracking_uri is not None:
        args["mlflow_tracking_uri"] = str(cli_args.mlflow_tracking_uri)
    if cli_args.mlflow_experiment_name is not None:
        args["mlflow_experiment_name"] = str(cli_args.mlflow_experiment_name)
    if cli_args.mlflow_run_name is not None:
        args["mlflow_run_name"] = str(cli_args.mlflow_run_name)
    args["config_path"] = str(cfg_path)

    # Reproducibility: ensure the same random stream across MPI ranks.
    # This is especially important for dataset partitioning (Dirichlet splits).
    seed = args["seed"]
    if seed is not None and int(seed) >= 0:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Ensure dataset root is under Split-Framework/datasets/downloads.
    data_dir = args["dataDir"] or str(PROJECT_ROOT / "datasets" / "downloads")
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = (PROJECT_ROOT / data_dir_path).resolve()
    args["dataDir"] = str(data_dir_path)
    
    comm = None
    process_id = None
    worker_number = None
    run_success = False
    run_error_message = None

    # 2. Initialize MPI communication
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id

    # Keep config consistent with the MPI world size.
    # Many algorithms (e.g., semaphore ring / finish broadcast) rely on max_rank being exact.
    # If `max_rank` is larger than `mpiexec -np - 1`, ranks will send to non-existent processes
    # and can deadlock or crash.
    variant_name_raw = str(args["variants_type"] or "").lower()
    uses_fed_server = variant_name_raw in {"splitfed", "splitfed2"}
    if variant_name_raw == "central":
        central_partition_count = args["central_partition_count"]
        if central_partition_count is None:
            central_partition_count = args["partition_client_number"]
        if central_partition_count is None:
            central_partition_count = args["max_rank"]
        if central_partition_count is None:
            central_partition_count = 1
        central_partition_count = max(1, int(central_partition_count))
        args["central_partition_count"] = central_partition_count
        args["client_number"] = central_partition_count
        args["partition_client_number"] = central_partition_count
        args["max_rank"] = 0
    else:
        actual_max_rank = int(worker_number) - (2 if uses_fed_server else 1)
        configured_max_rank = args["max_rank"]
        if configured_max_rank is None:
            args["max_rank"] = actual_max_rank
        elif int(configured_max_rank) != actual_max_rank:
            logging.warning(
                "Overriding config max_rank=%s to match MPI topology (active client max_rank=%s).",
                configured_max_rank,
                actual_max_rank,
            )
            args["max_rank"] = actual_max_rank

        args["client_number"] = args["max_rank"]
        args["partition_client_number"] = int(args["client_number"])

    # 2b. Build a per-run log filename using current settings.
    # Examples:
    #   5client_resnet18_50epochs_homo_09-02-2026_14-02.log
    #   10client_bert_tiny_50epochs_hetero_a0.1_11-02-2026_14-45.log
    #   5client_efficientnet_b0_50epochs_alpha0_11-02-2026_12-16.log
    # Use a shared run_id (broadcast from rank 0) so all ranks write to the same file.
    client_number = args["client_number"]
    model_name = str(args["model"] or "model").lower()
    epochs = args["epochs"]
    variant_name = str(args["variants_type"] or "default").lower()

    # Canonicalize model names for folder grouping.
    def _normalize_model_folder(name: str) -> str:
        n = (name or "").strip().lower()
        alias_map = {
            "mobilenetv3small": "densenet121",
            "mobilenet_v3_small": "densenet121",
            "densenet_121": "densenet121",
            "efficientnetb0": "efficientnet_b0",
            "bilstm": "bigru",
            "bi_lstm": "bigru",
            "bigru": "bigru",
            "bi_gru": "bigru",
        }
        if n in alias_map:
            return alias_map[n]
        return n or "model"

    normalized_model_name = _normalize_model_folder(model_name)
    model_name_tag = normalized_model_name
    group_dir_name = Path(variant_name) / normalized_model_name
    try:
        partition_method_raw = args["partition_method"]
    except Exception:
        partition_method_raw = None
    partition_method = str(partition_method_raw or "homo").lower()

    try:
        partition_alpha = args["partition_alpha"]
    except Exception:
        partition_alpha = None

    if partition_method == "hetero":
        if partition_alpha is None:
            partition_tag = "hetero"
        else:
            partition_tag = f"hetero_a{float(partition_alpha):g}"
    elif partition_method in ("alpha0", "a0"):
        partition_tag = "alpha0"
    else:
        partition_tag = partition_method

    # Store logs under results/logs/<algorithm>/<model>/...
    # New layout:
    #   results/logs/<algorithm>/<model>/baseline/
    #   results/logs/<algorithm>/<model>/reduce_comm_cost/quantization/<technique>/<direction>/<variant>/
    #   results/logs/<algorithm>/<model>/reduce_comm_cost/sparsity/
    #   results/logs/<algorithm>/<model>/reduce_comm_cost/dimensionality_reduction/
    quantize_forward = bool(args["quantize_forward"]) if args["quantize_forward"] is not None else bool(args["quantize_activations"]) if args["quantize_activations"] is not None else True
    quantize_backward = bool(args["quantize_backward"]) if args["quantize_backward"] is not None else False

    group_root = PROJECT_ROOT / "results" / "logs" / group_dir_name
    baseline_root = group_root / "baseline"
    reduce_root = group_root / "reduce_comm_cost"
    # Create the expected top-level directories (safe to call on every rank).
    os.makedirs(baseline_root, exist_ok=True)
    os.makedirs(reduce_root / "quantization", exist_ok=True)
    os.makedirs(reduce_root / "sparsity", exist_ok=True)
    os.makedirs(reduce_root / "dimensionality_reduction", exist_ok=True)

    if quantize_forward or quantize_backward:
        fwd_kind = str(args["forward_quantization"] or "").strip().lower()
        bwd_kind = str(args["backward_quantization"] or "").strip().lower()
        try:
            quantization_bits = int(args["quantization_bits"] or 8)
        except Exception:
            quantization_bits = 8
        if quantization_bits not in (4, 8):
            quantization_bits = 8
        try:
            dynamic_quantization = bool(args["dynamic_quantization"])
        except Exception:
            dynamic_quantization = False
        try:
            dynamic_quantization_mode = str(args["dynamic_quantization_mode"] or "baseline").strip().lower().replace("_", "-").replace(" ", "-")
        except Exception:
            dynamic_quantization_mode = "baseline"
        if dynamic_quantization_mode in {"dynamic-baseline", "baseline", ""}:
            dynamic_quantization_mode = "baseline"
        elif dynamic_quantization_mode in {"dynamic-seperate", "dynamic-separate", "seperate", "separate"}:
            dynamic_quantization_mode = "seperate"
        elif dynamic_quantization_mode in {"dynamic-test-1", "test-1", "test1"}:
            dynamic_quantization_mode = "test-1"
        elif dynamic_quantization_mode in {"dynamic-test-2", "test-2", "test2"}:
            dynamic_quantization_mode = "test-2"
        elif dynamic_quantization_mode in {"dynamic-test-3", "test-3", "test3"}:
            dynamic_quantization_mode = "test-3"
        elif dynamic_quantization_mode in {"dynamic-test-4", "test-4", "test4"}:
            dynamic_quantization_mode = "test-4"
        elif dynamic_quantization_mode in {"dynamic-test-5", "test-5", "test5"}:
            dynamic_quantization_mode = "test-5"
        else:
            dynamic_quantization_mode = "baseline"

        # Map the configured codec kind to the on-disk folder hierarchy.
        # Arithmetic-conversion codecs live under quantization/<bits>bit/arithmetic_conversion/<format>/...
        # while true truncation codecs live under quantization/<bits>bit/truncation/<format>/...
        def _tech_path_from_kind(kind: str) -> tuple[str, ...]:
            if kind in ("fp8_e4m3", "float8_e4m3", "e4m3", "float8"):
                return ("arithmetic_conversion", "fp4" if quantization_bits == 4 else "fp8")
            if kind in (
                "dynamic_symmetric_int8",
                "dynamic_int8",
                "int8",
                "dynamic_symmetric_int8_per_channel",
                "dynamic_int8_per_channel",
                "int8_per_channel",
                "per_channel_int8",
                "fixed_scale_int8",
                "fixed_int8",
            ):
                if kind in ("dynamic_symmetric_int8_per_channel", "dynamic_int8_per_channel", "int8_per_channel", "per_channel_int8"):
                    return ("arithmetic_conversion", "int4_per_channel" if quantization_bits == 4 else "int8_per_channel")
                return ("arithmetic_conversion", "int4" if quantization_bits == 4 else "int8")
            if kind in ("uniform_codebook_uint8", "uniform_codeword_uint8", "codeword_uniform_uint8", "codeword_uniform", "uniform"):
                return ("codeword", "uniform")
            if kind in ("uniform_per_channel_codebook_uint8", "uniform_per_channel_uint8", "codeword_uniform_per_channel_uint8", "codeword_uniform_per_channel", "uniform_per_channel"):
                return ("codeword", "uniform_per_channel")
            if kind in ("non_uniform_loyd_codebook_uint8", "non_uniform_loyd_uint8", "codeword_non_uniform_loyd", "non_uniform_loyd"):
                return ("codeword", "non_uniform", "loyd")
            if kind in ("non_uniform_loyd_per_channel_codebook_uint8", "non_uniform_loyd_per_channel_uint8", "codeword_non_uniform_loyd_per_channel", "non_uniform_loyd_per_channel", "loyd_per_channel"):
                return ("codeword", "non_uniform", "loyd_per_channel")
            if kind in ("mulaw_codebook_uint8", "mulaw_non_uniform_uint8", "codeword_non_uniform", "codeword_mulaw", "non_uniform_mlaw"):
                return ("codeword", "non_uniform", "mu_law")
            if kind in ("mulaw_per_channel_codebook_uint8", "mulaw_per_channel_uint8", "codeword_mulaw_per_channel", "non_uniform_mlaw_per_channel", "mu_law_per_channel", "mlaw_per_channel"):
                return ("codeword", "non_uniform", "mu_law_per_channel")
            if kind in ("lloyd_max_codebook_uint8", "lloyd_max_uint8", "codeword_lloyd_max", "non_uniform_lloyd_uint8", "lloyd_max"):
                return ("codeword", "non_uniform", "lloyd_max")
            if kind in ("trunc_noscale_int8", "trunc_bits_int8", "trunc_scale_int8", "truncation_int8"):
                return ("truncation", "int4" if quantization_bits == 4 else "int8")
            if not kind:
                return ("unknown",)
            return (kind,)

        technique_parts = _tech_path_from_kind(fwd_kind if quantize_forward else bwd_kind)
        if quantize_forward and quantize_backward:
            direction = "forward_backward"
        elif quantize_forward:
            direction = "forward"
        else:
            direction = "backward"

        if dynamic_quantization:
            if dynamic_quantization_mode == "baseline":
                bit_folder = "dynamic-baseline"
            elif dynamic_quantization_mode == "seperate":
                bit_folder = "dynamic-seperate"
            else:
                bit_folder = f"dynamic-{dynamic_quantization_mode}"
        else:
            bit_folder = f"{quantization_bits}bit"
        results_dir = reduce_root / "quantization" / bit_folder
        for part in technique_parts:
            results_dir = results_dir / part
        results_dir = results_dir / direction / variant_name
    else:
        results_dir = baseline_root
    os.makedirs(results_dir, exist_ok=True)

    if process_id == 0:
        base_run_id = datetime.now().strftime("%d-%m-%Y_%H-%M")
        run_id = base_run_id
        suffix = 1
        while (results_dir / f"{client_number}client_{model_name_tag}_{epochs}epochs_{partition_tag}_{run_id}.log").exists():
            run_id = f"{base_run_id}_{suffix}"
            suffix += 1
    else:
        run_id = None
    run_id = comm.bcast(run_id, root=0)

    args["log_save_path"] = str(
        results_dir / f"{client_number}client_{model_name_tag}_{epochs}epochs_{partition_tag}_{run_id}.log"
    )
    
    runtime_device = _configure_process_runtime(args, process_id, worker_number)

    # 3. Build dataset controller first so text models can populate
    # vocabulary/tokenizer metadata before the model is instantiated.
    dataset = datasetFactory(args).factory()
    if hasattr(dataset, "prepare_model_metadata"):
        dataset.prepare_model_metadata()

    # 4. Create model (client & server parts)
    unsupported_variants = {"ushaped", "parallel_u_shape", "taskagnostic", "taskagnostic2"}
    if variant_name_raw in unsupported_variants:
        raise ValueError(
            f"variants_type={args['variants_type']} still depends on retired local multi-part models and is no longer supported."
        )

    client_model, server_model = model_factory(args).create()
    args["client_model"] = client_model
    args["server_model"] = server_model
    
    # 5. Set up device (GPU/CPU)
    device = runtime_device
    args["device"] = device

    # Place only the model halves used by the current rank onto its assigned device.
    if process_id == args["server_rank"]:
        args["server_model"] = _move_model_obj_to_device(args["server_model"], device)
    elif process_id == args.as_dict().get("fed_server_rank"):
        pass
    else:
        args["client_model"] = _move_model_obj_to_device(args["client_model"], device)
        if "client_model_2" in args.as_dict():
            args["client_model_2"] = _move_model_obj_to_device(args["client_model_2"], device)
    
    # 6. Load and partition dataset.
    # SplitFed/SplitFed2 use an extra FedServer aggregation rank that should not
    # consume a client partition or build local dataloaders.
    fed_server_rank = args.as_dict().get("fed_server_rank")
    if process_id == fed_server_rank:
        train_data_num = 0
        train_data_global = None
        test_data_global = None
        local_data_num = 0
        train_data_local = None
        test_data_local = None
        class_num = 0
    else:
        train_data_num, train_data_global, test_data_global, \
            local_data_num, train_data_local, test_data_local, \
            class_num = dataset.load_partition_data(process_id)
    
    # 7. Initialize logging
    log = Log("main", args)
    log.info(f"Process {process_id} initialized with device: {device}")
    
    try:
        # 8. Run the split learning experiment
        SplitNN_distributed(process_id, args)
        run_success = True
    except Exception as exc:
        run_error_message = str(exc)
        raise
    finally:
        if comm is not None:
            try:
                comm.Barrier()
            except Exception:
                pass
        if process_id == 0 and mlflow_enabled(args):
            try:
                exported = export_run_to_mlflow(
                    args=args,
                    project_root=PROJECT_ROOT,
                    success=run_success,
                    error_message=run_error_message,
                )
                if not exported:
                    logging.warning(
                        "MLflow tracking is enabled but mlflow is not installed. Install it with 'pip install mlflow'."
                    )
            except Exception as exc:
                logging.warning("MLflow export failed for %s: %s", args["log_save_path"], exc)


if __name__ == '__main__':
    main()

