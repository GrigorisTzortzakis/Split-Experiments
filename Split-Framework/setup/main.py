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

from runtime.log import Log
from datasets.Dataset_Picker import datasetFactory
from runtime.config import yaml_config
from models.model_factory import model_factory
from runtime.MPI.start_MPI import SplitNN_distributed, SplitNN_init


def init_training_device(process_id, num_workers, gpu_num_per_machine):
    """
    Initialize the GPU/CPU device for this process.
    Server (process 0) gets GPU 0, clients are distributed across available GPUs.
    """
    if process_id == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    
    # Map client processes to GPUs
    gpu_index = (process_id - 1) % gpu_num_per_machine
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--split-layer", dest="split_layer", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="lr", type=float, default=None)
    parser.add_argument("--momentum", dest="momentum", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--log-step", dest="log_step", type=int, default=None)
    parser.add_argument("--seed", dest="seed", type=int, default=None)
    parser.add_argument("--max-rank", dest="max_rank", type=int, default=None)
    # Dataset partitioning can be decoupled from active MPI client count.
    # Example: partition_client_number=10 but run with 1/5/10 active clients to use 10%/50%/100% of data.
    parser.add_argument(
        "--partition-client-number",
        dest="partition_client_number",
        type=int,
        default=None,
        help="Total number of client data partitions (slices) to create, independent of MPI client processes.",
    )
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
    if cli_args.split_layer is not None:
        args["split_layer"] = cli_args.split_layer
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
    
    # 2. Initialize MPI communication
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id

    # Keep config consistent with the MPI world size.
    # Many algorithms (e.g., semaphore ring / finish broadcast) rely on max_rank being exact.
    # If `max_rank` is larger than `mpiexec -np - 1`, ranks will send to non-existent processes
    # and can deadlock or crash.
    actual_max_rank = int(worker_number) - 1
    configured_max_rank = args["max_rank"]
    if configured_max_rank is None:
        args["max_rank"] = actual_max_rank
    elif int(configured_max_rank) != actual_max_rank:
        logging.warning(
            "Overriding config max_rank=%s to match MPI world size (max_rank=%s).",
            configured_max_rank,
            actual_max_rank,
        )
        args["max_rank"] = actual_max_rank

    # Keep the number of *active* clients consistent with MPI.
    # Dataset partitioning can optionally use a different number via `partition_client_number`.
    args["client_number"] = args["max_rank"]

    # Safety: cannot have more active clients than partitions.
    partition_client_number = args["partition_client_number"]
    if partition_client_number is not None:
        partition_client_number = int(partition_client_number)
        if partition_client_number <= 0:
            raise ValueError("partition_client_number must be a positive integer")
        if int(args["client_number"]) > partition_client_number:
            raise ValueError(
                f"Active clients (client_number={int(args['client_number'])}) exceed partition_client_number={partition_client_number}."
            )

    # 2b. Build a per-run log filename using current settings.
    # Examples:
    #   5client_lenet_50epochs_homo_09-02-2026_14-02.log
    #   10client_lenet_50epochs_hetero_a0.1_11-02-2026_14-45.log
    #   5client_lenet_50epochs_alpha0_11-02-2026_12-16.log
    # Use a shared run_id (broadcast from rank 0) so all ranks write to the same file.
    client_number = args["max_rank"]
    model_name = str(args["model"] or "model").lower()
    epochs = args["epochs"]
    variant_name = str(args["variants_type"] or "default").lower()
    try:
        partition_method_raw = args["partition_method"]
    except Exception:
        partition_method_raw = None
    partition_method = str(partition_method_raw or "homo").lower()

    try:
        partition_alpha = args["partition_alpha"]
    except Exception:
        partition_alpha = None

    try:
        partition_client_number = args["partition_client_number"]
    except Exception:
        partition_client_number = None

    if partition_method == "hetero":
        if partition_alpha is None:
            partition_tag = "hetero"
        else:
            partition_tag = f"hetero_a{float(partition_alpha):g}"
    elif partition_method in ("alpha0", "a0"):
        partition_tag = "alpha0"
    else:
        partition_tag = partition_method

    # If partitions are decoupled from active MPI clients, encode it in the log name.
    if partition_client_number is not None and int(partition_client_number) != int(client_number):
        partition_tag = f"{partition_tag}_pc{int(partition_client_number)}"

    # Store logs under results/logs/<variant>-<model>/ so different variants/models don't mix.
    # Example folder: results/logs/vanilla-lenet/
    variant_model_dir = f"{variant_name}-{model_name}"
    results_dir = PROJECT_ROOT / "results" / "logs" / variant_model_dir
    os.makedirs(results_dir, exist_ok=True)

    if process_id == 0:
        base_run_id = datetime.now().strftime("%d-%m-%Y_%H-%M")
        run_id = base_run_id
        suffix = 1
        while (results_dir / f"{client_number}client_{model_name}_{epochs}epochs_{partition_tag}_{run_id}.log").exists():
            run_id = f"{base_run_id}_{suffix}"
            suffix += 1
    else:
        run_id = None
    run_id = comm.bcast(run_id, root=0)

    args["log_save_path"] = str(
        results_dir / f"{client_number}client_{model_name}_{epochs}epochs_{partition_tag}_{run_id}.log"
    )
    
    # 3. Create model (client & server parts)
    variant_name_raw = str(args["variants_type"] or "").lower()
    if variant_name_raw == "ushaped":
        from models.lenet5_Ushape import (
            LeNetClientNetworkPart1,
            LeNetServerNetwork_U,
            LeNetClientNetworkPart2,
        )

        args["client_model"] = LeNetClientNetworkPart1()
        args["server_model"] = LeNetServerNetwork_U()
        args["client_model_2"] = LeNetClientNetworkPart2()
    else:
        client_model, server_model = model_factory(args).create()
        args["client_model"] = client_model
        args["server_model"] = server_model
    
    # 4. Set up device (GPU/CPU)
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = device
    
    # 5. Load and partition dataset
    dataset = datasetFactory(args).factory()
    train_data_num, train_data_global, test_data_global, \
        local_data_num, train_data_local, test_data_local, \
        class_num = dataset.load_partition_data(process_id)
    
    # 6. Initialize logging
    log = Log("main", args)
    log.info(f"Process {process_id} initialized with device: {device}")
    
    # 7. Run the split learning experiment
    SplitNN_distributed(process_id, args)


if __name__ == '__main__':
    main()
