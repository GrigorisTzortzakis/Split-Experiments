"""MPI split-learning entrypoints.

What this file is for:
- `SplitNN_init`: initialize MPI, set ranks/sizes into the config.
- `SplitNN_distributed`: start server role (rank 0) or client role (others).

This file is MPI-specific.
"""

from mpi4py import MPI

from algorithms.algorithm_selector import AlgorithmSelector
from runtime.exports.log import Log


def SplitNN_init(parse):
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    configured_max_rank = parse["max_rank"]
    configured_partition_client_number = parse["partition_client_number"]
    parse["comm"] = comm
    parse["process_id"] = process_id
    parse["rank"] = process_id
    parse["worker_number"] = worker_number
    # Default topology: 1 server (rank 0) + N clients (ranks 1..N)
    parse["client_number"] = worker_number - 1
    parse["max_rank"] = parse["worker_number"] - 1

    if parse["variants_type"] in {"central", "Central"}:
        if worker_number != 1:
            raise ValueError("central requires a single process. Use python setup/main.py or mpiexec -np 1.")
        requested_partition_count = configured_max_rank
        if requested_partition_count is None:
            requested_partition_count = configured_partition_client_number
        if requested_partition_count is None:
            requested_partition_count = 1
        requested_partition_count = max(1, int(requested_partition_count))
        parse["central_partition_count"] = requested_partition_count
        parse["client_number"] = requested_partition_count
        parse["max_rank"] = 0
        return comm, process_id, worker_number

    # SplitFed / SplitFed2 topology: 1 collector server (rank 0) + N clients (ranks 1..N)
    # + 1 dedicated FedServer (rank N+1). This keeps client ranks stable.
    if parse["variants_type"] in {"SplitFed", "SplitFed2"}:
        parse["client_number"] = worker_number - 2
        parse["max_rank"] = worker_number - 2
        parse["fed_server_rank"] = parse["max_rank"] + 1

    if parse["variants_type"] == "TaskAgnostic":
        dataset_cur, cur_client_num = judge_client_dataset(parse)
        parse["dataset_cur"] = dataset_cur
        parse["cur_client_num"] = cur_client_num

    return comm, process_id, worker_number


def SplitNN_distributed(process_id, parse):
    logging = Log("SplitNN_distributed", parse)
    server_rank = 0
    if parse["variants_type"] in {"central", "Central"}:
        logging.info("process_id == server_rank : {}".format(process_id))
        init_server(parse)
        return

    fed_server_rank = parse["fed_server_rank"] if "fed_server_rank" in parse.as_dict() else None
    if process_id == server_rank:
        logging.info("process_id == server_rank : {}".format(process_id))
        init_server(parse)
    elif fed_server_rank is not None and process_id == fed_server_rank:
        logging.info("process_id == fed_server_rank : {}".format(process_id))
        init_fed_server(parse)
    else:
        logging.info("process_id == client_rank : {}".format(process_id))
        init_client(parse)


def init_server(args):
    logging = Log("init_server", args)
    server, server_manager = AlgorithmSelector(args["variants_type"], "server", args).factory()
    logging.info("Server run begin {}".format(args["variants_type"]))
    server_manager.run()
    logging.info("Server run end")


def init_client(args):
    logging = Log("init_client", args)
    client, client_manager = AlgorithmSelector(args["variants_type"], "client", args).factory()
    logging.info("Client {} run begin".format(args["rank"]))
    client_manager.run()
    logging.info("Client {} run end".format(args["rank"]))


def init_fed_server(args):
    logging = Log("init_fed_server", args)
    fed_server, fed_server_manager = AlgorithmSelector(args["variants_type"], "fed_server", args).factory()
    logging.info("FedServer run begin {}".format(args["variants_type"]))
    fed_server_manager.run()
    logging.info("FedServer run end")


def judge_client_dataset(parse):
    """Task-agnostic multi-dataset helper.

    Returns the dataset index and how many clients belong to that dataset.
    """

    client_num = parse["client_split"][0]
    for i in range(len(parse["client_split"])):
        if parse["rank"] <= parse["client_split"][i]:
            return i, client_num
        client_num = parse["client_split"][i + 1] - parse["client_split"][i]

