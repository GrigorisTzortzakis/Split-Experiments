"""Algorithm selector.

What this file is for:
- Maps the configured `variants_type` string to the corresponding algorithm implementation.
- Returns the correct (client/server) trainer + manager pair.

It does not run training itself; it only selects and instantiates the right classes.
"""

from __future__ import annotations

import logging

from runtime.exports.log import Log


class AlgorithmSelector:
    def __init__(self, variants_type: str, node_type: str, args):
        self.variants_type = variants_type
        self.node_type = node_type
        self.args = args
        self.log = Log("algorithm_selector", args)

    def factory(self):
        if self.variants_type == "vanilla":
            from algorithms.vanilla.server import SplitNNServer
            from algorithms.vanilla.client import SplitNNClient
            from algorithms.vanilla.manager import ServerManager
            from algorithms.vanilla.manager import ClientManager
        elif self.variants_type == "Ushaped":
            from algorithms.Ushaped.server import SplitNNServer
            from algorithms.Ushaped.client import SplitNNClient
            from algorithms.Ushaped.manager import ServerManager
            from algorithms.Ushaped.manager import ClientManager
        elif self.variants_type == "gnn":
            from algorithms.gnn.server import SplitNNServer
            from algorithms.gnn.client import SplitNNClient
            from algorithms.gnn.manager import ServerManager
            from algorithms.gnn.manager import ClientManager
        elif self.variants_type == "parallel_U_Shape":
            from algorithms.parallel_U_Shape.server import SplitNNServer
            from algorithms.parallel_U_Shape.client import SplitNNClient
            from algorithms.parallel_U_Shape.manager import ServerManager
            from algorithms.parallel_U_Shape.manager import ClientManager
        elif self.variants_type in {"asy_vanilla", "asyVanilla"}:
            from algorithms.asyVanilla.server import SplitNNServer
            from algorithms.asyVanilla.client import SplitNNClient
            from algorithms.asyVanilla.manager import ServerManager
            from algorithms.asyVanilla.manager import ClientManager
        elif self.variants_type in {"asy_vanilla2", "asyVanilla2"}:
            from algorithms.asyVanilla2.server import SplitNNServer
            from algorithms.asyVanilla2.client import SplitNNClient
            from algorithms.asyVanilla2.manager import ServerManager
            from algorithms.asyVanilla2.manager import ClientManager
        elif self.variants_type == "vertical":
            from algorithms.vertical.server import SplitNNServer
            from algorithms.vertical.client import SplitNNClient
            from algorithms.vertical.manager import ServerManager
            from algorithms.vertical.manager import ClientManager
        elif self.variants_type == "Asynchronous":
            from algorithms.Asynchronous.server import SplitNNServer
            from algorithms.Asynchronous.client import SplitNNClient
            from algorithms.Asynchronous.manager import ServerManager
            from algorithms.Asynchronous.manager import ClientManager
        elif self.variants_type == "SGLR":
            from algorithms.SGLR.server import SplitNNServer
            from algorithms.SGLR.client import SplitNNClient
            from algorithms.SGLR.manager import ServerManager
            from algorithms.SGLR.manager import ClientManager
        elif self.variants_type == "SplitFed":
            from algorithms.SplitFed.server import SplitNNServer
            from algorithms.SplitFed.client import SplitNNClient
            from algorithms.SplitFed.manager import MainServerManager as ServerManager
            from algorithms.SplitFed.manager import ClientManager
            from algorithms.SplitFed.manager import FedServer
            from algorithms.SplitFed.manager import FedServerManager
        elif self.variants_type == "SplitFed2":
            from algorithms.SplitFed2.server import SplitNNServer
            from algorithms.SplitFed2.client import SplitNNClient
            from algorithms.SplitFed2.manager import MainServerManager as ServerManager
            from algorithms.SplitFed2.manager import ClientManager
            from algorithms.SplitFed2.manager import FedServer
            from algorithms.SplitFed2.manager import FedServerManager
        elif self.variants_type == "parallel":
            from algorithms.parallel.server import SplitNNServer
            from algorithms.parallel.client import SplitNNClient
            from algorithms.parallel.manager import ServerManager
            from algorithms.parallel.manager import ClientManager
        elif self.variants_type == "TaskAgnostic":
            from algorithms.TaskAgnostic.server import SplitNNServer
            from algorithms.TaskAgnostic.client import SplitNNClient
            from algorithms.TaskAgnostic.manager import ServerManager
            from algorithms.TaskAgnostic.manager import ClientManager
        elif self.variants_type == "TaskAgnostic2":
            from algorithms.TaskAgnostic2.server import SplitNNServer
            from algorithms.TaskAgnostic2.client import SplitNNClient
            from algorithms.TaskAgnostic2.manager import ServerManager
            from algorithms.TaskAgnostic2.manager import ClientManager
        elif self.variants_type == "comp_model":
            from algorithms.comp_model.server import SplitNNServer
            from algorithms.comp_model.client import SplitNNClient
            from algorithms.comp_model.manager import ServerManager
            from algorithms.comp_model.manager import ClientManager
        elif self.variants_type == "fedavg":
            from algorithms.fedavg.server import SplitNNServer
            from algorithms.fedavg.client import SplitNNClient
            from algorithms.fedavg.manager import ServerManager
            from algorithms.fedavg.manager import ClientManager
        elif self.variants_type in {"fedprox", "FedProx"}:
            from algorithms.fedprox.server import SplitNNServer
            from algorithms.fedprox.client import SplitNNClient
            from algorithms.fedprox.manager import ServerManager
            from algorithms.fedprox.manager import ClientManager
        elif self.variants_type in {"central", "Central"}:
            if self.node_type != "server":
                raise ValueError("central only supports the server role")
            from algorithms.central.server import SplitNNServer
            from algorithms.central.manager import ServerManager
        else:
            logging.warning("variants_type: default as vanilla")
            from algorithms.vanilla.server import SplitNNServer
            from algorithms.vanilla.client import SplitNNClient
            from algorithms.vanilla.manager import ServerManager
            from algorithms.vanilla.manager import ClientManager

        if self.node_type == "server":
            server = SplitNNServer(self.args)
            server_manager = ServerManager(self.args, server)
            return server, server_manager

        if self.node_type == "fed_server":
            if self.variants_type not in {"SplitFed", "SplitFed2"}:
                raise ValueError(f"fed_server role not supported for variants_type={self.variants_type}")
            fed_server = FedServer(self.args)
            fed_server_manager = FedServerManager(self.args, fed_server)
            return fed_server, fed_server_manager

        if self.variants_type in {"central", "Central"}:
            raise ValueError("central does not define a client role")

        client = SplitNNClient(self.args)
        client_manager = ClientManager(self.args, client)
        return client, client_manager


# Backwards-compat alias (keeps older code working if still using the old class name)
variantsFactory = AlgorithmSelector

