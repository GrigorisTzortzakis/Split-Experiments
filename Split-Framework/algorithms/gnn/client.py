"""Client-side model/trainer for the `gnn` algorithm variant."""

import torch.optim as optim
import logging
from runtime.exports.log import Log


class SplitNNClient():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.rank = args["rank"]
        self.log = Log(self.__class__.__name__, args)
        self.MAX_RANK = args["max_rank"]
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 1 if self.rank == self.MAX_RANK else self.rank + 1
        self.epoch_count = 0
        self.adj = args["adj"]
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=args["lr"],
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0,
                                    amsgrad=False)

        self.device = args["device"]

    def forward_pass(self):
        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        # adj, x = inputs
        self.acts = self.model(x, self.adj)
        logging.info("{} forward_pass".format(self.rank))
        return self.acts, labels

    def backward_pass(self, grads):
        self.acts.backward(grads)

        self.optimizer.step()

    """
    If the model has dropout or batch norm layers, switch the model mode appropriately.
    """

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.model.eval()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.model.train()

    def print_com_size(self, com_manager):
        send_by_cat = dict(getattr(com_manager, "total_send_size_by_category", {}))
        recv_by_cat = dict(getattr(com_manager, "total_receive_size_by_category", {}))
        send_by_type = dict(getattr(com_manager, "total_send_size_by_type", {}))
        recv_by_type = dict(getattr(com_manager, "total_receive_size_by_type", {}))
        self.log.info(
            "worker_num={} epoch_send={} epoch_receive={} total_send={} total_receive={} send_by_category={} recv_by_category={} send_by_type={} recv_by_type={}".format(
                self.rank,
                com_manager.send_thread.tmp_send_size,
                com_manager.receive_thread.tmp_receive_size,
                com_manager.send_thread.total_send_size,
                com_manager.receive_thread.total_receive_size,
                send_by_cat,
                recv_by_cat,
                send_by_type,
                recv_by_type,
            )
        )

