"""Server-side model/trainer for the `SplitFed2` algorithm variant."""

import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.extend("../../../")

from runtime.exports.log import Log


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.rank = args["rank"]

        self.model_param_dict = dict()
        self.client_sample_dict = dict()
        self.acts_dict = dict()

        self.acts_num = 0
        self.model_param_num = 0
        self.sum_sample_number = 0

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # Log every N steps.
        self.train_mode()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_mode(self):
        self.model.train()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"

    def forward_pass(self, acts, labels):
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()


    def backward_pass(self):
        self.loss.backward()
        self.optimizer.step()
        # self.log.info(self.acts.grad.shape)
        return self.acts.grad

