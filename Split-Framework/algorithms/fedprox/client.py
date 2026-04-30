"""Client-side model/trainer for the `fedprox` algorithm variant."""

import logging

import torch
import torch.optim as optim
from torch import nn

from runtime.exports.log import Log


class SplitNNClient():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.SERVER_RANK = args["server_rank"]

        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = args["device"]
        try:
            self.local_sample_number = len(self.trainloader.dataset)
        except Exception:
            self.local_sample_number = len(self.trainloader)
        self.phase = "train"
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]

        self.log = Log(self.__class__.__name__, args)
        self.log_step = args["log_step"] if args["log_step"] else 50
        self.args = args
        prox_mu = args["fedprox_mu"]
        self.prox_mu = 0.01 if prox_mu is None else float(prox_mu)
        self.reference_params = []
        self._capture_reference_params()

    def _capture_reference_params(self):
        self.reference_params = [
            parameter.detach().clone().to(parameter.device)
            for parameter in self.model.parameters()
        ]

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def write_log(self):
        if (self.phase == "train" and self.step % self.log_step == 0) or self.phase == "validation":
            self.log.info(
                "phase={} acc={} loss={} epoch={} and step={}".format(
                    self.phase,
                    self.correct / self.total,
                    self.val_loss,
                    self.epoch_count,
                    self.step,
                )
            )

    def _proximal_penalty(self):
        if self.phase != "train" or not self.reference_params:
            return None

        penalty = None
        for parameter, reference in zip(self.model.parameters(), self.reference_params):
            ref = reference.to(parameter.device)
            term = torch.sum((parameter - ref) ** 2)
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return None
        return 0.5 * self.prox_mu * penalty

    def forward_pass(self):
        inputs, labels = next(self.dataloader)

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        proximal_penalty = self._proximal_penalty()
        if proximal_penalty is not None:
            self.loss = self.loss + proximal_penalty
        total = labels.size(0)
        correct = predictions.eq(labels).sum().item()
        val_loss = self.loss.item()

        return total, correct, val_loss

    def backward_pass(self):
        self.loss.backward()
        self.optimizer.step()

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.phase = "validation"
        self.model.eval()
        self.reset_local_params()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.phase = "train"
        self.model.train()
        self.reset_local_params()
