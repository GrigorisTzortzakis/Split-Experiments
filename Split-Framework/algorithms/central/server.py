"""Server-side trainer for the `central` algorithm variant."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from runtime.exports.log import Log


class SplitNNServer:
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.rank = args["rank"]
        self.model = args["server_model"]
        self.trainloader = args["train_data_global"]
        self.testloader = args["test_data_global"]
        self.device = args["device"]
        self.MAX_EPOCH_PER_NODE = int(args["epochs"])
        self.log_step = int(args["log_step"] or 50)

        if self.trainloader is None or self.testloader is None:
            raise ValueError("central requires global train/test loaders on rank 0")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(args["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_count = 0
        self.phase = "train"
        self.reset_local_params()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.loss_sum = 0.0
        self.val_loss = 0.0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.phase = "train"
        self.model.train()
        self.dataloader = self.trainloader
        self.reset_local_params()

    def eval_mode(self):
        self.phase = "validation"
        self.model.eval()
        self.dataloader = self.testloader
        self.reset_local_params()

    def run_batch(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if self.phase == "train":
            self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)

        if self.phase == "train":
            loss.backward()
            self.optimizer.step()

        _, predictions = logits.max(1)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        self.loss_sum += float(loss.item())
        self.batch_idx += 1
        self.step += 1
        self.val_loss = self.loss_sum / max(1, self.step)

    def write_log(self):
        if self.total == 0:
            return
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

