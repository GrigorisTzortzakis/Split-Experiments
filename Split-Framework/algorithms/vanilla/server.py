"""Server-side model/trainer for the `vanilla` algorithm variant."""

from runtime.exports.log import Log
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time

sys.path.extend("../../../")


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.epoch = 0
        # Log every N steps.
        self.log_step = args["log_step"] if args["log_step"] else 50
        self.active_node = 1
        self.last_train_metrics = None
        self.epoch_start_time = None
        self.train_mode()
        momentum = args["momentum"] if args["momentum"] is not None else 0.0
        weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0
        self.optimizer = optim.SGD(
            self.model.parameters(),
            args["lr"],
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.train_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()
        self.epoch_start_time = time.perf_counter()

    def eval_mode(self):
        if self.phase == "train" and self.step > 0 and self.total > 0:
            self.last_train_metrics = {
                "epoch": self.epoch,
                "train_acc": self.correct / self.total,
                "train_loss": self.train_loss / max(self.step, 1),
            }
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts, labels):
        # When activations are transmitted over MPI, they may arrive detached
        # (e.g., after compression/quantization codecs). Server-side backprop
        # requires grads w.r.t. the received activations.
        if isinstance(acts, torch.Tensor) and not acts.requires_grad:
            acts = acts.detach().requires_grad_(True)

        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        if self.phase == "train":
            self.train_loss += self.loss.item()
        if self.step % self.log_step == 0 and self.phase == "train":
            acc = self.correct / self.total
            self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format("train", acc, self.loss.item(), self.epoch, self.step))

            # Also log metrics (e.g., accuracy) here.
        if self.phase == "validation":
            # self.log.info("phase={} acc={} loss={} epoch={} and step={}"
            #               .format("train", acc, self.loss.item(), self.epoch, self.step))
            self.val_loss += self.loss.item()
            # torch.save(self.model, self.args["model_save_path"].format("server", self.epoch, ""))
        self.step += 1

    def backward_pass(self):
        self.loss.backward()
        self.optimizer.step()
        grads = self.acts.grad
        self.loss = None
        self.acts = None
        return grads

    def validation_over(self):
        # not precise estimation of validation loss
        self.val_loss /= max(self.step, 1)
        acc = self.correct / self.total
        train_metrics = self.last_train_metrics or {
            "epoch": self.epoch,
            "train_acc": 0.0,
            "train_loss": 0.0,
        }
        epoch_time = 0.0
        if self.epoch_start_time is not None:
            epoch_time = time.perf_counter() - self.epoch_start_time
        epoch_summary = {
            "epoch": int(train_metrics.get("epoch", self.epoch)),
            "train_acc": float(train_metrics.get("train_acc", 0.0)),
            "train_loss": float(train_metrics.get("train_loss", 0.0)),
            "val_acc": float(acc),
            "val_loss": float(self.val_loss),
            "epoch_time": float(epoch_time),
        }

        # Also log metrics (e.g., accuracy) here.
        self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                      .format(self.phase, acc, self.val_loss, self.epoch, self.step))
        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()
        return epoch_summary

    # def reset_local_params(self):
    #     self.total = 0
    #     self.correct = 0
    #     self.val_loss = 0
    #     self.step = 0
    #     self.batch_idx = 0

