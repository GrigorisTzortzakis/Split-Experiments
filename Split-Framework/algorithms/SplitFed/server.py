"""Server-side model/trainer for the `SplitFed` algorithm variant."""

import copy
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.extend("../../../")

from compression.comparison_papers.paper_top_k import transfer_paper_top_k_cache_id
from runtime.exports.log import Log


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.server_model_dict = dict()
        self.server_optimizer_state_dict = dict()
        self.client_sample_num_dict = dict()
        self.epoch = 0
        self.phase = "train"
        self.optimizer = self._build_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.reset_local_params()

    def _build_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args["lr"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0

    def train_mode(self):
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def ensure_client_state(self, client_id):
        if client_id not in self.server_model_dict:
            self.server_model_dict[client_id] = copy.deepcopy(self.model.state_dict())
        if client_id not in self.server_optimizer_state_dict:
            self.server_optimizer_state_dict[client_id] = copy.deepcopy(self.optimizer.state_dict())

    def load_client_state(self, client_id, phase):
        self.ensure_client_state(client_id)
        self.model.load_state_dict(copy.deepcopy(self.server_model_dict[client_id]))
        if phase == "train":
            self.train_mode()
            self.optimizer = self._build_optimizer()
            optimizer_state = self.server_optimizer_state_dict.get(client_id)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        else:
            self.eval_mode()

    def save_client_state(self, client_id):
        self.server_model_dict[client_id] = copy.deepcopy(self.model.state_dict())
        self.server_optimizer_state_dict[client_id] = copy.deepcopy(self.optimizer.state_dict())

    def forward_pass(self, acts, labels):
        self.acts = transfer_paper_top_k_cache_id(acts, acts.detach().clone().requires_grad_(True))
        self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(self.acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        return transfer_paper_top_k_cache_id(self.acts, self.acts.grad)

    def federate_server_models(self):
        if not self.server_model_dict:
            return

        ordered_client_ids = sorted(self.server_model_dict.keys())
        averaged_state = copy.deepcopy(self.server_model_dict[ordered_client_ids[0]])
        client_weights = {
            client_id: float(self.client_sample_num_dict.get(client_id, 1.0))
            for client_id in ordered_client_ids
        }
        total_weight = sum(client_weights.values())
        if total_weight <= 0:
            client_weights = {client_id: 1.0 for client_id in ordered_client_ids}
            total_weight = float(len(ordered_client_ids))

        for key in averaged_state.keys():
            for index, client_id in enumerate(ordered_client_ids):
                local_state = self.server_model_dict[client_id]
                weight = client_weights[client_id] / total_weight
                if index == 0:
                    averaged_state[key] = local_state[key].clone() * weight
                else:
                    averaged_state[key] += local_state[key] * weight

        self.model.load_state_dict(averaged_state)
        self.server_model_dict = {
            client_id: copy.deepcopy(averaged_state) for client_id in ordered_client_ids
        }
        self.server_optimizer_state_dict = {client_id: None for client_id in ordered_client_ids}

