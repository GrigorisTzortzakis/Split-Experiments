"""Server-side model/trainer for the `parallel` algorithm variant."""

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
        self.gradients_dict = dict()
        self.gradients_number = 0

        self.client_number = args["client_number"]

        self.validation_sign_number = 0

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # Log every N steps.
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
        # self.log.info(self.acts.grad.shape)
        return transfer_paper_top_k_cache_id(self.acts, self.acts.grad)

    def process_client_batch(self, client_id, acts, labels, client_phase):
        self.load_client_state(client_id, client_phase)
        self.forward_pass(acts, labels)
        grads = None
        if client_phase == "train":
            grads = self.backward_pass()
            self.save_client_state(client_id)

        return grads, (self.total, self.correct, self.val_loss)

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

    def process_client_batches(self, client_batches):
        if not client_batches:
            return {}

        client_phase = client_batches[0][3]
        if client_phase == "train":
            self.train_mode()
        else:
            self.eval_mode()

        prepared_acts = []
        labels_list = []
        batch_sizes = []

        self.optimizer.zero_grad()
        for _client_id, acts, labels, _phase in client_batches:
            if isinstance(acts, torch.Tensor) and not acts.requires_grad:
                acts = transfer_paper_top_k_cache_id(acts, acts.detach().requires_grad_(True))
            prepared_acts.append(acts)
            labels_list.append(labels)
            batch_sizes.append(labels.size(0))

        combined_acts = torch.cat(prepared_acts, dim=0)
        combined_acts.retain_grad()
        combined_labels = torch.cat(labels_list, dim=0)
        combined_logits = self.model(combined_acts)
        combined_loss = self.criterion(combined_logits, combined_labels)

        combined_grads = None
        if client_phase == "train":
            combined_loss.backward()
            self.optimizer.step()
            combined_grads = combined_acts.grad

        results = {}
        start_idx = 0
        for (client_id, _acts, labels, _phase), batch_size in zip(client_batches, batch_sizes):
            end_idx = start_idx + batch_size
            local_logits = combined_logits[start_idx:end_idx]
            _, predictions = local_logits.max(1)
            local_total = labels.size(0)
            local_correct = predictions.eq(labels).sum().item()
            local_loss = self.criterion(local_logits, labels).item()
            local_grads = None if combined_grads is None else combined_grads[start_idx:end_idx]
            local_grads = transfer_paper_top_k_cache_id(prepared_acts[len(results)], local_grads)
            results[client_id] = (local_grads, (local_total, local_correct, local_loss))
            start_idx = end_idx

        return results

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

