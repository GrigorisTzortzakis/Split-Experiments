"""Managers (client + server) and message types for `fedprox`."""


class MyMessage(object):
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_MODEL = 8

    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_SEND_MODEL = 7

    MSG_TYPE_C2C_SEMAPHORE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_AGR_KEY_RESULT = "result"

    MSG_TYPE_TEST_C2C = 9

    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"

    MSG_ARG_KEY_MODEL = "model"
    MSG_AGR_KEY_SAMPLE_NUM = "sample_num"


from mpi4py import MPI
import logging
import torch
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log


class ServerManager(MessageManager):
    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.active_node = -1
        self.finished_nodes = 0

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(
            MyMessage.MSG_AGR_KEY_RESULT,
            (self.trainer.total, self.trainer.correct, self.trainer.val_loss),
        )
        self.annotate_tensor_distribution_message(message, self.trainer)
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.handle_message_finish_protocol
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.handle_message_model_param
        )

    def handle_message_finish_protocol(self, msg_params=None):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def get_dict_n2_dist(x1: dict, x2: dict):
        pass

    def handle_message_model_param(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        sample_number = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        if self.trainer.last_param is None:
            self.trainer.last_param = model_param
        self.trainer.sum_sample_number += sample_number
        self.trainer.model_param_dict[sender] = (sample_number, model_param)
        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == self.trainer.MAX_RANK:
            self.log.info("get all model params ---- from rank {}".format(self.trainer.rank))

            self.trainer.model_param_num = 0
            model_avg = model_param
            dist = 0
            total_params = 0
            for key in model_avg.keys():
                for idx in range(1, self.trainer.MAX_RANK + 1):
                    local_sample_number, local_model_params = self.trainer.model_param_dict[idx]
                    weight = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        model_avg[key] = local_model_params[key] * weight
                    else:
                        model_avg[key] += local_model_params[key] * weight
                delta = self.trainer.last_param[key] - model_avg[key]
                size = 1
                for dimension in delta.shape:
                    size *= dimension
                total_params += size
                dist += torch.pow(delta, 2).sum(list(range(len(delta.shape))))

            self.log.info("Dist:{d}".format(d=dist / total_params))
            self.trainer.sum_sample_number = 0
            self.trainer.last_param = model_avg
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_model_param_to_fed_client(idx, model_avg)

    def send_model_param_to_fed_client(self, receive_id, model_avg_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, model_avg_param)
        self.send_message(message)


import torch
from .client import SplitNNClient


class ClientManager(MessageManager):
    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)

    def run(self):
        self.register_message_receive_handlers()
        self.run_forward_pass()
        if self._is_finished:
            return
        super(ClientManager, self).run()

    def _wait_for_message(self):
        msg_params = self.com_manager.q_receiver.get()
        self.com_manager.notify(msg_params)

    def run_forward_pass(self):
        while True:
            total, correct, val_loss = self.trainer.forward_pass()
            self.trainer.batch_idx += 1
            self.trainer.total += total
            self.trainer.correct += correct
            self.trainer.val_loss += val_loss
            self.trainer.step += 1
            if self.trainer.phase == "train":
                self.trainer.write_log()
                self.trainer.backward_pass()
                logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))

                if self.trainer.batch_idx == len(self.trainer.trainloader):
                    self.send_model_param_to_fed_server(0)
                    self._wait_for_message()
                    self.run_eval()
                    break
            else:
                if self.trainer.batch_idx == len(self.trainer.dataloader):
                    break

    def run_eval(self):
        self.trainer.eval_mode()
        self.run_forward_pass()
        self.trainer.write_log()
        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_MODEL, self.handle_message_model_param_from_server
        )

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.trainer.model.load_state_dict(model_param)
        self.trainer._capture_reference_params()

    def send_model_param_to_fed_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.model.state_dict())
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        self.send_message(message)
