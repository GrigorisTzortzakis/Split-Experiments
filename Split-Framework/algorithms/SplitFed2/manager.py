"""Managers (client + servers) and message types for `SplitFed2`.

Per your requirements:
- All MPI manager code lives in this single file.
- The split-learning collector runs as the "main server" (rank 0).
- The FedServer (aggregation) runs as a separate MPI process (rank = args['fed_server_rank']).
"""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_MODEL = 8

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_SEND_MODEL = 7

    # client to client
    MSG_TYPE_C2C_SEMAPHORE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_AGR_KEY_RESULT = "result"

    MSG_TYPE_TEST_C2C = 9

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"

    MSG_ARG_KEY_MODEL = "model"
    MSG_AGR_KEY_SAMPLE_NUM = "sample_num"

# --- Server managers ------------------------------------------------
import logging

from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log


class MainServerManager(MessageManager):
    """Collector/main server for SplitFed2."""

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"], args["worker_number"], backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.finished_nodes = 0
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

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
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.handle_message_acts)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.handle_message_finish_protocol
        )

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)

        if client_phase == "train":
            self.trainer.train_mode()
        else:
            self.trainer.eval_mode()

        self.trainer.forward_pass(acts, labels)

        grads = None
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()

        self.send_grads_to_client(sender, grads)

    def handle_message_finish_protocol(self, msg_params=None):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()


class FedServerManager(MessageManager):
    """FedServer aggregation process for SplitFed2."""

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"], args["worker_number"], backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.finished_nodes = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.handle_message_model_param)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.handle_message_finish_protocol
        )

    def handle_message_finish_protocol(self, msg_params=None):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_message_model_param(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        sample_num = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        self.trainer.model_param_dict[sender] = model_param
        self.trainer.sample_num_dict[sender] = sample_num

        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == self.trainer.MAX_RANK:
            self.log.info("get all model params ---- from rank {}".format(self.trainer.rank))
            self.trainer.model_param_num = 0
            ordered_client_ids = sorted(self.trainer.model_param_dict.keys())
            model_avg = self.trainer.model_param_dict[ordered_client_ids[0]]
            client_weights = {
                client_id: float(self.trainer.sample_num_dict.get(client_id, 1.0))
                for client_id in ordered_client_ids
            }
            total_weight = sum(client_weights.values())
            if total_weight <= 0:
                client_weights = {client_id: 1.0 for client_id in ordered_client_ids}
                total_weight = float(len(ordered_client_ids))
            for key in model_avg.keys():
                for index, client_id in enumerate(ordered_client_ids):
                    local_model_params = self.trainer.model_param_dict[client_id]
                    weight = client_weights[client_id] / total_weight
                    if index == 0:
                        model_avg[key] = local_model_params[key].clone() * weight
                    else:
                        model_avg[key] += local_model_params[key] * weight
            self.trainer.model_param_dict = dict()
            self.trainer.sample_num_dict = dict()
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_model_param_to_client(idx, model_avg)

    def send_model_param_to_client(self, receive_id, model_avg_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, model_avg_param)
        self.send_message(message)


class FedServer:
    """Minimal trainer/state holder for the FedServer process."""

    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.rank = args["rank"]

        self.MAX_RANK = args["max_rank"]

        self.model_param_dict = dict()
        self.sample_num_dict = dict()
        self.model_param_num = 0


# Backwards-compatible export: existing code expects `ServerManager`.
ServerManager = MainServerManager
# --- Client manager ------------------------------------------------
import logging
import torch
import time
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log
from .client import SplitNNClient


class ClientManager(MessageManager):
    """
    args must include MPI comm, rank, and max_rank (comm.size() - 1). Other fields are not required here.
    trainer is an instance of SplitNNClient.
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["worker_number"], backend)
        # self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        fed_server_rank = args["fed_server_rank"]
        self.fed_server_rank = 0 if fed_server_rank is None else fed_server_rank
        self.log_comm_stats = bool(args.get("log_comm_stats", False)) if isinstance(args, dict) else False

    def run(self):
        self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        if self.log_comm_stats:
            self.trainer.print_com_size(self.com_manager)
        for _ in range(len(self.trainer.testloader)):
            self.run_forward_pass()
            msg_params = self.com_manager.wait_for_message()
            self.com_manager.notify(msg_params)
        self.trainer.write_log()

        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.send_finish_to_server(self.fed_server_rank)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MODEL,
                                              self.handle_message_model_param_from_server)

    def handle_message_gradients(self, msg_params):
        tot, cor, vl = msg_params.get(MyMessage.MSG_AGR_KEY_RESULT)
        self.trainer.total += tot
        self.trainer.correct += cor
        self.trainer.val_loss += vl
        self.trainer.step += 1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(grads)

            # if self.trainer.batch_idx % 10 == 0 and self.trainer.batch_idx != len(self.trainer.trainloader):
            #     self.send_model_param_to_fed_server(0)
            #
            #     while True:
            #         if self.com_manager.q_receiver.qsize() > 0:
            #             msg_params = self.com_manager.q_receiver.get()
            #             # logging.info(msg_params)
            #
            #             self.com_manager.notify(msg_params)
            #             break
            #         else:
            #             time.sleep(0.5)
            #     self.run_forward_pass()

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                # torch.save(self.trainer.model, self.args["model_tmp_path"])
                self.send_model_param_to_fed_server(self.fed_server_rank)


                # while True:
                #     if self.com_manager.q_receiver.qsize() > 0:
                #         msg_params = self.com_manager.q_receiver.get()
                #         logging.info(msg_params)
                #
                #         self.com_manager.notify(msg_params)
                #         break
                #     else:
                #         time.sleep(0.5)
                #
                # self.run_eval()
            else:
                self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        self.annotate_tensor_distribution_message(message, self.trainer)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        # self.log.info(model_param["block1.0.weight"])
        self.trainer.model.load_state_dict(model_param)
        self.run_eval()

    def send_model_param_to_fed_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.model.state_dict())
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        self.send_message(message)

