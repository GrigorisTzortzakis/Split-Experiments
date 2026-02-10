"""Managers (client + server) and message types for `asyVanilla`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_START_VALIDATION = 8

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_VALIDATION = 7

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

# --- Server manager ------------------------------------------------
from mpi4py import MPI
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.log import Log


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.active_node = -1
        self.finished_nodes = 0
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(MyMessage.MSG_AGR_KEY_RESULT,
                           (self.trainer.res_dict[receive_id]))
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION,
                                              self.handle_validation_sign)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        if client_phase == "train":
            self.trainer.train_mode()
        else:
            self.trainer.eval_mode()
        self.trainer.act_dict[sender] = (acts, labels)
        self.trainer.act_number += 1
        # self.log.info("act_number: {} MAX_RANK: {}".format(self.trainer.act_number, self.trainer.MAX_RANK))
        if self.trainer.act_number == self.trainer.MAX_RANK:
            self.trainer.act_number = 0

            for i in range(1, self.trainer.MAX_RANK+1):
                self.trainer.forward_pass(self.trainer.act_dict[i][0], self.trainer.act_dict[i][1], i)
                grads = None
                if self.trainer.phase == "train":
                    grads = self.trainer.backward_pass()

                self.send_grads_to_client(i, grads)
            if self.trainer.phase == "train":
                self.trainer.optimizer.step()
                self.trainer.optimizer.zero_grad()

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_validation_sign(self, msg_params):
        self.trainer.validation_sign_number += 1
        if self.trainer.validation_sign_number == self.trainer.MAX_RANK:
            self.trainer.validation_sign_number = 0
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_validation_sign_to_client(idx)

    def send_validation_sign_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_START_VALIDATION, 0, receive_id)
        self.send_message(message)

# --- Client manager ------------------------------------------------
import logging
import torch
import time
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.log import Log
from .client import SplitNNClient

class ClientManager(MessageManager):
    """
    args must include MPI comm, rank, and max_rank (comm.size() - 1). Other fields are not required here.
    trainer is an instance of SplitNNClient.
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        #self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)

    def run(self):
        # logging.info("{} begin run_forward_pass".format(self.trainer.rank))
        self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        logging.info("{} end run_forward_pass act :{}".format(self.trainer.rank, acts.shape))
        logging.warning("rank {}".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            logging.warning("validate {}".format(i))
            self.run_forward_pass()
            while True:
                if self.com_manager.q_receiver.qsize()>0:
                    msg_params = self.com_manager.q_receiver.get()
                    self.com_manager.notify(msg_params)
                    break
                else:
                    time.sleep(0.03)
        self.trainer.write_log()
        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_START_VALIDATION,
                                              self.handle_validation_start_sign)

    def handle_message_gradients(self, msg_params):
        tot,cor,vl = msg_params.get(MyMessage.MSG_AGR_KEY_RESULT)
        self.trainer.total += tot
        self.trainer.correct += cor
        self.trainer.val_loss += vl
        self.trainer.step += 1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(grads)
            logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
            # if self.trainer.rank == 2 and self.trainer.batch_idx == len(self.trainer.trainloader) // 2:
            #     self.run_eval()

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                # torch.save(self.trainer.model, self.args["model_tmp_path"])
                self.send_validation_sign(0)
                # self.run_eval()
            else:
                self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        logging.warning("acts to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_validation_sign(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION, self.rank, receive_id)
        self.send_message(message)


    def handle_validation_start_sign(self, msg_params):
        self.run_eval()
