"""Managers (client + server) and message types for `Ushaped`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_ACTS = 8

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_SEND_GRADS = 7
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5

    # client to client
    MSG_TYPE_C2C_SEMAPHORE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_TYPE_TEST_C2C = 9

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"

# --- Server manager ------------------------------------------------
from mpi4py import MPI
from runtime.MPI.Messaging_MPI import Message, MessageManager
import logging


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0

        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRADS,
                                              self.handle_message_grads)

    def handle_message_acts(self, msg_params):
        # logging.warning("server recv acts")
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        acts2 = self.trainer.forward_pass(acts)
        self.send_acts_to_client(self.trainer.active_node, acts2)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        new_grad = self.trainer.backward_pass(grads)
        self.send_grads_to_client(self.trainer.active_node, new_grad)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()

    def handle_message_finish_protocol(self, msg_params):
        self.finish()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def send_acts_to_client(self, receive_id, acts):
        # logging.warning("server acts2 to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        self.send_message(message)

# --- Client manager ------------------------------------------------
import logging
import torch
import time
import sys
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log

class ClientManager(MessageManager):
    """
    args must include MPI comm, rank, and max_rank (comm.size() - 1). Other fields are not required here.
    trainer is an instance of SplitNNClient.
    """

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "client", args["comm"], args["rank"], args["max_rank"] + 1, backend)
        self.trainer = trainer

        self.log = Log(self.__class__.__name__, args)

    def run(self):
        if self.rank == 1:
            self.trainer.train_mode()
            self.run_forward_pass()
        super().run()  # Start the receive loop after initial sends; otherwise ranks may deadlock waiting for each other.

    def run_forward_pass(self):
        acts = self.trainer.forward_pass()
        logging.warning("send acts to server")
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()

        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
            # logging.warning("lis")
            while True:
                if self.com_manager.q_receiver.qsize() > 0:
                    msg_params = self.com_manager.q_receiver.get()
                    self.com_manager.notify(msg_params)
                    break
                else:
                    time.sleep(0.5)
        self.trainer.validation_over()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.send_semaphore_to_client(self.trainer.node_right)

        self.trainer.batch_idx = 0

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ACTS,
                                              self.handle_message_acts_from_server)

    def handle_message_semaphore(self, msg_params):
        logging.warning("client{} recv semapgore".format(self.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(type=0, grads=grads)
        logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            self.trainer.print_com_size(self.com_manager)
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.trainer.epoch_count))
            self.run_eval()
        else:
            self.run_forward_pass()

    def handle_message_acts_from_server(self, msg_params):
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        # logging.warning("QAQ3")
        self.trainer.forward_pass(type=1, inputs=acts)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass(type=1)
            self.send_grads_to_server(self.trainer.SERVER_RANK, grads)

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_to_server(self, acts, receive_id):
        #    logging.warning("{} acts to {}".format(self.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        self.send_message(message)

    def send_grads_to_server(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

