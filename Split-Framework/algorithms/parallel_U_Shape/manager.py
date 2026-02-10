"""Managers (client + server) and message types for `parallel_U_Shape`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_ACTS = 8
    MSG_TYPE_S2C_TEST = 10
    MSG_TYPE_S2C_TRAIN = 12

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_SEND_GRADS = 7
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_TEST_EMD = 11
    MSG_TYPE_C2S_TRAIN_END = 12

    # client to client
    MSG_TYPE_C2C_SEMAPHORE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_TYPE_TEST_C2C = 9

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"
    MSG_ARG_KEY_BATCH_END = "batch_end"

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
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_TEST_EMD,
                                              self.handle_test_end_from_clients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_TRAIN_END,
                                              self.handle_train_end_from_clients)

    def handle_message_acts(self, msg_params):
        # logging.warning("server recv acts")
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        batch_end = msg_params.get(MyMessage.MSG_ARG_KEY_BATCH_END)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        if phase == "train":
            self.trainer.train_mode()
        else:
            self.trainer.eval_mode()
            # logging.warning("server recv acts {}".format(self.trainer.is_waiting))
        # Always enqueue (sender, activations) first.
        self.trainer.client_act_queue.put((sender, acts, batch_end))
        if not self.trainer.is_waiting:
            # Forward pass
            # logging.warning("forwordpass")
            self.trainer.is_waiting = True
            sender, acts, batch_end = self.trainer.client_act_queue.get()
            acts2 = self.trainer.forward_pass(acts)
            self.send_acts_to_client(sender, acts2, batch_end)
        # else:
        #     acts2 = self.trainer.forward_pass(acts)
        #     self.send_acts_to_client(sender, acts2, batch_end)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        if grads is not None:
            new_grad = self.trainer.backward_pass(grads)

            self.send_grads_to_client(sender, new_grad)
            if self.trainer.client_act_queue.empty():
                # Queue is empty
                self.trainer.is_waiting = False
            else:
                # Queue is not empty
                sender, acts, batch_end = self.trainer.client_act_queue.get()

                acts2 = self.trainer.forward_pass(acts)
                self.send_acts_to_client(sender, acts2, batch_end)

                # if self.trainer.client_batch_end_num == self.trainer.client_number:
                #
                #     for i in range(1, self.trainer.client_number + 1):
                #         self.send_test_sign_to_client(i)
                #     self.trainer.client_batch_end_num = 0
                #     self.trainer.is_waiting = False
        else:
            self.send_grads_to_client(sender, None)
            if self.trainer.client_act_queue.empty():
                # Queue is empty
                self.trainer.is_waiting = False
            else:
                # Queue is not empty
                sender, acts, batch_end = self.trainer.client_act_queue.get()

                if batch_end:
                    self.trainer.client_batch_end_num += 1
                acts2 = self.trainer.forward_pass(acts)

                self.send_acts_to_client(sender, acts2, batch_end)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()

    def handle_message_finish_protocol(self):
        self.finish()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def send_acts_to_client(self, receive_id, acts, batch_end):
        # logging.warning("server acts2 to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        # message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, batch_end)
        self.send_message(message)

    def send_test_sign_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_TEST, self.rank, receive_id)
        self.send_message(message)

    def handle_test_end_from_clients(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # logging.info("{} end".format(sender))
        self.trainer.client_test_end_num += 1
        if self.trainer.client_test_end_num == self.trainer.client_number:
            self.trainer.print_com_size(self.com_manager)
            for i in range(1, self.trainer.client_number + 1):
                self.send_train_sign_to_clients(i)
            self.trainer.client_test_end_num = 0

    def handle_train_end_from_clients(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        # logging.info("{} train end".format(sender))
        self.trainer.client_train_end_num += 1
        # logging.info("{}, {}".format(self.trainer.client_train_end_num, self.trainer.client_number))
        if self.trainer.client_train_end_num == self.trainer.client_number:
            # logging.info("queue len {}".format(self.trainer.client_act_queue.qsize()))
            self.trainer.is_waiting = False
            for i in range(1, self.trainer.client_number + 1):
                self.send_test_sign_to_client(i)
            self.trainer.client_train_end_num = 0

    def send_train_sign_to_clients(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_TRAIN, self.rank, receive_id)
        self.send_message(message)

# --- Client manager ------------------------------------------------
import logging
import torch
import time
import sys
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.log import Log


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
        # if self.rank == 1:
        self.trainer.train_mode()
        self.run_forward_pass()
        super().run()  # Start the receive loop after initial sends; otherwise ranks may deadlock waiting for each other.

    def run_forward_pass(self):
        acts = self.trainer.forward_pass()
        logging.warning("{} send acts to server".format(self.trainer.rank))
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        acts = self.trainer.forward_pass()
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)

        # for i in range(len(self.trainer.dataloader)):
        #     logging.warning("validate {} from {} len {}".format(i, self.trainer.rank, len(self.trainer.dataloader)))
        #     self.run_forward_pass()
        #     while True:
        #         if self.com_manager.q_receiver.qsize() > 0:
        #             msg_params = self.com_manager.q_receiver.get()
        #             self.com_manager.notify(msg_params)
        #             break
        #         else:
        #             time.sleep(0.5)
        # self.trainer.write_log()
        # self.trainer.epoch_count += 1
        # if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
        #     self.send_finish_to_server(self.trainer.SERVER_RANK)
        #     self.finish()
        # else:
        #     # self.trainer.train_mode()
        #     # self.run_forward_pass()
        #     self.send_test_end_to_server(self.trainer.SERVER_RANK)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_TEST,
                                              self.handle_test_sign)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ACTS,
                                              self.handle_message_acts_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_TRAIN,
                                              self.handle_train_sign)

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        if grads is not None:
            self.trainer.backward_pass(type=0, grads=grads)
        logging.warning("batch: {} len {} from {}".format(self.trainer.batch_idx, len(self.trainer.dataloader),
                                                          self.trainer.rank))
        if self.trainer.batch_idx == len(self.trainer.dataloader):
            self.trainer.print_com_size(self.com_manager)
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.trainer.epoch_count))
            if self.trainer.phase == 'validation':

                self.trainer.write_log()
                self.trainer.epoch_count += 1
                if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE \
                        and self.trainer.rank == self.trainer.MAX_RANK:
                    self.send_finish_to_server(self.trainer.SERVER_RANK)
                    self.finish()
                self.send_test_end_to_server(self.trainer.SERVER_RANK)
            else:
                self.send_train_end_to_server(self.trainer.SERVER_RANK)


            # self.run_eval()
        else:

            self.run_forward_pass()

    def handle_message_acts_from_server(self, msg_params):
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        # logging.warning("QAQ3")
        self.trainer.forward_pass(type=1, inputs=acts)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass(type=1)
            self.send_grads_to_server(self.trainer.SERVER_RANK, grads)
        else:
            self.send_grads_to_server(self.trainer.SERVER_RANK, None)

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_to_server(self, acts, receive_id):
        #    logging.warning("{} acts to {}".format(self.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)

        message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, self.trainer.batch_idx + 1 == len(self.trainer.dataloader))
        self.send_message(message)

    def send_grads_to_server(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
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

    def handle_test_sign(self, msg_params):
        logging.info("{} handle_test_sign ".format(self.trainer.rank))
        self.trainer.eval_mode()
        self.run_forward_pass()

    def send_test_end_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_TEST_EMD, self.rank, receive_id)
        self.send_message(message)

    def send_train_end_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_TRAIN_END, self.rank, receive_id)
        self.send_message(message)

    def handle_train_sign(self, msg_params):
        logging.info("{} handle_train_sign ".format(self.trainer.rank))
        self.trainer.train_mode()
        self.run_forward_pass()
