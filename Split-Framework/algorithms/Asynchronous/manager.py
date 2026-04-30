"""Managers (client + server) and message types for `Asynchronous`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_SEND_MODEL = 8

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_SEND_MODEL = 7
    MSG_TYPE_C2S_NEXT_CLIENT = 11

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
    MSG_ARG_KEY_MODEL = "model"
    MSG_ARG_KEY_NEXT_CLIENT = 'next_client'
    MSG_ARG_KEY_STATE = 'state'

# --- Server manager ------------------------------------------------
from mpi4py import MPI
from runtime.MPI.Messaging_MPI import Message, MessageManager
import logging
from runtime.exports.log import Log

class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0
        self.log = Log(self.__class__.__name__, args)

        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.annotate_tensor_distribution_message(message, self.trainer)
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE,
                                              self.handle_message_validation_mode)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER,
                                              self.handle_message_validation_over)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_client_model)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_NEXT_CLIENT,
                                              self.handle_next_client)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            if self.trainer.state == "A":
                self.send_grads_to_client(self.trainer.active_node, grads)
            else:
                self.send_grads_to_client(self.trainer.active_node, None)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()
        self.advance_dynamic_quantization_for_trainer(self.trainer)

    def handle_message_finish_protocol(self, msg_params):
        self.finish()

    def handle_message_client_model(self, msg_params):
        model = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.trainer.client_model = model

    def send_model_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.client_model)
        message.add_params(MyMessage.MSG_ARG_KEY_STATE, self.trainer.state)
        self.send_message(message)

    def handle_next_client(self, msg_params):
        next_client = msg_params.get(MyMessage.MSG_ARG_KEY_NEXT_CLIENT)
        if next_client == 1:
            # new turn
            if self.trainer.state == "A":
                self.trainer.last_update_loss = self.trainer.total_loss / 156 / 3
            self.trainer.delta_loss = self.trainer.last_update_loss - self.trainer.total_loss / 156 / 3
            self.log.info("loss:{}, {}".format(self.trainer.last_update_loss, self.trainer.total_loss / 156 / 3))
            if self.trainer.delta_loss >= self.trainer.loss_thred:
                self.trainer.state = "A"
            else:
                if self.trainer.state == "A":
                    self.trainer.state = "B"
                else:
                    self.trainer.state = "C"
            self.trainer.total_loss = 0
        self.send_model_to_client(next_client)

    # def send_message_to_next_client(self, receive_id):
    #     message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODEL, self.rank, receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.client_model)
    #     self.send_message(message)

# --- Client manager ------------------------------------------------
import logging
import torch
import time
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
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)
        self.round_idx = 0

    def run(self):
        if self.rank == 1:
            logging.info("{} begin run_forward_pass".format(self.trainer.rank))
            self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        if self.trainer.server_state == "C":
            acts, labels = self.trainer.acts_last, self.trainer.labels_last
        else:
            acts, labels = self.trainer.forward_pass()
            self.trainer.acts_last, self.trainer.labels_last = acts, labels
        logging.info("{} run_forward_pass".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()
        self.trainer.print_com_size(self.com_manager)

        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.send_next_client_to_server(0, self.trainer.node_right)

        self.trainer.batch_idx = 0

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_MODEL,
                                              self.handle_model_from_server)

    def handle_message_semaphore(self, msg_params):
        # no point in checking the semaphore message
        logging.warning("client{} recv sema".format(self.rank))
        self.trainer.train_mode()
        # self.trainer.model.load_state_dict(torch.load(self.args["model_tmp_path"]))
        # self.trainer.model = torch.load(self.args["model_tmp_path"])
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        if grads is not None:
            self.trainer.backward_pass(grads)
        else:
            pass
        logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.round_idx))
            # torch.save(self.trainer.model.state_dict(), self.args["model_tmp_path"])
            # torch.save(self.trainer.model, self.args["model_tmp_path"])
            self.send_model_to_server(0)
            self.run_eval()
        else:
            self.run_forward_pass()

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
      #  logging.warning("acts to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.annotate_tensor_distribution_message(message, self.trainer)
        self.send_message(message)

    def send_next_client_to_server(self, receive_id, next_client):
        message = Message(MyMessage.MSG_TYPE_C2S_NEXT_CLIENT, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_NEXT_CLIENT, next_client)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        logging.warning("client {} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_model_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, self.trainer.model)
        self.send_message(message)

    def handle_model_from_server(self, msg_params):
        model = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.trainer.server_state = msg_params.get(MyMessage.MSG_ARG_KEY_STATE)
        self.trainer.model = model
        # self.log.info(model)
        # self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
        #                            weight_decay=5e-4)
        logging.warning("client{} recv sema".format(self.rank))
        self.trainer.train_mode()
        # self.trainer.model.load_state_dict(torch.load(self.args["model_tmp_path"]))
        # self.trainer.model = torch.load(self.args["model_tmp_path"])
        self.run_forward_pass()

