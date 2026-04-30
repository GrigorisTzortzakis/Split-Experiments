"""Managers (client + server) and message types for `parallel`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_START_VALIDATION = 8
    MSG_TYPE_S2C_START_NEXT_ROUND = 9

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5

    # client to client
    MSG_TYPE_C2C_SEMAPHORE = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"
    MSG_ARG_KEY_PHASE = "phase"
    MSG_AGR_KEY_RESULT = "result"
    MSG_AGR_KEY_SAMPLE_NUM = "sample_num"

    MSG_TYPE_TEST_C2C = 9

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_ACTS = "activations"
    MSG_ARG_KEY_GRADS = "activation_grads"

# --- Server manager ------------------------------------------------
from mpi4py import MPI
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.log = Log(self.__class__.__name__, args)
        self.trainer = trainer
        self.finished_nodes = 0
        self.validation_ready_nodes = set()
        self.validation_over_nodes = set()

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None, result=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        if result is None:
            result = (self.trainer.total, self.trainer.correct, self.trainer.val_loss)
        message.add_params(MyMessage.MSG_AGR_KEY_RESULT, result)
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

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        sample_num = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        if client_phase is None:
            client_phase = "train"
        if sample_num is not None:
            self.trainer.client_sample_num_dict[sender] = sample_num

        self.trainer.load_client_state(sender, client_phase)
        self.trainer.forward_pass(acts, labels)
        grads = None
        if client_phase == "train":
            grads = self.trainer.backward_pass()
            self.trainer.save_client_state(sender)
        self.send_grads_to_client(
            sender,
            grads,
            (self.trainer.total, self.trainer.correct, self.trainer.val_loss),
        )

    def handle_message_finish_protocol(self, msg_params=None):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_message_validation_mode(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        self.validation_ready_nodes.add(sender)
        if len(self.validation_ready_nodes) == self.trainer.MAX_RANK:
            self.validation_ready_nodes.clear()
            self.trainer.federate_server_models()
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_validation_sign_to_client(idx)

    def handle_message_validation_over(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        self.validation_over_nodes.add(sender)
        if len(self.validation_over_nodes) == self.trainer.MAX_RANK:
            self.validation_over_nodes.clear()
            self.trainer.epoch += 1
            if self.trainer.epoch < self.trainer.args["epochs"]:
                self.advance_dynamic_quantization_epoch(next_epoch=self.trainer.epoch)
                for idx in range(1, self.trainer.MAX_RANK + 1):
                    self.send_next_batch_sign(idx)

    def send_validation_sign_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_START_VALIDATION, 0, receive_id)
        self.send_message(message)

    def send_next_batch_sign(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_START_NEXT_ROUND, 0, receive_id)
        self.send_message(message)

# --- Client manager ------------------------------------------------
import logging
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.exports.log import Log
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
        if self._is_finished:
            return
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        logging.info("{} end run_forward_pass act :{}".format(self.trainer.rank, acts.shape))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def _wait_for_message(self):
        msg_params = self.com_manager.q_receiver.get()
        self.com_manager.notify(msg_params)

    def run_eval(self):
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            logging.warning("validate {}".format(i))
            self.run_forward_pass()
            self._wait_for_message()
        self.trainer.write_log()
        self.trainer.epoch_count += 1
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_START_VALIDATION,
                                              self.handle_validation_start_sign)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_START_NEXT_ROUND,
                                              self.handle_next_round_sign)

    def handle_message_gradients(self, msg_params):
        tot,cor,vl= msg_params.get(MyMessage.MSG_AGR_KEY_RESULT)
        self.trainer.total += tot
        self.trainer.correct += cor
        self.trainer.val_loss += vl
        self.trainer.step+=1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(grads)
            logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
            # if self.trainer.rank == 2 and self.trainer.batch_idx == len(self.trainer.trainloader) // 2:
            #     self.run_eval()

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
            else:
                self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        logging.warning("acts to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        self.annotate_tensor_distribution_message(message, self.trainer)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.rank, receive_id)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)


    def handle_validation_start_sign(self, msg_params):
        self.run_eval()

    def handle_next_round_sign(self, msg_params):
        self.trainer.train_mode()
        self.run_forward_pass()




