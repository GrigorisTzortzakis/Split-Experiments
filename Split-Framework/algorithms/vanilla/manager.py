"""Managers (client + server) and message types for `vanilla`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5

    # Vanilla paper Algorithm 2 (Step 6-7): client requests last-trained weights from Bob (server)
    MSG_TYPE_C2S_SEND_MODEL = 7
    MSG_TYPE_C2S_REQUEST_LAST_MODEL = 10

    # server to client
    MSG_TYPE_S2C_SEND_MODEL = 8

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

    MSG_ARG_KEY_MODEL_STATE_DICT = "model_state_dict"

# --- Server manager ------------------------------------------------
from mpi4py import MPI
from runtime.MPI.Messaging_MPI import Message
from runtime.MPI.Messaging_MPI import MessageManager
import logging


class ServerManager(MessageManager):

    def __init__(self, args, trainer, backend="MPI"):
        super().__init__(args, "server", args["comm"], args["rank"],
                         args["max_rank"] + 1, backend)
        self.trainer = trainer
        self.round_idx = 0

        # Bob stores the last-trained client weights (paper Algorithm 2 Step 6-7).
        self._last_trained_state_dict = None
        self._last_trained_rank = None

        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
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
                                              self.handle_message_client_model_update)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_REQUEST_LAST_MODEL,
                                              self.handle_message_request_last_model)

    def handle_message_client_model_update(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        state_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_STATE_DICT)
        self._last_trained_state_dict = state_dict
        self._last_trained_rank = sender

    def handle_message_request_last_model(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_MODEL, self.rank, sender)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_STATE_DICT, self._last_trained_state_dict)
        self.send_message(message)

    def handle_message_acts(self, msg_params):
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            self.send_grads_to_client(self.trainer.active_node, grads)

    def handle_message_validation_mode(self, msg_params):
        logging.warning("server recv vali mode")
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        # logging.warning("over")
        self.trainer.validation_over()

        # Robust termination: the protocol should end after each client completes
        # `epochs` validation phases (total validations = epochs * num_clients).
        # If the last client's FINISH message is delayed or lost, the default
        # token-passing scheme can deadlock and wait forever. This check lets the
        # server end deterministically.
        try:
            expected_total_validations = int(self.args["epochs"]) * int(self.args["max_rank"])
        except Exception:
            expected_total_validations = None

        if expected_total_validations is not None and getattr(self.trainer, "epoch", 0) >= expected_total_validations:
            logging.warning(
                "server reached expected total validations (%s); broadcasting finish",
                expected_total_validations,
            )
            self.handle_message_finish_protocol()

    def handle_message_finish_protocol(self, msg_params=None):
        # Broadcast finish to all clients so they can exit their receive loops.
        logging.warning("server recv finish -> broadcasting")
        for client_rank in range(1, self.args["max_rank"] + 1):
            message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, client_rank)
            self.send_message(message)
        self.finish()

# --- Client manager ------------------------------------------------
import logging
import time
from runtime.MPI.Messaging_MPI import Message
from runtime.MPI.Messaging_MPI import MessageManager
from runtime.log import Log


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
        self._awaiting_refresh_from_server = False

    def run(self):
        if self.rank == 1:
            logging.info("{} begin run_forward_pass".format(self.trainer.rank))
            self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        acts, labels = self.trainer.forward_pass()
        
        #logging.info("{} run_forward_pass".format(self.trainer.rank))
        self.send_activations_and_labels_to_server(acts, labels, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()
        self.trainer.print_com_size(self.com_manager)

        for i in range(len(self.trainer.testloader)):
            self.trainer.step += 1
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.round_idx += 1
        self.trainer.epoch_count+=1
        if self.round_idx == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            logging.warning("client{} send finish -> exiting".format(self.rank))
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            time.sleep(3)
            self.send_semaphore_to_client(self.trainer.node_right)

        self.trainer.batch_idx = 0

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEND_MODEL,
                                              self.handle_message_model_from_server)

    def handle_message_finish_protocol(self, msg_params=None):
        logging.warning("client{} recv finish".format(self.rank))
        self.finish()

    def handle_message_semaphore(self, msg_params):
        # Paper Algorithm 2 Step 6-7: request last-trained weights from Bob (server), then refresh.
        logging.warning("client{} recv sema".format(self.rank))
        self._awaiting_refresh_from_server = True
        self.request_last_model_from_server(self.trainer.SERVER_RANK)

    def handle_message_model_from_server(self, msg_params):
        state_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_STATE_DICT)
        # If server hasn't received any model yet (e.g., very first handover), skip refresh.
        if state_dict is not None:
            self.trainer.model.load_state_dict(state_dict)
        self._awaiting_refresh_from_server = False
        self.trainer.train_mode()
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        logging.warning("batch: {} len {}".format(self.trainer.batch_idx, len(self.trainer.trainloader)))
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            # torch.save(self.trainer.model, self.args["model_save_path"].format("client", self.trainer.rank,
            #                                                                    self.round_idx))
            self.send_model_update_to_server(self.trainer.SERVER_RANK)
            self.run_eval()
        else:
            self.run_forward_pass()

    def _cpu_state_dict(self):
        # Ensure tensors are on CPU for MPI serialization and cross-device safety.
        return {k: v.detach().cpu() for k, v in self.trainer.model.state_dict().items()}

    def send_model_update_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_STATE_DICT, self._cpu_state_dict())
        self.send_message(message)

    def request_last_model_from_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_REQUEST_LAST_MODEL, self.rank, receive_id)
        self.send_message(message)

    def send_message_test(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_TEST_C2C, self.rank, receive_id)
        self.send_message(message)

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
      #  logging.warning("acts to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
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
