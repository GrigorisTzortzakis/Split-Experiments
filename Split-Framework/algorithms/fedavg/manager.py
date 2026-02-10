"""Managers (client + server) and message types for `fedavg`."""

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

# --- Server manager ------------------------------------------------
from mpi4py import MPI
import logging
from runtime.MPI.Messaging_MPI import Message, MessageManager
from runtime.log import Log
import torch


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
                           (self.trainer.total, self.trainer.correct, self.trainer.val_loss))
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                                              self.handle_message_model_param)

    def handle_message_finish_protocol(self):
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
        # self.sender_list[sender] = True
        # self.trainer.client_sample_dict[sender] = sample_number
        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == self.trainer.MAX_RANK:
            # self.log.info(self.sender_list)
            # for key in self.trainer.model_param_dict.keys():
            #     logging.info("key: ".format(key))
            self.log.info(
                "get all model params ---- from rank {}".format(self.trainer.rank))

            self.trainer.model_param_num = 0
            model_avg = model_param
            dist = 0
            t_n = 0
            for key in model_avg.keys():
                for idx in range(1, self.trainer.MAX_RANK + 1):

                    # self.sender_list[idx] = False

                    local_sample_number, local_model_params = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        model_avg[key] = local_model_params[key] * w
                    else:
                        model_avg[key] += local_model_params[key] * w
                #self.log.info("key:{}\n value:{}".format(key, model_avg[key]))
                dif = self.trainer.last_param[key]-model_avg[key]
                sz = 1
                for v in dif.shape:
                    sz *= v
                t_n += sz
                dist += torch.pow(dif, 2).sum(list(range(len(dif.shape))))

            self.log.info("Dist:{d}".format(d=dist/t_n))
            self.trainer.sum_sample_number = 0
            self.trainer.last_param = model_avg
            for idx in range(1, self.trainer.MAX_RANK + 1):
                self.send_model_param_to_fed_client(idx, model_avg)

    def send_model_param_to_fed_client(self, receive_id, model_avg_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL, model_avg_param)
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
        super().__init__(args, "client",
                         args["comm"], args["rank"], args["max_rank"] + 1, backend)
        # self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)

    def run(self):
        # logging.info("{} begin run_forward_pass".format(self.trainer.rank))
        self.register_message_receive_handlers()
        self.run_forward_pass()
        super(ClientManager, self).run()

    def run_forward_pass(self):
        while True:
            tot, cor, vl = self.trainer.forward_pass()
            # logging.info("{} end run_forward_pass act :{}".format(self.trainer.rank, acts.shape))
            self.trainer.batch_idx += 1
            self.trainer.total += tot
            self.trainer.correct += cor
            self.trainer.val_loss += vl
            self.trainer.step += 1
            if self.trainer.phase == "train":
                self.trainer.write_log()
                self.trainer.backward_pass()
                logging.warning("batch: {} len {}".format(
                    self.trainer.batch_idx, len(self.trainer.trainloader)))

                if self.trainer.batch_idx == len(self.trainer.trainloader):
                    # torch.save(self.trainer.model, self.args["model_tmp_path"])
                    self.send_model_param_to_fed_server(0)
                    while True:
                        if self.com_manager.q_receiver.qsize() > 0:
                            msg_params = self.com_manager.q_receiver.get()

                            self.com_manager.notify(msg_params)
                            break
                        else:
                            time.sleep(0.5)

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
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            self.trainer.train_mode()
            self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MODEL,
                                              self.handle_message_model_param_from_server)

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        logging.warning("acts to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                          self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL)
        self.trainer.model.load_state_dict(model_param)

    def send_model_param_to_fed_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL,
                          self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL,
                           self.trainer.model.state_dict())
      #  self.log.info(self.trainer.model.state_dict())
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM,
                           self.trainer.local_sample_number)
        self.send_message(message)
