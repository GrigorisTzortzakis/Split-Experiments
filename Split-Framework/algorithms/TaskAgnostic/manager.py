"""Managers (client + server) and message types for `TaskAgnostic`."""

# --- Message types -------------------------------------------------
class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_GRADS = 1
    MSG_TYPE_S2C_MODEL = 10
    MSG_TYPE_S2C_ACTS = 8
    MSG_TYPE_S2C_SEMAPHORE = 6
    MSG_TYPE_S2C_READY_TO_GET_MODEL = 14

    # client to server
    MSG_TYPE_C2S_SEND_ACTS = 2
    MSG_TYPE_C2S_VALIDATION_MODE = 3
    MSG_TYPE_C2S_VALIDATION_OVER = 4
    MSG_TYPE_C2S_PROTOCOL_FINISHED = 5
    MSG_TYPE_C2S_SEND_GRADS = 7
    MSG_TYPE_C2S_MODEL = 11
    MSG_TYPE_C2S_NEXT_BATCH = 13
    MSG_TYPE_C2S_READY_TO_SEND_MODEL = 15
    # Priority queue wait type

    # c 2 c
    MSG_TYPE_C2C_SEMAPHORE = 12


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

    # MSG_AGR_KEY_MODEL = "model"
    MSG_AGR_HEAD_MODEL = "head_model"
    MSG_AGR_TAIL_MODEL = "tail_model"
    MSG_AGR_KEY_SAMPLE_NUM = "sample_num"
    # MSG_AGR_KEY_MODEL
    MSG_ARG_KEY_CLIENT_NUM = "client_num"
    MSG_ARG_KEY_CUR_DATASET_IDX = "dataset_cur"
    MSG_ARG_KEY_BATCH_END = "batch_end"

# --- Server manager ------------------------------------------------
from mpi4py import MPI
import logging
import time
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
        self.sender_list = dict()
        # logging.warning("server rank{} args{}".format(self.rank,args["rank"]))

    def run(self):
        super().run()

    def send_grads_to_client(self, receive_id, grads=None):
        message = Message(MyMessage.MSG_TYPE_S2C_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        # message.add_params(MyMessage.MSG_AGR_KEY_RESULT,
        #                    (self.trainer.total, self.trainer.correct, self.trainer.val_loss))
        self.send_message(message)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_ACTS,
                                              self.handle_message_acts)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
                                              self.handle_message_finish_protocol)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_GRADS,
                                              self.handle_message_grads)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_MODEL,
                                              self.handle_message_model_param)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_NEXT_BATCH,
                                              self.handle_next_batch)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_READY_TO_SEND_MODEL,
                                              self.handle_model_ready_sign)

    # def handle_message_acts(self, msg_params):
    #     acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
    #     self.active_node = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
    #     client_phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
    #     if client_phase == "train":
    #         self.trainer.train_mode()
    #     else:
    #         self.trainer.eval_mode()
    #     self.trainer.forward_pass(acts, labels)
    #     # self.log.info(acts.shape)
    #     # self.log.info(type(acts))
    #
    #
    #
    #     grads = None
    #     if self.trainer.phase == "train":
    #         grads = self.trainer.backward_pass()
    #
    #     self.send_grads_to_client(self.active_node, grads)
    def handle_message_acts(self, msg_params):
        # logging.warning("server recv acts")
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        self.trainer.client_act_dict[sender] = acts
        # self.trainer.client_act_check_dict[dataset_cur]
        if dataset_cur in self.trainer.client_act_check_dict:
            self.trainer.client_act_check_dict[dataset_cur] += 1
        else:
            self.trainer.client_act_check_dict[dataset_cur] = 1
        # Check whether we've received activations from all clients for this dataset shard.
        if self.trainer.client_act_check_dict[dataset_cur] == client_number:
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            # Forward pass

            for i in range(start, end):
                acts = self.trainer.client_act_dict[i]
                acts2 = self.trainer.forward_pass(acts)
                self.send_acts_to_client(i, acts2)
                # tmp = self.com_manager.q_receiver.qsize()
                if phase == "train":
                    while True:
                        # Wait for gradient backprop message
                        if self.com_manager.q_receiver.qsize() > 0:
                            msg_params = self.com_manager.q_receiver.get()
                            if msg_params.get(MyMessage.MSG_ARG_KEY_TYPE) == MyMessage.MSG_TYPE_C2S_SEND_GRADS:
                                logging.info(
                                    "server get grads from {} key {}".format(
                                        msg_params.get(MyMessage.MSG_ARG_KEY_SENDER),
                                        msg_params.get(MyMessage.MSG_ARG_KEY_TYPE))
                                )
                            self.com_manager.notify(msg_params)
                            break
                        else:
                            time.sleep(0.5)

            self.trainer.client_act_check_dict[dataset_cur] = 0

    def handle_message_finish_protocol(self):
        self.finished_nodes += 1
        if self.finished_nodes == self.trainer.MAX_RANK:
            self.finish()

    def handle_message_model_param(self, msg_params):
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        head_model_param = msg_params.get(MyMessage.MSG_AGR_HEAD_MODEL)
        tail_model_param = msg_params.get(MyMessage.MSG_AGR_TAIL_MODEL)
        sample_number = msg_params.get(MyMessage.MSG_AGR_KEY_SAMPLE_NUM)
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        self.trainer.sum_sample_number += sample_number
        self.trainer.model_param_dict[sender] = (sample_number, head_model_param, tail_model_param)

        self.sender_list[sender] = True
        # self.trainer.client_sample_dict[sender] = sample_number
        self.trainer.model_param_num += 1
        if self.trainer.model_param_num == client_number:
            self.log.info(self.sender_list)
            # self.log.info("get all model params ---- from rank {}".format(self.trainer.rank))
            self.trainer.model_param_num = 0
            head_model_avg = head_model_param
            tail_model_avg = tail_model_param
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            for key in head_model_param.keys():
                for idx in range(start, end):

                    self.sender_list[idx] = False

                    local_sample_number, local_model_params, _ = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        head_model_avg[key] = local_model_params[key] * w
                    else:
                        head_model_avg[key] += local_model_params[key] * w

            for key in tail_model_param.keys():
                for idx in range(start, end):

                    self.sender_list[idx] = False

                    local_sample_number, _, local_model_params = self.trainer.model_param_dict[idx]
                    w = local_sample_number / self.trainer.sum_sample_number
                    if idx == 1:
                        tail_model_avg[key] = local_model_params[key] * w
                    else:
                        tail_model_avg[key] += local_model_params[key] * w
            self.trainer.sum_sample_number = 0
            # self.log.info(head_model_avg)
            # self.log.info(tail_model_avg)
            for idx in range(start, end):
                # self.log.info("send_model_param_to_fed_client： {}".format(idx))
                self.send_model_param_to_fed_client(idx, head_model_avg, tail_model_avg)

    # avg server
    def send_model_param_to_fed_client(self, receive_id, model_head_param, model_tail_param):
        message = Message(MyMessage.MSG_TYPE_S2C_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_AGR_TAIL_MODEL, model_tail_param)
        message.add_params(MyMessage.MSG_AGR_HEAD_MODEL, model_head_param)
        self.send_message(message)

    def send_acts_to_client(self, receive_id, acts):
        # logging.warning("server acts2 to {}".format(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        self.send_message(message)

    def handle_message_grads(self, msg_params):
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        phase = msg_params.get(MyMessage.MSG_ARG_KEY_PHASE)
        batch_end = msg_params.get(MyMessage.MSG_ARG_KEY_BATCH_END)
        client_num = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        if batch_end == 1:
            self.trainer.client_batch_end += 1
            if self.trainer.client_batch_end == client_num:
                self.trainer.client_batch_end = 0
                start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
                end = self.args['client_split'][dataset_cur] + 1
                for i in range(start, end):
                    # self.log.info("all client have been trained one epoch".format(i))
                    self.send_ready_get_model(i)
        if phase == "train":
            new_grad = self.trainer.backward_pass(grads)
            self.send_grads_to_client(sender, new_grad)
        else:
            self.send_grads_to_client(sender, None)

    # MSG_TYPE_S2C_SEMAPHORE
    def send_semaphore_to_clients(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_S2C_SEMAPHORE, self.rank, receive_id)
        message.add_params("SEMAPHORE", 0)
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        # logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))

        message = Message(MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.rank, receive_id)
        self.send_message(message)

    # MSG_TYPE_S2C_READY_TO_GET_MODEL
    def send_ready_get_model(self, receive_id):
        # logging.warning("client{} send vali over to server{}".format(self.rank, self.trainer.SERVER_RANK))

        message = Message(MyMessage.MSG_TYPE_S2C_READY_TO_GET_MODEL, self.rank, receive_id)
        self.send_message(message)

    def handle_next_batch(self, msg_params):
        # client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
        end = self.args['client_split'][dataset_cur] + 1
        for i in range(start, end):
            # self.log.info("send sign to rank {}".format(i))
            self.send_semaphore_to_clients(i)

    def handle_model_ready_sign(self, msg_params):
        # Count how many clients have reported readiness.
        client_number = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_NUM)
        dataset_cur = msg_params.get(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX)
        self.trainer.model_ready_num += 1

        if self.trainer.model_ready_num == client_number:
            self.trainer.model_ready_num = 0
            start = 1 if dataset_cur == 0 else self.args['client_split'][dataset_cur - 1] + 1
            end = self.args['client_split'][dataset_cur] + 1
            for i in range(start, end):
                # self.log.info("all client have been trained one epoch".format(i))
                self.send_ready_get_model(i)

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
        # self.trainer = type(SplitNNClient)
        self.trainer = trainer
        self.trainer.train_mode()
        self.log = Log(self.__class__.__name__, args)

    def run(self):
        # logging.info("{} begin run_forward_pass".format(self.trainer.rank))
        if self.trainer.args["dataset_cur"] == 0:
            self.trainer.train_mode()
            self.run_forward_pass()
        super().run()

    def run_forward_pass(self):
        acts = self.trainer.forward_pass()
        logging.warning("rank {} send acts to server".format(self.args["rank"]))
        self.send_activations_to_server(acts, self.trainer.SERVER_RANK)
        self.trainer.batch_idx += 1

    def run_eval(self):
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            logging.warning("validate {}".format(i))
            self.run_forward_pass()
            while True:
                if self.com_manager.q_receiver.qsize() > 0:
                    msg_params = self.com_manager.q_receiver.get()
                    self.com_manager.notify(msg_params)
                    break
                else:
                    time.sleep(0.5)
        self.trainer.write_log()
        self.trainer.epoch_count += 1
        if self.trainer.epoch_count == self.trainer.MAX_EPOCH_PER_NODE and self.trainer.rank == self.trainer.MAX_RANK:
            self.send_finish_to_server(self.trainer.SERVER_RANK)
            self.finish()
        else:
            # self.send_semaphore_to_client(self.trainer.node_right)
            # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
            # message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
            if self.args['client_split'][self.args['dataset_cur']] == self.args['rank']:

                next_group_cur = (self.args['dataset_cur'] + 1) % len(self.args['client_split'])
                logging.warning("next_group_cur {}".format(next_group_cur))
                start = 1 if next_group_cur == 0 else self.args['client_split'][next_group_cur - 1] + 1
                end = self.args['client_split'][next_group_cur] + 1

                for i in range(start, end):
                    self.send_semaphore_to_client(i)

            # while True:
            #     # wait for sign
            #     if self.com_manager.q_receiver.qsize() > 0:
            #         msg_params = self.com_manager.q_receiver.get()
            #         self.com_manager.notify(msg_params)
            #         break
            #     else:
            #         time.sleep(0.5)

            # self.trainer.train_mode()
            # self.run_forward_pass()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_GRADS,
                                              self.handle_message_gradients)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_ACTS,
                                              self.handle_message_acts_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SEMAPHORE,
                                              self.handle_message_semaphore_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_MODEL,
                                              self.handle_message_model_param_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2C_SEMAPHORE,
                                              self.handle_message_semaphore)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_READY_TO_GET_MODEL,
                                              self.handle_ready_to_get_model)

    def handle_message_model_param_from_server(self, msg_params):
        model_param = msg_params.get(MyMessage.MSG_AGR_HEAD_MODEL)
        # self.log.info("rank {} get model param".format(self.args['rank']))
        self.trainer.model.load_state_dict(model_param)
        model_param = msg_params.get(MyMessage.MSG_AGR_TAIL_MODEL)
        # self.log.info(model_param["block1.0.weight"])
        self.trainer.model_2.load_state_dict(model_param)
        self.run_eval()

    # MSG_TYPE_S2C_SEMAPHORE
    def handle_message_semaphore_from_server(self, msg_params):
        self.run_forward_pass()

    def handle_message_gradients(self, msg_params):
        self.trainer.step += 1
        if self.trainer.phase == "train":
            self.trainer.write_log()
            grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
            self.trainer.backward_pass(type=0, grads=grads)
            logging.warning("rank: {} batch: {} len {}".format(self.args['rank'], self.trainer.batch_idx,
                                                               len(self.trainer.trainloader)))

            if self.trainer.batch_idx == len(self.trainer.trainloader):
                # torch.save(self.trainer.model, self.args["model_tmp_path"])
                # self.send_ready_to_send_model(0)
                pass

            else:
                if self.args["rank"] == self.args['client_split'][self.args['dataset_cur']]:
                    self.send_next_batch_to_server(0)
            # else:
            #     while True:
            #         if self.com_manager.q_receiver.qsize() > 0:
            #             msg_params = self.com_manager.q_receiver.get()
            #             logging.info("rank {} get sign".format(self.args['rank']))
            #
            #             self.com_manager.notify(msg_params)
            #             break
            #         else:
            #             time.sleep(0.5)
            #     self.run_forward_pass()
        else:
            self.run_forward_pass()

    # def send_activations_and_labels_to_server(self, acts, labels, receive_id):
    #     logging.warning("acts to {}".format(receive_id))
    #     message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
    #     message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
    #     message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
    #     self.send_message(message)

    def send_activations_to_server(self, acts, receive_id):
        #    logging.warning("{} acts to {}".format(self.rank,receive_id))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, acts)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])

        self.send_message(message)

    def handle_message_acts_from_server(self, msg_params):
        acts = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        # logging.warning("QAQ3")
        self.trainer.forward_pass(type=1, inputs=acts)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass(type=1)
            self.send_grads_to_server(self.trainer.SERVER_RANK, grads)

    # def handle_message_acts_from_server(self, msg_params):
    def handle_message_semaphore(self, msg_params):
        logging.warning("client {} recv semapgore".format(self.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def send_model_param_to_fed_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_AGR_HEAD_MODEL, self.trainer.model.state_dict())
        message.add_params(MyMessage.MSG_AGR_TAIL_MODEL, self.trainer.model_2.state_dict())
        message.add_params(MyMessage.MSG_AGR_KEY_SAMPLE_NUM, self.trainer.local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.rank, receive_id)
        self.send_message(message)

    def send_grads_to_server(self, receive_id, grads):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_GRADS, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        # message.add_params(MyMessage.MSG_ARG_KEY_PHASE, self.trainer.phase)
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, 1)
        else:
            message.add_params(MyMessage.MSG_ARG_KEY_BATCH_END, 0)
        # comm.send(data, dest=destination_process)
        # msg = Message()
        # msg.add(Message.MSG_ARG_KEY_TYPE, message.get_type())
        # msg.add(Message.MSG_ARG_KEY_SENDER, message.get_sender_id())
        # msg.add(Message.MSG_ARG_KEY_RECEIVER, message.get_receiver_id())
        # for key, value in message.get_params().items():
        #     # logging.info("%s == %s" % (key, value))
        #     msg.add(key, value)
        #
        # self.args['comm'].send(msg, dest=0)
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.rank, receive_id)
        self.send_message(message)

    def send_next_batch_to_server(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_NEXT_BATCH, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    # MSG_TYPE_C2S_READY_TO_SEND_MODEL
    def send_ready_to_send_model(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_C2S_READY_TO_SEND_MODEL, self.rank, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_NUM, self.args['cur_client_num'])
        message.add_params(MyMessage.MSG_ARG_KEY_CUR_DATASET_IDX, self.args['dataset_cur'])
        self.send_message(message)

    # MSG_TYPE_S2C_READY_TO_GET_MODEL
    def handle_ready_to_get_model(self, msg_params):
        self.send_model_param_to_fed_server(0)
