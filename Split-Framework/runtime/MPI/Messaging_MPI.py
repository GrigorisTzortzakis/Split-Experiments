"""MPI send/receive + message dispatch.

What this file is for:
- Defines the `Message` container exchanged between ranks.
- Runs MPI send/receive threads.
- Provides `MpiCommunicationManager` (queues + threads).
- Provides `MessageManager` (register handlers + dispatch received messages).

Everything in here is MPI-specific.
"""

from __future__ import annotations

import ctypes
import json
import logging
import queue
import sys
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional

import torch
from mpi4py import MPI
from collections import defaultdict


class Message(object):
    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    MSG_ARG_KEY_RECEIVE_PRIORITY = "receive_priority"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"

    def __init__(self, type=0, sender_id=0, receiver_id=0):
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {}
        self.msg_params[Message.MSG_ARG_KEY_TYPE] = type
        self.msg_params[Message.MSG_ARG_KEY_SENDER] = sender_id
        self.msg_params[Message.MSG_ARG_KEY_RECEIVER] = receiver_id

    def init(self, msg_params):
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        self.msg_params = json.loads(json_string)
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]

    def get_size(self):
        """Best-effort payload size estimate in bytes.

        Notes:
        - MPI uses Python pickling for `comm.send`, so the *true* serialized size
          can differ. This estimator is designed to be consistent and to
          correctly account for tensors nested in dicts/tuples/lists.
        - Tensors are counted by `numel * element_size` (not Python object
          overhead).
        """

        def _tensor_nbytes(t: torch.Tensor) -> int:
            try:
                return int(t.nelement() * t.element_size())
            except Exception:
                try:
                    return int(t.numel() * t.element_size())
                except Exception:
                    return 0

        def _estimate(obj) -> int:
            if obj is None:
                return 0
            if isinstance(obj, torch.Tensor):
                return _tensor_nbytes(obj)
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return len(obj)
            if isinstance(obj, str):
                return len(obj.encode("utf-8", errors="replace"))
            if isinstance(obj, (int, float, bool)):
                return sys.getsizeof(obj)
            if isinstance(obj, dict):
                # Count both keys and values.
                return sum(_estimate(k) + _estimate(v) for k, v in obj.items())
            if isinstance(obj, (list, tuple, set)):
                return sum(_estimate(v) for v in obj)
            # Fallback: object header + string repr
            try:
                return sys.getsizeof(obj)
            except Exception:
                return sys.getsizeof(str(obj))

        return int(sum(_estimate(v) for v in self.msg_params.values()))

    def get_comm_category(self) -> str:
        """Heuristic category for message-size breakdown."""
        keys = set(self.msg_params.keys())
        if "activation_grads" in keys:
            return "GRADS"
        if "activations" in keys:
            return "ACTS"
        # Common model payload keys across algorithms.
        if "model_state_dict" in keys or "model" in keys or "model_params" in keys:
            return "MODEL"
        return "OTHER"

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    def add_params(self, key, value):
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params

    def add(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        return self.msg_params[key]

    def get_type(self):
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self.msg_params

    def to_json(self):
        json_string = json.dumps(self.msg_params)
        print("json string size = " + str(sys.getsizeof(json_string)))
        return json_string

    def get_content(self):
        print_dict = self.msg_params.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def __lt__(self, other):
        return self.get(Message.MSG_ARG_KEY_RECEIVE_PRIORITY) > other.get(Message.MSG_ARG_KEY_RECEIVE_PRIORITY)


class MPISendThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPISendThread, self).__init__()
        self.daemon = True
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q
        self.total_send_size = 0
        self.tmp_send_size = 0
        self.total_send_size_by_type = defaultdict(int)
        self.tmp_send_size_by_type = defaultdict(int)
        self.total_send_size_by_category = defaultdict(int)
        self.tmp_send_size_by_category = defaultdict(int)

    def run(self):
        logging.info("Starting " + self.name + ". Process ID = " + str(self.rank))
        while not self.stopped():
            try:
                if not self.q.empty():
                    msg = self.q.get()
                    msg_str = msg.to_string()
                    size = msg.get_size()
                    msg_type = msg.get_type()
                    msg_cat = msg.get_comm_category()
                    self.tmp_send_size += size
                    self.total_send_size += size
                    self.tmp_send_size_by_type[msg_type] += size
                    self.total_send_size_by_type[msg_type] += size
                    self.tmp_send_size_by_category[msg_cat] += size
                    self.total_send_size_by_category[msg_cat] += size
                    dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
                    self.comm.send(msg_str, dest=dest_id)
                else:
                    time.sleep(0.03)
            except SystemExit:
                break
            except Exception:
                traceback.print_exc()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")


class MPIReceiveThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPIReceiveThread, self).__init__()
        self.daemon = True
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.total_receive_size = 0
        self.tmp_receive_size = 0
        self.total_receive_size_by_type = defaultdict(int)
        self.tmp_receive_size_by_type = defaultdict(int)
        self.total_receive_size_by_category = defaultdict(int)
        self.tmp_receive_size_by_category = defaultdict(int)
        self.q = q

    def run(self):
        logging.debug("Starting Thread:" + self.name + ". Process ID = " + str(self.rank))
        status = MPI.Status()
        while not self.stopped():
            try:
                if not self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
                    time.sleep(0.01)
                    continue

                msg_str = self.comm.recv(source=status.Get_source(), tag=status.Get_tag())

                msg = Message()
                msg.init(msg_str)
                self.q.put(msg)
                size = msg.get_size()
                msg_type = msg.get_type()
                msg_cat = msg.get_comm_category()
                self.tmp_receive_size += size
                self.total_receive_size += size
                self.tmp_receive_size_by_type[msg_type] += size
                self.total_receive_size_by_type[msg_type] += size
                self.tmp_receive_size_by_category[msg_cat] += size
                self.total_receive_size_by_category[msg_cat] += size
            except SystemExit:
                break
            except Exception:
                traceback.print_exc()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")


class Observer:
    def receive_message(self, msg_type, msg_params):
        return


class MpiCommunicationManager:
    def __init__(self, comm, rank, size, node_type="client"):
        self.comm = comm
        self.rank = rank
        self.size = size

        self._observers: List[Observer] = []
        self.send_thread = None
        self.receive_thread = None
        self.collective_thread = None

        if node_type == "client":
            self.q_sender, self.q_receiver = self.init_client_communication()
        elif node_type == "server":
            self.q_sender, self.q_receiver = self.init_server_communication()
        else:
            self.q_sender, self.q_receiver = self.init_client_communication()

        self.is_running = True
        self.reset_analysis_data()

    def init_server_communication(self):
        server_send_queue = queue.Queue()
        self.send_thread = MPISendThread(self.comm, self.rank, self.size, "ServerSendThread", server_send_queue)
        self.send_thread.start()

        server_receive_queue = queue.Queue()
        self.receive_thread = MPIReceiveThread(
            self.comm, self.rank, self.size, "ServerReceiveThread", server_receive_queue
        )
        self.receive_thread.start()
        return server_send_queue, server_receive_queue

    def init_client_communication(self):
        client_send_queue = queue.Queue()
        self.send_thread = MPISendThread(self.comm, self.rank, self.size, "ClientSendThread", client_send_queue)
        self.send_thread.start()

        client_receive_queue = queue.Queue()
        self.receive_thread = MPIReceiveThread(
            self.comm, self.rank, self.size, "ClientReceiveThread", client_receive_queue
        )
        self.receive_thread.start()
        return client_send_queue, client_receive_queue

    def reset_analysis_data(self):
        self.send_thread.tmp_send_size = 0
        self.receive_thread.tmp_receive_size = 0
        # Reset per-type and per-category rolling (tmp) counters.
        if hasattr(self.send_thread, "tmp_send_size_by_type"):
            self.send_thread.tmp_send_size_by_type = defaultdict(int)
        if hasattr(self.send_thread, "tmp_send_size_by_category"):
            self.send_thread.tmp_send_size_by_category = defaultdict(int)
        if hasattr(self.receive_thread, "tmp_receive_size_by_type"):
            self.receive_thread.tmp_receive_size_by_type = defaultdict(int)
        if hasattr(self.receive_thread, "tmp_receive_size_by_category"):
            self.receive_thread.tmp_receive_size_by_category = defaultdict(int)

    # Backwards-compatible aliases (some algorithms expect these on `com_manager`).
    @property
    def tmp_send_size(self) -> int:
        return int(getattr(self.send_thread, "tmp_send_size", 0))

    @property
    def tmp_receive_size(self) -> int:
        return int(getattr(self.receive_thread, "tmp_receive_size", 0))

    @property
    def total_send_size(self) -> int:
        return int(getattr(self.send_thread, "total_send_size", 0))

    @property
    def total_receive_size(self) -> int:
        return int(getattr(self.receive_thread, "total_receive_size", 0))

    @property
    def total_send_size_by_type(self):
        return getattr(self.send_thread, "total_send_size_by_type", {})

    @property
    def total_receive_size_by_type(self):
        return getattr(self.receive_thread, "total_receive_size_by_type", {})

    @property
    def total_send_size_by_category(self):
        return getattr(self.send_thread, "total_send_size_by_category", {})

    @property
    def total_receive_size_by_category(self):
        return getattr(self.receive_thread, "total_receive_size_by_category", {})

    def send_message(self, msg: Message, priority=100):
        msg.add_params(Message.MSG_ARG_KEY_RECEIVE_PRIORITY, priority)
        self.q_sender.put(msg)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self.is_running = True
        while self.is_running:
            try:
                # Block briefly to avoid busy-waiting. This also lets us drain bursts
                # of messages quickly (e.g., during validation), so shutdown messages
                # like "protocol finished" aren't delayed behind a large backlog.
                msg_params = self.q_receiver.get(timeout=0.1)
            except queue.Empty:
                continue
            self.notify(msg_params)

    def flush_sends(self, timeout: float = 30.0):
        if not self.send_thread:
            return

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.q_sender.empty():
                return
            if not self.send_thread.is_alive():
                return
            time.sleep(0.01)

    def wait_for_message(self):
        while True:
            try:
                return self.q_receiver.get(timeout=0.1)
            except queue.Empty:
                continue

    def stop_receive_message(self):
        self.is_running = False
        # Let outbound messages drain before stopping threads. This is especially
        # important at shutdown: the last client enqueues `validation_over` and
        # `finish` right before calling `finish()`. If we stop/kill the send thread
        # too early, the server may never receive the termination signals and the
        # whole MPI job can hang.
        self.flush_sends(timeout=30.0)

        self.__stop_thread(self.send_thread)
        self.__stop_thread(self.receive_thread)
        self.__stop_thread(self.collective_thread)

    def notify(self, msg_params):
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def __stop_thread(self, thread):
        if thread:
            stop_fn = getattr(thread, "stop", None)
            if callable(stop_fn):
                stop_fn()
            # Prefer a graceful stop first. Only force an async exception if the
            # thread refuses to stop in time.
            thread.join(timeout=2)
            if thread.is_alive():
                try:
                    thread.raise_exception()
                except Exception:
                    pass
                thread.join(timeout=2)


class MessageManager(Observer):
    def __init__(
        self,
        args,
        node_type: str,
        comm,
        rank: int,
        size: int,
        backend: str = "MPI",
    ):
        self.args = args
        self.node_type = node_type
        self.comm = comm
        self.rank = rank
        self.size = size
        self.backend = backend

        self._message_handlers: Dict[int, Callable] = {}
        self._is_finished = False

        # Ensure comm breakdown is written into the experiment log file.
        try:
            from runtime.log import Log

            self._comm_log = Log("CommBreakdown", args)
        except Exception:
            self._comm_log = None

        if backend != "MPI":
            raise ValueError(f"Unsupported backend: {backend}")

        # Minimal per-msg_type accounting at the single choke points where all
        # messages pass through (send_message/receive_message).
        self.bytes_sent_by_type = defaultdict(int)
        self.bytes_received_by_type = defaultdict(int)

        self.com_manager = MpiCommunicationManager(comm, rank, size, node_type=node_type)
        self.com_manager.add_observer(self)

    def _safe_msg_type_and_size(self, message) -> Tuple[Optional[int], int]:
        """Best-effort extraction of (msg_type, size_bytes) without raising."""
        try:
            msg_type = message.get_type() if hasattr(message, "get_type") else None
        except Exception:
            msg_type = None
        try:
            size = int(message.get_size()) if hasattr(message, "get_size") else 0
        except Exception:
            size = 0
        return msg_type, size

    def register_message_receive_handlers(self):
        return

    def register_message_receive_handler(self, msg_type: int, handler: Callable):
        self._message_handlers[msg_type] = handler

    def receive_message(self, msg_type, msg_params):
        # Per-msg_type byte accounting (receive side) at the central dispatch point.
        try:
            _msg_type, size = self._safe_msg_type_and_size(msg_params)
            self.bytes_received_by_type[msg_type] += int(size)
        except Exception:
            pass

        handler = self._message_handlers.get(msg_type)
        if handler is None:
            logging.debug("No handler for msg_type=%s on rank=%s", msg_type, self.rank)
            return
        handler(msg_params)

    def send_message(self, message, priority: Optional[int] = None):
        # Per-msg_type byte accounting (send side) at the central send entrypoint.
        try:
            msg_type, size = self._safe_msg_type_and_size(message)
            if msg_type is not None:
                self.bytes_sent_by_type[msg_type] += int(size)
        except Exception:
            pass

        if priority is None:
            self.com_manager.send_message(message)
        else:
            self.com_manager.send_message(message, priority=priority)

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def finish(self):
        if self._is_finished:
            return
        self._is_finished = True

        # Log a final breakdown before shutting down threads.
        try:
            self.com_manager.flush_sends(timeout=5.0)

            def _sorted_nonzero(d):
                try:
                    items = [(k, int(v)) for k, v in dict(d).items() if int(v) != 0]
                except Exception:
                    return {}
                items.sort(key=lambda kv: (-kv[1], str(kv[0])))
                return {k: v for k, v in items}

            send_by_cat = _sorted_nonzero(getattr(self.com_manager, "total_send_size_by_category", {}))
            recv_by_cat = _sorted_nonzero(getattr(self.com_manager, "total_receive_size_by_category", {}))
            send_by_type = _sorted_nonzero(getattr(self.com_manager, "total_send_size_by_type", {}))
            recv_by_type = _sorted_nonzero(getattr(self.com_manager, "total_receive_size_by_type", {}))
            mm_send_by_type = _sorted_nonzero(getattr(self, "bytes_sent_by_type", {}))
            mm_recv_by_type = _sorted_nonzero(getattr(self, "bytes_received_by_type", {}))
            total_send = int(getattr(self.com_manager, "total_send_size", 0))
            total_recv = int(getattr(self.com_manager, "total_receive_size", 0))

            if self._comm_log is not None:
                self._comm_log.info(
                    "rank={} node_type={} total_send={} total_receive={} send_by_category={} recv_by_category={} send_by_type={} recv_by_type={} mm_send_by_type={} mm_recv_by_type={}".format(
                        self.rank,
                        self.node_type,
                        total_send,
                        total_recv,
                        send_by_cat,
                        recv_by_cat,
                        send_by_type,
                        recv_by_type,
                        mm_send_by_type,
                        mm_recv_by_type,
                    )
                )
        except Exception:
            pass
        self.com_manager.stop_receive_message()
