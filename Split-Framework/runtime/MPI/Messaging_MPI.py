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
import math
import queue
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from mpi4py import MPI
from collections import defaultdict


def _get_bool_arg(args, key: str, default: bool) -> bool:
    try:
        if isinstance(args, dict):
            return bool(args.get(key, default))
    except Exception:
        pass
    # Many parts of this framework pass a dict-like Config object that supports
    # `args["key"]` for both declared fields and extra YAML keys.
    try:
        if hasattr(args, "__getitem__"):
            v = args[key]
            if v is not None:
                return bool(v)
    except Exception:
        pass
    try:
        return bool(getattr(args, key, default))
    except Exception:
        return bool(default)


def _get_float_arg(args, key: str, default):
    """Best-effort float arg reader.

    Supports dict-like Config objects and plain objects.
    If `default` is None, returns None when the key is missing/unparseable.
    """

    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    try:
        if isinstance(args, dict) and key in args:
            out = _to_float(args.get(key))
            return out if out is not None else default
    except Exception:
        pass

    try:
        if hasattr(args, "__getitem__"):
            out = _to_float(args[key])
            return out if out is not None else default
    except Exception:
        pass

    try:
        out = _to_float(getattr(args, key, default))
        return out if out is not None else default
    except Exception:
        return default


def _get_str_arg(args, key: str, default: str) -> str:
    try:
        if isinstance(args, dict) and key in args:
            v = args[key]
            return default if v is None else str(v)
    except Exception:
        pass
    try:
        if hasattr(args, "__getitem__"):
            v = args[key]
            return default if v is None else str(v)
    except Exception:
        pass
    try:
        v = getattr(args, key, default)
        return default if v is None else str(v)
    except Exception:
        return str(default)


class ComposedCodec:
    def __init__(self, steps):
        self.steps = list(steps)

    def _encode_step(self, current, codec):
        if isinstance(current, torch.Tensor):
            return codec.encode(current)
        if isinstance(current, dict) and "q" in current:
            next_payload = dict(current)
            next_payload["q"] = self._encode_step(current["q"], codec)
            return next_payload
        raise TypeError(f"Cannot apply {type(codec).__name__} to payload type {type(current)!r}")

    def encode(self, x: torch.Tensor):
        current = x
        for codec in self.steps:
            current = self._encode_step(current, codec)
        return {"codec": "composed", "payload": current}

    def _decode_step(self, current, codec, *, device, dtype):
        if isinstance(current, dict):
            try:
                return codec.decode(current, device=device, dtype=dtype)
            except Exception:
                if "q" in current and isinstance(current["q"], dict):
                    next_payload = dict(current)
                    next_payload["q"] = self._decode_step(current["q"], codec, device=device, dtype=dtype)
                    return next_payload
        return current

    def decode(self, payload, *, device=None, dtype=torch.float32):
        if not isinstance(payload, dict) or "payload" not in payload:
            raise TypeError("ComposedCodec.decode expects a dict payload with 'payload'")
        current = payload["payload"]
        for codec in reversed(self.steps):
            current = self._decode_step(current, codec, device=device, dtype=dtype)
        return current


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
        self.total_send_time = 0.0
        self.tmp_send_time = 0.0
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
                    send_started = time.perf_counter()
                    self.comm.send(msg_str, dest=dest_id)
                    send_elapsed = time.perf_counter() - send_started
                    self.tmp_send_time += send_elapsed
                    self.total_send_time += send_elapsed
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
        self.total_receive_time = 0.0
        self.tmp_receive_time = 0.0
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

                recv_started = time.perf_counter()
                msg_str = self.comm.recv(source=status.Get_source(), tag=status.Get_tag())
                recv_elapsed = time.perf_counter() - recv_started

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
                self.tmp_receive_time += recv_elapsed
                self.total_receive_time += recv_elapsed
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
        self.send_thread.tmp_send_time = 0.0
        self.receive_thread.tmp_receive_time = 0.0
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
    def tmp_send_time(self) -> float:
        return float(getattr(self.send_thread, "tmp_send_time", 0.0))

    @property
    def tmp_receive_time(self) -> float:
        return float(getattr(self.receive_thread, "tmp_receive_time", 0.0))

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
            from runtime.exports.log import Log

            self._comm_log = Log("CommBreakdown", args)
        except Exception:
            self._comm_log = None

        if backend != "MPI":
            raise ValueError(f"Unsupported backend: {backend}")

        # Minimal per-msg_type accounting at the single choke points where all
        # messages pass through (send_message/receive_message).
        self.bytes_sent_by_type = defaultdict(int)
        self.bytes_received_by_type = defaultdict(int)
        self._comm_analysis_key = "_comm_analysis"
        self._epoch_comm_stats = defaultdict(float)
        self._tensor_distribution_logging = _get_bool_arg(args, "tensor_distribution_logging", True)
        self._tensor_distribution_log_validation = _get_bool_arg(args, "tensor_distribution_log_validation", False)
        self._tensor_distribution_logged = set()
        total_epochs = _get_float_arg(args, "epochs", 0)
        self._tensor_distribution_samples = self._build_tensor_distribution_samples(int(total_epochs or 0))
        # --- Communication compression hooks (forward/backward quantization) ---
        # Backward-compat: `quantize_activations` previously controlled forward-only.
        quantize_acts_legacy = _get_bool_arg(args, "quantize_activations", False)

        # New switches:
        # - quantize_forward: quantize messages carrying "activations"
        # - quantize_backward: quantize messages carrying "activation_grads"
        self._quantize_forward = _get_bool_arg(args, "quantize_forward", quantize_acts_legacy)
        self._quantize_backward = _get_bool_arg(args, "quantize_backward", False)

        self._forward_quantization = _get_str_arg(args, "forward_quantization", "int").strip().lower()
        self._backward_quantization = _get_str_arg(args, "backward_quantization", "int").strip().lower()
        quantization_bits = int(round(_get_float_arg(args, "quantization_bits", 8.0) or 8.0))
        if quantization_bits not in (2, 3, 4, 6, 8, 16, 32):
            quantization_bits = 8
        quantization_granularity = _get_str_arg(args, "quantization_granularity", "per_tensor").strip().lower().replace("-", "_")
        sparsity_k = _get_float_arg(args, "sparsity_k", None)
        dimensionality_reduction_ratio = _get_float_arg(args, "dimensionality_reduction_ratio", 0.25)

        def _normalize_sparsity_percent(value: Optional[float], default: int) -> int:
            if value is None:
                return int(default)
            numeric = float(value)
            if 0.0 < numeric < 1.0:
                numeric *= 100.0
            normalized = int(round(numeric))
            if normalized not in (1, 5, 10, 25, 50):
                return int(default)
            return normalized

        def _normalize_dimensionality_ratio(value: Optional[float], default: float) -> float:
            if value is None:
                return float(default)
            numeric = float(value)
            if numeric > 1.0:
                numeric /= 100.0
            if numeric <= 0.0 or numeric > 1.0:
                return float(default)
            return float(numeric)
        quantization_group_size = int(round(_get_float_arg(args, "quantization_group_size", 32.0) or 32.0))
        if quantization_group_size <= 0:
            quantization_group_size = 32

        self._fwd_codec = None
        self._bwd_codec = None
        self._probed_forward_epoch: Optional[int] = None
        self._probed_backward_epoch: Optional[int] = None

        def _granularity_for_kind(kind: str) -> str:
            if kind in (
                "dynamic_symmetric_int8_per_channel",
                "dynamic_int8_per_channel",
                "int8_per_channel",
                "per_channel_int8",
                "uniform_per_channel_codebook_uint8",
                "uniform_per_channel_uint8",
                "codeword_uniform_per_channel_uint8",
                "codeword_uniform_per_channel",
                "uniform_per_channel",
                "non_uniform_loyd_per_channel_codebook_uint8",
                "non_uniform_loyd_per_channel_uint8",
                "codeword_non_uniform_loyd_per_channel",
                "non_uniform_loyd_per_channel",
                "loyd_per_channel",
                "mulaw_per_channel_codebook_uint8",
                "mulaw_per_channel_uint8",
                "codeword_mulaw_per_channel",
                "non_uniform_mlaw_per_channel",
                "mu_law_per_channel",
                "mlaw_per_channel",
            ):
                return "per_channel"
            return quantization_granularity

        def _build_single_codec(kind: str, *, scale_key_prefix: str, bits_override: Optional[int] = None):
            if kind in ("none", "off", "false", "0"):
                return None
            codec_bits = quantization_bits if bits_override is None else int(bits_override)
            codec_granularity = _granularity_for_kind(kind)
            reduction_storage_bits = 32
            # Keep imports local so MPI can still run if compression module is absent.
            from compression.quantization.arithmetic_conversion import (
                IntCodec,
                TruncationFloatCodec as ArithmeticFloatCodec,
            )
            from compression.dimensionality_reduction import (
                LowRankPCAProjectionCodec,
                RandomProjectionCodec,
            )
            from compression.comparison_papers import PaperTopKSparsityCodec, SplitFCCodec
            from compression.sparsity import RandomTopKSparsityCodec, TopKSparsityCodec
            from compression.quantization.codeword import (
                LloydMaxCodebookUInt8Codec,
                MuLawCodebookCodec,
                NonUniformLoydCodebookCodec,
                UniformCodebookCodec,
            )
            from compression.quantization.Truncation import TruncationIntCodec

            if kind in ("dynamic_symmetric_int8", "dynamic_int8", "int8", "int"):
                return IntCodec(num_bits=codec_bits, granularity=codec_granularity, group_size=quantization_group_size)
            if kind in ("random_projection", "autoencoder"):
                return RandomProjectionCodec(
                    reduction_ratio=_normalize_dimensionality_ratio(dimensionality_reduction_ratio, 0.25),
                    storage_bits=reduction_storage_bits,
                )
            if kind in ("low_rank_pca", "low_rank_projection", "pca_projection", "pca", "low_rank"):
                return LowRankPCAProjectionCodec(
                    reduction_ratio=_normalize_dimensionality_ratio(dimensionality_reduction_ratio, 0.25),
                    storage_bits=reduction_storage_bits,
                )
            if kind in ("top_k", "topk", "top_k_sparsity"):
                return TopKSparsityCodec(k_percent=_normalize_sparsity_percent(sparsity_k, 1), storage_bits=reduction_storage_bits)
            if kind in ("random_top_k", "random_topk", "random_top_k_sparsity"):
                return RandomTopKSparsityCodec(
                    k_percent=_normalize_sparsity_percent(sparsity_k, 5),
                    storage_bits=reduction_storage_bits,
                )
            if kind in ("paper_top_k", "paper_topk", "paper_top_k_sparsity"):
                return PaperTopKSparsityCodec(
                    k_percent=_normalize_sparsity_percent(sparsity_k, 5),
                    alpha=_get_float_arg(self.args, "paper_top_k_alpha", 0.1),
                    storage_bits=reduction_storage_bits,
                )
            if kind in ("split_fc", "splitfc"):
                split_fc_ratio = _get_float_arg(self.args, "split_fc_reduction_ratio", None)
                if split_fc_ratio is None:
                    split_fc_ratio = 16.0
                return SplitFCCodec(
                    reduction_ratio=float(split_fc_ratio),
                    feature_bits_per_entry=float(max(1, codec_bits)),
                    gradient_bits_per_entry=float(max(1, codec_bits)),
                    endpoint_levels=int(_get_float_arg(self.args, "split_fc_endpoint_levels", 200) or 200),
                    seed=_get_float_arg(self.args, "seed", None),
                )
            if kind in ("dynamic_symmetric_int8_per_channel", "dynamic_int8_per_channel", "int8_per_channel", "per_channel_int8"):
                return IntCodec(num_bits=codec_bits, granularity="per_channel", group_size=quantization_group_size)
            if kind in ("fixed_scale_int8", "fixed_int8"):
                fixed_scale = _get_float_arg(self.args, f"{scale_key_prefix}_truncation_scale", None)
                if fixed_scale is None:
                    fixed_scale = _get_float_arg(self.args, "truncation_scale", None)
                return IntCodec(
                    fixed_scale=fixed_scale,
                    num_bits=codec_bits,
                    granularity=codec_granularity,
                    group_size=quantization_group_size,
                )
            if kind in ("fp8_e4m3", "float8_e4m3", "e4m3", "float8", "float"):
                return ArithmeticFloatCodec(layout=("e2m1" if codec_bits == 4 else "e4m3"))
            if kind in ("uniform_codebook_uint8", "uniform_codeword_uint8", "codeword_uniform_uint8", "codeword_uniform", "uniform"):
                return UniformCodebookCodec(num_bits=codec_bits, granularity=codec_granularity, group_size=quantization_group_size)
            if kind in ("uniform_per_channel_codebook_uint8", "uniform_per_channel_uint8", "codeword_uniform_per_channel_uint8", "codeword_uniform_per_channel", "uniform_per_channel"):
                return UniformCodebookCodec(num_bits=codec_bits, granularity="per_channel", group_size=quantization_group_size)
            if kind in ("non_uniform_loyd_codebook_uint8", "non_uniform_loyd_uint8", "codeword_non_uniform_loyd", "non_uniform_loyd"):
                return NonUniformLoydCodebookCodec(num_bits=codec_bits, granularity=codec_granularity, group_size=quantization_group_size)
            if kind in ("non_uniform_loyd_per_channel_codebook_uint8", "non_uniform_loyd_per_channel_uint8", "codeword_non_uniform_loyd_per_channel", "non_uniform_loyd_per_channel", "loyd_per_channel"):
                return NonUniformLoydCodebookCodec(num_bits=codec_bits, granularity="per_channel", group_size=quantization_group_size)
            if kind in ("mulaw_codebook_uint8", "mulaw_non_uniform_uint8", "codeword_non_uniform", "codeword_mulaw", "non_uniform_mlaw"):
                return MuLawCodebookCodec(num_bits=codec_bits, granularity=codec_granularity, group_size=quantization_group_size)
            if kind in ("mulaw_per_channel_codebook_uint8", "mulaw_per_channel_uint8", "codeword_mulaw_per_channel", "non_uniform_mlaw_per_channel", "mu_law_per_channel", "mlaw_per_channel"):
                return MuLawCodebookCodec(num_bits=codec_bits, granularity="per_channel", group_size=quantization_group_size)
            if kind in ("lloyd_max_codebook_uint8", "lloyd_max_uint8", "codeword_lloyd_max", "non_uniform_lloyd_uint8", "lloyd_max"):
                return LloydMaxCodebookUInt8Codec(num_bits=codec_bits)
            if kind in ("trunc_noscale_int", "trunc_noscale_int8", "trunc_bits_int8", "trunc_scale_int8", "truncation_int", "truncation_int8"):
                return TruncationIntCodec(num_bits=codec_bits, granularity=codec_granularity, group_size=quantization_group_size)
            raise ValueError(f"Unsupported quantization kind: {kind!r}")

        def _build_codec(kind: str, *, scale_key_prefix: str, bits_override: Optional[int] = None):
            parts = [part.strip() for part in str(kind or "").strip().lower().split("+") if part.strip()]
            if not parts:
                return None
            if len(parts) == 1:
                return _build_single_codec(parts[0], scale_key_prefix=scale_key_prefix, bits_override=bits_override)
            return ComposedCodec([
                _build_single_codec(part, scale_key_prefix=scale_key_prefix, bits_override=bits_override)
                for part in parts
            ])


        try:
            if self._quantize_forward:
                self._fwd_codec = _build_codec(self._forward_quantization, scale_key_prefix="forward", bits_override=quantization_bits)
                if self._fwd_codec is None:
                    raise ValueError("Forward quantization was requested but no forward codec was created.")
            if self._quantize_backward:
                shared_sparse_kind = str(self._forward_quantization or "").strip().lower() in {"paper_top_k", "paper_topk", "paper_top_k_sparsity", "split_fc", "splitfc"}
                if self._quantize_forward and shared_sparse_kind and str(self._backward_quantization or "").strip().lower() == str(self._forward_quantization or "").strip().lower():
                    self._bwd_codec = self._fwd_codec
                else:
                    self._bwd_codec = _build_codec(self._backward_quantization, scale_key_prefix="backward", bits_override=quantization_bits)
                if self._bwd_codec is None:
                    raise ValueError("Backward quantization was requested but no backward codec was created.")
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize requested quantization codecs. "
                f"forward_enabled={self._quantize_forward}, forward_kind={self._forward_quantization!r}, "
                f"backward_enabled={self._quantize_backward}, backward_kind={self._backward_quantization!r}, "
                f"quantization_bits={quantization_bits}"
            ) from exc

        logging.info(
            "Quantization initialized: forward_enabled=%s forward_codec=%s backward_enabled=%s backward_codec=%s bits=%s",
            self._quantize_forward,
            (type(self._fwd_codec).__name__ if self._fwd_codec is not None else None),
            self._quantize_backward,
            (type(self._bwd_codec).__name__ if self._bwd_codec is not None else None),
            quantization_bits,
        )

        self.com_manager = MpiCommunicationManager(comm, rank, size, node_type=node_type)
        self.com_manager.add_observer(self)

    def _build_tensor_distribution_samples(self, total_epochs: int) -> Dict[int, str]:
        if total_epochs <= 0:
            return {}

        merged = defaultdict(list)
        for epoch_idx, label in (
            (0, "early"),
            (max(0, total_epochs // 2), "middle"),
            (max(0, total_epochs - 1), "late"),
        ):
            merged[int(epoch_idx)].append(label)
        return {epoch_idx: "/".join(labels) for epoch_idx, labels in merged.items()}

    def annotate_tensor_distribution_message(self, message, trainer, *, phase=None, epoch=None):
        if phase is None:
            phase = getattr(trainer, "phase", None)
        if epoch is None:
            epoch = getattr(trainer, "epoch_count", None)
        if epoch is None:
            epoch = getattr(trainer, "epoch", None)
        if phase is not None:
            message.add_params("_tensor_dist_phase", str(phase))
        if epoch is not None:
            try:
                message.add_params("_tensor_dist_epoch", int(epoch))
            except Exception:
                pass
        if not self._tensor_distribution_logging:
            return

    def _collect_logged_tensors(self, obj, prefix: str):
        items = []
        if isinstance(obj, torch.Tensor):
            return [(prefix, obj)]
        if isinstance(obj, tuple):
            if prefix == "activations" and len(obj) == 2:
                return self._collect_logged_tensors(obj[0], f"{prefix}[0]")
            for idx, value in enumerate(obj):
                items.extend(self._collect_logged_tensors(value, f"{prefix}[{idx}]"))
            return items
        if isinstance(obj, list):
            if prefix == "activations" and len(obj) == 2:
                return self._collect_logged_tensors(obj[0], f"{prefix}[0]")
            for idx, value in enumerate(obj):
                items.extend(self._collect_logged_tensors(value, f"{prefix}[{idx}]"))
            return items
        if isinstance(obj, dict):
            if self._is_quant_payload(obj):
                return []
            for key, value in obj.items():
                items.extend(self._collect_logged_tensors(value, f"{prefix}.{key}"))
        return items

    def _estimate_nbytes(self, obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, torch.Tensor):
            try:
                return int(obj.nelement() * obj.element_size())
            except Exception:
                try:
                    return int(obj.numel() * obj.element_size())
                except Exception:
                    return 0
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return len(obj)
        if isinstance(obj, str):
            return len(obj.encode("utf-8", errors="replace"))
        if isinstance(obj, (int, float, bool)):
            return sys.getsizeof(obj)
        if isinstance(obj, dict):
            return sum(self._estimate_nbytes(key) + self._estimate_nbytes(value) for key, value in obj.items())
        if isinstance(obj, (list, tuple, set)):
            return sum(self._estimate_nbytes(value) for value in obj)
        try:
            return sys.getsizeof(obj)
        except Exception:
            return sys.getsizeof(str(obj))

    def _summarize_tensor_payload_bytes(self, obj, *, preserve_second_if_pair: bool):
        raw_bytes = 0
        quantized_bytes = 0
        metadata_bytes = 0

        if obj is None:
            return raw_bytes, quantized_bytes, metadata_bytes
        if isinstance(obj, torch.Tensor):
            raw_bytes = self._estimate_nbytes(obj)
            return raw_bytes, raw_bytes, 0
        if isinstance(obj, tuple):
            values = (obj[0],) if preserve_second_if_pair and len(obj) == 2 else obj
            for value in values:
                child_raw, child_quantized, child_metadata = self._summarize_tensor_payload_bytes(
                    value,
                    preserve_second_if_pair=False,
                )
                raw_bytes += child_raw
                quantized_bytes += child_quantized
                metadata_bytes += child_metadata
            return raw_bytes, quantized_bytes, metadata_bytes
        if isinstance(obj, list):
            values = [obj[0]] if preserve_second_if_pair and len(obj) == 2 else obj
            for value in values:
                child_raw, child_quantized, child_metadata = self._summarize_tensor_payload_bytes(
                    value,
                    preserve_second_if_pair=False,
                )
                raw_bytes += child_raw
                quantized_bytes += child_quantized
                metadata_bytes += child_metadata
            return raw_bytes, quantized_bytes, metadata_bytes
        if isinstance(obj, dict):
            if self._is_quant_payload(obj):
                payload_key = "payload" if obj.get("codec") == "composed" else "q"
                quantized_bytes = self._estimate_nbytes(obj.get(payload_key))
                metadata_bytes = sum(
                    self._estimate_nbytes(key) + self._estimate_nbytes(value)
                    for key, value in obj.items()
                    if key != payload_key
                )
                return 0, quantized_bytes, metadata_bytes
            for value in obj.values():
                child_raw, child_quantized, child_metadata = self._summarize_tensor_payload_bytes(
                    value,
                    preserve_second_if_pair=False,
                )
                raw_bytes += child_raw
                quantized_bytes += child_quantized
                metadata_bytes += child_metadata
            return raw_bytes, quantized_bytes, metadata_bytes
        return 0, 0, 0

    def _record_epoch_comm_stats(self, tensor_key: str, *, raw_bytes: int, quantized_bytes: int, metadata_bytes: int, quant_time: float):
        prefix = "acts" if tensor_key == "activations" else "grads"
        self._epoch_comm_stats[f"{prefix}_raw_bytes"] += int(raw_bytes)
        self._epoch_comm_stats[f"{prefix}_quantized_bytes"] += int(quantized_bytes)
        self._epoch_comm_stats[f"{prefix}_metadata_bytes"] += int(metadata_bytes)
        self._epoch_comm_stats[f"{prefix}_quant_time"] += float(quant_time)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 1.0 if numerator <= 0 else 0.0
        return float(numerator) / float(denominator)

    def _consume_comm_analysis(self, params: dict, tensor_key: str) -> bool:
        analysis = params.get(self._comm_analysis_key)
        if not isinstance(analysis, dict):
            return False
        stats = analysis.get(tensor_key)
        if not isinstance(stats, dict):
            return False
        self._record_epoch_comm_stats(
            tensor_key,
            raw_bytes=int(stats.get("raw_bytes", 0) or 0),
            quantized_bytes=int(stats.get("quantized_bytes", 0) or 0),
            metadata_bytes=int(stats.get("metadata_bytes", 0) or 0),
            quant_time=float(stats.get("quant_time", 0.0) or 0.0),
        )
        return True

    def get_epoch_comm_summary(self):
        acts_raw_bytes = int(self._epoch_comm_stats.get("acts_raw_bytes", 0))
        acts_quantized_bytes = int(self._epoch_comm_stats.get("acts_quantized_bytes", 0))
        acts_metadata_bytes = int(self._epoch_comm_stats.get("acts_metadata_bytes", 0))
        grads_raw_bytes = int(self._epoch_comm_stats.get("grads_raw_bytes", 0))
        grads_quantized_bytes = int(self._epoch_comm_stats.get("grads_quantized_bytes", 0))
        grads_metadata_bytes = int(self._epoch_comm_stats.get("grads_metadata_bytes", 0))
        acts_total_encoded = acts_quantized_bytes + acts_metadata_bytes
        grads_total_encoded = grads_quantized_bytes + grads_metadata_bytes
        total_raw_bytes = acts_raw_bytes + grads_raw_bytes
        total_encoded_bytes = acts_total_encoded + grads_total_encoded
        return {
            "raw_acts_bytes": acts_raw_bytes,
            "quantized_acts_bytes": acts_quantized_bytes,
            "acts_metadata_bytes": acts_metadata_bytes,
            "raw_grads_bytes": grads_raw_bytes,
            "quantized_grads_bytes": grads_quantized_bytes,
            "grads_metadata_bytes": grads_metadata_bytes,
            "acts_compression": self._safe_ratio(acts_raw_bytes, acts_total_encoded),
            "grads_compression": self._safe_ratio(grads_raw_bytes, grads_total_encoded),
            "total_compression": self._safe_ratio(total_raw_bytes, total_encoded_bytes),
            "acts_quant_time": float(self._epoch_comm_stats.get("acts_quant_time", 0.0)),
            "grads_quant_time": float(self._epoch_comm_stats.get("grads_quant_time", 0.0)),
            "send_time": float(getattr(self.com_manager, "tmp_send_time", 0.0)),
            "recv_time": float(getattr(self.com_manager, "tmp_receive_time", 0.0)),
        }

    def log_epoch_summary(self, *, epoch: int, train_acc: float, train_loss: float, val_acc: float, val_loss: float, epoch_time: float):
        if self._comm_log is not None:
            summary = self.get_epoch_comm_summary()
            self._comm_log.info(
                "epoch_summary rank={} node_type={} epoch={} train_acc={} train_loss={} val_acc={} val_loss={} raw_acts_bytes={} quantized_acts_bytes={} acts_metadata_bytes={} raw_grads_bytes={} quantized_grads_bytes={} grads_metadata_bytes={} acts_compression={} grads_compression={} total_compression={} acts_quant_time={} grads_quant_time={} send_time={} recv_time={} epoch_time={}".format(
                    self.rank,
                    self.node_type,
                    int(epoch),
                    format(float(train_acc), ".6g"),
                    format(float(train_loss), ".6g"),
                    format(float(val_acc), ".6g"),
                    format(float(val_loss), ".6g"),
                    summary["raw_acts_bytes"],
                    summary["quantized_acts_bytes"],
                    summary["acts_metadata_bytes"],
                    summary["raw_grads_bytes"],
                    summary["quantized_grads_bytes"],
                    summary["grads_metadata_bytes"],
                    format(summary["acts_compression"], ".6g"),
                    format(summary["grads_compression"], ".6g"),
                    format(summary["total_compression"], ".6g"),
                    format(summary["acts_quant_time"], ".6g"),
                    format(summary["grads_quant_time"], ".6g"),
                    format(summary["send_time"], ".6g"),
                    format(summary["recv_time"], ".6g"),
                    format(float(epoch_time), ".6g"),
                )
            )
        self._epoch_comm_stats = defaultdict(float)
        self.com_manager.reset_analysis_data()

    def _bucketize_distribution_metric(self, value: float, low_cutoff: float, high_cutoff: float) -> str:
        if value < low_cutoff:
            return "low"
        if value < high_cutoff:
            return "medium"
        return "high"

    def _summarize_tensor_distribution(self, tensor: torch.Tensor):
        try:
            with torch.no_grad():
                flat = tensor.detach()
                if not flat.dtype.is_floating_point:
                    flat = flat.float()
                flat = flat.reshape(-1).to(device="cpu", dtype=torch.float32)
                if flat.numel() == 0:
                    return None
                flat_abs = flat.abs()
                mean = float(flat.mean().item())
                std = float(flat.std(unbiased=False).item())
                min_value = float(flat.min().item())
                max_value = float(flat.max().item())
                mean_abs = float(flat_abs.mean().item())
                max_abs = float(flat_abs.max().item())
                q90 = float(torch.quantile(flat_abs, 0.90).item())
                p95_abs = float(torch.quantile(flat_abs, 0.95).item())
                p99_abs = float(torch.quantile(flat_abs, 0.99).item())
                spread_score = std / (mean_abs + 1e-12)
                outlier_score = p99_abs / (q90 + 1e-12)
                near_zero_threshold = max(1e-8, 0.1 * (mean_abs + 1e-12))
                near_zero_fraction = float((flat_abs <= near_zero_threshold).float().mean().item())

                centered = flat - mean
                centered2 = centered * centered
                centered3 = centered2 * centered
                centered4 = centered2 * centered2
                skewness = float(centered3.mean().item() / ((std ** 3) + 1e-12))
                kurtosis = float(centered4.mean().item() / ((std ** 4) + 1e-12))

                channel_tensor = tensor.detach()
                if not channel_tensor.dtype.is_floating_point:
                    channel_tensor = channel_tensor.float()
                channel_tensor = channel_tensor.to(device="cpu", dtype=torch.float32)
                if channel_tensor.ndim == 0:
                    channel_flat = channel_tensor.reshape(1, 1)
                else:
                    channel_axis = 0 if channel_tensor.ndim < 2 else 1
                    perm = (channel_axis, *[idx for idx in range(channel_tensor.ndim) if idx != channel_axis])
                    channel_flat = channel_tensor.permute(perm).contiguous().reshape(channel_tensor.shape[channel_axis], -1)
                channel_std = channel_flat.std(dim=1, unbiased=False)
                channel_scale = channel_flat.abs().max(dim=1).values
                mean_channel_std = float(channel_std.mean().item())
                std_channel_std = float(channel_std.std(unbiased=False).item())
                min_channel_scale = float(channel_scale.min().item()) if channel_scale.numel() > 0 else 0.0
                max_channel_scale = float(channel_scale.max().item()) if channel_scale.numel() > 0 else 0.0
                max_channel_scale_over_min_channel_scale = max_channel_scale / (min_channel_scale + 1e-12)
                return {
                    "shape": tuple(tensor.shape),
                    "mean": mean,
                    "std": std,
                    "min": min_value,
                    "max": max_value,
                    "mean_abs": mean_abs,
                    "max_abs": max_abs,
                    "p95_abs": p95_abs,
                    "p99_abs": p99_abs,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "mean_channel_std": mean_channel_std,
                    "std_channel_std": std_channel_std,
                    "max_channel_scale_over_min_channel_scale": max_channel_scale_over_min_channel_scale,
                    "spread_score": spread_score,
                    "spread_bucket": self._bucketize_distribution_metric(spread_score, 0.5, 1.25),
                    "outlier_score": outlier_score,
                    "outlier_bucket": self._bucketize_distribution_metric(outlier_score, 1.5, 3.0),
                    "near_zero_fraction": near_zero_fraction,
                    "near_zero_bucket": self._bucketize_distribution_metric(near_zero_fraction, 0.2, 0.6),
                }
        except Exception:
            return None

    def _maybe_log_tensor_distribution(self, message) -> None:
        if not self._tensor_distribution_logging or self._comm_log is None:
            return
        try:
            params = message.get_params() if hasattr(message, "get_params") else None
        except Exception:
            params = None
        if not isinstance(params, dict):
            return

        phase = str(params.get("_tensor_dist_phase", "")).strip().lower()
        if phase != "train" and not self._tensor_distribution_log_validation:
            return
        epoch = params.get("_tensor_dist_epoch")
        try:
            epoch = int(epoch)
        except Exception:
            return

        sample_label = self._tensor_distribution_samples.get(epoch)
        if not sample_label:
            return

        for key in ("activations", "activation_grads"):
            if key not in params:
                continue
            for tensor_name, tensor in self._collect_logged_tensors(params[key], key):
                summary = self._summarize_tensor_distribution(tensor)
                if summary is None:
                    continue
                dedupe_key = (self.rank, self.node_type, phase, epoch, sample_label, tensor_name)
                if dedupe_key in self._tensor_distribution_logged:
                    continue
                self._tensor_distribution_logged.add(dedupe_key)
                self._comm_log.info(
                    "tensor_distribution rank={} node_type={} phase={} epoch={} sample={} tensor={} shape={} mean={} std={} min={} max={} mean_abs={} max_abs={} p95_abs={} p99_abs={} skewness={} kurtosis={} mean_channel_std={} std_channel_std={} max_channel_scale_over_min_channel_scale={} spread={} spread_score={} outliers={} outlier_score={} near_zero={} near_zero_fraction={}".format(
                        self.rank,
                        self.node_type,
                        phase,
                        epoch,
                        sample_label,
                        tensor_name,
                        summary["shape"],
                        format(summary["mean"], ".6g"),
                        format(summary["std"], ".6g"),
                        format(summary["min"], ".6g"),
                        format(summary["max"], ".6g"),
                        format(summary["mean_abs"], ".6g"),
                        format(summary["max_abs"], ".6g"),
                        format(summary["p95_abs"], ".6g"),
                        format(summary["p99_abs"], ".6g"),
                        format(summary["skewness"], ".6g"),
                        format(summary["kurtosis"], ".6g"),
                        format(summary["mean_channel_std"], ".6g"),
                        format(summary["std_channel_std"], ".6g"),
                        format(summary["max_channel_scale_over_min_channel_scale"], ".6g"),
                        summary["spread_bucket"],
                        format(summary["spread_score"], ".6g"),
                        summary["outlier_bucket"],
                        format(summary["outlier_score"], ".6g"),
                        summary["near_zero_bucket"],
                        format(summary["near_zero_fraction"], ".6g"),
                    )
                )

    def _is_quant_payload(self, obj) -> bool:
        if not isinstance(obj, dict):
            return False
        # Prefer explicit codec marker, but also accept minimal required keys.
        if obj.get("codec") in (
            "fp8_e4m3",
            "fp4_e2m1",
            "dynamic_symmetric_int",
            "dynamic_symmetric_int8",
            "dynamic_symmetric_int8_per_channel",
            "uniform_codebook",
            "uniform_codebook_uint8",
            "uniform_per_channel_codebook_uint8",
            "non_uniform_loyd_codebook",
            "non_uniform_loyd_codebook_uint8",
            "mulaw_codebook",
            "mulaw_codebook_uint8",
            "non_uniform_loyd_per_channel_codebook_uint8",
            "mulaw_per_channel_codebook_uint8",
            "lloyd_max_codebook_uint8",
            "trunc_noscale_int",
            "trunc_noscale_int8",
            "trunc_bits_int8",
            "trunc_scale_int8",
            "minmax_affine_uint8",
            "uniform_asymmetric_uint8",
            "top_k_sparsity",
            "random_top_k_sparsity",
            "paper_top_k_sparsity",
            "split_fc",
            "autoencoder",
            "random_projection",
            "low_rank_pca_projection",
            "composed",
        ):
            return True
        if obj.get("codec") == "composed" and "payload" in obj:
            return True
        if "q" in obj and "shape" in obj and "indices" in obj:
            return True
        if "q" in obj and "shape" in obj and ("basis" in obj or "mean" in obj or "block_size" in obj):
            return True
        if "q" in obj and "shape" in obj and ("scale" in obj or "codebook" in obj or "layout" in obj):
            return True
        return "q" in obj and "scale" in obj and "zero_point" in obj

    def _encode_tensor_obj(self, obj, codec, *, preserve_second_if_pair: bool):
        if codec is None:
            return obj

        if isinstance(obj, torch.Tensor):
            # Only quantize floating-point tensors (activations). Labels are typically int64.
            if obj.dtype is not None and obj.dtype.is_floating_point:
                try:
                    return codec.encode(obj)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to encode tensor with codec {type(codec).__name__} for dtype={obj.dtype} shape={tuple(obj.shape)}"
                    ) from exc
            return obj

        # Common pattern across algorithms: (acts, labels) stored under the
        # "activations" key. When requested, quantize acts only; never touch labels.
        if isinstance(obj, tuple):
            if preserve_second_if_pair and len(obj) == 2:
                return (self._encode_tensor_obj(obj[0], codec, preserve_second_if_pair=preserve_second_if_pair), obj[1])
            return tuple(self._encode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for v in obj)
        if isinstance(obj, list):
            if preserve_second_if_pair and len(obj) == 2:
                return [self._encode_tensor_obj(obj[0], codec, preserve_second_if_pair=preserve_second_if_pair), obj[1]]
            return [self._encode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for v in obj]
        if isinstance(obj, dict):
            # Avoid double-encoding if someone already sent a payload.
            if self._is_quant_payload(obj):
                return obj
            return {k: self._encode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for k, v in obj.items()}
        return obj

    def _decode_tensor_obj(self, obj, codec, *, preserve_second_if_pair: bool):
        if codec is None:
            return obj

        if self._is_quant_payload(obj):
            try:
                # Decode onto the receiver's configured device when possible.
                # Falls back to CPU if args['device'] is missing/unavailable.
                dev = None
                try:
                    dev = self.args["device"]
                except Exception:
                    dev = None
                return codec.decode(obj, device=(dev if dev is not None else "cpu"), dtype=torch.float32)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to decode payload with codec {type(codec).__name__}"
                ) from exc

        if isinstance(obj, tuple):
            if preserve_second_if_pair and len(obj) == 2:
                return (self._decode_tensor_obj(obj[0], codec, preserve_second_if_pair=preserve_second_if_pair), obj[1])
            return tuple(self._decode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for v in obj)
        if isinstance(obj, list):
            if preserve_second_if_pair and len(obj) == 2:
                return [self._decode_tensor_obj(obj[0], codec, preserve_second_if_pair=preserve_second_if_pair), obj[1]]
            return [self._decode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for v in obj]
        if isinstance(obj, dict):
            return {k: self._decode_tensor_obj(v, codec, preserve_second_if_pair=preserve_second_if_pair) for k, v in obj.items()}
        return obj

    def _move_tensor_obj(self, obj, device):
        if isinstance(obj, torch.Tensor):
            try:
                return obj.to(device)
            except Exception:
                return obj
        if isinstance(obj, tuple):
            return tuple(self._move_tensor_obj(v, device) for v in obj)
        if isinstance(obj, list):
            return [self._move_tensor_obj(v, device) for v in obj]
        if isinstance(obj, dict):
            if self._is_quant_payload(obj):
                return obj
            return {k: self._move_tensor_obj(v, device) for k, v in obj.items()}
        return obj

    def _current_device(self):
        try:
            return self.args["device"]
        except Exception:
            return "cpu"

    def _maybe_quantize_message(self, message) -> None:
        try:
            params = message.get_params() if hasattr(message, "get_params") else None
        except Exception:
            params = None
        if not isinstance(params, dict):
            return

        analysis = params.get(self._comm_analysis_key)
        if not isinstance(analysis, dict):
            analysis = {}
            params[self._comm_analysis_key] = analysis

        # Forward-path activations
        if "activations" in params:
            raw_bytes, _, _ = self._summarize_tensor_payload_bytes(params["activations"], preserve_second_if_pair=True)
            quant_time = 0.0
            if self._quantize_forward and self._fwd_codec is not None:
                quant_started = time.perf_counter()
                params["activations"] = self._encode_tensor_obj(
                    params["activations"],
                    self._fwd_codec,
                    preserve_second_if_pair=True,
                )
                quant_time = time.perf_counter() - quant_started
            _, quantized_bytes, metadata_bytes = self._summarize_tensor_payload_bytes(
                params["activations"],
                preserve_second_if_pair=True,
            )
            analysis["activations"] = {
                "raw_bytes": int(raw_bytes),
                "quantized_bytes": int(quantized_bytes),
                "metadata_bytes": int(metadata_bytes),
                "quant_time": float(quant_time),
            }
            self._record_epoch_comm_stats(
                "activations",
                raw_bytes=raw_bytes,
                quantized_bytes=quantized_bytes,
                metadata_bytes=metadata_bytes,
                quant_time=quant_time,
            )
        # Backward-path gradients
        if "activation_grads" in params:
            raw_bytes, _, _ = self._summarize_tensor_payload_bytes(params["activation_grads"], preserve_second_if_pair=False)
            quant_time = 0.0
            if self._quantize_backward and self._bwd_codec is not None:
                quant_started = time.perf_counter()
                params["activation_grads"] = self._encode_tensor_obj(
                    params["activation_grads"],
                    self._bwd_codec,
                    preserve_second_if_pair=False,
                )
                quant_time = time.perf_counter() - quant_started
            _, quantized_bytes, metadata_bytes = self._summarize_tensor_payload_bytes(
                params["activation_grads"],
                preserve_second_if_pair=False,
            )
            analysis["activation_grads"] = {
                "raw_bytes": int(raw_bytes),
                "quantized_bytes": int(quantized_bytes),
                "metadata_bytes": int(metadata_bytes),
                "quant_time": float(quant_time),
            }
            self._record_epoch_comm_stats(
                "activation_grads",
                raw_bytes=raw_bytes,
                quantized_bytes=quantized_bytes,
                metadata_bytes=metadata_bytes,
                quant_time=quant_time,
            )

        # MPI comm.send pickles Python objects. Stage any raw tensors on CPU
        # before enqueueing so CUDA tensors are not sent directly.
        for key in ("activations", "activation_grads"):
            if key in params:
                try:
                    params[key] = self._move_tensor_obj(params[key], "cpu")
                except Exception:
                    pass

    def _maybe_dequantize_message(self, msg_params) -> None:
        try:
            params = msg_params.get_params() if hasattr(msg_params, "get_params") else None
        except Exception:
            params = None
        if not isinstance(params, dict):
            return
        for key in ("activations", "activation_grads"):
            if key in params:
                self._consume_comm_analysis(params, key)
        # Forward-path activations
        if self._quantize_forward and self._fwd_codec is not None and "activations" in params:
            params["activations"] = self._decode_tensor_obj(
                params["activations"],
                self._fwd_codec,
                preserve_second_if_pair=True,
            )
        # Backward-path gradients
        if self._quantize_backward and self._bwd_codec is not None and "activation_grads" in params:
            params["activation_grads"] = self._decode_tensor_obj(
                params["activation_grads"],
                self._bwd_codec,
                preserve_second_if_pair=False,
            )

        # Restore raw tensors to the receiver's configured device.
        device = self._current_device()
        for key in ("activations", "activation_grads"):
            if key in params:
                try:
                    params[key] = self._move_tensor_obj(params[key], device)
                except Exception:
                    pass
        params.pop(self._comm_analysis_key, None)

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

        # Decode activations after comm accounting, before dispatch to algorithm handlers.
        self._maybe_dequantize_message(msg_params)

        handler = self._message_handlers.get(msg_type)
        if handler is None:
            logging.debug("No handler for msg_type=%s on rank=%s", msg_type, self.rank)
            return
        handler(msg_params)

    def send_message(self, message, priority: Optional[int] = None):
        self._maybe_log_tensor_distribution(message)

        # Quantize activations before accounting and enqueueing for send.
        self._maybe_quantize_message(message)

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

