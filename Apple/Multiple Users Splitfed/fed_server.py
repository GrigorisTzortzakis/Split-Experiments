from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import websockets


def _state_dict_to_b64(state: Dict[str, torch.Tensor]) -> str:
    cpu_state = {str(k): v.detach().cpu() for k, v in state.items()}
    bio = io.BytesIO()
    torch.save(cpu_state, bio)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _b64_to_state_dict(payload_b64: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(payload_b64.encode("ascii"))
    bio = io.BytesIO(raw)
    obj = torch.load(bio, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("decoded state is not a dict")
    out: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"state_dict value for {k} is not a Tensor")
        out[str(k)] = v
    return out


def _fedavg(states: Dict[int, Dict[str, torch.Tensor]], weights: Dict[int, int]) -> Dict[str, torch.Tensor]:
    total = sum(int(weights[cid]) for cid in states.keys())
    if total <= 0:
        raise ValueError("total weight must be > 0")
    keys = next(iter(states.values())).keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for cid, sd in states.items():
            w = float(weights[cid]) / float(total)
            t = sd[k].detach().float() * w
            acc = t if acc is None else (acc + t)
        assert acc is not None
        out[k] = acc
    # preserve dtypes
    ref = next(iter(states.values()))
    for k in out.keys():
        out[k] = out[k].to(dtype=ref[k].dtype)
    return out


@dataclass
class FedServerConfig:
    host: str
    port: int
    expected_clients: int
    max_message_mb: int


class SplitFedServer:
    def __init__(self, cfg: FedServerConfig, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

        self._clients: Dict[int, websockets.WebSocketServerProtocol] = {}

        # global client-front weights per round. By convention:
        # - round 0 weights are empty (clients start from deterministic init)
        # - after aggregating updates for round r, we publish weights for round r+1
        self._wc_by_round: Dict[int, Dict[str, torch.Tensor]] = {}

        # round_id -> client_id -> (nk, state_dict) for updates *submitted for that round*
        self._round_updates: Dict[int, Dict[int, tuple[int, Dict[str, torch.Tensor]]]] = {}

        self._lock = asyncio.Lock()

    async def _send(self, ws: websockets.WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
        await ws.send(json.dumps(msg))

    async def _broadcast(self, msg: Dict[str, Any]) -> None:
        dead: list[int] = []
        for cid, ws in self._clients.items():
            try:
                await self._send(ws, msg)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self._clients.pop(cid, None)

    async def handle(self, ws: websockets.WebSocketServerProtocol) -> None:
        client_id: Optional[int] = None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "hello":
                    client_id = int(msg.get("client_id"))
                    async with self._lock:
                        self._clients[client_id] = ws
                    await self._send(
                        ws,
                        {
                            "type": "hello_ack",
                            "expected_clients": self.cfg.expected_clients,
                            "server": "fed_server",
                        },
                    )
                    continue

                if mtype == "get_global_client_weights":
                    round_id = int(msg.get("round_id", 0))
                    async with self._lock:
                        wc = self._wc_by_round.get(round_id)
                        await self._send(
                            ws,
                            {
                                "type": "global_client_weights",
                                "round_id": round_id,
                                "state_b64": _state_dict_to_b64(wc) if wc is not None else "",
                            },
                        )
                    continue

                if mtype == "submit_client_update":
                    cid = int(msg["client_id"])
                    round_id = int(msg["round_id"])
                    nk = int(msg["nk"])
                    state_b64 = str(msg["state_b64"])
                    if not state_b64:
                        raise ValueError("state_b64 required")
                    wc_k = _b64_to_state_dict(state_b64)

                    async with self._lock:
                        bucket = self._round_updates.setdefault(round_id, {})
                        bucket[cid] = (nk, wc_k)
                        got = len(bucket)
                        await self._send(ws, {"type": "submit_client_update_ack", "round_id": round_id, "received": got})
                        self.logger.info("round=%d received update client_id=%d (%d/%d)", round_id, cid, got, self.cfg.expected_clients)

                        if got >= self.cfg.expected_clients:
                            states = {c: s for c, (n, s) in bucket.items()}
                            weights = {c: n for c, (n, s) in bucket.items()}
                            wc_next = _fedavg(states, weights)
                            next_round_id = round_id + 1
                            self._wc_by_round[next_round_id] = wc_next
                            self.logger.info("round=%d aggregated; published global weights for round=%d", round_id, next_round_id)
                            await self._broadcast(
                                {
                                    "type": "global_client_weights",
                                    "round_id": next_round_id,
                                    "state_b64": _state_dict_to_b64(wc_next),
                                }
                            )
                    continue

                await self._send(ws, {"type": "error", "message": f"unknown type: {mtype}"})

        finally:
            if client_id is not None:
                async with self._lock:
                    self._clients.pop(client_id, None)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SplitFed FedServer (aggregates client-front weights)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--expected-clients", type=int, default=2)
    p.add_argument("--max-message-mb", type=int, default=128)
    return p.parse_args(argv)


async def _run(cfg: FedServerConfig, logger: logging.Logger) -> None:
    max_size = cfg.max_message_mb * 1024 * 1024
    server = SplitFedServer(cfg, logger)
    async with websockets.serve(server.handle, cfg.host, cfg.port, max_size=max_size):
        logger.info("fed_server listening on ws://%s:%d expected_clients=%d", cfg.host, cfg.port, cfg.expected_clients)
        await asyncio.Future()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("fed_server")
    cfg = FedServerConfig(
        host=args.host,
        port=args.port,
        expected_clients=args.expected_clients,
        max_message_mb=args.max_message_mb,
    )
    asyncio.run(_run(cfg, logger))


if __name__ == "__main__":
    main()
