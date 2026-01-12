from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol

from .codec import decode_video_message, encode_video_message
from .tracker import SortWithExtras


@dataclass(frozen=True)
class BridgeConfig:
    upstream_host: str
    upstream_video_port: int
    upstream_nmea_port: int
    upstream_control_port: int

    downstream_bind: str
    downstream_video_port: int
    downstream_nmea_port: int
    downstream_control_port: int

    reconnect_delay_s: float = 1.0


class BroadcastHub:
    def __init__(self, name: str):
        self.name = name
        self._clients: Set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message) -> None:
        async with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        dead: list[WebSocketServerProtocol] = []
        for ws in clients:
            try:
                await ws.send(message)
            except ConnectionClosed:
                dead.append(ws)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


def _ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}"


async def _upstream_video_loop(cfg: BridgeConfig, tracker: SortWithExtras, video_hub: BroadcastHub) -> None:
    url = _ws_url(cfg.upstream_host, cfg.upstream_video_port)
    while True:
        try:
            async with websockets.connect(url, max_size=None) as upstream:
                print(f"[sort-ws] Connected upstream video: {url}")
                async for payload in upstream:
                    if not isinstance(payload, (bytes, bytearray)):
                        continue
                    vm = decode_video_message(payload)
                    bboxes = vm.metadata.get("bboxes", [])
                    if not isinstance(bboxes, list):
                        bboxes = []

                    assigned = tracker.assign(bboxes)
                    tracked_bboxes = []
                    for det, track_id in zip(bboxes, assigned):
                        if not isinstance(det, dict):
                            continue
                        # If the detection isn't confirmed by the tracker yet, drop it entirely.
                        # This avoids leaking upstream `obj_id` values downstream.
                        if track_id is None:
                            continue
                        det2 = dict(det)
                        det2.setdefault("source_obj_id", det2.get("obj_id"))
                        det2["obj_id"] = int(track_id)
                        tracked_bboxes.append(det2)

                    vm.metadata["bboxes"] = tracked_bboxes
                    vm.metadata["tracked"] = True
                    out_payload = encode_video_message(vm.metadata, vm.jpeg_bytes)
                    await video_hub.broadcast(out_payload)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[sort-ws] Upstream video error ({url}): {e}", file=sys.stderr)
            await asyncio.sleep(cfg.reconnect_delay_s)


async def _upstream_nmea_loop(cfg: BridgeConfig, nmea_hub: BroadcastHub) -> None:
    url = _ws_url(cfg.upstream_host, cfg.upstream_nmea_port)
    while True:
        try:
            async with websockets.connect(url, max_size=None) as upstream:
                print(f"[sort-ws] Connected upstream NMEA: {url}")
                async for msg in upstream:
                    # NMEA is JSON text in replay; forward as-is.
                    await nmea_hub.broadcast(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[sort-ws] Upstream NMEA error ({url}): {e}", file=sys.stderr)
            await asyncio.sleep(cfg.reconnect_delay_s)


async def _upstream_control_loop(
    cfg: BridgeConfig, control_hub: BroadcastHub, outbound_to_upstream: "asyncio.Queue[str]"
) -> None:
    url = _ws_url(cfg.upstream_host, cfg.upstream_control_port)
    while True:
        try:
            async with websockets.connect(url, max_size=None) as upstream:
                print(f"[sort-ws] Connected upstream control: {url}")

                async def sender() -> None:
                    while True:
                        msg = await outbound_to_upstream.get()
                        await upstream.send(msg)

                sender_task = asyncio.create_task(sender())
                try:
                    async for msg in upstream:
                        await control_hub.broadcast(msg)
                finally:
                    sender_task.cancel()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[sort-ws] Upstream control error ({url}): {e}", file=sys.stderr)
            await asyncio.sleep(cfg.reconnect_delay_s)


async def _downstream_video_handler(ws: WebSocketServerProtocol, video_hub: BroadcastHub) -> None:
    await video_hub.register(ws)
    try:
        # Clients generally don't send anything on the video socket; drain if they do.
        async for _ in ws:
            pass
    finally:
        await video_hub.unregister(ws)


async def _downstream_nmea_handler(ws: WebSocketServerProtocol, nmea_hub: BroadcastHub) -> None:
    await nmea_hub.register(ws)
    try:
        async for _ in ws:
            pass
    finally:
        await nmea_hub.unregister(ws)


async def _downstream_control_handler(
    ws: WebSocketServerProtocol,
    control_hub: BroadcastHub,
    outbound_to_upstream: "asyncio.Queue[str]",
) -> None:
    await control_hub.register(ws)
    try:
        async for msg in ws:
            # Forward downstream control commands upstream (pause/play/seek).
            if isinstance(msg, (bytes, bytearray)):
                try:
                    msg = msg.decode("utf-8")
                except Exception:
                    continue
            await outbound_to_upstream.put(str(msg))
    finally:
        await control_hub.unregister(ws)


async def run_bridge(cfg: BridgeConfig, tracker: SortWithExtras) -> None:
    video_hub = BroadcastHub("video")
    nmea_hub = BroadcastHub("nmea")
    control_hub = BroadcastHub("control")
    outbound_to_upstream: asyncio.Queue[str] = asyncio.Queue()

    # Start downstream servers and keep the returned server objects alive.
    video_server = await websockets.serve(
        lambda ws: _downstream_video_handler(ws, video_hub),
        cfg.downstream_bind,
        cfg.downstream_video_port,
        max_size=None,
    )
    nmea_server = await websockets.serve(
        lambda ws: _downstream_nmea_handler(ws, nmea_hub),
        cfg.downstream_bind,
        cfg.downstream_nmea_port,
        max_size=None,
    )
    control_server = await websockets.serve(
        lambda ws: _downstream_control_handler(ws, control_hub, outbound_to_upstream),
        cfg.downstream_bind,
        cfg.downstream_control_port,
        max_size=None,
    )

    print("[sort-ws] Starting downstream websocket servers:")
    print(f"  video   ws://{cfg.downstream_bind}:{cfg.downstream_video_port}")
    print(f"  nmea    ws://{cfg.downstream_bind}:{cfg.downstream_nmea_port}")
    print(f"  control ws://{cfg.downstream_bind}:{cfg.downstream_control_port}")
    print("[sort-ws] Connecting to upstream websocket servers:")
    print(f"  video   {_ws_url(cfg.upstream_host, cfg.upstream_video_port)}")
    print(f"  nmea    {_ws_url(cfg.upstream_host, cfg.upstream_nmea_port)}")
    print(f"  control {_ws_url(cfg.upstream_host, cfg.upstream_control_port)}")

    try:
        await asyncio.gather(
            _upstream_video_loop(cfg, tracker, video_hub),
            _upstream_nmea_loop(cfg, nmea_hub),
            _upstream_control_loop(cfg, control_hub, outbound_to_upstream),
        )
    finally:
        video_server.close()
        nmea_server.close()
        control_server.close()
        await video_server.wait_closed()
        await nmea_server.wait_closed()
        await control_server.wait_closed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SORT websocket bridge: consume replay streams, run SORT, re-broadcast with tracker IDs."
    )
    p.add_argument("--upstream-host", type=str, default="127.0.0.1")
    p.add_argument("--upstream-video-port", type=int, default=5001)
    p.add_argument("--upstream-nmea-port", type=int, default=3636)
    p.add_argument("--upstream-control-port", type=int, default=6001)

    p.add_argument("--downstream-bind", type=str, default="0.0.0.0")
    p.add_argument("--downstream-video-port", type=int, default=5002)
    p.add_argument("--downstream-nmea-port", type=int, default=3637)
    p.add_argument("--downstream-control-port", type=int, default=6002)

    # Tracker params
    # p.add_argument("--max-age", type=int, default=10)
    # p.add_argument("--min-hits", type=int, default=1)
    # p.add_argument("--iou-threshold", type=float, default=0.3)
    # p.add_argument("--alpha-distance", type=float, default=0.15)
    # p.add_argument("--beta-heading", type=float, default=0.10)
    # p.add_argument("--gamma-confidence", type=float, default=0.05)
    # p.add_argument("--new-track-min-confidence", type=float, default=0.0)
    p.add_argument("--max-age", type=int, default=40)
    p.add_argument("--min-hits", type=int, default=40)
    p.add_argument("--iou-threshold", type=float, default=0.1)
    p.add_argument("--alpha-distance", type=float, default=0.15)
    p.add_argument("--beta-heading", type=float, default=0.0)
    p.add_argument("--gamma-confidence", type=float, default=0.0)
    p.add_argument("--new-track-min-confidence", type=float, default=0.0)

    p.add_argument("--reconnect-delay-s", type=float, default=1.0)
    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = BridgeConfig(
        upstream_host=args.upstream_host,
        upstream_video_port=args.upstream_video_port,
        upstream_nmea_port=args.upstream_nmea_port,
        upstream_control_port=args.upstream_control_port,
        downstream_bind=args.downstream_bind,
        downstream_video_port=args.downstream_video_port,
        downstream_nmea_port=args.downstream_nmea_port,
        downstream_control_port=args.downstream_control_port,
        reconnect_delay_s=args.reconnect_delay_s,
    )

    tracker = SortWithExtras(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        alpha_distance=args.alpha_distance,
        beta_heading=args.beta_heading,
        gamma_confidence=args.gamma_confidence,
        new_track_min_confidence=args.new_track_min_confidence,
    )

    try:
        asyncio.run(run_bridge(cfg, tracker))
    except KeyboardInterrupt:
        pass


