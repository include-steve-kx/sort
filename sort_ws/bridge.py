from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Optional, Set, List, Dict, Any

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol

from .codec import decode_video_message, encode_video_message
from .ego import EgoSmoother, EgoStateStore
from .image_space_tracker import ImageSpaceSort
from .world_space_tracker import WorldSpaceSort
from .world_transform import detection_to_world_latlon


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

    # Tracking mode: "world_space" or "image_space"
    mode: str = "world_space"

    # World-tracking / camera params (used when mode == "world_space")
    world_space_cv_width_px: int = 1920
    world_space_fov_x_deg: float = 90.0
    world_space_camera_yaw_offset_deg: float = 0.0

    # Logging
    # If > 0, print a simple progress line every N upstream video frames.
    log_every_n_frames: int = 0


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


def _track_image_space(bboxes: list, tracker: ImageSpaceSort) -> List[Dict[str, Any]]:
    assigned = tracker.assign(bboxes)
    tracked_bboxes: List[Dict[str, Any]] = []
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
    return tracked_bboxes


async def _track_world_space(
    *,
    bboxes: list,
    world_tracker: WorldSpaceSort,
    ego_store: EgoStateStore,
    cfg: BridgeConfig,
) -> List[Dict[str, Any]]:
    ego = await ego_store.get()
    if ego.latlon is None or ego.heading_deg is None or ego.ref_latlon is None:
        return []

    fov_x_rad = float(cfg.world_space_fov_x_deg) * 3.141592653589793 / 180.0
    local_ref = ego.ref_latlon

    world_dets: List[Dict[str, Any]] = []
    # keep mapping back to original list order if needed later
    for det in bboxes:
        if not isinstance(det, dict):
            continue
        try:
            x = float(det.get("x"))
            w = float(det.get("width", det.get("w")))
            distance_m = float(det.get("distance"))
        except Exception:
            continue
        if not (distance_m > 0.0) or not (w > 0.0):
            continue

        x_bottom_center = x + w / 2.0
        world_latlon, enu_from_ref, enu_rel_ego = detection_to_world_latlon(
            ego_latlon=ego.latlon,
            ego_heading_deg=float(ego.heading_deg),
            x_bottom_center_px=float(x_bottom_center),
            distance_m=float(distance_m),
            image_width_px=float(cfg.world_space_cv_width_px),
            fov_x_rad=fov_x_rad,
            camera_yaw_offset_deg=float(cfg.world_space_camera_yaw_offset_deg),
            local_ref=local_ref,
        )
        det2 = dict(det)
        det2["world_latitude"] = float(world_latlon.lat)
        det2["world_longitude"] = float(world_latlon.lon)
        det2["world_east_m"] = float(enu_from_ref.east_m)
        det2["world_north_m"] = float(enu_from_ref.north_m)
        det2["world_rel_ego_east_m"] = float(enu_rel_ego.east_m)
        det2["world_rel_ego_north_m"] = float(enu_rel_ego.north_m)
        world_dets.append(det2)

    assigned = world_tracker.assign(world_dets)
    tracked_bboxes: List[Dict[str, Any]] = []
    for det, track_id in zip(world_dets, assigned):
        if track_id is None:
            continue
        det2 = dict(det)
        det2.setdefault("source_obj_id", det2.get("obj_id"))
        det2["obj_id"] = int(track_id)
        tracked_bboxes.append(det2)
    return tracked_bboxes

async def _upstream_video_loop(
    cfg: BridgeConfig,
    image_tracker: ImageSpaceSort,
    world_tracker: Optional[WorldSpaceSort],
    ego_store: Optional[EgoStateStore],
    video_hub: BroadcastHub,
) -> None:
    url = _ws_url(cfg.upstream_host, cfg.upstream_video_port)
    upstream_frame_count = 0
    while True:
        try:
            async with websockets.connect(url, max_size=None) as upstream:
                print(f"[sort-ws] Connected upstream video: {url}")
                async for payload in upstream:
                    if not isinstance(payload, (bytes, bytearray)):
                        continue
                    upstream_frame_count += 1
                    vm = decode_video_message(payload)
                    bboxes = vm.metadata.get("bboxes", [])
                    if not isinstance(bboxes, list):
                        bboxes = []

                    if cfg.mode == "world_space" and world_tracker is not None and ego_store is not None:
                        tracked_bboxes = await _track_world_space(
                            bboxes=bboxes,
                            world_tracker=world_tracker,
                            ego_store=ego_store,
                            cfg=cfg,
                        )
                        vm.metadata["tracked_space"] = "world_space"
                    else:
                        tracked_bboxes = _track_image_space(bboxes, image_tracker)
                        vm.metadata["tracked_space"] = "image_space"

                    vm.metadata["bboxes"] = tracked_bboxes
                    vm.metadata["tracked"] = True
                    out_payload = encode_video_message(vm.metadata, vm.jpeg_bytes)
                    await video_hub.broadcast(out_payload)

                    if cfg.log_every_n_frames and upstream_frame_count % int(cfg.log_every_n_frames) == 0:
                        active = world_tracker if (cfg.mode == "world_space" and world_tracker is not None) else image_tracker
                        try:
                            active_tracks = len(active.trackers)
                        except Exception:
                            active_tracks = -1
                        try:
                            active_frame_count = int(getattr(active, "frame_count", -1))
                        except Exception:
                            active_frame_count = -1
                        try:
                            active_max_age = int(getattr(active, "max_age", -1))
                        except Exception:
                            active_max_age = -1
                        print(
                            f"[sort-ws] frames upstream={upstream_frame_count} "
                            f"tracker_frames={active_frame_count} tracks={active_tracks} max_age={active_max_age}"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[sort-ws] Upstream video error ({url}): {e}", file=sys.stderr)
            await asyncio.sleep(cfg.reconnect_delay_s)


async def _upstream_nmea_loop(
    cfg: BridgeConfig, nmea_hub: BroadcastHub, ego_store: Optional[EgoStateStore]
) -> None:
    url = _ws_url(cfg.upstream_host, cfg.upstream_nmea_port)
    while True:
        try:
            async with websockets.connect(url, max_size=None) as upstream:
                print(f"[sort-ws] Connected upstream NMEA: {url}")
                async for msg in upstream:
                    # NMEA is JSON text in replay; forward as-is.
                    if cfg.mode == "world_space" and ego_store is not None and isinstance(msg, str):
                        await ego_store.update_from_nmea_json(msg)
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


async def run_bridge(
    cfg: BridgeConfig,
    image_tracker: ImageSpaceSort,
    world_tracker: Optional[WorldSpaceSort] = None,
    ego_store: Optional[EgoStateStore] = None,
) -> None:
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
            _upstream_video_loop(cfg, image_tracker, world_tracker, ego_store, video_hub),
            _upstream_nmea_loop(cfg, nmea_hub, ego_store),
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

    # Image-space tracker params (shown in --help)
    p.add_argument("--image-space-max-age", dest="image_space_max_age", type=int, default=40)
    p.add_argument("--image-space-min-hits", dest="image_space_min_hits", type=int, default=300)
    p.add_argument("--image-space-iou-threshold", dest="image_space_iou_threshold", type=float, default=0.1)
    p.add_argument("--image-space-alpha-distance", dest="image_space_alpha_distance", type=float, default=0.15)
    p.add_argument("--image-space-beta-heading", dest="image_space_beta_heading", type=float, default=0.0)
    p.add_argument("--image-space-gamma-confidence", dest="image_space_gamma_confidence", type=float, default=0.0)
    p.add_argument(
        "--image-space-new-track-min-confidence",
        dest="image_space_new_track_min_confidence",
        type=float,
        default=0.0,
    )

    p.add_argument("--reconnect-delay-s", type=float, default=1.0)

    # World-space tracking
    p.add_argument(
        "--mode",
        type=str,
        default="world_space",
        choices=["world_space", "image_space"],
        help='Tracking mode. "world_space" uses NMEA ego lat/lon/heading and tracks in ENU meters. "image_space" runs image-space SORT.',
    )
    p.add_argument(
        "--world-space-cv-width-px",
        dest="world_space_cv_width_px",
        type=int,
        default=1920,
        help="CV image width in pixels (used to compute bearing from x pixel).",
    )
    p.add_argument(
        "--world-space-fov-x-deg",
        dest="world_space_fov_x_deg",
        type=float,
        default=55.3,
        help="Camera horizontal field-of-view in degrees.",
    )
    p.add_argument(
        "--world-space-camera-yaw-offset-deg",
        dest="world_space_camera_yaw_offset_deg",
        type=float,
        default=0.0,
        help="Yaw offset between camera forward and boat forward (degrees).",
    )

    # World-space tracker params (shown in --help)
    p.add_argument("--world-space-max-age", dest="world_space_max_age", type=int, default=300)
    p.add_argument("--world-space-min-hits", dest="world_space_min_hits", type=int, default=40)
    p.add_argument(
        "--world-space-max-distance-m",
        dest="world_space_max_distance_m",
        type=float,
        default=50.0,
        help="Max world-space (meters) to allow a detection to match a track.",
    )
    p.add_argument(
        "--world-space-beta-heading",
        dest="world_space_beta_heading",
        type=float,
        default=0.0,
        help="Weight for target heading difference in world association cost.",
    )
    p.add_argument(
        "--world-space-gamma-confidence",
        dest="world_space_gamma_confidence",
        type=float,
        default=0.0,
        help="Weight for (1-confidence) in world association cost.",
    )
    p.add_argument(
        "--world-space-new-track-min-confidence",
        dest="world_space_new_track_min_confidence",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--log-every-n-frames",
        type=int,
        default=0,
        help="If > 0, print a progress line every N upstream video frames (includes tracker frame_count / max_age).",
    )

    return p


def cli_main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    mode = str(args.mode)
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
        mode=mode,
        world_space_cv_width_px=int(args.world_space_cv_width_px),
        world_space_fov_x_deg=float(args.world_space_fov_x_deg),
        world_space_camera_yaw_offset_deg=float(args.world_space_camera_yaw_offset_deg),
        log_every_n_frames=int(args.log_every_n_frames),
    )

    image_tracker = ImageSpaceSort(
        image_space_max_age=args.image_space_max_age,
        image_space_min_hits=args.image_space_min_hits,
        image_space_iou_threshold=args.image_space_iou_threshold,
        image_space_alpha_distance=args.image_space_alpha_distance,
        image_space_beta_heading=args.image_space_beta_heading,
        image_space_gamma_confidence=args.image_space_gamma_confidence,
        image_space_new_track_min_confidence=args.image_space_new_track_min_confidence,
    )

    world_tracker: Optional[WorldSpaceSort] = None
    ego_store: Optional[EgoStateStore] = None
    if cfg.mode == "world_space":
        ego_store = EgoStateStore(EgoSmoother())
        world_tracker = WorldSpaceSort(
            world_space_max_age=args.world_space_max_age,
            world_space_min_hits=args.world_space_min_hits,
            world_space_max_distance_m=args.world_space_max_distance_m,
            world_space_beta_heading=args.world_space_beta_heading,
            world_space_gamma_confidence=args.world_space_gamma_confidence,
            world_space_new_track_min_confidence=args.world_space_new_track_min_confidence,
        )

    try:
        asyncio.run(run_bridge(cfg, image_tracker, world_tracker, ego_store))
    except KeyboardInterrupt:
        pass


