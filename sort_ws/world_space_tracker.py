from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

from .image_space_tracker import linear_assignment  # reuse lap/scipy wrapper
from .world_transform import heading_diff_deg


@dataclass
class WorldTrackExtras:
    confidence: float = float("nan")
    category: Optional[str] = None
    heading_deg: float = float("nan")  # if the detector provides a target heading estimate


class KalmanPointTracker:
    """
    Track a single target in world space (local ENU meters), using a constant-velocity KF.

    State: [east, north, ve, vn]
    Measurement: [east, north]
    """

    count = 0

    def __init__(self, meas_enu: np.ndarray, extras: WorldTrackExtras):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # covariances: tuned for stability (similar spirit to SORT)
        self.kf.R = np.eye(2, dtype=np.float32) * 4.0
        self.kf.P = np.eye(4, dtype=np.float32) * 10.0
        self.kf.P[2:, 2:] *= 1000.0  # velocities initially very uncertain

        q = 1.0
        self.kf.Q = np.array(
            [[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2], [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]],
            dtype=np.float32,
        ) * q

        self.kf.x = np.array([[float(meas_enu[0])], [float(meas_enu[1])], [0.0], [0.0]], dtype=np.float32)

        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.extras = extras

    def update(self, meas_enu: np.ndarray, extras: WorldTrackExtras) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = np.array([[float(meas_enu[0])], [float(meas_enu[1])]], dtype=np.float32)
        self.kf.update(z)
        self.extras = extras

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        # return [east, north]
        return self.kf.x[:2, 0].astype(np.float32)


def _euclid_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,2), b: (M,2) -> (N,M) euclidean distances
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    aa = a[:, None, :]  # (N,1,2)
    bb = b[None, :, :]  # (1,M,2)
    d = aa - bb
    return np.sqrt(np.sum(d * d, axis=2)).astype(np.float32)


def associate_world_detections_to_trackers(
    det_xy: np.ndarray,
    trk_xy: np.ndarray,
    det_heading: np.ndarray,
    trk_heading: np.ndarray,
    det_conf: np.ndarray,
    *,
    max_distance_m: float = 30.0,
    beta_heading: float = 0.0,
    gamma_confidence: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Association in world space.

    Gating: distance <= max_distance_m
    Cost (Hungarian): distance + beta*heading_cost + gamma*(1-confidence)
    """
    if trk_xy.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(det_xy.shape[0], dtype=int),
            np.empty((0,), dtype=int),
        )

    dist = _euclid_dist_matrix(det_xy, trk_xy)  # (N,M)
    cost = dist.copy()

    if beta_heading and det_heading.size and trk_heading.size:
        heading_cost = np.zeros_like(cost, dtype=np.float32)
        for i in range(det_xy.shape[0]):
            if np.isnan(det_heading[i]):
                continue
            for j in range(trk_xy.shape[0]):
                if np.isnan(trk_heading[j]):
                    continue
                heading_cost[i, j] = abs(heading_diff_deg(float(det_heading[i]), float(trk_heading[j]))) / 180.0
        cost = cost + float(beta_heading) * heading_cost

    if gamma_confidence and det_conf.size:
        det_conf_clipped = np.clip(det_conf.astype(np.float32), 0.0, 1.0)
        conf_penalty = 1.0 - det_conf_clipped
        cost = cost + float(gamma_confidence) * conf_penalty.reshape((-1, 1))

    gated = dist <= float(max_distance_m)
    if not np.any(gated):
        return (
            np.empty((0, 2), dtype=int),
            np.arange(det_xy.shape[0], dtype=int),
            np.arange(trk_xy.shape[0], dtype=int),
        )

    huge = 1e6
    cost_gated = np.where(gated, cost, huge)
    matched_indices = linear_assignment(cost_gated)

    unmatched_dets: List[int] = []
    for d in range(det_xy.shape[0]):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)
    unmatched_trks: List[int] = []
    for t in range(trk_xy.shape[0]):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)

    matches: List[np.ndarray] = []
    for m in matched_indices:
        if not gated[m[0], m[1]]:
            unmatched_dets.append(int(m[0]))
            unmatched_trks.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches_arr = np.empty((0, 2), dtype=int)
    else:
        matches_arr = np.concatenate(matches, axis=0).astype(int)
    return matches_arr, np.array(unmatched_dets, dtype=int), np.array(unmatched_trks, dtype=int)


class WorldSpaceSort:
    """
    SORT-like tracker operating on world-space (ENU) point measurements.

    Input detections: dicts containing at least:
      - world_east_m, world_north_m (meters, local tangent plane)
    Optionally:
      - confidence, category, heading

    Output: list of assigned IDs aligned with input order (None if unconfirmed).
    IDs are 1-based (MOT-style).
    """

    def __init__(
        self,
        *,
        # World-space tracker parameters (prefixed for clarity at call sites / CLI wiring).
        world_space_max_age: int = 20,
        world_space_min_hits: int = 5,
        world_space_max_distance_m: float = 30.0,
        world_space_beta_heading: float = 0.0,
        world_space_gamma_confidence: float = 0.0,
        world_space_new_track_min_confidence: float = 0.0,
    ):
        # Keep attribute names short/stable for runtime introspection/logging.
        self.max_age = int(world_space_max_age)
        self.min_hits = int(world_space_min_hits)
        self.max_distance_m = float(world_space_max_distance_m)
        self.beta_heading = float(world_space_beta_heading)
        self.gamma_confidence = float(world_space_gamma_confidence)
        self.new_track_min_confidence = float(world_space_new_track_min_confidence)

        self.trackers: List[KalmanPointTracker] = []
        self.frame_count = 0

    def _det_to_xy(self, det: Dict[str, Any]) -> Tuple[np.ndarray, WorldTrackExtras]:
        e = float(det.get("world_east_m"))
        n = float(det.get("world_north_m"))
        conf = det.get("confidence", det.get("score", float("nan")))
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = float("nan")
        heading = det.get("heading", det.get("target_heading", float("nan")))
        try:
            heading_f = float(heading)
        except Exception:
            heading_f = float("nan")
        extras = WorldTrackExtras(confidence=conf_f, category=det.get("category"), heading_deg=heading_f)
        return np.array([e, n], dtype=np.float32), extras

    def assign(self, detections: List[Dict[str, Any]]) -> List[Optional[int]]:
        self.frame_count += 1

        if len(detections) == 0:
            # age/predict existing trackers
            to_del: List[int] = []
            for t in range(len(self.trackers)):
                pos = self.trackers[t].predict()
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            for t in reversed(to_del):
                self.trackers.pop(t)
            # remove dead
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return []

        det_xy = np.zeros((len(detections), 2), dtype=np.float32)
        det_heading = np.full((len(detections),), np.nan, dtype=np.float32)
        det_conf = np.full((len(detections),), np.nan, dtype=np.float32)
        det_extras: List[WorldTrackExtras] = []
        for i, d in enumerate(detections):
            xy, ex = self._det_to_xy(d)
            det_xy[i, :] = xy
            det_extras.append(ex)
            det_heading[i] = ex.heading_deg
            det_conf[i] = ex.confidence

        trk_xy = np.zeros((len(self.trackers), 2), dtype=np.float32)
        trk_heading = np.full((len(self.trackers),), np.nan, dtype=np.float32)
        trk_conf = np.full((len(self.trackers),), np.nan, dtype=np.float32)
        to_del: List[int] = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()
            trk_xy[t, :] = pos.astype(np.float32)
            if np.any(np.isnan(pos)):
                to_del.append(t)
            trk_heading[t] = float(self.trackers[t].extras.heading_deg)
            trk_conf[t] = float(self.trackers[t].extras.confidence)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trk_xy = np.delete(trk_xy, t, axis=0)
            trk_heading = np.delete(trk_heading, t, axis=0)
            trk_conf = np.delete(trk_conf, t, axis=0)

        matches, unmatched_dets, _unmatched_trks = associate_world_detections_to_trackers(
            det_xy,
            trk_xy,
            det_heading,
            trk_heading,
            det_conf,
            max_distance_m=self.max_distance_m,
            beta_heading=self.beta_heading,
            gamma_confidence=self.gamma_confidence,
        )

        assigned: List[Optional[int]] = [None] * len(detections)

        for det_idx, trk_idx in matches:
            self.trackers[int(trk_idx)].update(det_xy[int(det_idx), :], det_extras[int(det_idx)])
            trk = self.trackers[int(trk_idx)]
            track_id = int(trk.id) + 1
            if trk.hit_streak >= self.min_hits:
                assigned[int(det_idx)] = track_id

        for det_idx in unmatched_dets:
            conf = float(det_extras[int(det_idx)].confidence)
            if not np.isnan(conf) and conf < self.new_track_min_confidence:
                continue
            trk = KalmanPointTracker(det_xy[int(det_idx), :], det_extras[int(det_idx)])
            self.trackers.append(trk)
            track_id = int(trk.id) + 1
            if trk.hit_streak >= self.min_hits:
                assigned[int(det_idx)] = track_id

        # remove dead
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return assigned

