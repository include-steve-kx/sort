from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes IOU between two sets of bboxes in [x1,y1,x2,y2] format.
    Returns an (N,M) matrix for N test boxes and M gt boxes.
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype=np.float32)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
        + 1e-9
    )
    return o


def convert_bbox_to_z(bbox_xyxy: np.ndarray) -> np.ndarray:
    """
    bbox in [x1,y1,x2,y2] -> z in [x,y,s,r] (center x,y, scale=area, r=aspect)
    """
    w = bbox_xyxy[2] - bbox_xyxy[0]
    h = bbox_xyxy[3] - bbox_xyxy[1]
    x = bbox_xyxy[0] + w / 2.0
    y = bbox_xyxy[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-9)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    x in [x,y,s,r] center form -> [x1,y1,x2,y2]
    """
    w = np.sqrt(max(0.0, x[2] * x[3]))
    h = x[2] / (w + 1e-9)
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape(
        (1, 4)
    )


def _heading_diff_norm_deg(a: float, b: float) -> float:
    """
    Normalized heading difference in degrees -> [0,1].
    Handles wraparound (e.g., 359 vs 1 degrees).
    """
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d) / 180.0


def _dist_diff_norm(a: float, b: float) -> float:
    """
    Normalized absolute distance difference -> [0,1] (clipped).
    """
    denom = max(abs(a), abs(b), 1e-3)
    return float(min(1.0, abs(a - b) / denom))


def associate_detections_to_trackers(
    dets_xyxy: np.ndarray,
    trks_xyxy: np.ndarray,
    det_dist: np.ndarray,
    trk_dist: np.ndarray,
    det_heading: np.ndarray,
    trk_heading: np.ndarray,
    det_conf: np.ndarray,
    iou_threshold: float = 0.3,
    alpha_distance: float = 0.15,
    beta_heading: float = 0.10,
    gamma_confidence: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      matches: (K,2) array of (det_idx, trk_idx)
      unmatched_dets: (U,) det indices
      unmatched_trks: (V,) trk indices

    Matching is gated by IOU threshold; among valid pairs, we minimize:
      cost = (1 - IOU) + alpha * dist_cost + beta * heading_cost
    """
    if trks_xyxy.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(dets_xyxy.shape[0], dtype=int),
            np.empty((0,), dtype=int),
        )

    iou_matrix = iou_batch(dets_xyxy, trks_xyxy)

    # base cost for Hungarian is (1 - iou) in [0,1]
    cost = 1.0 - iou_matrix

    # Add distance/heading costs where available.
    if det_dist.size and trk_dist.size:
        dist_cost = np.zeros_like(cost, dtype=np.float32)
        for i in range(dets_xyxy.shape[0]):
            if np.isnan(det_dist[i]):
                continue
            for j in range(trks_xyxy.shape[0]):
                if np.isnan(trk_dist[j]):
                    continue
                dist_cost[i, j] = _dist_diff_norm(float(det_dist[i]), float(trk_dist[j]))
        cost = cost + alpha_distance * dist_cost

    if det_heading.size and trk_heading.size:
        heading_cost = np.zeros_like(cost, dtype=np.float32)
        for i in range(dets_xyxy.shape[0]):
            if np.isnan(det_heading[i]):
                continue
            for j in range(trks_xyxy.shape[0]):
                if np.isnan(trk_heading[j]):
                    continue
                heading_cost[i, j] = _heading_diff_norm_deg(
                    float(det_heading[i]), float(trk_heading[j])
                )
        cost = cost + beta_heading * heading_cost

    # Prefer matching high-confidence detections when there are more detections than tracks.
    # This doesn't depend on tracker state; it only influences which detections "claim" tracks.
    if det_conf.size:
        det_conf_clipped = np.clip(det_conf.astype(np.float32), 0.0, 1.0)
        # penalty in [0,1]
        conf_penalty = 1.0 - det_conf_clipped
        cost = cost + gamma_confidence * conf_penalty.reshape((-1, 1))

    # Gate invalid pairs by IOU.
    gated = iou_matrix >= float(iou_threshold)
    if not np.any(gated):
        return (
            np.empty((0, 2), dtype=int),
            np.arange(dets_xyxy.shape[0], dtype=int),
            np.arange(trks_xyxy.shape[0], dtype=int),
        )

    huge = 1e6
    cost_gated = np.where(gated, cost, huge)

    matched_indices = linear_assignment(cost_gated)

    unmatched_detections: List[int] = []
    for d in range(dets_xyxy.shape[0]):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers: List[int] = []
    for t in range(trks_xyxy.shape[0]):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches that were only possible due to huge cost (i.e. gated out).
    matches: List[np.ndarray] = []
    for m in matched_indices:
        if not gated[m[0], m[1]]:
            unmatched_detections.append(int(m[0]))
            unmatched_trackers.append(int(m[1]))
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches_arr = np.empty((0, 2), dtype=int)
    else:
        matches_arr = np.concatenate(matches, axis=0).astype(int)

    return matches_arr, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


@dataclass
class TrackExtras:
    distance: float = float("nan")
    heading: float = float("nan")  # degrees
    confidence: float = float("nan")  # [0,1]
    category: Optional[str] = None  # we currently don't use this for association


class KalmanBoxTracker:
    """
    Internal tracker for 2D bbox with a constant-velocity model.
    We additionally store "extras" (distance/heading/category) from the last matched detection.
    """

    count = 0

    def __init__(self, bbox_xyxy: np.ndarray, extras: TrackExtras):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox_xyxy)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[np.ndarray] = []
        # Count the initial detection as the first "hit" so min_hits behaves intuitively
        # without needing the original SORT warm-up exception (frame_count <= min_hits).
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        self.extras = extras

    def update(self, bbox_xyxy: np.ndarray, extras: TrackExtras) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox_xyxy))
        self.extras = extras

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


class SortWithExtras:
    """
    SORT wrapper that assigns stable IDs to incoming per-frame detections.

    Input detections are dicts with (at least):
      x, y, width, height, confidence
    Optionally:
      distance, heading, category

    Output:
      a list of assigned track IDs aligned with input detection order (len = #dets).
    """

    def __init__(
        self,
        # max_age: int = 10,
        # min_hits: int = 1,
        max_age: int = 20,
        min_hits: int = 20,
        iou_threshold: float = 0.1,
        # alpha_distance: float = 0.15,
        alpha_distance: float = 0.0,
        # beta_heading: float = 0.10,
        beta_heading: float = 0.0,
        # gamma_confidence: float = 0.05,
        gamma_confidence: float = 0.0,
        new_track_min_confidence: float = 0.0,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.alpha_distance = float(alpha_distance)
        self.beta_heading = float(beta_heading)
        self.gamma_confidence = float(gamma_confidence)
        self.new_track_min_confidence = float(new_track_min_confidence)

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def _det_to_xyxy(self, det: Dict[str, Any]) -> Tuple[np.ndarray, TrackExtras]:
        x = float(det.get("x", 0.0))
        y = float(det.get("y", 0.0))
        w = float(det.get("width", 0.0))
        h = float(det.get("height", 0.0))

        distance = float(det.get("distance", float("nan")))
        heading = float(det.get("heading", float("nan")))
        confidence = float(det.get("confidence", float("nan")))
        category = det.get("category", "other")

        extras = TrackExtras(
            distance=distance,
            heading=heading,
            confidence=confidence,
            category=category,
        )

        x2 = x + w
        y2 = y + h
        
        return np.array([x, y, x2, y2], dtype=np.float32), extras

    def assign(self, detections: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Returns assigned track IDs for each detection (1-based IDs, like MOT format).
        If there are no detections, returns [].
        """
        self.frame_count += 1

        if len(detections) == 0:
            # Still age/predict existing trackers.
            to_del = []
            for t in range(len(self.trackers)):
                pos = self.trackers[t].predict()[0]
                if np.any(np.isnan(pos)):
                    to_del.append(t)
            for t in reversed(to_del):
                self.trackers.pop(t)
            # Remove dead tracklets.
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                i -= 1
                if trk.time_since_update > self.max_age:
                    self.trackers.pop(i)
            return []

        det_xyxy = np.zeros((len(detections), 4), dtype=np.float32)
        det_dist = np.full((len(detections),), np.nan, dtype=np.float32)
        det_heading = np.full((len(detections),), np.nan, dtype=np.float32)
        det_conf = np.full((len(detections),), np.nan, dtype=np.float32)
        det_extras: List[TrackExtras] = []
        for i, d in enumerate(detections):
            xyxy, extras = self._det_to_xyxy(d)
            det_xyxy[i, :] = xyxy
            det_extras.append(extras)
            det_dist[i] = extras.distance
            det_heading[i] = extras.heading
            det_conf[i] = extras.confidence

        # Predict trackers and build tracker arrays.
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        trk_dist = np.full((len(self.trackers),), np.nan, dtype=np.float32)
        trk_heading = np.full((len(self.trackers),), np.nan, dtype=np.float32)
        trk_conf = np.full((len(self.trackers),), np.nan, dtype=np.float32)
        to_del: List[int] = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = pos.astype(np.float32)
            if np.any(np.isnan(pos)):
                to_del.append(t)
            trk_dist[t] = float(self.trackers[t].extras.distance)
            trk_heading[t] = float(self.trackers[t].extras.heading)
            trk_conf[t] = float(self.trackers[t].extras.confidence)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)
            trk_dist = np.delete(trk_dist, t, axis=0)
            trk_heading = np.delete(trk_heading, t, axis=0)
            trk_conf = np.delete(trk_conf, t, axis=0)

        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_xyxy,
            trks,
            det_dist,
            trk_dist,
            det_heading,
            trk_heading,
            det_conf,
            iou_threshold=self.iou_threshold,
            alpha_distance=self.alpha_distance,
            beta_heading=self.beta_heading,
            gamma_confidence=self.gamma_confidence,
        )

        assigned_ids: List[Optional[int]] = [None] * len(detections)

        # Update matched trackers with assigned detections.
        for det_idx, trk_idx in matches:
            self.trackers[int(trk_idx)].update(det_xyxy[int(det_idx), :], det_extras[int(det_idx)])
            trk = self.trackers[int(trk_idx)]
            track_id = int(trk.id) + 1
            if trk.hit_streak >= self.min_hits:
                assigned_ids[int(det_idx)] = track_id

        # Create and initialize new trackers for unmatched detections.
        for det_idx in unmatched_dets:
            conf = float(det_extras[int(det_idx)].confidence)
            if not np.isnan(conf) and conf < self.new_track_min_confidence:
                continue
            trk = KalmanBoxTracker(det_xyxy[int(det_idx), :], det_extras[int(det_idx)])
            self.trackers.append(trk)
            track_id = int(trk.id) + 1
            if trk.hit_streak >= self.min_hits:
                assigned_ids[int(det_idx)] = track_id

        # Remove dead tracklets.
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return assigned_ids


