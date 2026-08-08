"""Adaptive board detection (Phase 7 / section 11 research extension).

Proposes a rectangular board boundary automatically so the user can
confirm it with one click instead of tapping four corners every
session. Classical CV pipeline: edges -> contours -> quadrilateral
candidates -> scoring by area + rectangularity. Falls back to "no
proposal" (caller should ask for manual corners) when nothing scores
well -- this is explicitly NOT meant to replace manual calibration,
only to speed it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2

Point = Tuple[float, float]


@dataclass
class BoardCandidate:
    corners: List[Point]      # 4 points, arbitrary order
    score: float
    area_fraction: float


def _order_quad(pts: np.ndarray) -> List[Point]:
    pts = pts.reshape(4, 2).astype(np.float64)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def detect_board_candidates(frame_bgr: np.ndarray, max_candidates: int = 3,
                             min_area_fraction: float = 0.08) -> List[BoardCandidate]:
    """Return a ranked list of quadrilateral board candidates found in the
    frame. Empty list means "nothing convincing found"."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

    candidates: List[BoardCandidate] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < frame_area * min_area_fraction:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        rect_score = _rectangularity_score(approx.reshape(4, 2))
        area_fraction = area / frame_area
        # favor large, rectangular, roughly-centered regions
        score = 0.6 * rect_score + 0.4 * min(1.0, area_fraction / 0.5)

        candidates.append(BoardCandidate(
            corners=_order_quad(approx),
            score=float(score),
            area_fraction=float(area_fraction),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def _rectangularity_score(pts: np.ndarray) -> float:
    """1.0 = perfect rectangle (right angles, opposite sides equal),
    lower for skewed/irregular quads. Robust to perspective skew since we
    check angle closeness to 90 degrees loosely."""
    ordered = np.array(_order_quad(pts))
    angles = []
    for i in range(4):
        p_prev = ordered[(i - 1) % 4]
        p_curr = ordered[i]
        p_next = ordered[(i + 1) % 4]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)
    deviation = np.mean([abs(a - 90) for a in angles])
    return max(0.0, 1.0 - deviation / 45.0)


def best_candidate(frame_bgr: np.ndarray) -> Optional[BoardCandidate]:
    candidates = detect_board_candidates(frame_bgr)
    return candidates[0] if candidates else None
