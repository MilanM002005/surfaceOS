"""Homography computation and coordinate mapping (implementation.md 7.1).

Camera pixel <-> normalized board coordinate (u, v) in [0, 1]^2.
This is the single source of truth for "where is the board" that every
other module (touch classifier, renderer, self-healing tracker) relies
on.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import cv2

Point = Tuple[float, float]


class BoardNotCalibratedError(RuntimeError):
    pass


def is_convex_quad(corners: List[Point]) -> bool:
    """Validate that four points, IN THE GIVEN ORDER, form a simple convex
    quadrilateral (guards against a user misclicking corners out of order,
    e.g. clicking diagonally which produces a self-intersecting 'bowtie').

    Checks two things:
      1. The four points are in "convex position" (none is inside the
         triangle formed by the other three) -- via convex hull size.
      2. Walking the points IN ORDER always turns the same rotational
         direction (all cross products of consecutive edges share sign) --
         this is what actually catches a bowtie ordering, since a bowtie's
         points are still in convex position but visiting them in the
         given order self-intersects.
    """
    if len(corners) != 4:
        return False
    pts = np.array(corners, dtype=np.float64)

    hull = cv2.convexHull(pts.astype(np.float32))
    if len(hull) != 4:
        return False

    cross_signs = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            continue
        cross_signs.append(cross > 0)

    if not cross_signs:
        return False
    return all(s == cross_signs[0] for s in cross_signs)


def order_corners_clockwise(corners: List[Point]) -> List[Point]:
    """Return corners ordered as [top-left, top-right, bottom-right,
    bottom-left] regardless of click order, using centroid angle sort."""
    pts = np.array(corners, dtype=np.float64)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # angles sorted CCW starting from a somewhat arbitrary point; rotate so
    # that the top-left (smallest x+y) is first.
    sums = ordered[:, 0] + ordered[:, 1]
    start = int(np.argmin(sums))
    ordered = np.roll(ordered, -start, axis=0)
    return [tuple(p) for p in ordered]


class BoardHomography:
    """Holds the current camera<->board homography and provides mapping
    helpers. Board coordinates are normalized to [0, 1]^2 regardless of the
    physical aspect ratio; callers that need pixel-accurate rectified
    canvases multiply by (canvas_width, canvas_height) themselves.
    """

    def __init__(self):
        self.H_cam_to_board: Optional[np.ndarray] = None
        self.H_board_to_cam: Optional[np.ndarray] = None
        self.corners_cam: Optional[List[Point]] = None
        self.aspect_ratio: float = 1.0

    @property
    def is_calibrated(self) -> bool:
        return self.H_cam_to_board is not None

    def calibrate_from_corners(self, corners_cam: List[Point],
                                aspect_ratio: Optional[float] = None) -> None:
        """corners_cam must be 4 points in camera pixel space, ordered
        [top-left, top-right, bottom-right, bottom-left]."""
        if len(corners_cam) != 4:
            raise ValueError("Exactly 4 corners are required")
        if not is_convex_quad(corners_cam):
            raise ValueError("Corners do not form a convex quadrilateral")

        if aspect_ratio is None:
            aspect_ratio = self._estimate_aspect_ratio(corners_cam)
        self.aspect_ratio = aspect_ratio

        # Destination is normalized board space [0,1] x [0,1/aspect] so
        # that a single 'aspect_ratio' float fully describes board shape.
        board_h = 1.0 / aspect_ratio if aspect_ratio >= 1.0 else 1.0
        board_w = 1.0 if aspect_ratio >= 1.0 else aspect_ratio
        dst = np.array([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]],
                        dtype=np.float32)
        src = np.array(corners_cam, dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("Homography computation failed (degenerate corners)")
        self.H_cam_to_board = H
        self.H_board_to_cam = np.linalg.inv(H)
        self.corners_cam = list(corners_cam)

    @staticmethod
    def _estimate_aspect_ratio(corners_cam: List[Point]) -> float:
        pts = np.array(corners_cam, dtype=np.float64)
        top = np.linalg.norm(pts[1] - pts[0])
        bottom = np.linalg.norm(pts[2] - pts[3])
        left = np.linalg.norm(pts[3] - pts[0])
        right = np.linalg.norm(pts[2] - pts[1])
        width = (top + bottom) / 2.0
        height = (left + right) / 2.0
        if height < 1e-6:
            return 1.0
        return max(width / height, 1e-3)

    def cam_to_board(self, x: float, y: float) -> Point:
        if self.H_cam_to_board is None:
            raise BoardNotCalibratedError("Board has not been calibrated yet")
        pt = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_cam_to_board)[0, 0]
        # normalize into [0,1] against the longer of width/height axis
        bw = 1.0 if self.aspect_ratio >= 1.0 else self.aspect_ratio
        bh = 1.0 / self.aspect_ratio if self.aspect_ratio >= 1.0 else 1.0
        return (float(out[0] / bw), float(out[1] / bh))

    def board_to_cam(self, u: float, v: float) -> Point:
        if self.H_board_to_cam is None:
            raise BoardNotCalibratedError("Board has not been calibrated yet")
        bw = 1.0 if self.aspect_ratio >= 1.0 else self.aspect_ratio
        bh = 1.0 / self.aspect_ratio if self.aspect_ratio >= 1.0 else 1.0
        pt = np.array([[[u * bw, v * bh]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_board_to_cam)[0, 0]
        return (float(out[0]), float(out[1]))

    def update(self, new_H_cam_to_board: np.ndarray) -> None:
        """Replace the current homography (used by self-healing recovery
        after RANSAC feature re-registration)."""
        self.H_cam_to_board = new_H_cam_to_board
        self.H_board_to_cam = np.linalg.inv(new_H_cam_to_board)

    def rectify(self, frame: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        """Warp the raw camera frame into a fronto-parallel board view of
        size (width, height) pixels -- used for the 'rectified preview'."""
        if self.H_cam_to_board is None:
            raise BoardNotCalibratedError("Board has not been calibrated yet")
        w, h = out_size
        bw = 1.0 if self.aspect_ratio >= 1.0 else self.aspect_ratio
        bh = 1.0 / self.aspect_ratio if self.aspect_ratio >= 1.0 else 1.0
        # scale board-normalized homography to pixel-space destination
        S = np.array([[w / bw, 0, 0], [0, h / bh, 0], [0, 0, 1]], dtype=np.float64)
        H_px = S @ self.H_cam_to_board
        return cv2.warpPerspective(frame, H_px, (w, h))
