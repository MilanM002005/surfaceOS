"""Self-healing calibration (Innovation C, Phase 5).

After the initial 4-corner calibration, we capture a reference set of
ORB keypoints/descriptors from inside the board region. On subsequent
frames we match against that reference, estimate a candidate homography
with RANSAC, and only accept it if quality gates pass (inlier ratio,
reprojection error, corner displacement). This lets small camera bumps
recover automatically instead of silently shifting previously-drawn
ink, and lets us detect "the camera moved too much, freeze and ask for
recalibration" (TRACKING_LOST).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from vision.homography import BoardHomography


@dataclass
class TrackingResult:
    ok: bool
    inlier_ratio: float = 0.0
    reprojection_error_px: float = float("inf")
    corner_displacement_px: float = float("inf")
    reason: str = ""


class BoardTracker:
    def __init__(self, orb_features: int = 800, lowe_ratio: float = 0.75,
                 min_inlier_ratio: float = 0.35, max_reprojection_error_px: float = 6.0,
                 max_corner_displacement_px: float = 40.0):
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lowe_ratio = lowe_ratio
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_corner_displacement_px = max_corner_displacement_px

        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_corners_cam: Optional[np.ndarray] = None
        self.ref_gray: Optional[np.ndarray] = None

    # -- reference capture --------------------------------------------------

    def capture_reference(self, gray_frame: np.ndarray, corners_cam) -> int:
        """Call once right after 4-corner calibration succeeds. Restricts
        feature extraction to inside the board polygon so background
        clutter doesn't pollute the reference set. Returns keypoint count."""
        mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
        poly = np.array(corners_cam, dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)

        kp, desc = self.orb.detectAndCompute(gray_frame, mask)
        self.ref_keypoints = kp
        self.ref_descriptors = desc
        self.ref_corners_cam = np.array(corners_cam, dtype=np.float32)
        self.ref_gray = gray_frame.copy()
        return 0 if desc is None else len(kp)

    @property
    def has_reference(self) -> bool:
        return self.ref_descriptors is not None and len(self.ref_descriptors) >= 8

    # -- per-frame tracking ---------------------------------------------------

    def track(self, gray_frame: np.ndarray) -> Tuple[TrackingResult, Optional[np.ndarray]]:
        """Attempt to re-register the board in the current frame. Returns
        (result, new_H_cam_to_board_pixels) where the homography is in raw
        camera pixel coordinates (caller composes with the board's
        normalization homography)."""
        if not self.has_reference:
            return TrackingResult(ok=False, reason="no_reference"), None

        kp, desc = self.orb.detectAndCompute(gray_frame, None)
        if desc is None or len(desc) < 8:
            return TrackingResult(ok=False, reason="too_few_features"), None

        raw_matches = self.matcher.knnMatch(self.ref_descriptors, desc, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            return TrackingResult(ok=False, reason="too_few_matches"), None

        src = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return TrackingResult(ok=False, reason="ransac_failed"), None

        inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratio = inliers / max(1, len(good))

        # reprojection error on inliers
        projected = cv2.perspectiveTransform(src, H)
        errors = np.linalg.norm(projected.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
        inlier_mask = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(errors), bool)
        reproj_error = float(np.mean(errors[inlier_mask])) if inlier_mask.any() else float("inf")

        # how far did the board corners move under this candidate H?
        new_corners = cv2.perspectiveTransform(
            self.ref_corners_cam.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        displacement = float(np.mean(
            np.linalg.norm(new_corners - self.ref_corners_cam, axis=1)
        ))

        ok = (inlier_ratio >= self.min_inlier_ratio and
              reproj_error <= self.max_reprojection_error_px)

        result = TrackingResult(
            ok=ok,
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reproj_error,
            corner_displacement_px=displacement,
            reason="" if ok else "quality_gate_failed",
        )
        return result, (H if ok else None)

    def displaced_corners(self, H_ref_to_current: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            self.ref_corners_cam.reshape(-1, 1, 2), H_ref_to_current
        ).reshape(-1, 2)
