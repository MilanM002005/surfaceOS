"""Calibration persistence + Surface Fingerprint (section 12).

Two responsibilities:

1. Save/load the numeric calibration (corner points, aspect ratio,
   personalized touch thresholds) so a user doesn't have to redo
   4-corner + touch calibration every single launch on the *same*
   camera position.

2. Surface Fingerprint: a compact ORB descriptor set for the board
   region, saved under a fingerprint id. On a later startup (even after
   the camera was unplugged and repositioned close to the same spot),
   SurfaceOS can match the current view against saved fingerprints,
   figure out "this is the board I drew on last time", restore that
   board's homography, and reload its BoardDocument -- persistent
   physical-surface memory (section 16, stretch goal).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

Point = Tuple[float, float]


@dataclass
class CalibrationRecord:
    corners_cam: List[Point]
    aspect_ratio: float
    z_hover_reference: float
    z_touch_reference: float
    board_fingerprint_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "corners_cam": [list(p) for p in self.corners_cam],
            "aspect_ratio": self.aspect_ratio,
            "z_hover_reference": self.z_hover_reference,
            "z_touch_reference": self.z_touch_reference,
            "board_fingerprint_id": self.board_fingerprint_id,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "CalibrationRecord":
        return CalibrationRecord(
            corners_cam=[tuple(p) for p in d["corners_cam"]],
            aspect_ratio=d["aspect_ratio"],
            z_hover_reference=d.get("z_hover_reference", -0.12),
            z_touch_reference=d.get("z_touch_reference", -0.02),
            board_fingerprint_id=d.get("board_fingerprint_id"),
            created_at=d.get("created_at", time.time()),
        )


class CalibrationStore:
    def __init__(self, calibration_dir: str = "data/calibration",
                 fingerprints_dir: str = "data/fingerprints"):
        self.calibration_dir = Path(calibration_dir)
        self.fingerprints_dir = Path(fingerprints_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)
        self._orb = cv2.ORB_create(nfeatures=800)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # -- numeric calibration --------------------------------------------------

    def save_calibration(self, name: str, record: CalibrationRecord) -> Path:
        path = self.calibration_dir / f"{name}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))
        return path

    def load_calibration(self, name: str) -> Optional[CalibrationRecord]:
        path = self.calibration_dir / f"{name}.json"
        if not path.exists():
            return None
        return CalibrationRecord.from_dict(json.loads(path.read_text()))

    def list_calibrations(self) -> List[str]:
        return [p.stem for p in self.calibration_dir.glob("*.json")]

    # -- surface fingerprint --------------------------------------------------

    def save_fingerprint(self, gray_frame: np.ndarray, corners_cam: List[Point],
                          fingerprint_id: Optional[str] = None) -> str:
        fingerprint_id = fingerprint_id or uuid.uuid4().hex[:12]
        mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
        poly = np.array(corners_cam, dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        kp, desc = self._orb.detectAndCompute(gray_frame, mask)

        if desc is None:
            raise RuntimeError("No features found inside board region to fingerprint")

        pts = np.array([k.pt for k in kp], dtype=np.float32)
        np.savez(
            self.fingerprints_dir / f"{fingerprint_id}.npz",
            descriptors=desc,
            keypoints_xy=pts,
            corners_cam=np.array(corners_cam, dtype=np.float32),
            frame_shape=np.array(gray_frame.shape[:2]),
        )
        return fingerprint_id

    def match_fingerprint(self, gray_frame: np.ndarray, lowe_ratio: float = 0.75,
                           min_inliers: int = 15) -> Optional[Tuple[str, np.ndarray]]:
        """Try every saved fingerprint against the current frame. Returns
        (fingerprint_id, H_fingerprint_to_current) for the best match, or
        None if nothing matches well enough."""
        kp, desc = self._orb.detectAndCompute(gray_frame, None)
        if desc is None or len(desc) < 8:
            return None

        best_id, best_H, best_inliers = None, None, min_inliers - 1

        for path in self.fingerprints_dir.glob("*.npz"):
            data = np.load(path)
            ref_desc = data["descriptors"]
            ref_pts = data["keypoints_xy"]
            if ref_desc is None or len(ref_desc) < 8:
                continue

            matches = self._matcher.knnMatch(ref_desc, desc, k=2)
            good = [m for pair in matches if len(pair) == 2
                    for m, n in [pair] if m.distance < lowe_ratio * n.distance]
            if len(good) < 8:
                continue

            src = np.float32([ref_pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                continue
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers > best_inliers:
                best_inliers = inliers
                best_id = path.stem
                best_H = H

        if best_id is None:
            return None
        return best_id, best_H
