"""MediaPipe Hand Landmarker wrapper (Phase 1).

Wraps the MediaPipe Tasks Hand Landmarker API and exposes just the
landmarks this project cares about (index tip/DIP/PIP/MCP, thumb tip,
middle tip, wrist), converted to normalized image coordinates plus the
MediaPipe-provided relative z.

mediapipe is an optional runtime dependency: importing this module
without mediapipe installed will only fail when you actually try to
construct a HandTracker, so the rest of the codebase (which the test
suite exercises) stays importable without a full CV environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

LANDMARK_NAMES = {
    0: "wrist",
    4: "thumb_tip",
    5: "index_mcp",
    6: "index_pip",
    7: "index_dip",
    8: "index_tip",
    12: "middle_tip",
}


@dataclass
class HandLandmarks:
    handedness: str                         # "Left" | "Right"
    points_norm: np.ndarray                 # (21, 3) -> x, y in [0,1], z relative
    points_px: np.ndarray                   # (21, 2) -> pixel coords
    score: float

    def get(self, index: int) -> np.ndarray:
        return self.points_norm[index]

    def named(self, name: str) -> np.ndarray:
        for idx, n in LANDMARK_NAMES.items():
            if n == name:
                return self.points_norm[idx]
        raise KeyError(name)


class HandTracker:
    def __init__(self, model_path: Optional[str] = None, num_hands: int = 1,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for live hand tracking. "
                "Install it with `pip install mediapipe`. "
                "(Pure logic modules -- touch classifier, whiteboard model, "
                "homography, tests -- work without it.)"
            ) from exc

        self._mp = mp
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{model_path}'. Download "
                "hand_landmarker.task from Google's MediaPipe model zoo and "
                "place it there (see README setup instructions)."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandLandmarks]:
        mp = self._mp
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        hands: List[HandLandmarks] = []
        if not result.hand_landmarks:
            return hands

        for i, landmarks in enumerate(result.hand_landmarks):
            pts_norm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
            pts_px = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
            handedness = "Right"
            score = 1.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                handedness = cat.category_name
                score = cat.score
            hands.append(HandLandmarks(handedness=handedness, points_norm=pts_norm,
                                        points_px=pts_px, score=score))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
