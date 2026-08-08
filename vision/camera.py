"""Camera acquisition (Phase 0): stream + FPS display + timestamps.

Kept deliberately thin -- a context-managed wrapper around cv2.VideoCapture
with rolling FPS measurement, since the rest of the pipeline (touch
confidence, self-healing calibration) needs accurate per-frame dt.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2


@dataclass
class Frame:
    image: "object"          # np.ndarray, BGR
    timestamp: float
    frame_index: int
    fps_estimate: float


class Camera:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720,
                 target_fps: int = 30, fourcc: Optional[str] = "MJPG"):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.fourcc = fourcc
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._timestamps: Deque[float] = deque(maxlen=30)

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        if self.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.device_index}. "
                f"Check that a webcam is connected and not in use by another app."
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("Camera not opened; use `with Camera(...) as cam:`")
        ok, image = self._cap.read()
        now = time.time()
        if not ok:
            raise RuntimeError("Failed to read frame from camera (device disconnected?)")
        self._timestamps.append(now)
        self._frame_index += 1
        return Frame(image=image, timestamp=now, frame_index=self._frame_index,
                     fps_estimate=self._estimate_fps())

    def _estimate_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._cap is None:
            return (self.width, self.height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
