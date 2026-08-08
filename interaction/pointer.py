"""FingerSample history (implementation.md 7.2) and the Pointer that turns
raw hand landmarks into a smoothed, board-mapped, feature-rich stream
the Touch Confidence Engine consumes.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from vision.filters import OneEuroFilter, EMAFilter, VelocityEstimator
from vision.homography import BoardHomography, BoardNotCalibratedError


@dataclass
class FingerSample:
    timestamp: float
    x_cam: float
    y_cam: float
    z_mp: float
    u_board: float
    v_board: float
    velocity: float
    acceleration: float
    dz: float
    finger_extension: float
    touch_probability: float = 0.0
    on_board: bool = True


class Pointer:
    """Consumes raw index-fingertip landmarks per frame, produces a
    smoothed board-space FingerSample, and keeps a short rolling history
    for the touch classifier's temporal features (dwell, trajectory
    flattening, velocity/acceleration).
    """

    def __init__(self, homography: BoardHomography, smoothing_method: str = "one_euro",
                 ema_alpha: float = 0.35, one_euro_min_cutoff: float = 1.0,
                 one_euro_beta: float = 0.02, one_euro_d_cutoff: float = 1.0,
                 history_seconds: float = 2.0):
        self.homography = homography
        self.smoothing_method = smoothing_method
        self._ema = EMAFilter(alpha=ema_alpha)
        self._one_euro = OneEuroFilter(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta,
                                        d_cutoff=one_euro_d_cutoff)
        self._velocity = VelocityEstimator(window=5)
        self.history: Deque[FingerSample] = deque()
        self.history_seconds = history_seconds
        self._last_z: Optional[float] = None
        self._last_speed: Optional[float] = None
        self._last_v_time: Optional[float] = None

    def reset(self) -> None:
        self._ema.reset()
        self._one_euro.reset()
        self._velocity.reset()
        self.history.clear()
        self._last_z = None
        self._last_speed = None
        self._last_v_time = None

    def update(self, x_cam: float, y_cam: float, z_mp: float,
               index_dip_px: Optional[tuple] = None, index_mcp_px: Optional[tuple] = None,
               t: Optional[float] = None) -> Optional[FingerSample]:
        """x_cam, y_cam are pixel coordinates of the index fingertip.
        z_mp is MediaPipe's relative depth for that landmark.
        index_dip_px / index_mcp_px (optional) are used to compute finger
        extension angle (feature f4)."""
        t = t if t is not None else time.time()

        if self.smoothing_method == "ema":
            sx, sy = self._ema.filter(x_cam, y_cam)
        else:
            sx, sy = self._one_euro.filter(x_cam, y_cam, t)

        try:
            u, v = self.homography.cam_to_board(sx, sy)
            on_board = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
        except BoardNotCalibratedError:
            u, v, on_board = 0.0, 0.0, False

        speed = self._velocity.speed(sx, sy, t)
        accel = 0.0
        if self._last_speed is not None and self._last_v_time is not None:
            dt = max(1e-6, t - self._last_v_time)
            accel = (speed - self._last_speed) / dt
        self._last_speed, self._last_v_time = speed, t

        dz = 0.0 if self._last_z is None else (z_mp - self._last_z)
        self._last_z = z_mp

        extension = 1.0
        if index_dip_px is not None and index_mcp_px is not None:
            extension = _finger_extension_ratio((sx, sy), index_dip_px, index_mcp_px)

        sample = FingerSample(
            timestamp=t, x_cam=sx, y_cam=sy, z_mp=z_mp, u_board=u, v_board=v,
            velocity=speed, acceleration=accel, dz=dz, finger_extension=extension,
            on_board=on_board,
        )
        self.history.append(sample)
        self._trim_history(t)
        return sample

    def _trim_history(self, now: float) -> None:
        while self.history and now - self.history[0].timestamp > self.history_seconds:
            self.history.popleft()

    def dwell_time(self, window_seconds: float, radius_board_units: float = 0.02) -> float:
        """Approximate how long the fingertip has stayed within a small
        board-space radius, used as touch feature f7. Returns seconds."""
        if not self.history:
            return 0.0
        recent = [s for s in self.history if self.history[-1].timestamp - s.timestamp <= window_seconds]
        if len(recent) < 2:
            return 0.0
        ref_u, ref_v = recent[-1].u_board, recent[-1].v_board
        still = [s for s in recent
                 if math.hypot(s.u_board - ref_u, s.v_board - ref_v) <= radius_board_units]
        if not still:
            return 0.0
        return still[-1].timestamp - still[0].timestamp

    @property
    def latest(self) -> Optional[FingerSample]:
        return self.history[-1] if self.history else None


def _finger_extension_ratio(tip, dip, mcp) -> float:
    """1.0 = fully extended finger (tip, dip, mcp roughly collinear),
    lower values indicate a curled/bent finger. Used as feature f4/f5."""
    v1 = (dip[0] - mcp[0], dip[1] - mcp[1])
    v2 = (tip[0] - dip[0], tip[1] - dip[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    # angle 0 -> collinear/extended -> 1.0 ; angle pi -> folded back -> 0.0
    angle = math.acos(cos_angle)
    return max(0.0, 1.0 - angle / math.pi)
