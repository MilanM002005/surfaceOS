"""Smoothing filters for the fingertip pointer.

Raw MediaPipe landmarks jitter frame to frame. implementation.md 7.3
warns against oversmoothing (it adds visible ink latency), so the
default is a One Euro Filter, which adapts its cutoff to the current
speed: slow motion gets heavily smoothed (steady cursor), fast motion
gets lightly smoothed (low lag while drawing quickly).
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple


class EMAFilter:
    """Simple exponential moving average. Cheapest option, used for the
    MVP baseline (Phase 0-3)."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._value: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._value = None

    def filter(self, x: float, y: float) -> Tuple[float, float]:
        if self._value is None:
            self._value = (x, y)
        else:
            px, py = self._value
            self._value = (
                self.alpha * x + (1 - self.alpha) * px,
                self.alpha * y + (1 - self.alpha) * py,
            )
        return self._value


class _LowPassFilter:
    def __init__(self):
        self._y: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = alpha * x + (1 - alpha) * self._y
        return self._y

    @property
    def value(self) -> Optional[float]:
        return self._y


class OneEuroFilter:
    """1-euro filter (Casiez et al. 2012) applied independently to x and y.

    min_cutoff: lower -> more smoothing at low speed (less jitter, more lag)
    beta:       higher -> less lag at high speed (cutoff increases with speed)
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_filters = (_LowPassFilter(), _LowPassFilter())
        self._dx_filters = (_LowPassFilter(), _LowPassFilter())
        self._last_time: Optional[float] = None
        self._last_raw: Optional[Tuple[float, float]] = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self._x_filters = (_LowPassFilter(), _LowPassFilter())
        self._dx_filters = (_LowPassFilter(), _LowPassFilter())
        self._last_time = None
        self._last_raw = None

    def filter(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        t = t if t is not None else time.time()
        if self._last_time is not None:
            dt = max(1e-6, t - self._last_time)
            self.freq = 1.0 / dt
        self._last_time = t

        out = []
        raw = (x, y)
        prev_raw = self._last_raw or raw
        for i, val in enumerate(raw):
            dx = (val - prev_raw[i]) * self.freq
            edx = self._dx_filters[i].filter(dx, self._alpha(self.d_cutoff, self.freq))
            cutoff = self.min_cutoff + self.beta * abs(edx)
            filtered = self._x_filters[i].filter(val, self._alpha(cutoff, self.freq))
            out.append(filtered)
        self._last_raw = raw
        return (out[0], out[1])


class VelocityEstimator:
    """Simple finite-difference velocity/acceleration tracker over a short
    rolling window, used as touch-classifier features (f3 = xy speed)."""

    def __init__(self, window: int = 5):
        self.window = window
        self._samples: Deque[Tuple[float, float, float]] = deque(maxlen=window)  # (t, x, y)

    def reset(self) -> None:
        self._samples.clear()

    def update(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        t = t if t is not None else time.time()
        self._samples.append((t, x, y))
        if len(self._samples) < 2:
            return (0.0, 0.0)
        (t0, x0, y0), (t1, x1, y1) = self._samples[0], self._samples[-1]
        dt = max(1e-6, t1 - t0)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        return (vx, vy)

    def speed(self, x: float, y: float, t: Optional[float] = None) -> float:
        vx, vy = self.update(x, y, t)
        return math.hypot(vx, vy)
