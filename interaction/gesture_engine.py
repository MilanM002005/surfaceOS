"""Gesture engine (implementation.md Innovation E / section 4).

Gestures are deliberately state-machine driven and kept separate from
the drawing pipeline so a normal stroke can never be misread as a
command. Implements:
  - pinch-to-draw (Phase 3 MVP pen-down, before the Touch Confidence
    Engine takes over in Phase 4)
  - two-finger contact -> eraser mode
  - closed-fist hold (0.6s) -> command mode
  - two-finger tap -> undo
  - lasso gesture flag (driven externally by Selection, this module
    just detects the triggering hand shape)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Gesture(Enum):
    NONE = "none"
    PINCH = "pinch"                 # thumb+index pinch (temporary pen-down)
    TWO_FINGER_CONTACT = "two_finger_contact"
    FIST_HOLD = "fist_hold"
    TWO_FINGER_TAP = "two_finger_tap"


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PinchDetector:
    """Distance between thumb tip and index tip in normalized board units.
    Used only as the Phase-3 placeholder pen-down before real touch
    inference replaces it (Phase 4)."""

    def __init__(self, threshold: float = 0.045):
        self.threshold = threshold
        self.active = False

    def update(self, index_tip_board, thumb_tip_board) -> bool:
        d = _dist(index_tip_board, thumb_tip_board)
        self.active = d < self.threshold
        return self.active


class TwoFingerDetector:
    """Index + middle tip close together and both near the board plane ->
    eraser mode, distinct from a single-finger touch/draw."""

    def __init__(self, threshold: float = 0.06, tap_window_seconds: float = 0.4):
        self.threshold = threshold
        self.tap_window_seconds = tap_window_seconds
        self.active = False
        self._contact_start: Optional[float] = None
        self._last_release: Optional[float] = None
        self._tap_pending = False

    def update(self, index_tip_board, middle_tip_board, t: Optional[float] = None) -> Gesture:
        t = t if t is not None else time.time()
        d = _dist(index_tip_board, middle_tip_board)
        was_active = self.active
        self.active = d < self.threshold

        if self.active and not was_active:
            self._contact_start = t
        if not self.active and was_active:
            # a brief two-finger contact counts as a "tap" -> undo
            if self._contact_start is not None and (t - self._contact_start) < self.tap_window_seconds:
                self._last_release = t
                return Gesture.TWO_FINGER_TAP
            self._contact_start = None

        return Gesture.TWO_FINGER_CONTACT if self.active else Gesture.NONE


class FistHoldDetector:
    """All four non-thumb fingertips curled close to the palm/MCPs for a
    sustained duration -> enter command mode. Uses average distance from
    fingertip to its own MCP as a curl proxy (small = curled)."""

    def __init__(self, hold_seconds: float = 0.6, curl_threshold: float = 0.06):
        self.hold_seconds = hold_seconds
        self.curl_threshold = curl_threshold
        self._fist_start: Optional[float] = None
        self.triggered = False

    def update(self, fingertip_to_mcp_distances: list, t: Optional[float] = None) -> bool:
        """fingertip_to_mcp_distances: list of normalized distances for
        index/middle/ring/pinky (tip-to-mcp). Returns True exactly once
        when the hold threshold is first crossed."""
        t = t if t is not None else time.time()
        is_fist = all(d < self.curl_threshold for d in fingertip_to_mcp_distances)

        if not is_fist:
            self._fist_start = None
            self.triggered = False
            return False

        if self._fist_start is None:
            self._fist_start = t

        held = t - self._fist_start
        if held >= self.hold_seconds and not self.triggered:
            self.triggered = True
            return True
        return False


class GestureEngine:
    """Aggregates the individual detectors into one place the state
    machine can query per frame."""

    def __init__(self, pinch_threshold: float = 0.045, two_finger_threshold: float = 0.06,
                 fist_hold_seconds: float = 0.6, undo_tap_window_seconds: float = 0.4):
        self.pinch = PinchDetector(threshold=pinch_threshold)
        self.two_finger = TwoFingerDetector(threshold=two_finger_threshold,
                                             tap_window_seconds=undo_tap_window_seconds)
        self.fist = FistHoldDetector(hold_seconds=fist_hold_seconds)

    def update(self, *, index_tip_board=None, thumb_tip_board=None,
               middle_tip_board=None, fingertip_to_mcp_distances=None,
               t: Optional[float] = None) -> Gesture:
        t = t if t is not None else time.time()

        if fingertip_to_mcp_distances is not None:
            if self.fist.update(fingertip_to_mcp_distances, t):
                return Gesture.FIST_HOLD

        if index_tip_board is not None and middle_tip_board is not None:
            g = self.two_finger.update(index_tip_board, middle_tip_board, t)
            if g in (Gesture.TWO_FINGER_TAP, Gesture.TWO_FINGER_CONTACT):
                return g

        if index_tip_board is not None and thumb_tip_board is not None:
            if self.pinch.update(index_tip_board, thumb_tip_board):
                return Gesture.PINCH

        return Gesture.NONE
