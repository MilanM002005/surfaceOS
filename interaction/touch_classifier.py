"""Touch Confidence Engine (implementation.md Innovation B / section 7.4).

Combines several weak cues into P(touch | observations) in [0,1], then
applies hysteresis so a single noisy frame can't flicker the state
between HOVER and TOUCH:

    P < 0.35                -> HOVER
    P > 0.70 for N frames    -> TOUCH
    stays TOUCH until P < 0.45

Ships as a rule-based scorer (debuggable, no training data needed) but
exposes learn-friendly per-frame features so a later logistic
regression / random forest / MLP classifier (section 7.4, "later
collect labelled examples") can be swapped in without touching the
rest of the pipeline -- see `LearnedTouchClassifier` at the bottom,
which implements the same `score(sample_features) -> float` interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from collections import deque

from interaction.pointer import FingerSample, Pointer


@dataclass
class TouchFeatures:
    z_level: float              # normalized closeness to board plane, 1=touching
    z_velocity: float           # how fast z is decreasing (approaching)
    xy_speed: float             # normalized inverse speed (slow = more touch-like)
    finger_extension: float     # 0..1
    dwell: float                 # normalized dwell time
    board_plane_consistency: float  # 1 if on_board and near plane, decays otherwise


@dataclass
class TouchWeights:
    z_level: float = 0.28
    z_velocity: float = 0.18
    xy_speed: float = 0.14
    finger_extension: float = 0.10
    dwell: float = 0.15
    board_plane_consistency: float = 0.15

    def normalized(self) -> Dict[str, float]:
        total = (self.z_level + self.z_velocity + self.xy_speed + self.finger_extension +
                 self.dwell + self.board_plane_consistency)
        if total <= 0:
            total = 1.0
        return {
            "z_level": self.z_level / total,
            "z_velocity": self.z_velocity / total,
            "xy_speed": self.xy_speed / total,
            "finger_extension": self.finger_extension / total,
            "dwell": self.dwell / total,
            "board_plane_consistency": self.board_plane_consistency / total,
        }


class TouchState:
    HOVER = "HOVER"
    TOUCH = "TOUCH"


class TouchConfidenceEngine:
    def __init__(self, enter_touch_threshold: float = 0.70, enter_touch_frames: int = 3,
                 exit_touch_threshold: float = 0.45, hover_threshold: float = 0.35,
                 weights: Optional[TouchWeights] = None,
                 dwell_window_seconds: float = 0.25,
                 z_touch_reference: float = -0.02, z_hover_reference: float = -0.12):
        self.enter_touch_threshold = enter_touch_threshold
        self.enter_touch_frames = enter_touch_frames
        self.exit_touch_threshold = exit_touch_threshold
        self.hover_threshold = hover_threshold
        self.weights = weights or TouchWeights()
        self.dwell_window_seconds = dwell_window_seconds
        # MediaPipe z is negative-ish toward camera / relative to wrist; these
        # references are tunable per-user via calibration/touch_calibration.py
        self.z_touch_reference = z_touch_reference
        self.z_hover_reference = z_hover_reference

        self.state = TouchState.HOVER
        self._consecutive_high = 0
        self.last_probability = 0.0

    def calibrate_z_references(self, hover_z_samples, touch_z_samples) -> None:
        if hover_z_samples:
            self.z_hover_reference = sum(hover_z_samples) / len(hover_z_samples)
        if touch_z_samples:
            self.z_touch_reference = sum(touch_z_samples) / len(touch_z_samples)

    def extract_features(self, sample: FingerSample, pointer: Pointer) -> TouchFeatures:
        # z_level: 1.0 at touch reference, 0.0 at (or beyond) hover reference
        span = self.z_hover_reference - self.z_touch_reference
        if abs(span) < 1e-6:
            z_level = 0.5
        else:
            z_level = (self.z_hover_reference - sample.z_mp) / span
        z_level = max(0.0, min(1.0, z_level))

        # z_velocity: negative dz (moving toward touch reference) is good
        z_velocity = max(0.0, min(1.0, -sample.dz * 25.0 + 0.5))

        # xy_speed: slow lateral motion is more touch-consistent (contact
        # dampens tangential speed -- "trajectory flattening", cue 7)
        xy_speed = max(0.0, 1.0 - min(1.0, sample.velocity / 400.0))

        finger_extension = max(0.0, min(1.0, sample.finger_extension))

        dwell_seconds = pointer.dwell_time(self.dwell_window_seconds)
        dwell = max(0.0, min(1.0, dwell_seconds / self.dwell_window_seconds))

        board_plane_consistency = 1.0 if sample.on_board else 0.0

        return TouchFeatures(
            z_level=z_level, z_velocity=z_velocity, xy_speed=xy_speed,
            finger_extension=finger_extension, dwell=dwell,
            board_plane_consistency=board_plane_consistency,
        )

    def score(self, features: TouchFeatures) -> float:
        w = self.weights.normalized()
        p = (w["z_level"] * features.z_level +
             w["z_velocity"] * features.z_velocity +
             w["xy_speed"] * features.xy_speed +
             w["finger_extension"] * features.finger_extension +
             w["dwell"] * features.dwell +
             w["board_plane_consistency"] * features.board_plane_consistency)
        return max(0.0, min(1.0, p))

    def update(self, sample: FingerSample, pointer: Pointer) -> str:
        """Compute this frame's probability, apply hysteresis, update and
        return the new state (TouchState.HOVER | TouchState.TOUCH)."""
        features = self.extract_features(sample, pointer)
        p = self.score(features)
        self.last_probability = p
        sample.touch_probability = p

        if self.state == TouchState.HOVER:
            if p > self.enter_touch_threshold:
                self._consecutive_high += 1
                if self._consecutive_high >= self.enter_touch_frames:
                    self.state = TouchState.TOUCH
                    self._consecutive_high = 0
            else:
                self._consecutive_high = 0
        else:  # TOUCH
            if p < self.exit_touch_threshold:
                self.state = TouchState.HOVER
                self._consecutive_high = 0

        return self.state

    def reset(self) -> None:
        self.state = TouchState.HOVER
        self._consecutive_high = 0
        self.last_probability = 0.0


class LearnedTouchClassifier:
    """Drop-in replacement for the rule-based scorer once labelled hover/
    touch examples have been collected (section 7.4: "train a tiny
    classifier such as logistic regression, random forest, or MLP").
    Requires scikit-learn.
    """

    FEATURE_ORDER = ["z_level", "z_velocity", "xy_speed", "finger_extension",
                      "dwell", "board_plane_consistency"]

    def __init__(self, model=None):
        self.model = model

    def fit(self, feature_rows, labels) -> None:
        from sklearn.linear_model import LogisticRegression
        X = [[getattr(f, name) for name in self.FEATURE_ORDER] for f in feature_rows]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def score(self, features: TouchFeatures) -> float:
        if self.model is None:
            raise RuntimeError("LearnedTouchClassifier has not been fit yet")
        x = [[getattr(features, name) for name in self.FEATURE_ORDER]]
        return float(self.model.predict_proba(x)[0][1])

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
