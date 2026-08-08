"""Per-session touch calibration (Phase 4):

    10s hover
    10s touch
    5 horizontal touch strokes
    5 vertical touch strokes

Collects labelled z / feature samples used to (a) set personalized
z_hover_reference / z_touch_reference on the rule-based
TouchConfidenceEngine, and (b) optionally train a LearnedTouchClassifier
once enough sessions have been logged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from interaction.pointer import FingerSample
from interaction.touch_classifier import TouchConfidenceEngine, TouchFeatures


class CalibrationStage(Enum):
    HOVER = "hover"
    TOUCH = "touch"
    HORIZONTAL_STROKES = "horizontal_strokes"
    VERTICAL_STROKES = "vertical_strokes"
    DONE = "done"

    @classmethod
    def order(cls) -> List["CalibrationStage"]:
        return [cls.HOVER, cls.TOUCH, cls.HORIZONTAL_STROKES, cls.VERTICAL_STROKES, cls.DONE]


@dataclass
class TouchCalibrationSession:
    hover_seconds: float = 10.0
    touch_seconds: float = 10.0
    strokes_per_direction: int = 5

    stage: CalibrationStage = CalibrationStage.HOVER
    _stage_start: Optional[float] = None
    hover_z_samples: List[float] = field(default_factory=list)
    touch_z_samples: List[float] = field(default_factory=list)
    touch_feature_rows: List[TouchFeatures] = field(default_factory=list)
    touch_labels: List[int] = field(default_factory=list)
    horizontal_stroke_count: int = 0
    vertical_stroke_count: int = 0

    def start(self) -> None:
        self.stage = CalibrationStage.HOVER
        self._stage_start = time.time()
        self.hover_z_samples.clear()
        self.touch_z_samples.clear()
        self.touch_feature_rows.clear()
        self.touch_labels.clear()
        self.horizontal_stroke_count = 0
        self.vertical_stroke_count = 0

    def feed(self, sample: FingerSample, engine: TouchConfidenceEngine, pointer) -> None:
        if self.stage == CalibrationStage.HOVER:
            self.hover_z_samples.append(sample.z_mp)
            features = engine.extract_features(sample, pointer)
            self.touch_feature_rows.append(features)
            self.touch_labels.append(0)
        elif self.stage == CalibrationStage.TOUCH:
            self.touch_z_samples.append(sample.z_mp)
            features = engine.extract_features(sample, pointer)
            self.touch_feature_rows.append(features)
            self.touch_labels.append(1)
        elif self.stage in (CalibrationStage.HORIZONTAL_STROKES,
                             CalibrationStage.VERTICAL_STROKES):
            features = engine.extract_features(sample, pointer)
            self.touch_feature_rows.append(features)
            self.touch_labels.append(1)

    def register_stroke_completed(self) -> None:
        if self.stage == CalibrationStage.HORIZONTAL_STROKES:
            self.horizontal_stroke_count += 1
        elif self.stage == CalibrationStage.VERTICAL_STROKES:
            self.vertical_stroke_count += 1

    def maybe_advance(self, now: Optional[float] = None) -> bool:
        """Advance to the next stage if the current stage's exit condition
        is met. Returns True if a stage transition occurred."""
        now = now if now is not None else time.time()
        if self._stage_start is None:
            self._stage_start = now
        elapsed = now - self._stage_start

        advanced = False
        if self.stage == CalibrationStage.HOVER and elapsed >= self.hover_seconds:
            self.stage = CalibrationStage.TOUCH
            advanced = True
        elif self.stage == CalibrationStage.TOUCH and elapsed >= self.touch_seconds:
            self.stage = CalibrationStage.HORIZONTAL_STROKES
            advanced = True
        elif (self.stage == CalibrationStage.HORIZONTAL_STROKES and
              self.horizontal_stroke_count >= self.strokes_per_direction):
            self.stage = CalibrationStage.VERTICAL_STROKES
            advanced = True
        elif (self.stage == CalibrationStage.VERTICAL_STROKES and
              self.vertical_stroke_count >= self.strokes_per_direction):
            self.stage = CalibrationStage.DONE
            advanced = True

        if advanced:
            self._stage_start = now
        return advanced

    @property
    def is_done(self) -> bool:
        return self.stage == CalibrationStage.DONE

    def apply_to_engine(self, engine: TouchConfidenceEngine) -> None:
        engine.calibrate_z_references(self.hover_z_samples, self.touch_z_samples)

    def progress(self) -> dict:
        return {
            "stage": self.stage.value,
            "hover_samples": len(self.hover_z_samples),
            "touch_samples": len(self.touch_z_samples),
            "horizontal_strokes": self.horizontal_stroke_count,
            "vertical_strokes": self.vertical_stroke_count,
        }
