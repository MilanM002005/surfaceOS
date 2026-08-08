"""Interaction state machine (implementation.md section 9).

    UNCALIBRATED
        v
    CALIBRATING
        v
    READY/HOVER
        v touch confidence high
    DRAWING
        v confidence low
    READY/HOVER

    READY
     +-- eraser gesture -> ERASING
     +-- lasso gesture  -> SELECTING
     +-- board lost     -> TRACKING_LOST
     +-- camera moved   -> RECOVERING_CALIBRATION

Implemented explicitly (not scattered ifs) so accidental strokes are
structurally impossible: e.g. ERASING and DRAWING can never be active
at once, and nothing can enter DRAWING except from HOVER on a rising
touch-confidence edge.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, Optional


class State(Enum):
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATING = "CALIBRATING"
    HOVER = "HOVER"
    DRAWING = "DRAWING"
    ERASING = "ERASING"
    SELECTING = "SELECTING"
    COMMAND = "COMMAND"
    TRACKING_LOST = "TRACKING_LOST"
    RECOVERING_CALIBRATION = "RECOVERING_CALIBRATION"


# Explicit adjacency list of legal transitions. Any transition attempted
# outside this table is rejected (raises ValueError in strict mode, or
# silently ignored otherwise) rather than happening implicitly.
_TRANSITIONS: Dict[State, set] = {
    State.UNCALIBRATED: {State.CALIBRATING},
    State.CALIBRATING: {State.HOVER, State.UNCALIBRATED},
    State.HOVER: {State.DRAWING, State.ERASING, State.SELECTING, State.COMMAND,
                  State.TRACKING_LOST, State.RECOVERING_CALIBRATION},
    State.DRAWING: {State.HOVER, State.TRACKING_LOST, State.RECOVERING_CALIBRATION},
    State.ERASING: {State.HOVER, State.TRACKING_LOST, State.RECOVERING_CALIBRATION},
    State.SELECTING: {State.HOVER, State.TRACKING_LOST, State.RECOVERING_CALIBRATION},
    State.COMMAND: {State.HOVER, State.TRACKING_LOST, State.RECOVERING_CALIBRATION},
    State.TRACKING_LOST: {State.RECOVERING_CALIBRATION, State.HOVER, State.UNCALIBRATED},
    State.RECOVERING_CALIBRATION: {State.HOVER, State.TRACKING_LOST, State.UNCALIBRATED},
}


class InteractionStateMachine:
    def __init__(self, on_transition: Optional[Callable[[State, State], None]] = None,
                 strict: bool = True):
        self.state = State.UNCALIBRATED
        self.on_transition = on_transition
        self.strict = strict
        self._history = [self.state]

    def can_transition(self, target: State) -> bool:
        return target in _TRANSITIONS.get(self.state, set())

    def transition(self, target: State) -> bool:
        if target == self.state:
            return True
        if not self.can_transition(target):
            if self.strict:
                raise ValueError(f"Illegal transition {self.state} -> {target}")
            return False
        prev = self.state
        self.state = target
        self._history.append(target)
        if self.on_transition:
            self.on_transition(prev, target)
        return True

    def is_drawing_allowed(self) -> bool:
        return self.state == State.DRAWING

    def is_interactive(self) -> bool:
        """False while calibrating, uncalibrated, or recovering -- input
        should be ignored / frozen (implementation.md: 'freeze drawing and
        request recalibration when confidence is unsafe')."""
        return self.state not in (State.UNCALIBRATED, State.CALIBRATING,
                                   State.TRACKING_LOST, State.RECOVERING_CALIBRATION)

    @property
    def history(self):
        return list(self._history)
