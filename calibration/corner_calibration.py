"""Four-point board calibration workflow (Phase 2 / section 4 setup steps
3-6): collect four corner clicks (or an auto-detected proposal the user
confirms), validate them, and produce a BoardHomography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from vision.homography import BoardHomography, is_convex_quad, order_corners_clockwise

Point = Tuple[float, float]


@dataclass
class CornerCalibrationSession:
    collected: List[Point] = field(default_factory=list)
    confirmed: bool = False

    def reset(self) -> None:
        self.collected = []
        self.confirmed = False

    def add_click(self, x: float, y: float) -> bool:
        """Add a corner click. Returns True once the 4th corner completes
        the set (caller should then call finalize())."""
        if len(self.collected) >= 4:
            return True
        self.collected.append((x, y))
        return len(self.collected) == 4

    def undo_last(self) -> None:
        if self.collected:
            self.collected.pop()

    @property
    def is_ready(self) -> bool:
        return len(self.collected) == 4

    def propose_from_detector(self, corners: List[Point]) -> None:
        """Seed the session from an automatic board-detector candidate
        (section 11: 'proposes a board boundary and asks for confirmation
        instead of requiring four manual points every session')."""
        self.collected = list(corners[:4])

    def finalize(self, homography: BoardHomography,
                 aspect_ratio: Optional[float] = None) -> None:
        if not self.is_ready:
            raise ValueError("Need exactly 4 corners before finalizing")
        ordered = order_corners_clockwise(self.collected)
        if not is_convex_quad(ordered):
            raise ValueError(
                "Selected corners do not form a valid convex quadrilateral; "
                "please re-select the four board corners in order."
            )
        homography.calibrate_from_corners(ordered, aspect_ratio=aspect_ratio)
        self.confirmed = True
