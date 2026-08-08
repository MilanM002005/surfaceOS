import numpy as np
import cv2

from vision.board_detector import detect_board_candidates, best_candidate


def _synthetic_board_frame(w=640, h=480):
    frame = np.full((h, w, 3), 40, dtype=np.uint8)  # dark background
    cv2.rectangle(frame, (120, 90), (520, 390), (230, 230, 230), thickness=-1)
    # a bit of texture so edges are crisp for Canny
    cv2.rectangle(frame, (120, 90), (520, 390), (200, 200, 200), thickness=3)
    return frame


def test_detect_board_candidates_finds_rectangle():
    frame = _synthetic_board_frame()
    candidates = detect_board_candidates(frame)
    assert len(candidates) >= 1
    top = candidates[0]
    assert top.score > 0.5
    # corners should roughly bound the drawn rectangle
    xs = [c[0] for c in top.corners]
    ys = [c[1] for c in top.corners]
    assert min(xs) < 160 and max(xs) > 480
    assert min(ys) < 130 and max(ys) > 350


def test_best_candidate_returns_none_for_blank_frame():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = best_candidate(frame)
    assert result is None
