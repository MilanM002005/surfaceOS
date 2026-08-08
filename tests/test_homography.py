import math
import pytest
import numpy as np

from vision.homography import (
    BoardHomography, BoardNotCalibratedError, is_convex_quad, order_corners_clockwise,
)


def test_is_convex_quad_true_for_rectangle():
    corners = [(0, 0), (100, 0), (100, 80), (0, 80)]
    assert is_convex_quad(corners) is True


def test_is_convex_quad_false_for_self_intersecting():
    # bowtie shape: crossing diagonals order
    corners = [(0, 0), (100, 80), (100, 0), (0, 80)]
    assert is_convex_quad(corners) is False


def test_order_corners_clockwise_from_scrambled_input():
    tl, tr, br, bl = (0, 0), (100, 0), (100, 80), (0, 80)
    scrambled = [br, tl, bl, tr]
    ordered = order_corners_clockwise(scrambled)
    assert ordered[0] == tl


def test_calibrate_and_roundtrip_corners_map_near_unit_square():
    h = BoardHomography()
    corners = [(100, 100), (500, 100), (500, 400), (100, 400)]  # 400x300 -> ar=1.333
    h.calibrate_from_corners(corners)
    assert h.is_calibrated

    # each physical corner should map close to its canonical board corner
    u, v = h.cam_to_board(100, 100)
    assert math.isclose(u, 0.0, abs_tol=1e-3)
    assert math.isclose(v, 0.0, abs_tol=1e-3)

    u, v = h.cam_to_board(500, 400)
    assert math.isclose(u, 1.0, abs_tol=1e-3)
    assert math.isclose(v, 1.0, abs_tol=1e-3)


def test_cam_to_board_and_back_is_consistent():
    h = BoardHomography()
    corners = [(50, 50), (450, 60), (440, 380), (40, 370)]  # slightly oblique
    h.calibrate_from_corners(corners)

    u, v = h.cam_to_board(240, 200)
    x, y = h.board_to_cam(u, v)
    u2, v2 = h.cam_to_board(x, y)
    assert math.isclose(u, u2, abs_tol=1e-4)
    assert math.isclose(v, v2, abs_tol=1e-4)


def test_uncalibrated_raises():
    h = BoardHomography()
    with pytest.raises(BoardNotCalibratedError):
        h.cam_to_board(10, 10)


def test_rectify_produces_expected_size():
    h = BoardHomography()
    corners = [(0, 0), (400, 0), (400, 300), (0, 300)]
    h.calibrate_from_corners(corners)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    rectified = h.rectify(frame, (200, 150))
    assert rectified.shape[:2] == (150, 200)


def test_calibrate_rejects_non_convex():
    h = BoardHomography()
    bad_corners = [(0, 0), (100, 80), (100, 0), (0, 80)]
    with pytest.raises(ValueError):
        h.calibrate_from_corners(bad_corners)
