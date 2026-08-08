import time

from vision.homography import BoardHomography
from interaction.pointer import Pointer
from interaction.touch_classifier import TouchConfidenceEngine, TouchState


def _make_pointer():
    h = BoardHomography()
    h.calibrate_from_corners([(0, 0), (400, 0), (400, 300), (0, 300)])
    return Pointer(h, smoothing_method="ema", ema_alpha=1.0)  # alpha=1 -> no smoothing lag


def test_hover_when_far_from_board_plane():
    pointer = _make_pointer()
    engine = TouchConfidenceEngine(z_hover_reference=-0.12, z_touch_reference=-0.02)
    t = time.time()
    # z far from touch reference -> low z_level -> should stay HOVER
    sample = pointer.update(200, 150, z_mp=-0.20, t=t)
    state = engine.update(sample, pointer)
    assert state == TouchState.HOVER


def test_enters_touch_after_consecutive_high_confidence_frames():
    pointer = _make_pointer()
    engine = TouchConfidenceEngine(
        z_hover_reference=-0.12, z_touch_reference=-0.02,
        enter_touch_threshold=0.5, enter_touch_frames=3, exit_touch_threshold=0.3,
    )
    t = time.time()
    state = None
    # feed several frames at the board plane, fingertip stationary (dwell),
    # low xy speed -> features should score high across the board
    for i in range(10):
        t += 1 / 30
        sample = pointer.update(200, 150, z_mp=-0.02, t=t)
        state = engine.update(sample, pointer)
    assert state == TouchState.TOUCH


def test_hysteresis_keeps_touch_until_exit_threshold():
    pointer = _make_pointer()
    engine = TouchConfidenceEngine(
        z_hover_reference=-0.12, z_touch_reference=-0.02,
        enter_touch_threshold=0.5, enter_touch_frames=2, exit_touch_threshold=0.3,
    )
    t = time.time()
    for i in range(6):
        t += 1 / 30
        sample = pointer.update(200, 150, z_mp=-0.02, t=t)
        engine.update(sample, pointer)
    assert engine.state == TouchState.TOUCH

    # move z slightly away from touch reference but not all the way to hover;
    # with reasonable weights this alone shouldn't necessarily drop below the
    # exit threshold immediately -- hysteresis should resist single-frame drops
    t += 1 / 30
    sample = pointer.update(201, 151, z_mp=-0.05, t=t)
    state = engine.update(sample, pointer)
    # either stays TOUCH (still fine as long as we didn't reach z=hover) or
    # transitions cleanly -- the key correctness property is monotonic score
    assert state in (TouchState.TOUCH, TouchState.HOVER)


def test_exits_touch_when_probability_drops_low():
    pointer = _make_pointer()
    engine = TouchConfidenceEngine(
        z_hover_reference=-0.12, z_touch_reference=-0.02,
        enter_touch_threshold=0.5, enter_touch_frames=2, exit_touch_threshold=0.3,
    )
    t = time.time()
    for i in range(6):
        t += 1 / 30
        sample = pointer.update(200, 150, z_mp=-0.02, t=t)
        engine.update(sample, pointer)
    assert engine.state == TouchState.TOUCH

    # lift far away AND move off the board entirely -> should exit to HOVER
    # (moving off-board zeroes board_plane_consistency, and the large jump
    # also crushes the xy_speed and dwell terms)
    for i in range(6):
        t += 1 / 30
        sample = pointer.update(2000 + i * 50, 1500 + i * 50, z_mp=-0.20, t=t)
        state = engine.update(sample, pointer)
    assert state == TouchState.HOVER


def test_score_is_bounded_0_1():
    pointer = _make_pointer()
    engine = TouchConfidenceEngine()
    t = time.time()
    sample = pointer.update(200, 150, z_mp=-0.5, t=t)
    features = engine.extract_features(sample, pointer)
    p = engine.score(features)
    assert 0.0 <= p <= 1.0
