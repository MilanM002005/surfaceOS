from interaction.gesture_engine import (
    PinchDetector, TwoFingerDetector, FistHoldDetector, GestureEngine, Gesture,
)


def test_pinch_detector_triggers_within_threshold():
    d = PinchDetector(threshold=0.05)
    assert d.update((0.5, 0.5), (0.51, 0.51)) is True
    assert d.update((0.5, 0.5), (0.9, 0.9)) is False


def test_two_finger_contact_and_tap():
    d = TwoFingerDetector(threshold=0.05, tap_window_seconds=0.5)
    t = 0.0
    g = d.update((0.5, 0.5), (0.52, 0.52), t=t)
    assert g == Gesture.TWO_FINGER_CONTACT
    # release quickly -> tap
    t += 0.1
    g = d.update((0.5, 0.5), (0.9, 0.9), t=t)
    assert g == Gesture.TWO_FINGER_TAP


def test_two_finger_long_contact_is_not_a_tap():
    d = TwoFingerDetector(threshold=0.05, tap_window_seconds=0.3)
    t = 0.0
    d.update((0.5, 0.5), (0.52, 0.52), t=t)
    t += 1.0  # held long past the tap window
    g = d.update((0.5, 0.5), (0.9, 0.9), t=t)
    assert g == Gesture.NONE


def test_fist_hold_triggers_once_after_duration():
    d = FistHoldDetector(hold_seconds=0.5, curl_threshold=0.06)
    curled = [0.02, 0.02, 0.02, 0.02]
    t = 0.0
    assert d.update(curled, t=t) is False
    t += 0.6
    assert d.update(curled, t=t) is True
    # still curled -- should not re-trigger every frame
    t += 0.1
    assert d.update(curled, t=t) is False


def test_fist_hold_resets_when_hand_opens():
    d = FistHoldDetector(hold_seconds=0.3, curl_threshold=0.06)
    t = 0.0
    d.update([0.02] * 4, t=t)
    t += 0.1
    d.update([0.3] * 4, t=t)  # hand opens, resets timer
    t += 0.25
    # not enough sustained curl time since re-closing hasn't happened
    assert d.update([0.3] * 4, t=t) is False


def test_gesture_engine_priority_fist_over_pinch():
    engine = GestureEngine(fist_hold_seconds=0.1)
    t = 0.0
    curled = [0.01, 0.01, 0.01, 0.01]
    engine.update(fingertip_to_mcp_distances=curled, t=t)
    t += 0.2
    g = engine.update(
        index_tip_board=(0.5, 0.5), thumb_tip_board=(0.51, 0.51),
        fingertip_to_mcp_distances=curled, t=t,
    )
    assert g == Gesture.FIST_HOLD
