import pytest

from interaction.state_machine import InteractionStateMachine, State


def test_initial_state_is_uncalibrated():
    sm = InteractionStateMachine()
    assert sm.state == State.UNCALIBRATED
    assert sm.is_interactive() is False


def test_normal_flow_to_drawing():
    sm = InteractionStateMachine()
    sm.transition(State.CALIBRATING)
    sm.transition(State.HOVER)
    assert sm.is_interactive() is True
    sm.transition(State.DRAWING)
    assert sm.is_drawing_allowed() is True
    sm.transition(State.HOVER)
    assert sm.is_drawing_allowed() is False


def test_illegal_transition_raises_in_strict_mode():
    sm = InteractionStateMachine(strict=True)
    with pytest.raises(ValueError):
        sm.transition(State.DRAWING)  # can't draw straight from UNCALIBRATED


def test_illegal_transition_returns_false_in_non_strict_mode():
    sm = InteractionStateMachine(strict=False)
    ok = sm.transition(State.DRAWING)
    assert ok is False
    assert sm.state == State.UNCALIBRATED


def test_cannot_be_both_drawing_and_erasing():
    sm = InteractionStateMachine()
    sm.transition(State.CALIBRATING)
    sm.transition(State.HOVER)
    sm.transition(State.DRAWING)
    # must return to HOVER before erasing -- direct DRAWING->ERASING is illegal
    assert sm.can_transition(State.ERASING) is False


def test_tracking_lost_freezes_interaction():
    sm = InteractionStateMachine()
    sm.transition(State.CALIBRATING)
    sm.transition(State.HOVER)
    sm.transition(State.TRACKING_LOST)
    assert sm.is_interactive() is False
    assert sm.is_drawing_allowed() is False


def test_on_transition_callback_fires():
    calls = []
    sm = InteractionStateMachine(on_transition=lambda a, b: calls.append((a, b)))
    sm.transition(State.CALIBRATING)
    sm.transition(State.HOVER)
    assert calls == [(State.UNCALIBRATED, State.CALIBRATING), (State.CALIBRATING, State.HOVER)]
