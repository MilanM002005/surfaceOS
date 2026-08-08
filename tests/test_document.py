import json
import math
import tempfile
from pathlib import Path

from whiteboard.document import BoardDocument
from whiteboard.stroke import Stroke


def _make_stroke(doc, points):
    s = doc.start_stroke()
    for u, v in points:
        s.add_point(u, v)
    doc.commit_stroke(s)
    return s


def test_commit_and_undo_stroke():
    doc = BoardDocument()
    s = _make_stroke(doc, [(0.1, 0.1), (0.2, 0.2)])
    assert len(doc.visible_strokes()) == 1
    assert doc.undo() is True
    assert len(doc.visible_strokes()) == 0
    assert doc.redo() is True
    assert len(doc.visible_strokes()) == 1


def test_erase_near():
    doc = BoardDocument()
    _make_stroke(doc, [(0.5, 0.5), (0.51, 0.51)])
    erased = doc.erase_near(0.5, 0.5, radius=0.05)
    assert len(erased) == 1
    assert len(doc.visible_strokes()) == 0
    doc.undo()
    assert len(doc.visible_strokes()) == 1


def test_clear_is_undoable():
    doc = BoardDocument()
    _make_stroke(doc, [(0.1, 0.1), (0.2, 0.1)])
    _make_stroke(doc, [(0.3, 0.3), (0.4, 0.3)])
    doc.clear()
    assert len(doc.visible_strokes()) == 0
    doc.undo()
    assert len(doc.visible_strokes()) == 2


def test_move_strokes():
    doc = BoardDocument()
    s = _make_stroke(doc, [(0.1, 0.1), (0.2, 0.1)])
    doc.move_strokes([s.id], 0.05, 0.0)
    assert math.isclose(s.points[0][0], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    doc.undo()
    assert math.isclose(s.points[0][0], 0.1, rel_tol=1e-9, abs_tol=1e-9)


def test_save_and_load_round_trip():
    doc = BoardDocument()
    _make_stroke(doc, [(0.1, 0.1), (0.9, 0.9)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.json"
        doc.save(path)
        assert path.exists()
        loaded = BoardDocument.load(path)
        assert len(loaded.visible_strokes()) == 1
        assert loaded.visible_strokes()[0].points == doc.visible_strokes()[0].points


def test_history_max_depth_and_redo_cleared_on_new_action():
    doc = BoardDocument()
    s1 = _make_stroke(doc, [(0.0, 0.0), (0.1, 0.1)])
    doc.undo()
    # redo stack has s1; committing something new should clear it
    s2 = _make_stroke(doc, [(0.5, 0.5), (0.6, 0.6)])
    assert doc.redo() is False
    assert len(doc.visible_strokes()) == 1
    assert doc.visible_strokes()[0].id == s2.id
