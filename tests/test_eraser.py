from whiteboard.document import BoardDocument
from whiteboard.eraser import PointEraser, PathEraser


def _make_stroke(doc, points):
    s = doc.start_stroke()
    for u, v in points:
        s.add_point(u, v)
    doc.commit_stroke(s)
    return s


def test_point_eraser_erases_within_radius():
    doc = BoardDocument()
    s = _make_stroke(doc, [(0.5, 0.5), (0.55, 0.5)])
    eraser = PointEraser(doc, radius=0.03)
    hits = eraser.erase_at(0.5, 0.5)
    assert s.id in hits
    assert len(doc.visible_strokes()) == 0


def test_path_eraser_batches_into_single_undo_step():
    doc = BoardDocument()
    s1 = _make_stroke(doc, [(0.1, 0.1), (0.15, 0.1)])
    s2 = _make_stroke(doc, [(0.2, 0.1), (0.25, 0.1)])
    eraser = PathEraser(doc, radius=0.03)
    eraser.begin()
    eraser.feed(0.1, 0.1)
    eraser.feed(0.2, 0.1)
    ids = eraser.end()
    assert set(ids) == {s1.id, s2.id}
    assert len(doc.visible_strokes()) == 0

    doc.undo()
    assert len(doc.visible_strokes()) == 2


def test_path_eraser_noop_when_nothing_hit():
    doc = BoardDocument()
    _make_stroke(doc, [(0.9, 0.9), (0.95, 0.95)])
    eraser = PathEraser(doc, radius=0.01)
    eraser.begin()
    eraser.feed(0.1, 0.1)
    ids = eraser.end()
    assert ids == []
    assert len(doc.visible_strokes()) == 1
