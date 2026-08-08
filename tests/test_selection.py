from whiteboard.document import BoardDocument
from whiteboard.selection import Selection


def _make_stroke(doc, points):
    s = doc.start_stroke()
    for u, v in points:
        s.add_point(u, v)
    doc.commit_stroke(s)
    return s


def test_lasso_selects_enclosed_stroke():
    doc = BoardDocument()
    inside = _make_stroke(doc, [(0.4, 0.4), (0.5, 0.5), (0.6, 0.4)])
    outside = _make_stroke(doc, [(0.9, 0.9), (0.95, 0.95)])

    sel = Selection(doc)
    sel.begin_lasso()
    for p in [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]:
        sel.feed_lasso(*p)
    hits = sel.end_lasso()

    assert inside.id in hits
    assert outside.id not in hits


def test_lasso_too_few_points_selects_nothing():
    doc = BoardDocument()
    sel = Selection(doc)
    sel.begin_lasso()
    sel.feed_lasso(0.1, 0.1)
    hits = sel.end_lasso()
    assert hits == []


def test_select_at_picks_nearest_within_radius():
    doc = BoardDocument()
    s = _make_stroke(doc, [(0.2, 0.2), (0.25, 0.2)])
    sel = Selection(doc)
    hit = sel.select_at(0.21, 0.2, radius=0.05)
    assert hit == s.id
    miss = sel.select_at(0.9, 0.9, radius=0.05)
    assert miss is None


def test_drag_moves_selected_strokes():
    doc = BoardDocument()
    s = _make_stroke(doc, [(0.2, 0.2), (0.25, 0.2)])
    sel = Selection(doc)
    sel.selected_ids = [s.id]
    sel.begin_drag(0.2, 0.2)
    sel.drag_to(0.3, 0.2)
    assert s.points[0][0] > 0.2


def test_group_requires_at_least_two_selected():
    doc = BoardDocument()
    s1 = _make_stroke(doc, [(0.1, 0.1), (0.15, 0.1)])
    sel = Selection(doc)
    sel.selected_ids = [s1.id]
    assert sel.group() is None
    s2 = _make_stroke(doc, [(0.2, 0.2), (0.25, 0.2)])
    sel.selected_ids = [s1.id, s2.id]
    gid = sel.group()
    assert gid is not None
    assert s1.group_id == gid and s2.group_id == gid
