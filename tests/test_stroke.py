import math
from whiteboard.stroke import Stroke


def test_add_point_filters_near_duplicates():
    s = Stroke()
    assert s.add_point(0.1, 0.1) is True
    assert s.add_point(0.1001, 0.1001, min_distance=0.01) is False
    assert s.add_point(0.2, 0.2, min_distance=0.01) is True
    assert len(s.points) == 2


def test_add_point_clamps_to_unit_square():
    s = Stroke()
    s.add_point(-0.5, 1.5)
    assert s.points[0] == (0.0, 1.0)


def test_length_straight_line():
    s = Stroke()
    s.points = [(0.0, 0.0), (0.0, 1.0)]
    assert math.isclose(s.length(), 1.0)


def test_bounding_box():
    s = Stroke()
    s.points = [(0.1, 0.5), (0.9, 0.2), (0.4, 0.8)]
    assert s.bounding_box() == (0.1, 0.2, 0.9, 0.8)


def test_distance_to_point_on_segment():
    s = Stroke()
    s.points = [(0.0, 0.0), (1.0, 0.0)]
    d = s.distance_to_point(0.5, 0.1)
    assert math.isclose(d, 0.1, rel_tol=1e-6)


def test_translate_and_scale():
    s = Stroke()
    s.points = [(0.0, 0.0), (1.0, 1.0)]
    s.translate(0.1, 0.1)
    assert s.points == [(0.1, 0.1), (1.1, 1.1)]
    s.points = [(0.0, 0.0), (2.0, 0.0)]
    s.scale(2.0, origin=(0.0, 0.0))
    assert s.points == [(0.0, 0.0), (4.0, 0.0)]


def test_round_trip_dict():
    s = Stroke(tool="pen", thickness=0.01, color="#ff0000")
    s.add_point(0.2, 0.3)
    s.add_point(0.4, 0.5)
    d = s.to_dict()
    s2 = Stroke.from_dict(d)
    assert s2.tool == s.tool
    assert s2.points == s.points
    assert s2.id == s.id
