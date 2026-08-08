import tempfile
import numpy as np
import cv2

from calibration.calibration_store import CalibrationStore, CalibrationRecord
from calibration.corner_calibration import CornerCalibrationSession
from vision.homography import BoardHomography


def _textured_board_frame(w=640, h=480, offset=(0, 0)):
    rng = np.random.default_rng(42)
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    ox, oy = offset
    x0, y0, x1, y1 = 100 + ox, 80 + oy, 540 + ox, 400 + oy
    # random textured patches inside the board region give ORB something to key on
    region = frame[y0:y1, x0:x1]
    noise = rng.integers(0, 255, size=region.shape, dtype=np.uint8)
    # sparse blocky pattern (looks more like real-world texture than pure noise)
    block = 16
    for by in range(0, region.shape[0], block):
        for bx in range(0, region.shape[1], block):
            if rng.random() > 0.5:
                region[by:by+block, bx:bx+block] = noise[by:by+block, bx:bx+block]
    frame[y0:y1, x0:x1] = region
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return frame, corners


def test_calibration_record_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        store = CalibrationStore(calibration_dir=tmp + "/cal", fingerprints_dir=tmp + "/fp")
        record = CalibrationRecord(
            corners_cam=[(10, 10), (400, 12), (398, 300), (8, 298)],
            aspect_ratio=1.33, z_hover_reference=-0.12, z_touch_reference=-0.02,
        )
        store.save_calibration("test_session", record)
        loaded = store.load_calibration("test_session")
        assert loaded is not None
        assert loaded.aspect_ratio == record.aspect_ratio
        assert loaded.corners_cam == record.corners_cam
        assert "test_session" in store.list_calibrations()


def test_load_missing_calibration_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = CalibrationStore(calibration_dir=tmp + "/cal", fingerprints_dir=tmp + "/fp")
        assert store.load_calibration("does_not_exist") is None


def test_fingerprint_save_and_match_same_scene():
    with tempfile.TemporaryDirectory() as tmp:
        store = CalibrationStore(calibration_dir=tmp + "/cal", fingerprints_dir=tmp + "/fp")
        frame, corners = _textured_board_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fid = store.save_fingerprint(gray, corners)

        # match against a slightly shifted version of the same scene, simulating
        # the user returning later with the camera in a nearby position
        shifted, _ = _textured_board_frame(offset=(6, 4))
        shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        match = store.match_fingerprint(shifted_gray, min_inliers=4)
        assert match is not None
        matched_id, H = match
        assert matched_id == fid
        assert H.shape == (3, 3)


def test_fingerprint_no_match_on_unrelated_scene():
    with tempfile.TemporaryDirectory() as tmp:
        store = CalibrationStore(calibration_dir=tmp + "/cal", fingerprints_dir=tmp + "/fp")
        frame, corners = _textured_board_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        store.save_fingerprint(gray, corners)

        blank = np.full((480, 640), 128, dtype=np.uint8)
        match = store.match_fingerprint(blank, min_inliers=15)
        assert match is None


def test_corner_calibration_session_flow():
    session = CornerCalibrationSession()
    assert session.is_ready is False
    session.add_click(10, 10)
    session.add_click(400, 12)
    session.add_click(398, 300)
    done = session.add_click(8, 298)
    assert done is True
    assert session.is_ready is True

    h = BoardHomography()
    session.finalize(h)
    assert h.is_calibrated
    assert session.confirmed is True
