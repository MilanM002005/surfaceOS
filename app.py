"""SurfaceOS main application.

Wires together every module in vision/, interaction/, whiteboard/, and
calibration/ into the runnable pipeline described in implementation.md
section 5 (Architecture) and section 11 (Implementation Plan).

Run:
    python app.py [--config config/default.yaml] [--doc data/documents/my_board.json]

Controls (shown on-screen as an on-screen HUD too):
    Mouse click x4      Select board corners during CALIBRATING
    r                    Restart / redo corner calibration
    t                    Run per-session touch calibration sequence
    p                    Toggle pinch-mode vs touch-confidence mode
    e                    Hold to force eraser mode (also: two-finger gesture)
    z                    Undo
    y                    Undo-redo (redo)
    s                    Save document
    x                    Export SVG
    c                    Clear board (asks nothing -- undoable)
    q / ESC              Quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from vision.camera import Camera
from vision.hand_tracker import HandTracker
from vision.homography import BoardHomography
from vision.board_tracker import BoardTracker
from vision.board_detector import best_candidate

from interaction.pointer import Pointer
from interaction.touch_classifier import TouchConfidenceEngine, TouchWeights
from interaction.gesture_engine import GestureEngine, Gesture
from interaction.state_machine import InteractionStateMachine, State

from whiteboard.document import BoardDocument, BoardMeta
from whiteboard.stroke import Stroke
from whiteboard.eraser import PathEraser
from whiteboard.selection import Selection
from whiteboard.renderer import Renderer

from calibration.corner_calibration import CornerCalibrationSession
from calibration.touch_calibration import TouchCalibrationSession
from calibration.calibration_store import CalibrationStore, CalibrationRecord


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SurfaceOSApp:
    def __init__(self, config: dict, doc_path: Optional[str] = None):
        self.cfg = config
        cam_cfg = config["camera"]
        self.camera = Camera(
            device_index=cam_cfg["device_index"], width=cam_cfg["width"],
            height=cam_cfg["height"], target_fps=cam_cfg["target_fps"],
            fourcc=cam_cfg.get("fourcc"),
        )

        ht_cfg = config["hand_tracking"]
        self.hand_tracker: Optional[HandTracker] = None
        self._hand_tracker_cfg = ht_cfg  # constructed lazily in run() to give a
                                          # clean error if the model file is missing

        self.homography = BoardHomography()
        st_cfg = config["self_healing_calibration"]
        self.board_tracker = BoardTracker(
            orb_features=st_cfg["orb_features"], lowe_ratio=st_cfg["lowe_ratio"],
            min_inlier_ratio=st_cfg["min_inlier_ratio"],
            max_reprojection_error_px=st_cfg["max_reprojection_error_px"],
            max_corner_displacement_px=st_cfg["max_corner_displacement_px"],
        )
        self.self_healing_enabled = st_cfg["enabled"]
        self.recheck_every_n_frames = st_cfg["recheck_every_n_frames"]

        sm_cfg = config["smoothing"]
        self.pointer = Pointer(
            self.homography, smoothing_method=sm_cfg["method"],
            ema_alpha=sm_cfg["ema_alpha"],
            one_euro_min_cutoff=sm_cfg["one_euro"]["min_cutoff"],
            one_euro_beta=sm_cfg["one_euro"]["beta"],
            one_euro_d_cutoff=sm_cfg["one_euro"]["d_cutoff"],
        )

        tc_cfg = config["touch_confidence"]
        self.touch_engine = TouchConfidenceEngine(
            enter_touch_threshold=tc_cfg["enter_touch_threshold"],
            enter_touch_frames=tc_cfg["enter_touch_frames"],
            exit_touch_threshold=tc_cfg["exit_touch_threshold"],
            hover_threshold=tc_cfg["hover_threshold"],
            weights=TouchWeights(**tc_cfg["weights"]),
            dwell_window_seconds=tc_cfg["dwell_window_seconds"],
        )

        ge_cfg = config["gesture_engine"]
        self.gestures = GestureEngine(
            pinch_threshold=ge_cfg["pinch_distance_threshold"],
            two_finger_threshold=ge_cfg["two_finger_distance_threshold"],
            fist_hold_seconds=ge_cfg["fist_hold_seconds"],
            undo_tap_window_seconds=ge_cfg["undo_tap_window_seconds"],
        )

        self.sm = InteractionStateMachine(strict=False)

        stroke_cfg = config["stroke"]
        self.default_thickness = stroke_cfg["default_thickness"]
        self.min_point_distance = stroke_cfg["min_point_distance"]

        board_cfg = config["board"]
        self.renderer = Renderer(board_cfg["canonical_width"], board_cfg["canonical_height"])

        self.doc_path = Path(doc_path) if doc_path else Path(config["paths"]["documents_dir"]) / "session.json"
        self.document = BoardDocument.load(self.doc_path) if self.doc_path.exists() else BoardDocument()

        self.calibration_store = CalibrationStore(
            calibration_dir=config["paths"]["calibration_dir"],
            fingerprints_dir=config["paths"]["fingerprints_dir"],
        )

        self.corner_session = CornerCalibrationSession()
        self.touch_calib_session: Optional[TouchCalibrationSession] = None
        self.path_eraser = PathEraser(self.document, radius=0.025)
        self.selection = Selection(self.document)

        self.use_touch_confidence = False  # start in pinch mode (Phase 3), toggled with 'p'
        self._active_stroke: Optional[Stroke] = None
        self._frame_count = 0
        self._last_click_xy = None

        self.window_name = "SurfaceOS"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    # -- setup ---------------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.sm.state == State.CALIBRATING:
            done = self.corner_session.add_click(x, y)
            if done:
                try:
                    self.corner_session.finalize(self.homography)
                    self.sm.transition(State.HOVER)
                    print("[SurfaceOS] Calibration complete.")
                except ValueError as e:
                    print(f"[SurfaceOS] Calibration failed: {e}. Press 'r' to retry.")
                    self.corner_session.reset()

    def _ensure_hand_tracker(self) -> None:
        if self.hand_tracker is None:
            self.hand_tracker = HandTracker(**self._hand_tracker_cfg_kwargs())

    def _hand_tracker_cfg_kwargs(self) -> dict:
        c = self._hand_tracker_cfg
        return dict(
            model_path=c["model_path"], num_hands=c["num_hands"],
            min_hand_detection_confidence=c["min_hand_detection_confidence"],
            min_hand_presence_confidence=c["min_hand_presence_confidence"],
            min_tracking_confidence=c["min_tracking_confidence"],
        )

    # -- main loop -----------------------------------------------------------

    def run(self) -> None:
        self._ensure_hand_tracker()
        self.sm.transition(State.CALIBRATING)
        print("[SurfaceOS] Click the four board corners: TL, TR, BR, BL.")

        with self.camera:
            try:
                while True:
                    frame = self.camera.read()
                    self._frame_count += 1
                    image = frame.image
                    self._step(image, frame.timestamp, frame.fps_estimate)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    self._handle_key(key)
            finally:
                self.hand_tracker.close()
                cv2.destroyAllWindows()

    def _step(self, image: np.ndarray, t: float, fps: float) -> None:
        display = image.copy()

        if self.sm.state == State.CALIBRATING:
            self._draw_calibration_overlay(display)
            cv2.imshow(self.window_name, display)
            return

        if self.sm.state in (State.UNCALIBRATED,):
            cv2.imshow(self.window_name, display)
            return

        # self-healing calibration check
        if self.self_healing_enabled and self._frame_count % self.recheck_every_n_frames == 0:
            self._run_self_healing(image)

        if self.sm.state == State.TRACKING_LOST:
            self._draw_hud(display, fps, extra="TRACKING LOST -- hold board still or press 'r'")
            cv2.imshow(self.window_name, display)
            return

        hands = self.hand_tracker.process(image, int(t * 1000))
        sample = None
        gesture = Gesture.NONE

        if hands:
            hand = hands[0]
            idx_tip = hand.points_px[8]
            idx_dip = tuple(hand.points_px[7])
            idx_mcp = tuple(hand.points_px[5])
            thumb_tip = hand.points_norm[4]
            middle_tip = hand.points_norm[12]

            sample = self.pointer.update(
                idx_tip[0], idx_tip[1], hand.points_norm[8][2],
                index_dip_px=idx_dip, index_mcp_px=idx_mcp, t=t,
            )

            if self.use_touch_confidence:
                touch_state = self.touch_engine.update(sample, self.pointer)
            else:
                touch_state = None  # driven by pinch gesture below instead

            try:
                idx_board = self.homography.cam_to_board(*idx_tip)
                thumb_board = self.homography.cam_to_board(*hand.points_px[4])
                middle_board = self.homography.cam_to_board(*hand.points_px[12])
            except Exception:
                idx_board = thumb_board = middle_board = (0.0, 0.0)

            gesture = self.gestures.update(
                index_tip_board=idx_board, thumb_tip_board=thumb_board,
                middle_tip_board=middle_board, t=t,
            )

            self._apply_gesture_and_touch(sample, gesture, touch_state)
        else:
            self._end_active_stroke()
            if self.sm.state == State.DRAWING:
                self.sm.transition(State.HOVER)

        overlay = self.renderer.render_overlay(
            self.document, image, self.homography.H_board_to_cam,
            highlight_ids=self.selection.selected_ids,
        )
        self._draw_pointer(overlay, sample)
        self._draw_hud(overlay, fps)
        cv2.imshow(self.window_name, overlay)

        canvas = self.renderer.render_canvas(self.document,
                                              highlight_ids=self.selection.selected_ids)
        cv2.imshow(self.window_name + " - Board", canvas)

    def _apply_gesture_and_touch(self, sample, gesture: Gesture, touch_state) -> None:
        if gesture == Gesture.TWO_FINGER_TAP:
            self.document.undo()
            return

        if gesture == Gesture.FIST_HOLD:
            print("[SurfaceOS] Command mode gesture detected.")
            self.sm.transition(State.COMMAND)
            self.sm.transition(State.HOVER)
            return

        if gesture == Gesture.TWO_FINGER_CONTACT:
            self._end_active_stroke()
            if self.sm.state != State.ERASING:
                self.sm.transition(State.ERASING)
                self.path_eraser.begin()
            self.path_eraser.feed(sample.u_board, sample.v_board)
            return
        elif self.sm.state == State.ERASING:
            self.path_eraser.end()
            self.sm.transition(State.HOVER)

        pen_down = (gesture == Gesture.PINCH) if not self.use_touch_confidence \
            else (touch_state == "TOUCH")

        if pen_down and sample.on_board:
            if self.sm.state != State.DRAWING:
                self.sm.transition(State.DRAWING)
                self._active_stroke = self.document.start_stroke(
                    thickness=self.default_thickness)
            self._active_stroke.add_point(sample.u_board, sample.v_board,
                                           min_distance=self.min_point_distance)
        else:
            self._end_active_stroke()
            if self.sm.state == State.DRAWING:
                self.sm.transition(State.HOVER)

    def _end_active_stroke(self) -> None:
        if self._active_stroke is not None:
            if len(self._active_stroke.points) >= 2:
                self.document.commit_stroke(self._active_stroke)
            else:
                # discard accidental single-point "strokes"
                if self._active_stroke in self.document.strokes:
                    self.document.strokes.remove(self._active_stroke)
            self._active_stroke = None

    def _run_self_healing(self, image: np.ndarray) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.board_tracker.has_reference:
            if self.homography.corners_cam:
                self.board_tracker.capture_reference(gray, self.homography.corners_cam)
            return

        result, H = self.board_tracker.track(gray)
        if result.ok and H is not None:
            # H maps reference-frame pixels -> current-frame pixels; compose
            # with the original cam->board homography (H is inverse of that
            # drift) to keep board coordinates stable.
            H_inv = np.linalg.inv(H)
            new_cam_to_board_px = self.homography.H_cam_to_board @ H_inv
            self.homography.update(new_cam_to_board_px)
            if self.sm.state == State.TRACKING_LOST:
                self.sm.transition(State.HOVER)
        else:
            if result.corner_displacement_px > self.board_tracker.max_corner_displacement_px * 3:
                # movement too large to trust an automatic fix; freeze ink
                self.sm.transition(State.TRACKING_LOST)

    # -- key handling ----------------------------------------------------------

    def _handle_key(self, key: int) -> None:
        if key == ord("r"):
            self.corner_session.reset()
            self.board_tracker.ref_descriptors = None
            self.sm.transition(State.CALIBRATING)
            print("[SurfaceOS] Recalibrating -- click four corners.")
        elif key == ord("p"):
            self.use_touch_confidence = not self.use_touch_confidence
            mode = "touch-confidence" if self.use_touch_confidence else "pinch"
            print(f"[SurfaceOS] Pen-down mode: {mode}")
        elif key == ord("z"):
            self.document.undo()
        elif key == ord("y"):
            self.document.redo()
        elif key == ord("s"):
            self.document.save(self.doc_path)
            print(f"[SurfaceOS] Saved to {self.doc_path}")
        elif key == ord("x"):
            svg_path = str(self.doc_path.with_suffix(".svg"))
            self.renderer.export_svg(self.document, svg_path)
            print(f"[SurfaceOS] Exported SVG to {svg_path}")
        elif key == ord("c"):
            self.document.clear()
        elif key == ord("t"):
            self.touch_calib_session = TouchCalibrationSession()
            self.touch_calib_session.start()
            print("[SurfaceOS] Touch calibration started: hover, then touch, then trace strokes.")

    # -- drawing helpers (HUD / overlays) ---------------------------------------

    def _draw_calibration_overlay(self, display: np.ndarray) -> None:
        for i, (x, y) in enumerate(self.corner_session.collected):
            cv2.circle(display, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "Click 4 board corners: TL, TR, BR, BL",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _draw_pointer(self, display: np.ndarray, sample) -> None:
        if sample is None:
            return
        color = (0, 255, 0) if self.sm.state == State.DRAWING else (255, 200, 0)
        cv2.circle(display, (int(sample.x_cam), int(sample.y_cam)), 8, color, 2)
        cv2.putText(display, f"P(touch)={sample.touch_probability:.2f}",
                    (int(sample.x_cam) + 12, int(sample.y_cam)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_hud(self, display: np.ndarray, fps: float, extra: str = "") -> None:
        lines = [
            f"FPS: {fps:.1f}  State: {self.sm.state.value}  "
            f"Mode: {'touch' if self.use_touch_confidence else 'pinch'}",
            "[r]calibrate [t]touch-calib [p]mode [z]undo [y]redo [s]save [x]svg [c]clear [q]quit",
        ]
        if extra:
            lines.append(extra)
        for i, line in enumerate(lines):
            cv2.putText(display, line, (20, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="SurfaceOS -- self-calibrating spatial whiteboard")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--doc", default=None, help="Path to a .json board document to load/save")
    args = parser.parse_args()

    config = load_config(args.config)
    app = SurfaceOSApp(config, doc_path=args.doc)
    app.run()


if __name__ == "__main__":
    main()
