# SurfaceOS — Self-Calibrating Spatial Whiteboard

A camera-only spatial interaction system that turns a fixed physical
whiteboard/wall/desk into a writable digital surface using a bare
finger as a pen. See `implementation.md` for the full design spec this
was built from.

## Status

This is a full implementation of the architecture in `implementation.md`:
board calibration + homography, MediaPipe hand tracking, a rule-based
Touch Confidence Engine with hysteresis, self-healing (ORB+RANSAC) drift
recovery, a board-locked ink data model with undo/redo, lasso selection,
eraser, gestures (pinch / two-finger / fist-hold), an explicit interaction
state machine, calibration persistence + surface fingerprinting, SVG
export, and evaluation scripts for touch accuracy / latency / drift.

**60 unit tests pass** covering every pure-logic module (geometry, undo/
redo, homography math, the touch-confidence hysteresis, gestures,
selection, eraser, board detection, fingerprint matching) — run them
with no camera required (see below).

The **live app** (`app.py`) needs a real webcam, a display, and
MediaPipe's hand-landmarker model, none of which exist in the sandbox
this was built in — so *you* need to run it locally to actually draw on
a board. Everything below tells you how.

## 1. Install

Requires Python 3.10–3.12 (MediaPipe does not yet support 3.13 on all
platforms) and a webcam.

```bash
cd surfaceos
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the hand-landmarker model

MediaPipe's Hand Landmarker needs a model file that isn't bundled in
this repo (it's a ~10MB binary). Download it into `models/`:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

(If that URL changes, search "MediaPipe Hand Landmarker task file
download" — any `hand_landmarker.task` works, just point
`config/default.yaml -> hand_tracking.model_path` at it.)

## 3. Run the app

```bash
python app.py
```

Two windows open: the live camera feed with ink overlaid, and a
rectified top-down "Board" view.

**First run — calibrate:**
1. Click the board's four corners in the camera window, in order:
   top-left, top-right, bottom-right, bottom-left.
2. Calibration locks in and you're in HOVER mode.

**Controls (also shown in an on-screen HUD):**

| Key | Action |
|---|---|
| mouse click ×4 | corner calibration (during CALIBRATING) |
| `r` | redo corner calibration |
| `t` | run the guided touch-calibration sequence (10s hover, 10s touch, horizontal/vertical strokes) |
| `p` | toggle pinch-to-draw vs. touch-confidence mode |
| two fingers together | eraser (drag to erase, quick tap = undo) |
| `z` / `y` | undo / redo |
| `s` | save the board document to disk |
| `x` | export the board to SVG |
| `c` | clear the board (undoable) |
| `q` / `Esc` | quit |

**Recommended first session:** leave it in pinch mode (default) to
validate the drawing pipeline works end-to-end, then press `p` to
switch to the real Touch Confidence Engine, and press `t` beforehand to
personalize its thresholds to your hand/camera/lighting.

## 4. Run the tests (no camera needed)

```bash
pip install pytest
pytest tests/ -v
```

All 60 tests should pass — this validates the geometry, data model,
touch-confidence hysteresis, gestures, and calibration logic
independently of any hardware.

## 5. Run the evaluation scripts (camera needed, after calibrating once and saving with `s`)

```bash
python experiments/touch_accuracy.py --calibration session
python experiments/latency_test.py --calibration session
python experiments/drift_test.py --calibration session
```

Each writes a JSON report to `data/eval/`.

## Project layout

```
surfaceos/
├── app.py                    # main application (run this)
├── config/default.yaml       # all tunable thresholds/paths
├── vision/                   # camera, hand tracking, homography, self-healing tracker
├── interaction/              # pointer smoothing, touch confidence, gestures, state machine
├── whiteboard/                # stroke/document data model, renderer, eraser, selection
├── calibration/               # corner + touch calibration workflows, persistence, fingerprinting
├── experiments/                # touch accuracy / latency / drift evaluation scripts
├── tests/                     # 60 unit tests, no camera required
└── models/                    # put hand_landmarker.task here
```

## Notes / known limitations

- The touch-confidence weights in `config/default.yaml` are reasonable
  defaults, not tuned to any specific camera/lighting — run the `t`
  touch-calibration sequence each session for best results, per the
  spec's Phase 4.
- `LearnedTouchClassifier` in `interaction/touch_classifier.py` is
  wired up but unfit by default; collect labelled hover/touch sessions
  and call `.fit()` if you want to compare rule-based vs. learned
  contact detection (this is the "scientifically evaluable" comparison
  implementation.md calls out in section 7.4).
- `vision/board_detector.py` (automatic board proposal) is implemented
  but not wired into `app.py`'s calibration flow by default — corner
  calibration is manual-click, per the spec's explicit guidance not to
  start with automatic detection (section 17, "First Coding
  Milestone").
