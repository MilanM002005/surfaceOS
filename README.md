# surfaceOS
Self-Calibrating Spatial Whiteboard - A camera-only spatial interaction system that turns a fixed physical
whiteboard/wall/desk into a writable digital surface using a bare
finger as a pen.Board calibration + homography, MediaPipe hand tracking, a rule-based
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

