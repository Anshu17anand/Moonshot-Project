"""
working_enhanced.py

Augments your existing working.py with a Surgical Intelligence Layer:
- SurgicalState object (shared state updated every frame)
- Monocular distance-from-known-size estimation
- Phase detection (SEARCH / ALIGN / APPROACH / EXCISE / RETREAT)
- Context-aware guidance text
- Vision–Language style brief narration on state changes
- Real-time AI Panel displayed as a side box inside the same OpenCV window
- Event logging and automatic end-of-procedure report (JSON)

HOW TO INTEGRATE:
- This file is intentionally self-contained and provides a `run()` loop.
- Replace `run_inference(frame)` implementation with your existing inference call from working.py (or set INFERENCE_URL).
- Replace robot control hooks (`send_robot_command`) with your existing robot API calls; existing code does not modify robot behaviour, only suggests guidance strings.

Dependencies: opencv-python, numpy, requests (if using Roboflow HTTP). Optional: pyttsx3 for TTS (disabled by default).

"""

import cv2
import numpy as np
import time
import math
import json
import requests
from collections import deque

# =========================
# CONFIG
# =========================
# --- Camera / UI ---
CAMERA_PORT = 0
PANEL_WIDTH = 360  # width of the side AI panel (pixels)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Inference ---
# Option A: Use your local Roboflow HTTP inference server (example)
INFERENCE_URL = "http://localhost:9001/moonshot-project-xfdfs/3"  # set to your endpoint
USE_ROBOFLOW_HTTP = True

# --- Known object (thermocol ball) geometry for monocular depth ---
REAL_BALL_DIAMETER_MM = 12.0  # set exact real diameter of your thermocol ball in mm
# Focal length in pixels (approx). You can calibrate, or estimate using a reference distance.
# Rough heuristic: obtain f_px by placing the ball at a known distance and measuring its pixel size:
# f_px = (observed_px * known_distance_mm) / REAL_BALL_DIAMETER_MM
FOCAL_LENGTH_PX = 800.0  # start here and calibrate to your camera
TARGET_EXCISION_DEPTH_MM = 0.0  # for prototype, this can be 0 (we only measure distance to object)

# --- Phase thresholds (tweak for your setup) ---
APPROACH_DISTANCE_THRESHOLD_MM = 60.0  # > -> APPROACH
EXCISE_DISTANCE_THRESHOLD_MM = 25.0    # <= -> EXCISE
CENTER_PIX_TOLERANCE = 30  # pixels from frame center to be considered centered
STABLE_COUNT_REQUIRED = 5  # number of consecutive frames of stable bbox to lock EXCISE

# --- Logging / report ---
REPORT_PATH = "surgery_report.json"

# --- Optional TTS (disabled by default) ---
USE_TTS = False
try:
    if USE_TTS:
        import pyttsx3
        tts_engine = pyttsx3.init()
except Exception:
    USE_TTS = False

# =========================
# SURGICAL STATE
# =========================
class SurgicalState:
    def __init__(self):
        self.tumor_detected = False
        self.tumor_centered = False
        self.bbox = None  # (x,y,w,h)
        self.confidence = 0.0

        self.distance_mm = None
        self.depth_remaining_mm = None

        self.robot_dx = 0.0
        self.robot_dy = 0.0
        self.robot_speed = 0.0

        self.phase = "SEARCH"
        self.warnings = []
        self.timestamp = time.time()

        # internal helpers
        self._stable_counter = 0
        self._last_phase = None
        self.event_log = []  # list of (timestamp, event_str)

    def log_event(self, msg):
        t = time.time()
        self.event_log.append({"t": t, "msg": msg})

    def to_report(self):
        duration = None
        if len(self.event_log) >= 2:
            duration = self.event_log[-1]["t"] - self.event_log[0]["t"]
        return {
            "events": self.event_log,
            "final_phase": self.phase,
            "duration_s": duration,
            "warnings": self.warnings,
        }

# =========================
# INFERENCE (placeholder - replace with your own function)
# =========================

def run_inference(frame):
    """
    Expected return format (one detection for tumor):
    {
        'bbox': (x, y, w, h),  # pixels
        'confidence': float (0..1)
    }

    This implementation tries Roboflow HTTP local server.
    Replace or adapt to call your working.py inference function directly for best performance.
    """
    if not USE_ROBOFLOW_HTTP:
        # fallback: no inference
        return None

    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(INFERENCE_URL, files=files, timeout=1.0)
        r.raise_for_status()
        data = r.json()
        # parse results (format depends on your server). Example expected structure:
        # data['predictions'] = [{'x':..., 'y':..., 'width':..., 'height':..., 'confidence':..., 'class': 'vertebral_body_tumor'}]
        preds = data.get('predictions', [])
        if len(preds) == 0:
            return None
        # choose top confidence detection
        top = max(preds, key=lambda p: p.get('confidence', 0))
        # Roboflow local sometimes gives center x,y and width,height; convert to x,y,w,h
        cx = top.get('x')
        cy = top.get('y')
        w = top.get('width')
        h = top.get('height')
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        bbox = (int(x), int(y), int(w), int(h))
        conf = float(top.get('confidence', 0))
        return {'bbox': bbox, 'confidence': conf}
    except Exception as e:
        # network or parsing error; in your local integration you may instead import detection from working.py
        # print('Inference error', e)
        return None

# =========================
# GEOMETRY / DEPTH
# =========================

def estimate_distance_mm(observed_diameter_px, f_px=FOCAL_LENGTH_PX, real_diameter_mm=REAL_BALL_DIAMETER_MM):
    if observed_diameter_px <= 0:
        return None
    return (real_diameter_mm * f_px) / float(observed_diameter_px)

# =========================
# PHASE DETECTION & GUIDANCE
# =========================

def detect_phase(state: SurgicalState):
    if not state.tumor_detected:
        return 'SEARCH'
    if not state.tumor_centered:
        return 'ALIGN'
    # tumor centered now
    if state.distance_mm is None:
        return 'APPROACH'
    if state.distance_mm > APPROACH_DISTANCE_THRESHOLD_MM:
        return 'APPROACH'
    if state.distance_mm <= EXCISE_DISTANCE_THRESHOLD_MM:
        # require stability for excise
        state._stable_counter += 1
        if state._stable_counter >= STABLE_COUNT_REQUIRED:
            return 'EXCISE'
        else:
            return 'APPROACH'
    return 'APPROACH'


def guidance_for_state(state: SurgicalState):
    phase = state.phase
    if phase == 'SEARCH':
        return 'Scan slowly to find tumor.'
    if phase == 'ALIGN':
        # give pixel dx/dy guidance
        if state.bbox is None:
            return 'No bbox. Reposition camera.'
        x, y, w, h = state.bbox
        frame_center_x = state.frame_w // 2
        frame_center_y = state.frame_h // 2
        bbox_cx = x + w // 2
        bbox_cy = y + h // 2
        dx = bbox_cx - frame_center_x
        dy = bbox_cy - frame_center_y
        txt = ''
        if abs(dx) > CENTER_PIX_TOLERANCE:
            dir_x = 'right' if dx > 0 else 'left'
            txt += f'Move camera {dir_x} {abs(dx)} px. '
        if abs(dy) > CENTER_PIX_TOLERANCE:
            dir_y = 'down' if dy > 0 else 'up'
            txt += f'Move {dir_y} {abs(dy)} px.'
        return txt.strip() or 'Small adjustments to center the tumor.'
    if phase == 'APPROACH':
        if state.distance_mm is None:
            return 'Approach slowly. Distance unknown.'
        return f'Approach {max(0, state.distance_mm - EXCISE_DISTANCE_THRESHOLD_MM):.1f} mm more.'
    if phase == 'EXCISE':
        return 'Hold steady. Begin excision when ready. Maintain minimal movement.'
    if phase == 'RETREAT':
        return 'Retract instrument slowly.'
    return ''

# =========================
# DRAW PANEL & OVERLAYS
# =========================

def draw_ai_panel(screen, state: SurgicalState):
    # screen is the combined image that already reserves right PANEL_WIDTH columns for panel
    h, w = screen.shape[:2]
    panel_x0 = w - PANEL_WIDTH
    # panel background
    cv2.rectangle(screen, (panel_x0, 0), (w - 1, h - 1), (30, 30, 30), -1)

    # padding
    x = panel_x0 + 12
    y = 24
    line_h = 22

    def put(text, yy=None, scale=0.5, color=(220, 220, 220), bold=False):
        nonlocal y
        if yy is not None:
            y = yy
        cv2.putText(screen, text, (x, y), FONT, scale, color, 1 if not bold else 2, cv2.LINE_AA)
        y += line_h

    put('SURGICAL AI PANEL', scale=0.6, bold=True)
    put('')
    put(f'Phase: {state.phase}', scale=0.55, bold=True)
    put(f'Confidence: {state.confidence*100:.0f}%')
    if state.distance_mm is not None:
        put(f'Distance: {state.distance_mm:.1f} mm')
        if state.depth_remaining_mm is not None:
            put(f'Depth remaining: {state.depth_remaining_mm:.1f} mm')
    else:
        put('Distance: --')
    put('')
    put('Guidance:', bold=True)
    # wrap guidance text to panel width
    guidance = guidance_for_state(state)
    max_chars = 32
    for i in range(0, len(guidance), max_chars):
        put(guidance[i:i+max_chars])
    put('')
    if state.warnings:
        put('WARNINGS:', bold=True, color=(0, 180, 255))
        for wmsg in state.warnings[-4:]:
            for i in range(0, len(wmsg), max_chars):
                put(wmsg[i:i+max_chars], scale=0.5, color=(0, 200, 255))

# =========================
# ROBOT HOOK (placeholder)
# =========================

def send_robot_command(cmd: str):
    """
    Replace this with your robot command sender (socket, HTTP, etc.).
    For now we only log suggested commands; we do NOT issue motion commands automatically.
    """
    print(f'[ROBOT SUGGESTION] {cmd}')

# =========================
# MAIN LOOP
# =========================

def run():
    cap = cv2.VideoCapture(CAMERA_PORT)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')

    state = SurgicalState()
    last_narration = ''
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # create a wider canvas so we can draw a right-side panel
        h0, w0 = frame.shape[:2]
        canvas = np.zeros((h0, w0 + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[:, :w0] = frame.copy()

        # update state frame dims (used by guidance)
        state.frame_w = w0
        state.frame_h = h0

        # Run detection
        det = run_inference(frame)
        if det is None:
            state.tumor_detected = False
            state.bbox = None
            state.confidence = 0.0
        else:
            bbox = det['bbox']
            conf = det['confidence']
            state.tumor_detected = True
            state.bbox = bbox
            state.confidence = conf

            x, y, w, h = bbox
            # draw bbox on the live feed
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(canvas, f'Tumor {conf*100:.0f}%', (x, y - 6), FONT, 0.6, (0, 165, 255), 2)

            # estimate diameter in px (use max of w,h as proxy)
            observed_px = max(w, h)
            dist_mm = estimate_distance_mm(observed_px)
            state.distance_mm = dist_mm
            state.depth_remaining_mm = None if dist_mm is None else max(0.0, dist_mm - TARGET_EXCISION_DEPTH_MM)

            # centering check
            frame_cx = w0 // 2
            frame_cy = h0 // 2
            bbox_cx = x + w // 2
            bbox_cy = y + h // 2
            dx = bbox_cx - frame_cx
            dy = bbox_cy - frame_cy
            state.tumor_centered = (abs(dx) <= CENTER_PIX_TOLERANCE and abs(dy) <= CENTER_PIX_TOLERANCE)

        # detect phase
        prev_phase = state.phase
        new_phase = detect_phase(state)
        state.phase = new_phase
        if new_phase != prev_phase:
            state._stable_counter = 0
            ev = f'Phase changed {prev_phase} -> {new_phase}'
            state.log_event(ev)
            # narration (only on change)
            narration = ''
            if new_phase == 'SEARCH':
                narration = 'Searching for tumor.'
            elif new_phase == 'ALIGN':
                narration = 'Tumor detected; aligning.'
            elif new_phase == 'APPROACH':
                narration = 'Approaching tumor.'
            elif new_phase == 'EXCISE':
                narration = 'Excision zone reached. Hold steady.'
            elif new_phase == 'RETREAT':
                narration = 'Retreating.'
            if narration:
                last_narration = narration
                print('[NARRATION]', narration)
                if USE_TTS:
                    try:
                        tts_engine.say(narration)
                        tts_engine.runAndWait()
                    except Exception:
                        pass

        # show textual AI panel
        draw_ai_panel(canvas, state)

        # draw center crosshair
        cx, cy = w0 // 2, h0 // 2
        cv2.drawMarker(canvas, (cx, cy), (200, 200, 200), cv2.MARKER_CROSS, 10, 1)

        # show the composed canvas
        cv2.imshow('Surgery - Live Feed + AI Panel', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # Save snapshot + state
            ts = int(time.time())
            cv2.imwrite(f'snapshot_{ts}.jpg', canvas)
            print('Snapshot saved')

    cap.release()
    cv2.destroyAllWindows()

    # Save report
    report = state.to_report()
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print('Report saved to', REPORT_PATH)

if __name__ == '__main__':
    run()
