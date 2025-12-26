# =========================
# working.py (FINAL – REAL ROBOT + VOICE + AI)
# =========================

import cv2
import numpy as np
import socket
import time
import json
import shutil
import subprocess
import threading
import queue
import sounddevice as sd
import json as js
import os
import google.generativeai as genai

from vosk import Model, KaldiRecognizer
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import pyttsx3

# =========================
# CONFIG
# =========================
TRACK_CLASS = "vertebral_body_tumor"
CONFIDENCE_THRESH = 0.55
CAMERA_PORT = 0

AI_MODE = True
VOICE_MODE = False

# ⚠️ CHANGE THIS TO REAL IP (recommended) OR EXPORT ROBOT_HOST=1.2.3.4
ROBOT_HOST = os.getenv("ROBOT_HOST", "127.0.0.1")  # default to localhost so home testing is quiet
ROBOT_PORT = int(os.getenv("ROBOT_PORT", "5005"))
ROBOT_ON = os.getenv("ROBOT_ON", "0") != "0"  # default off at home; set ROBOT_ON=1 when connected
ROBOT_DRY_RUN = os.getenv("ROBOT_SIM", "0") == "1" or os.getenv("DRY_RUN", "0") == "1"
ROBOT_RETRY_SECS = float(os.getenv("ROBOT_RETRY_SECS", "8"))
ROBOT_ON = ROBOT_ON and not ROBOT_DRY_RUN
VOICE_MOVE_STEP = 0.4  # manual voice move delta
VOICE_Y_SIGN = -1.0    # set to 1.0 if your robot uses non-inverted Y (default: up is negative)
USE_OS_TTS_FIRST = True  # prefer macOS `say` if available for audible feedback
_robot_host_candidates = [ROBOT_HOST]
if ROBOT_HOST.endswith(".local"):
    base = ROBOT_HOST[:-6]
    _robot_host_candidates.append(f"{base}.lan")
elif ROBOT_HOST.endswith(".lan"):
    base = ROBOT_HOST[:-4]
    _robot_host_candidates.append(f"{base}.local")
_robot_host_idx = 0
_robot_disabled_until = 0.0

CENTER_TOL_PX = 30
APPROACH_MM = 10.0
EXCISE_MM = 2.0

REPORT_FILE = "surgery_report.json"

# ---- Distance calibration (relative) ----
REF_PX_DIAM = 500
MM_PER_PX_SCALE = 0.02

# =========================
# VOICE CONFIG
# =========================
VOICE_MODEL_PATH = "vosk_models/vosk-model-small-en-us-0.15"

VOICE_COMMANDS = [
    "move left", "move right", "move up", "move down",
    "stop", "lock target", "release lock",
    "enable voice", "disable voice"
]

voice_queue = queue.Queue()
last_move_cmd = None
target_locked = False
def parse_voice_commands(text):
    # Return only the latest command phrase inside the transcript to avoid early false positives
    txt = " ".join(text.split())
    best = None
    best_idx = None
    for phrase in VOICE_COMMANDS:
        start = 0
        while True:
            idx = txt.find(phrase, start)
            if idx == -1:
                break
            if best_idx is None or idx >= best_idx:
                best_idx = idx
                best = phrase
            start = idx + len(phrase)
    return [best] if best else []

# =========================
# TEXT TO SPEECH
# =========================
_last_spoken = None
_last_ts = 0.0
speech_queue = queue.Queue()
_system_say = shutil.which("say")

def _tts_worker():
    # Create the engine inside the worker so runAndWait executes on the same thread
    tts_engine = None
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 175)
    except Exception as e:
        print(f"TTS init failed: {e}")

    while True:
        text = speech_queue.get()
        if text is None:
            speech_queue.task_done()
            break

        ok = False
        # Prefer OS TTS (macOS `say`) if available and configured
        if USE_OS_TTS_FIRST and _system_say:
            try:
                subprocess.Popen([_system_say, text])
                ok = True
            except Exception as e:
                print(f"OS TTS error: {e}")

        if not ok and tts_engine:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
                ok = True
            except Exception as e:
                print(f"TTS engine error: {e}")
                try:
                    tts_engine.stop()
                except Exception:
                    pass
                tts_engine = None  # fall back to OS TTS

        if not ok and _system_say:
            # macOS fallback if pyttsx3 driver has an issue
            try:
                subprocess.Popen([_system_say, text])
                ok = True
            except Exception as e:
                print(f"OS TTS fallback error: {e}")

        if not ok:
            print("TTS failed: no engine available")
        speech_queue.task_done()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text):
    global _last_spoken, _last_ts
    now = time.time()
    if text == _last_spoken and (now - _last_ts) < 1.2:
        return
    _last_spoken = text
    _last_ts = now
    print("🗣️", text)
    speech_queue.put(text)

# =========================
# ROBOT SOCKET + PROTOCOL
# =========================
if ROBOT_ON:
    sock = None
    ROBOT_ADDR = None
else:
    sock = None
_last_robot_err_ts = 0.0
_next_robot_resolve = 0.0
_robot_suspended_until = 0.0
_robot_fail_count = 0
_robot_disabled = False

def _ensure_robot_socket():
    """
    Ensure we have a UDP socket and a resolved destination.
    If resolution is failing, we retry periodically and skip sends until it succeeds.
    """
    global sock, ROBOT_ADDR, _next_robot_resolve, _robot_host_idx, _robot_fail_count, _robot_disabled
    now = time.time()
    if ROBOT_DRY_RUN:
        return False
    if now < _robot_suspended_until:
        return False
    if _robot_disabled and now < _robot_disabled_until:
        return False
    if _robot_disabled and now >= _robot_disabled_until:
        _robot_disabled = False
        _robot_fail_count = 0
        ROBOT_ADDR = None
        sock = None
    if not ROBOT_ON:
        return False
    if sock is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            _robot_err(f"Robot socket create failed: {e}")
            return False
    if ROBOT_ADDR is None:
        now = time.time()
        if now >= _next_robot_resolve:
            _next_robot_resolve = now + 2.0
            host = _robot_host_candidates[_robot_host_idx]
            try:
                info = socket.getaddrinfo(host, ROBOT_PORT, socket.AF_INET, socket.SOCK_DGRAM)
                ROBOT_ADDR = info[0][4]
                _robot_fail_count = 0
            except (socket.gaierror, OSError) as e:
                if _robot_host_idx + 1 < len(_robot_host_candidates):
                    _robot_host_idx += 1
                    _robot_err(f"Robot resolve failed for {host}, trying {_robot_host_candidates[_robot_host_idx]}")
                else:
                    _robot_fail_count += 1
                    if _robot_fail_count >= 3 and not _robot_disabled:
                        _robot_disabled = True
                        _robot_disabled_until = now + ROBOT_RETRY_SECS
                        _robot_err(f"Robot network unreachable; retrying in {int(ROBOT_RETRY_SECS)}s. Set ROBOT_HOST to the robot IP or disable robot with ROBOT_ON=0.")
                    _robot_err(f"Robot address resolve failed: {e}")
                return False
        else:
            return False
    return True

def robot_init():
    if ROBOT_DRY_RUN:
        print("Robot dry run: skipping robot_init")
        return
    if not _ensure_robot_socket():
        return
    try:
        sock.sendto(b"ai_mode on", ROBOT_ADDR)
        sock.sendto(b"ai_speed 12", ROBOT_ADDR)
        sock.sendto(b"ai_speed_inc 0.04", ROBOT_ADDR)
        time.sleep(0.1)
    except OSError as e:
        _robot_err(f"Robot init failed: {e}")

def robot_stop():
    if ROBOT_DRY_RUN:
        return
    if not _ensure_robot_socket():
        return
    try:
        sock.sendto(b"move 0 0", ROBOT_ADDR)
    except OSError as e:
        _robot_fail(e, "stop")

def send_move(dx, dy):
    if ROBOT_DRY_RUN:
        return
    if not _ensure_robot_socket():
        return
    cmd = f"move {dx:.2f} {dy:.2f}"
    # debug trace to verify voice axis values
    # print("SEND", cmd)
    try:
        sock.sendto(cmd.encode(), ROBOT_ADDR)
    except OSError as e:
        _robot_fail(e, "move")

def _robot_err(msg):
    global _last_robot_err_ts, _robot_suspended_until
    now = time.time()
    _robot_suspended_until = now + 5.0  # back off sends briefly on errors
    if now - _last_robot_err_ts > 2.0:  # avoid flooding
        speak(msg)
        _last_robot_err_ts = now

def _robot_fail(exc, action):
    global _robot_fail_count, _robot_disabled, _robot_disabled_until
    _robot_fail_count += 1
    if _robot_fail_count >= 3 and not _robot_disabled:
        _robot_disabled = True
        _robot_disabled_until = time.time() + ROBOT_RETRY_SECS
        _robot_err(f"Robot network unreachable; retrying in {int(ROBOT_RETRY_SECS)}s. Check ROBOT_HOST or connect and restart.")
    else:
        _robot_err(f"Robot {action} failed: {exc}")

# =========================
# ROBOFLOW CLIENT
# =========================
config = InferenceConfiguration(confidence_threshold=CONFIDENCE_THRESH)
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=""
)
client.configure(config)
MODEL_ID = "moonshot-project-xfdfs/3"
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "roboflow").lower()
_last_infer_err_ts = 0.0
_gemini_model = None
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _ensure_gemini_model():
    """
    Lazy-init Gemini model. Returns None if not available.
    """
    global _gemini_model, _last_infer_err_ts
    if _gemini_model:
        return _gemini_model
    try:
        import google.generativeai as genai
    except ImportError:
        now = time.time()
        if now - _last_infer_err_ts > 3.0:
            _last_infer_err_ts = now
            print("Gemini backend requested but google-generativeai is not installed. pip install google-generativeai")
        return None
    if not GEMINI_API_KEY:
        now = time.time()
        if now - _last_infer_err_ts > 3.0:
            _last_infer_err_ts = now
            print("Gemini backend requested but GEMINI_API_KEY is not set.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)
        return _gemini_model
    except Exception as e:
        now = time.time()
        if now - _last_infer_err_ts > 3.0:
            _last_infer_err_ts = now
            print(f"Gemini configure failed: {e}")
        return None

def _gemini_infer(frame):
    """
    Send frame to Gemini vision model and expect a structured JSON response:
    {"predictions":[{"class":"vertebral_body_tumor","confidence":0.73,"x":..,"y":..,"width":..,"height":..}]}
    x,y,width,height are in pixels, x/y are top-left.
    """
    model = _ensure_gemini_model()
    if model is None:
        return []
    try:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return []
        prompt = (
            "You are a surgical vision model. Detect the vertebral body tumor in the image. "
            "Return JSON with a 'predictions' array. Each prediction must have: "
            "class (string), confidence (0-1), x, y, width, height (all in pixels, x/y top-left). "
            "If nothing is found, return an empty predictions array."
        )
        resp = model.generate_content(
            [
                {"inline_data": {"mime_type": "image/jpeg", "data": buf.tobytes()}},
                prompt,
            ],
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        if not resp or not resp.text:
            return []
        data = json.loads(resp.text)
        preds = data.get("predictions", [])
        if TRACK_CLASS:
            preds = [p for p in preds if p.get("class") == TRACK_CLASS]
        # normalize numeric types
        for p in preds:
            for k in ("x", "y", "width", "height", "confidence"):
                if k in p:
                    try:
                        p[k] = float(p[k])
                    except Exception:
                        pass
        preds.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return preds
    except Exception as e:
        now = time.time()
        if now - _last_infer_err_ts > 3.0:
            _last_infer_err_ts = now
            print(f"Inference error (gemini): {e}")
        return []

def run_inference(frame):
    """
    Default: Roboflow local HTTP (fast). Set INFERENCE_BACKEND=gemini to plug in Gemini later.
    Return list of prediction dicts compatible with existing downstream code.
    """
    global _last_infer_err_ts
    backend = INFERENCE_BACKEND
    if backend == "none":
        return []
    if backend == "gemini":
        return _gemini_infer(frame)

    try:
        preds = client.infer(frame, model_id=MODEL_ID).get("predictions", [])
        if TRACK_CLASS:
            preds = [p for p in preds if p.get("class") == TRACK_CLASS]
        preds.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        return preds
    except Exception as e:
        now = time.time()
        if now - _last_infer_err_ts > 3.0:
            _last_infer_err_ts = now
            print(f"Inference error ({backend}): {e}")
        return []

# =========================
# SURGICAL STATE
# =========================
class SurgicalState:
    def __init__(self):
        self.phase = "SEARCH"
        self.distance_mm = None
        self.centered = False
        self.confidence = 0.0
        self.events = []

    def log(self, msg):
        self.events.append({"time": round(time.time(), 2), "event": msg})

# =========================
# UTILS
# =========================
def estimate_distance(px):
    return max(0.0, (REF_PX_DIAM - px) * MM_PER_PX_SCALE)

def detect_phase(s):
    if s.distance_mm is None:
        return "SEARCH"
    if not s.centered:
        return "ALIGN"
    if s.distance_mm > APPROACH_MM:
        return "APPROACH"
    return "EXCISE" if s.distance_mm <= EXCISE_MM else "APPROACH"

def guidance_text(s):
    if s.phase == "SEARCH":
        return "Scanning for target"
    if s.phase == "ALIGN":
        return "Align to center"
    if s.phase == "APPROACH":
        return f"Advance {s.distance_mm:.1f} mm" if s.distance_mm is not None else "Advance"
    if s.phase == "EXCISE":
        return "Hold steady – excision zone"
    return ""

# =========================
# VOICE THREAD
# =========================
def voice_listener():
    model = Model(VOICE_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000, js.dumps(VOICE_COMMANDS))

    def cb(indata, frames, t, status):
        if rec.AcceptWaveform(bytes(indata)):
            txt = js.loads(rec.Result()).get("text", "")
            if txt:
                voice_queue.put(txt)
                print(f"[voice heard] {txt}")

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        while True:
            time.sleep(0.1)

threading.Thread(target=voice_listener, daemon=True).start()
speak("Voice listener started")

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(CAMERA_PORT)
state = SurgicalState()
state.log("session_start")
had_detection = False
last_centered = False

if ROBOT_ON:
    robot_init()
elif ROBOT_DRY_RUN:
    speak("Robot dry run: commands will be logged only")

speak("System initialized. Say enable voice to begin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2

    state.distance_mm = None
    state.centered = False
    state.confidence = 0.0

    # ---------- INFERENCE ----------
    preds = run_inference(frame) if AI_MODE else []

    if preds:
        p = preds[0]
        x = int(p["x"] - p["width"] / 2)
        y = int(p["y"] - p["height"] / 2)
        w = int(p["width"])
        h = int(p["height"])

        state.confidence = p["confidence"]
        px = max(w, h)
        state.distance_mm = estimate_distance(px)

        bx, by = x + w // 2, y + h // 2
        state.centered = abs(bx - cx) < CENTER_TOL_PX and abs(by - cy) < CENTER_TOL_PX

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        if not had_detection:
            state.log("detection:first")
            had_detection = True
    else:
        if had_detection:
            state.log("detection:lost")
            had_detection = False

    prev_phase = state.phase
    state.phase = detect_phase(state)
    if state.phase != prev_phase:
        state.log(f"phase:{state.phase}")
    if state.centered and not last_centered:
        state.log("target:centered")
    if not state.centered and last_centered:
        state.log("target:off_center")
    last_centered = state.centered

    # ---------- AI AUTO TRACK ----------
    if ROBOT_ON and AI_MODE and not VOICE_MODE and not target_locked:
        if preds and state.confidence > 0.6:
            dx = max(-1, min(1, (bx - cx) / cx))
            dy = max(-1, min(1, (by - cy) / cy))
            send_move(0, 0) if state.centered else send_move(dx, dy)
        else:
            robot_stop()

    # ---------- VOICE ----------
    while not voice_queue.empty():
        cmd_raw = voice_queue.get()
        cmd_norm = cmd_raw.strip().lower().replace(".", "")
        commands = parse_voice_commands(cmd_norm)
        if not commands:
            continue

        for cmd in commands:
            if cmd == "enable voice":
                VOICE_MODE = True
                speak("Voice control enabled")
                last_move_cmd = None
                state.log("voice:enable")
                continue

            if cmd == "release lock":
                target_locked = False
                AI_MODE = True
                speak("Target lock released. AI mode enabled")
                last_move_cmd = None
                state.log("voice:release_lock")
                continue

            if target_locked:
                continue

            if cmd == "disable voice":
                VOICE_MODE = False
                speak("Voice control disabled")
                last_move_cmd = None
                state.log("voice:disable")
                # clear any pending commands so we don't chatter while voice is off
                while not voice_queue.empty():
                    voice_queue.get()
                    voice_queue.task_done()
                continue

            if not VOICE_MODE:
                # when voice is off, ignore everything except future "enable voice"
                continue

            if state.phase == "EXCISE":
                speak("Voice disabled during excision")
                continue

            if cmd == "move up":
                state.log("voice:move_up")
                if last_move_cmd != "move up":
                    speak("Moving up")
                    last_move_cmd = "move up"
                send_move(0, VOICE_MOVE_STEP * VOICE_Y_SIGN)
            elif cmd == "move down":
                state.log("voice:move_down")
                if last_move_cmd != "move down":
                    speak("Moving down")
                    last_move_cmd = "move down"
                send_move(0, -VOICE_MOVE_STEP * VOICE_Y_SIGN)
            elif cmd == "move left":
                state.log("voice:move_left")
                if last_move_cmd != "move left":
                    speak("Moving left")
                    last_move_cmd = "move left"
                send_move(-VOICE_MOVE_STEP, 0)
            elif cmd == "move right":
                state.log("voice:move_right")
                if last_move_cmd != "move right":
                    speak("Moving right")
                    last_move_cmd = "move right"
                send_move(VOICE_MOVE_STEP, 0)
            elif cmd == "stop":
                speak("Stopping")
                last_move_cmd = None
                state.log("voice:stop")
                robot_stop()
            elif cmd == "lock target":
                if state.centered:
                    target_locked = True
                    AI_MODE = False
                    VOICE_MODE = False
                    robot_stop()
                    speak("Target locked")
                    last_move_cmd = None
                    state.log("voice:lock_target")
                else:
                    speak("Target not centered")

    # ---------- PANEL ----------
    panel = np.zeros((H, 320, 3), dtype=np.uint8)
    y = [30]
    def put(txt):
        cv2.putText(panel, txt, (10, y[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y[0] += 28

    put("SURGICAL AI PANEL")
    put(f"AI MODE: {'ON' if AI_MODE else 'OFF'}")
    put(f"VOICE MODE: {'ON' if VOICE_MODE else 'OFF'}")
    put(f"Phase: {state.phase}")
    if state.distance_mm is not None:
        put(f"Distance: {state.distance_mm:.1f} mm")
    put(f"Confidence: {state.confidence:.2f}")
    put("Guidance:")
    put(guidance_text(state))

    cv2.imshow("Surgery Assist", np.hstack([frame, panel]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
if ROBOT_ON and _ensure_robot_socket():
    try:
        sock.sendto(b"ai_mode off", ROBOT_ADDR)
    except OSError as e:
        _robot_err(f"Robot shutdown send failed: {e}")

cap.release()
cv2.destroyAllWindows()

state.log("session_end")
with open(REPORT_FILE, "w") as f:
    json.dump(state.events, f, indent=2)

speak("Surgery session ended. Report saved.")
speech_queue.put(None)
speech_queue.join()

