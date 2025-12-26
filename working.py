# =========================
# working.py — FINAL
# Real Robot + Voice Control + Gemini Jarvis (READ-ONLY)
# =========================

import cv2
import numpy as np
import socket
import time
import json
import threading
import queue
import sounddevice as sd
import json as js
import os
import subprocess
import shutil
import datetime
import re

from vosk import Model, KaldiRecognizer
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import pyttsx3
import google.genai as genai
from google.genai import types as genai_types

# =========================
# CONFIG
# =========================
TRACK_CLASS = "vertebral_body_tumor"
CONFIDENCE_THRESH = 0.55
CAMERA_PORT = 0

AI_MODE = True
VOICE_MODE = False
target_locked = False
ai_suspended_until = 0.0
VOICE_DEBUG = os.getenv("VOICE_DEBUG", "0") == "1"

ROBOT_HOST = "moonshot1.local"  # set your robot IP/hostname
ROBOT_PORT = int(os.getenv("ROBOT_PORT", "5005"))
ROBOT_ON = True


CENTER_TOL_PX = 30
APPROACH_MM = 10.0
EXCISE_MM = 2.0

REPORT_FILE = "surgery_report.json"

REF_PX_DIAM = 500
MM_PER_PX_SCALE = 0.02
VOICE_MOVE_STEP = 0.4
AI_SUSPEND_SECS = 4.0

# =========================
# VOICE COMMANDS (CONTROL)
# =========================
VOICE_COMMANDS = {
    "move left": ["move left", "left"],
    "move right": ["move right", "right"],
    "move up": ["move up", "up", "go up", "forward"],
    "move down": ["move down", "down", "go down", "back", "backward"],
    "stop": ["stop", "halt"],
    "lock target": ["lock target", "lock on", "lock"],
    "release lock": ["release lock", "unlock"],
    "enable voice": ["enable voice", "voice on"],
    "disable voice": ["disable voice", "voice off"],
}

voice_queue = queue.Queue()
GEMINI_ARMED = False  # keyboard-armed for next Jarvis question

def parse_control_command(text: str):
    txt = text.lower()
    for cmd, phrases in VOICE_COMMANDS.items():
        for p in phrases:
            if p in txt:
                return cmd
    return None

QUESTION_HINTS = ["?", "what", "why", "how", "who", "where", "jarvis", "tell me", "explain", "should", "can you"]
def looks_like_question(text: str):
    t = text.lower()
    return any(h in t for h in QUESTION_HINTS)

# =========================
# TEXT TO SPEECH (ASYNC)
# =========================
speech_queue = queue.Queue()
_last_spoken = None
_last_ts = 0.0
_system_say = shutil.which("say")
TTS_VOICE_NAME = os.getenv("TTS_VOICE", "Samantha")

def speak(text):
    global _last_spoken, _last_ts
    now = time.time()
    if text == _last_spoken and (now - _last_ts) < 1.0:
        return
    _last_spoken = text
    _last_ts = now
    print("🗣️", text)
    speech_queue.put(text)

def tts_worker():
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        if TTS_VOICE_NAME:
            voices = engine.getProperty("voices") or []
            chosen = None
            for v in voices:
                nm = (v.name or "").lower()
                if TTS_VOICE_NAME.lower() in nm or "female" in nm:
                    chosen = v.id
                    break
            if chosen:
                engine.setProperty("voice", chosen)
    except Exception as e:
        print(f"TTS init failed: {e}")

    while True:
        text = speech_queue.get()
        if text is None:
            break
        ok = False
        if _system_say:
            try:
                cmd = [_system_say]
                if TTS_VOICE_NAME:
                    cmd += ["-v", TTS_VOICE_NAME]
                cmd.append(text)
                subprocess.Popen(cmd)
                ok = True
            except Exception as e:
                print(f"OS TTS error: {e}")
        if not ok and engine:
            try:
                engine.say(text)
                engine.runAndWait()
                ok = True
            except Exception as e:
                print(f"TTS engine error: {e}")
        if not ok:
            print("TTS failed: no engine available")

threading.Thread(target=tts_worker, daemon=True).start()

# =========================
# ROBOT SOCKET
# =========================
sock = None
ROBOT_ADDR = None

def robot_init():
    global sock, ROBOT_ADDR
    if not ROBOT_ON:
        return
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ROBOT_ADDR = (ROBOT_HOST, ROBOT_PORT)
        sock.sendto(b"ai_mode on", ROBOT_ADDR)
        sock.sendto(b"ai_speed 12", ROBOT_ADDR)
        sock.sendto(b"ai_speed_inc 0.04", ROBOT_ADDR)
    except Exception as e:
        print(f"Robot init failed: {e}")

def send_move(dx, dy):
    if not ROBOT_ON:
        return
    if not sock or not ROBOT_ADDR:
        print("Robot socket not ready; check ROBOT_HOST/ROBOT_ON.")
        return
    try:
        # debug trace for verification
        print(f"[robot] send move {dx:.2f} {dy:.2f} -> {ROBOT_ADDR}")
        sock.sendto(f"move {dx:.2f} {dy:.2f}".encode(), ROBOT_ADDR)
    except Exception as e:
        print(f"Robot move send failed: {e}")

def robot_stop():
    send_move(0, 0)

def suspend_ai(reason=""):
    """
    Pause auto-tracking briefly after a manual intervention so we do not fight the operator.
    """
    global ai_suspended_until
    ai_suspended_until = time.time() + AI_SUSPEND_SECS
    if reason:
        print(f"[auto] AI paused for {AI_SUSPEND_SECS}s ({reason})")
    robot_stop()

# =========================
# ROBOFLOW
# =========================
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="cH5e05vZG9OkbOJtMAuH"
)
client.configure(InferenceConfiguration(confidence_threshold=CONFIDENCE_THRESH))
MODEL_ID = "moonshot-project-xfdfs/3"

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

state = SurgicalState()

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
# GEMINI (JARVIS)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_DISABLED = os.getenv("GEMINI_OFF", "0") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # offline Jarvis if set
OLLAMA_BIN = shutil.which("ollama")
_gemini_client = None
try:
    if not GEMINI_DISABLED and GEMINI_API_KEY:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Gemini client init failed: {e}")

_gemini_model = None
_gemini_tried = False
_gemini_model_id = os.getenv("GEMINI_MODEL")

def _gemini_candidate_models():
    """
    Prefer models returned by the API that support generateContent, with the env override at the front if provided.
    """
    discovered = []
    if _gemini_client:
        try:
            for m in _gemini_client.models.list():
                methods = getattr(m, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    discovered.append(m.name)
        except Exception as e:
            print(f"Gemini list_models failed: {e}")
    env_ids = [_gemini_model_id] if _gemini_model_id else []

    seen = set()
    ordered = []
    # prefer flash variants first for speed/quota, then pro
    discovered_sorted = sorted(discovered, key=lambda x: (0 if "flash" in x else 1, x))
    for mid in env_ids + discovered_sorted:
        if not mid:
            continue
        if mid in seen:
            continue
        seen.add(mid)
        ordered.append(mid)
    if not ordered:
        ordered = [
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemma-3-12b-it",
            "models/gemma-3-4b-it",
            "models/gemini-2.5-pro",
            "models/gemini-flash-latest",
            "models/gemini-pro-latest",
        ]
    return ordered

def _load_gemini():
    global _gemini_model, _gemini_tried, GEMINI_DISABLED, _gemini_client
    if GEMINI_DISABLED:
        return None
    if _gemini_model is not None:
        return _gemini_model
    if not GEMINI_API_KEY:
        print("Gemini: no API key; set GEMINI_API_KEY.")
        return None
    if _gemini_client is None:
        # try to (re)create client lazily
        try:
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            print(f"Gemini: client unavailable; check installation or key. {e}")
            return None
    if _gemini_tried:
        return None
    _gemini_tried = True
    for mid in _gemini_candidate_models():
        try:
            # ping the model to ensure it exists
            _gemini_client.models.generate_content(model=mid, contents=["ping"], config={"response_mime_type": "text/plain"})
            _gemini_model = mid
            print(f"Gemini using model: {mid}")
            return _gemini_model
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                print(f"Gemini init quota limit for {mid}: {msg}")
                continue
            print(f"Gemini init failed for {mid}: {msg}")
    print("Gemini: no available models.")
    return None

def ask_gemini(question: str):
    if GEMINI_DISABLED:
        if not ask_offline_llm(question):
            speak("Jarvis is unavailable.")
        return
    model = _load_gemini()
    if model is None:
        if not ask_offline_llm(question):
            speak("Jarvis is unavailable.")
        return
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    context = f"""
Phase: {state.phase}
Distance to tumor: {state.distance_mm}
Centered: {state.centered}
Confidence: {state.confidence}
Recent events: {state.events[-5:]}
Current datetime: {now}
"""
    try:
        resp = _gemini_client.models.generate_content(
            model=model,
            contents=[
                context,
                "Question: " + question,
                "Answer briefly. Use the provided current datetime exactly for any date/time answers. If the question is unrelated to surgical context, answer succinctly.",
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.2,
                max_output_tokens=80,
            )
        )
        if resp and getattr(resp, "text", None):
            speak(resp.text.strip())
        else:
            if not ask_offline_llm(question):
                speak("I did not get a response.")
    except Exception as e:
        msg = str(e)
        print(f"Gemini error: {msg}")
        # reset so next question tries the next candidate
        globals()["_gemini_model"] = None
        globals()["_gemini_tried"] = False
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            if not ask_offline_llm(question):
                speak("Jarvis hit a quota limit. Try again in a minute or switch models.")
        else:
            if not ask_offline_llm(question):
                speak("I am unable to answer that right now.")

def ask_offline_llm(question: str):
    """
    Optional offline Jarvis using a local Ollama model. Set OLLAMA_MODEL env and install ollama.
    """
    if not OLLAMA_BIN or not OLLAMA_MODEL:
        return False
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    prompt = f"""You are Jarvis, a concise surgical assistant.
Current phase: {state.phase}
Distance to tumor: {state.distance_mm}
Centered: {state.centered}
Confidence: {state.confidence}
Recent events: {state.events[-5:]}
Current datetime: {now}
Answer briefly: {question}
"""
    try:
        proc = subprocess.run(
            [OLLAMA_BIN, "run", OLLAMA_MODEL],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            speak(proc.stdout.strip()[:400])
            return True
        print(f"Ollama returned {proc.returncode}: {proc.stderr}")
    except Exception as e:
        print(f"Ollama error: {e}")
    return False

# =========================
# VOICE LISTENER
# =========================
def voice_listener():
    model = Model("vosk_models/vosk-model-small-en-us-0.15")
    rec = KaldiRecognizer(model, 16000)

    def cb(indata, frames, t, status):
        if rec.AcceptWaveform(bytes(indata)):
            txt = js.loads(rec.Result()).get("text", "")
            if txt:
                if VOICE_DEBUG:
                    print(f"[voice] heard: {txt}")
                voice_queue.put(txt.lower())

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

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(CAMERA_PORT)
robot_init()
speak("Hello Anshuman, how can I help you today?")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2

    state.distance_mm = None
    state.centered = False
    state.confidence = 0.0

    preds = client.infer(frame, model_id=MODEL_ID).get("predictions", []) if AI_MODE else []

    if preds:
        p = preds[0]
        x = int(p["x"] - p["width"] / 2)
        y = int(p["y"] - p["height"] / 2)
        w = int(p["width"])
        h = int(p["height"])

        state.confidence = p["confidence"]
        px = max(w, h)
        state.distance_mm = max(0, (REF_PX_DIAM - px) * MM_PER_PX_SCALE)

        bx, by = x + w // 2, y + h // 2
        state.centered = abs(bx - cx) < CENTER_TOL_PX and abs(by - cy) < CENTER_TOL_PX

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        label = f"{p.get('class', TRACK_CLASS)} {state.confidence:.2f}"
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Phase
    if state.distance_mm is None:
        state.phase = "SEARCH"
    elif not state.centered:
        state.phase = "ALIGN"
    elif state.distance_mm > APPROACH_MM:
        state.phase = "APPROACH"
    else:
        state.phase = "EXCISE"

    # AI auto-track
    auto_allowed = ROBOT_ON and AI_MODE and not VOICE_MODE and not target_locked and time.time() >= ai_suspended_until
    if auto_allowed:
        if preds and state.confidence > 0.6:
            dx = (bx - cx) / cx
            dy = (by - cy) / cy
            send_move(0, 0) if state.centered else send_move(dx, dy)
        else:
            robot_stop()
    elif time.time() < ai_suspended_until:
        robot_stop()

    # Voice handling
    while not voice_queue.empty():
        spoken = voice_queue.get()
        cmd = parse_control_command(spoken)

        if cmd:
            # CONTROL PATH
            if cmd == "enable voice":
                VOICE_MODE = True
                speak("Voice control enabled")
            elif cmd == "disable voice":
                VOICE_MODE = False
                suspend_ai("voice disabled")
                speak("Voice control disabled")
            elif not VOICE_MODE:
                continue
            elif cmd == "move up":
                speak("Moving up")
                send_move(0, -VOICE_MOVE_STEP)
            elif cmd == "move down":
                speak("Moving down")
                send_move(0, VOICE_MOVE_STEP)
            elif cmd == "move left":
                speak("Moving left")
                send_move(-VOICE_MOVE_STEP, 0)
            elif cmd == "move right":
                speak("Moving right")
                send_move(VOICE_MOVE_STEP, 0)
            elif cmd == "stop":
                speak("Stopping")
                suspend_ai("manual stop")
            elif cmd == "lock target" and state.centered:
                target_locked = True
                AI_MODE = False
                VOICE_MODE = False
                robot_stop()
                speak("Target locked")
            elif cmd == "release lock":
                target_locked = False
                AI_MODE = True
                speak("Target released")
        else:
            # Only send to Jarvis if armed via keyboard
            if GEMINI_ARMED:
                ask_gemini(spoken)
                GEMINI_ARMED = False

    # ----- PANEL -----
    panel = np.zeros((H, 320, 3), dtype=np.uint8)
    panel[:] = (12, 18, 30)
    cv2.rectangle(panel, (0, 0), (320, 60), (20, 45, 80), -1)
    cv2.putText(panel, "SURGICAL AI", (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)

    def put_row(y_pos, label, value, color=(230, 230, 230)):
        cv2.putText(panel, label, (14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 175, 190), 1)
        cv2.putText(panel, value, (140, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    status_color = (50, 200, 120) if AI_MODE else (60, 150, 255)
    voice_color = (50, 200, 120) if VOICE_MODE else (100, 100, 240)

    y = 90
    put_row(y, "AI MODE", "ON" if AI_MODE else "OFF", status_color); y += 28
    put_row(y, "VOICE", "ON" if VOICE_MODE else "OFF", voice_color); y += 28
    put_row(y, "PHASE", state.phase, (255, 215, 160)); y += 28
    if state.distance_mm is not None:
        put_row(y, "DIST", f"{state.distance_mm:.1f} mm", (180, 220, 255)); y += 28
    put_row(y, "CONF", f"{state.confidence:.2f}", (180, 220, 255)); y += 32

    cv2.rectangle(panel, (12, y - 6), (308, y + 60), (24, 60, 110), -1)
    cv2.putText(panel, "GUIDANCE", (20, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 140), 2)
    cv2.putText(panel, guidance_text(state), (20, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    shortcut_y = y + 84
    cv2.putText(panel, "Keys: q quit | v voice | a AI | g Jarvis", (14, shortcut_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 160, 200), 1)

    cv2.imshow("Surgery Assist", np.hstack([frame, panel]))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("v"):
        VOICE_MODE = not VOICE_MODE
        if not VOICE_MODE:
            suspend_ai("voice disabled (keyboard)")
        speak(f"Voice control {'enabled' if VOICE_MODE else 'disabled'} by keyboard")
    if key == ord("a"):
        AI_MODE = not AI_MODE
        if not AI_MODE:
            robot_stop()
            suspend_ai("AI mode off (keyboard)")
        speak(f"AI mode {'enabled' if AI_MODE else 'disabled'} by keyboard")
    if key == ord("g"):
        GEMINI_ARMED = True
        speak("Jarvis is listening for your next question")

# =========================
# CLEANUP
# =========================
if ROBOT_ON:
    sock.sendto(b"ai_mode off", ROBOT_ADDR)

cap.release()
cv2.destroyAllWindows()

state.log("session_end")
with open(REPORT_FILE, "w") as f:
    json.dump(state.events, f, indent=2)

speak("Surgery session ended. Report saved.")
speech_queue.put(None)
