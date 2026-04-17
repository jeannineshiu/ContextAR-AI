"""
ContextAR - Adaptive Museum Companion
FastAPI server that merges hand, crowd, and noise signals into a single /state endpoint.

Usage:
    python server.py
    # or
    uvicorn server:app --reload
"""

import os
import time
import threading
import uuid
import cv2
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from hand_detector import is_hand_holding, FINGER_TIPS
import mediapipe as mp

from crowd_detector import get_crowd_status, MODEL_PATH, PERSON_CLASS_ID
from noise_detector import NoiseDetector
from rag_engine import RAGEngine
import qa_pipeline

# ---------------------------------------------------------------------------
# Shared state (updated by background thread)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_latest_state: dict = {}

# RAGEngine singleton — loaded once at startup, shared across requests
_rag: RAGEngine | None = None

# Directory for generated TTS audio files served to Unity
AUDIO_DIR = "static/audio"


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class HandState(BaseModel):
    detected: bool              # True if any hand is visible in the frame
    both_holding: bool          # True if both hands are gripping something
    per_hand: list[str]         # ["holding", "free"] per detected hand


class CrowdState(BaseModel):
    count: int                  # number of people detected
    level: str                  # "low" | "moderate" | "crowded"


class NoiseState(BaseModel):
    db: float                   # dBFS
    level: str                  # "quiet" | "moderate" | "noisy"
    centroid_hz: float          # spectral centroid in Hz


class ContextState(BaseModel):
    timestamp: float
    hands: HandState
    crowd: CrowdState
    noise: NoiseState
    suggestion: str             # high-level hint for the AR layer


class AskStateInput(BaseModel):
    crowd:        str  = "low"    # "low" | "moderate" | "crowded"
    noise:        str  = "quiet"  # "quiet" | "moderate" | "noisy"
    detected:     bool = False    # any hand visible in frame
    both_holding: bool = False    # both hands gripping something


class AskRequest(BaseModel):
    question:     str
    image_base64: str | None = None   # base64 JPEG/PNG from camera; omit to skip recognition
    state:        AskStateInput = AskStateInput()


class AskResponse(BaseModel):
    mode:      str          # "FULL_VOICE" | "BRIEF_TEXT" | "XR_MENU"
    answer:    str          # text answer; empty for XR_MENU
    audio_url: str          # relative URL to mp3 file; empty if no TTS
    exhibit:   str          # recognised painting name; empty if not identified


# ---------------------------------------------------------------------------
# Suggestion logic
# ---------------------------------------------------------------------------

def _make_suggestion(hands: dict, crowd: dict, noise: dict) -> str:
    """
    Derive a simple AR display suggestion from the three signals.
    Extend this with richer rules as the project grows.
    """
    if hands["both_holding"]:
        return "show_overlay"       # hands occupied → show info on screen
    if crowd["level"] == "crowded" or noise["level"] == "noisy":
        return "minimal_ui"         # busy environment → reduce visual noise
    return "full_ui"                # calm environment, hands free → full UI


# ---------------------------------------------------------------------------
# Background sensing loop
# ---------------------------------------------------------------------------

def _sensing_loop(noise_detector: NoiseDetector, camera_index: int = 0):
    """
    Runs in a daemon thread.
    Captures one frame per iteration, runs hand + crowd inference,
    merges with the latest noise reading, then writes to _latest_state.
    """
    mp_hands_mod = mp.solutions.hands
    yolo_model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(camera_index)

    with mp_hands_mod.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as mp_hands_inst:

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # --- Hand detection ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = mp_hands_inst.process(rgb)
            hand_list = hand_results.multi_hand_landmarks or []

            per_hand = ["holding" if is_hand_holding(lm) else "free" for lm in hand_list]
            both_holding = len(per_hand) == 2 and all(s == "holding" for s in per_hand)

            hands_state = {
                "detected": len(hand_list) > 0,
                "both_holding": both_holding,
                "per_hand": per_hand,
            }

            # --- Crowd detection ---
            crowd_state = get_crowd_status(frame, yolo_model)
            crowd_state.pop("boxes", None)  # don't expose raw boxes in API

            # --- Noise (already running in its own thread) ---
            noise_state = noise_detector.get_status()

            suggestion = _make_suggestion(hands_state, crowd_state, noise_state)

            with _state_lock:
                _latest_state.update({
                    "timestamp": time.time(),
                    "hands": hands_state,
                    "crowd": crowd_state,
                    "noise": noise_state,
                    "suggestion": suggestion,
                })

    cap.release()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag

    # Load RAG index once (expensive — do not repeat per request)
    _rag = RAGEngine()

    # Start noise detector
    noise_det = NoiseDetector()
    noise_det.start()

    # Start sensing loop in background thread
    t = threading.Thread(target=_sensing_loop, args=(noise_det,), daemon=True)
    t.start()

    # Give sensors a moment to produce the first reading
    time.sleep(1.5)

    yield  # server is running

    noise_det.stop()


app = FastAPI(title="ContextAR", version="0.1.0", lifespan=lifespan)
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/state", response_model=ContextState)
def get_state():
    """Return the latest merged sensor state."""
    with _state_lock:
        if not _latest_state:
            # Sensors not ready yet
            return ContextState(
                timestamp=time.time(),
                hands=HandState(detected=False, both_holding=False, per_hand=[]),
                crowd=CrowdState(count=0, level="low"),
                noise=NoiseState(db=-80.0, level="quiet", centroid_hz=0.0),
                suggestion="full_ui",
            )
        s = _latest_state.copy()

    return ContextState(
        timestamp=s["timestamp"],
        hands=HandState(**s["hands"]),
        crowd=CrowdState(**s["crowd"]),
        noise=NoiseState(**s["noise"]),
        suggestion=s["suggestion"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Full QA pipeline endpoint for Unity.

    Unity sends the visitor's question, an optional camera frame (base64),
    and the current sensor state. Returns the response mode, text answer,
    a URL to the TTS audio file, and the identified exhibit name.
    """
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready yet")

    result = qa_pipeline.run(
        question=req.question,
        image_b64=req.image_base64,
        api_state=req.state.model_dump(),
        rag=_rag,
    )

    # Save TTS audio and build a URL Unity can GET
    audio_url = ""
    if result["audio_bytes"]:
        filename  = f"{uuid.uuid4().hex}.mp3"
        filepath  = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(result["audio_bytes"])
        audio_url = f"/audio/{filename}"

    return AskResponse(
        mode      = result["mode"],
        answer    = result["answer"],
        audio_url = audio_url,
        exhibit   = result["exhibit"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
