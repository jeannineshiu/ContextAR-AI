"""
ContextAR - QA Pipeline
Orchestrates the four existing modules in order:

  exhibit_recognizer  →  identify painting from camera frame
  rag_engine          →  retrieve exhibit knowledge
  context_router      →  decide mode (FULL_VOICE / BRIEF_TEXT / XR_MENU)
  tts_engine          →  synthesise audio if mode requires it

This module contains no AI logic of its own — it is a coordinator.
The RAGEngine instance is injected by the caller (server.py) so the
FAISS index is only loaded once at startup.

Usage (standalone test):
    python qa_pipeline.py
"""

import base64
import numpy as np
import cv2

from exhibit_recognizer import recognize_exhibit
from rag_engine import RAGEngine
from context_router import route
from tts_engine import speak


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _b64_to_frame(image_b64: str) -> np.ndarray | None:
    """
    Decode a base64 image string to a BGR numpy array (cv2 format).
    Returns None if decoding fails.
    """
    try:
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame  # None if imdecode fails
    except Exception:
        return None


def _normalize_state(api_state: dict) -> dict:
    """
    Convert the flat state dict sent by Unity into the nested internal
    format expected by context_router._decide_mode().

    Unity sends:
        {"crowd": "low", "noise": "quiet",
         "detected": false, "both_holding": false}

    Internal format:
        {"crowd":  {"level": "low"},
         "noise":  {"level": "quiet", "db": -80.0, "centroid_hz": 0.0},
         "hands":  {"detected": False, "both_holding": False, "per_hand": []}}
    """
    return {
        "crowd": {
            "level": api_state.get("crowd", "low"),
        },
        "noise": {
            "level":       api_state.get("noise", "quiet"),
            "db":          -80.0,
            "centroid_hz": 0.0,
        },
        "hands": {
            "detected":     api_state.get("detected", False),
            "both_holding": api_state.get("both_holding", False),
            "per_hand":     [],
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    question:   str,
    image_b64:  str | None,
    api_state:  dict,
    rag:        RAGEngine,
    tts_backend: str = "gtts",
) -> dict:
    """
    Full QA pipeline. No hardware access — everything is passed in.

    Args:
        question:    Visitor's natural-language question.
        image_b64:   Base64-encoded JPEG/PNG of the current camera frame.
                     Pass None or "" to skip exhibit recognition.
        api_state:   Flat state dict from Unity (crowd/noise/detected/both_holding).
        rag:         Pre-loaded RAGEngine singleton (injected by server.py).
        tts_backend: "gtts" (default) or "elevenlabs".

    Returns:
        {
            "mode":        str,    # "FULL_VOICE" | "BRIEF_TEXT" | "XR_MENU"
            "answer":      str,    # empty string for XR_MENU
            "audio_bytes": bytes,  # mp3 bytes; empty if mode has no TTS
            "exhibit":     str,    # recognised exhibit name or ""
        }
    """
    # ------------------------------------------------------------------
    # Step 1: Identify exhibit from camera frame (optional)
    # ------------------------------------------------------------------
    exhibit_name = ""
    if image_b64:
        frame = _b64_to_frame(image_b64)
        if frame is not None:
            recognition = recognize_exhibit(frame)
            if (
                "error" not in recognition
                and recognition.get("confidence") in ("high", "medium")
                and recognition.get("name", "unknown").lower() != "unknown"
            ):
                exhibit_name = recognition["name"]

    # ------------------------------------------------------------------
    # Step 2: Enrich question with exhibit context if recognised
    # ------------------------------------------------------------------
    enriched_question = (
        f"[Regarding: {exhibit_name}] {question}"
        if exhibit_name
        else question
    )

    # ------------------------------------------------------------------
    # Step 3: Route through context router → mode + answer
    # ------------------------------------------------------------------
    internal_state = _normalize_state(api_state)
    decision = route(
        question=enriched_question,
        rag=rag,
        state=internal_state,
        play_audio=False,   # server handles audio separately
    )

    # ------------------------------------------------------------------
    # Step 4: Synthesise TTS audio if the mode requires it
    # ------------------------------------------------------------------
    audio_bytes = b""
    if decision.use_tts and decision.answer:
        audio_bytes = speak(decision.answer, backend=tts_backend, play=False)

    return {
        "mode":        decision.mode,
        "answer":      decision.answer,
        "audio_bytes": audio_bytes,
        "exhibit":     exhibit_name,
    }


# ---------------------------------------------------------------------------
# Standalone smoke test (no Unity needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading RAG engine...")
    rag = RAGEngine()

    test_cases = [
        {
            "label": "Scene A — FULL_VOICE",
            "state": {"crowd": "low",     "noise": "quiet", "detected": True,  "both_holding": False},
        },
        {
            "label": "Scene B — BRIEF_TEXT",
            "state": {"crowd": "crowded", "noise": "noisy", "detected": False, "both_holding": False},
        },
        {
            "label": "Scene C — XR_MENU",
            "state": {"crowd": "low",     "noise": "quiet", "detected": True,  "both_holding": True},
        },
    ]

    for tc in test_cases:
        result = run(
            question="Tell me about this painting",
            image_b64=None,
            api_state=tc["state"],
            rag=rag,
        )
        print(f"\n{tc['label']}")
        print(f"  Mode   : {result['mode']}")
        print(f"  Answer : {result['answer'][:80]}{'…' if len(result['answer']) > 80 else ''}")
        print(f"  Audio  : {len(result['audio_bytes'])} bytes")
