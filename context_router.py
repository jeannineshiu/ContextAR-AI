"""
ContextAR - Context Router
Reads the merged state from server.py /state endpoint and decides
how the AR layer should respond to a visitor question.

Decision table:
┌─────────┬───────┬──────────┬──────────────┬────────────┐
│ crowd   │ noise │ detected │ both_holding │ mode       │
├─────────┼───────┼──────────┼──────────────┼────────────┤
│ crowded │ noisy │ True     │ True         │ XR_MENU    │
│ crowded │ noisy │ True     │ False        │ BRIEF_TEXT │
│ crowded │ noisy │ False    │ —            │ BRIEF_TEXT │
│ crowded │ quiet │ True     │ True         │ XR_MENU    │
│ crowded │ quiet │ True     │ False        │ FULL_VOICE │
│ crowded │ quiet │ False    │ —            │ FULL_VOICE │
│ low     │ quiet │ True     │ True         │ XR_MENU    │
│ low     │ quiet │ True     │ False        │ FULL_VOICE │
│ low     │ quiet │ False    │ —            │ FULL_VOICE │
└─────────┴───────┴──────────┴──────────────┴────────────┘
Priority: both_holding → XR_MENU; noisy → BRIEF_TEXT; quiet → FULL_VOICE

Usage (standalone demo):
    python context_router.py --question "Tell me about this painting"
"""

import argparse
import requests
from dataclasses import dataclass
from rag_engine import RAGEngine
from tts_engine import speak

STATE_URL = "http://localhost:8000/state"

# Answer length limits (characters)
BRIEF_MAX = 160
FULL_MAX  = None   # no limit


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    mode: str              # BRIEF_TEXT | FULL_VOICE | XR_MENU
    answer: str            # text answer (may be empty for XR_MENU / MAP)
    use_tts: bool          # should we speak the answer?
    tts_backend: str       # "gtts" or "elevenlabs"
    xr_action: str | None  # extra hint for the AR layer ("show_menu", "show_map", None)
    reason: str            # human-readable explanation of why this mode was chosen


# ---------------------------------------------------------------------------
# State fetcher
# ---------------------------------------------------------------------------

def fetch_state() -> dict:
    """
    Fetch latest sensor state from the FastAPI server.
    Returns a fallback (calm, empty) state if server is unreachable.
    """
    try:
        resp = requests.get(STATE_URL, timeout=1.0)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return {
            "hands":  {"detected": False, "both_holding": False, "per_hand": []},
            "crowd":  {"count": 0, "level": "low"},
            "noise":  {"db": -80.0, "level": "quiet", "centroid_hz": 0.0},
            "suggestion": "full_ui",
        }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _decide_mode(sensor_state: dict) -> tuple[str, str, str | None]:
    """
    Returns (mode, reason, xr_action) based on sensor state.
    Pure function — no side effects.
    """
    noise        = sensor_state["noise"]["level"]          # "quiet" | "moderate" | "noisy"
    hands        = sensor_state["hands"]
    both_holding = hands.get("both_holding", False)        # bool

    # Highest priority: both hands occupied → XR menu regardless of environment
    if both_holding:
        return (
            "XR_MENU",
            "both hands occupied → show XR menu overlay",
            "show_menu",
        )

    # Noisy environment → brief text only
    if noise in ("noisy", "moderate"):
        return (
            "BRIEF_TEXT",
            f"noise={noise} → short text only",
            None,
        )

    # Quiet environment, hands free → full immersive audio guide
    return (
        "FULL_VOICE",
        "quiet + hands free → full detail + voice",
        None,
    )


def route(question: str, rag: RAGEngine, state: dict | None = None,
          tts_backend: str = "gtts", play_audio: bool = True) -> RouterDecision:
    """
    Main entry point.

    Args:
        question:    visitor's question
        rag:         RAGEngine instance (pre-loaded)
        state:       sensor state dict; if None, fetches from server
        tts_backend: "gtts" or "elevenlabs"
        play_audio:  actually play TTS; set False for testing

    Returns:
        RouterDecision with answer, mode, and XR action
    """
    if state is None:
        state = fetch_state()

    mode, reason, xr_action = _decide_mode(state)

    # XR_MENU does not need a text answer
    if mode == "XR_MENU":
        return RouterDecision(
            mode=mode, answer="", use_tts=False,
            tts_backend=tts_backend, xr_action=xr_action, reason=reason,
        )

    # Determine answer length
    max_len = BRIEF_MAX if mode == "BRIEF_TEXT" else FULL_MAX
    rag_result = rag.query(question, max_length=max_len)
    answer = rag_result["answer"]

    # Speak only in FULL_VOICE mode
    use_tts = mode == "FULL_VOICE"
    if use_tts and play_audio:
        speak(answer, backend=tts_backend)

    return RouterDecision(
        mode=mode,
        answer=answer,
        use_tts=use_tts,
        tts_backend=tts_backend,
        xr_action=xr_action,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

_MODE_ICONS = {
    "BRIEF_TEXT": "📄",
    "FULL_VOICE": "📖🔊",
    "XR_MENU":    "🎛️ XR",
}


def _print_decision(d: RouterDecision):
    icon = _MODE_ICONS.get(d.mode, d.mode)
    print(f"\n  Mode   : {icon}  ({d.mode})")
    print(f"  Reason : {d.reason}")
    if d.xr_action:
        print(f"  XR     : {d.xr_action}")
    if d.answer:
        print(f"  Answer : {d.answer}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextAR Context Router")
    parser.add_argument("--question", type=str,
                        default="Tell me about this exhibit.")
    parser.add_argument("--backend", type=str, default="gtts",
                        choices=["gtts", "elevenlabs"])
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip TTS playback (text only)")

    # Manual state overrides for testing without running all sensors
    parser.add_argument("--crowd",  type=str, default=None,
                        choices=["low", "moderate", "crowded"])
    parser.add_argument("--noise",  type=str, default=None,
                        choices=["quiet", "moderate", "noisy"])
    parser.add_argument("--hands",  type=str, default=None,
                        choices=["none", "free", "holding"],
                        help="none=not detected, free=detected+idle, holding=both holding")
    args = parser.parse_args()

    # Build or override state
    if args.crowd or args.noise or args.hands:
        detected     = args.hands in ("free", "holding")   # bool
        both_holding = args.hands == "holding"
        cli_state = {
            "crowd": {"level": args.crowd or "low"},
            "noise": {"level": args.noise or "quiet"},
            "hands": {
                "detected": detected,
                "both_holding": both_holding,
                "per_hand": ["holding", "holding"] if both_holding else
                            (["free"] if detected else []),
            },
        }
        print(f"[manual state] crowd={args.crowd or 'low'}, "
              f"noise={args.noise or 'quiet'}, hands={args.hands or 'none'}")
    else:
        print("[fetching live state from server...]")
        cli_state = fetch_state()
        print(f"  crowd={cli_state['crowd']['level']}, "
              f"noise={cli_state['noise']['level']}, "
              f"hands={'holding' if cli_state['hands']['both_holding'] else 'free'}")

    rag_engine = RAGEngine()
    decision = route(
        question=args.question,
        rag=rag_engine,
        state=cli_state,
        tts_backend=args.backend,
        play_audio=not args.no_audio,
    )
    _print_decision(decision)
