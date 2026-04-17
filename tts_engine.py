"""
ContextAR - TTS Engine
Supports two backends:
  - "elevenlabs"  : high quality, requires ELEVENLABS_API_KEY in .env
  - "gtts"        : free, requires internet, uses Google TTS

Usage:
    python tts_engine.py --text "Hello, welcome to the museum." --backend gtts
    python tts_engine.py --text "Hello, welcome to the museum." --backend elevenlabs
"""

import os
import io
import subprocess
import tempfile
import argparse
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # "George" — calm, museum-friendly
ELEVENLABS_MODEL = "eleven_turbo_v2_5"   # low latency model

GTTS_LANG = "en"


# ---------------------------------------------------------------------------
# Playback helper
# ---------------------------------------------------------------------------

def _play_bytes(audio_bytes: bytes, fmt: str = "mp3"):
    """Play raw audio bytes using ffmpeg → afplay pipeline (macOS)."""
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        subprocess.run(["afplay", tmp_path], check=True)
    finally:
        os.unlink(tmp_path)


def _play_file(path: str):
    """Play an audio file using afplay (macOS)."""
    subprocess.run(["afplay", path], check=True)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _speak_gtts(text: str, play: bool = True) -> bytes:
    """
    Synthesise speech with Google TTS (free, no API key needed).
    Returns mp3 bytes. Plays immediately if play=True.
    """
    from gtts import gTTS
    tts = gTTS(text=text, lang=GTTS_LANG, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    audio_bytes = buf.getvalue()
    if play:
        _play_bytes(audio_bytes, fmt="mp3")
    return audio_bytes


def _speak_elevenlabs(text: str, play: bool = True) -> bytes:
    """
    Synthesise speech with ElevenLabs (high quality).
    Returns mp3 bytes. Plays immediately if play=True.
    Requires ELEVENLABS_API_KEY in .env
    """
    from elevenlabs import ElevenLabs

    if not ELEVENLABS_API_KEY:
        raise ValueError(
            "ELEVENLABS_API_KEY not set. Add it to .env or use backend='gtts'."
        )

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_iter = client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL,
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_iter)
    if play:
        _play_bytes(audio_bytes, fmt="mp3")
    return audio_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str, backend: str = "gtts", play: bool = True) -> bytes:
    """
    Synthesise and optionally play speech.

    Args:
        text:    text to speak
        backend: "gtts" (free) or "elevenlabs" (high quality)
        play:    if True, play audio immediately; if False, just return bytes

    Returns:
        mp3 audio as bytes (for saving or streaming)

    Example:
        speak("Welcome to the museum!", backend="gtts")
        audio = speak("Mona Lisa was painted by Leonardo.", play=False)
    """
    text = text.strip()
    if not text:
        return b""

    if backend == "elevenlabs":
        return _speak_elevenlabs(text, play=play)
    else:
        return _speak_gtts(text, play=play)


def save(text: str, path: str, backend: str = "gtts"):
    """
    Synthesise speech and save to file (no playback).

    Args:
        text:    text to speak
        path:    output file path, e.g. "output.mp3"
        backend: "gtts" or "elevenlabs"
    """
    audio_bytes = speak(text, backend=backend, play=False)
    with open(path, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextAR TTS Engine")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--backend", type=str, default="gtts",
                        choices=["gtts", "elevenlabs"], help="TTS backend")
    parser.add_argument("--save", type=str, default=None,
                        help="Save audio to this file path instead of playing")
    args = parser.parse_args()

    if args.save:
        save(args.text, args.save, backend=args.backend)
    else:
        print(f"[{args.backend.upper()}] Speaking: {args.text}")
        speak(args.text, backend=args.backend)
