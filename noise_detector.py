import numpy as np
import sounddevice as sd
import librosa
import threading
import queue
import time

# Audio config
SAMPLE_RATE = 22050       # Hz — librosa default
CHUNK_DURATION = 0.5      # seconds per analysis window
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Noise level thresholds (dBFS, higher = louder)
QUIET_DB   = -40          # below this → quiet
MODERATE_DB = -25         # below this → moderate
# above MODERATE_DB → loud/noisy


class NoiseDetector:
    """
    Continuous microphone noise analyser for ContextAR.

    Captures audio in a background thread and exposes:
      - current dBFS level
      - noise label: "quiet" | "moderate" | "noisy"
      - spectral centroid (indicates pitch / tone character of noise)
    """

    def __init__(self, device=None):
        self._q = queue.Queue(maxsize=10)
        self._lock = threading.Lock()
        self._running = False
        self._device = device

        self._db = -80.0
        self._level = "quiet"
        self._centroid = 0.0
        self._stream = None
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background capture and analysis threads."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            dtype="float32",
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._analyse_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop capture and analysis."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def get_status(self) -> dict:
        """
        Thread-safe snapshot of current noise status.

        Returns:
            {
                "db": float,          # dBFS, e.g. -32.5
                "level": str,         # "quiet" | "moderate" | "noisy"
                "centroid_hz": float  # spectral centroid in Hz
            }
        """
        with self._lock:
            return {
                "db": round(self._db, 1),
                "level": self._level,
                "centroid_hz": round(self._centroid, 1),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice on each chunk — just enqueue."""
        if not self._q.full():
            self._q.put(indata[:, 0].copy())

    def _analyse_loop(self):
        while self._running:
            try:
                chunk = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            db = _rms_db(chunk)
            level = _classify_level(db)
            centroid = _spectral_centroid(chunk, SAMPLE_RATE)

            with self._lock:
                self._db = db
                self._level = level
                self._centroid = centroid


# ------------------------------------------------------------------
# DSP helpers
# ------------------------------------------------------------------

def _rms_db(chunk: np.ndarray) -> float:
    """RMS amplitude → dBFS. Returns -80 for silence."""
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 1e-9:
        return -80.0
    return float(20 * np.log10(rms))


def _classify_level(db: float) -> str:
    if db < QUIET_DB:
        return "quiet"
    elif db < MODERATE_DB:
        return "moderate"
    else:
        return "noisy"


def _spectral_centroid(chunk: np.ndarray, sr: int) -> float:
    """Mean spectral centroid in Hz for this chunk."""
    if np.max(np.abs(chunk)) < 1e-9:
        return 0.0
    centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
    return float(np.mean(centroid))


# ------------------------------------------------------------------
# Standalone demo
# ------------------------------------------------------------------

LEVEL_COLORS = {
    "quiet":    "\033[92m",   # green
    "moderate": "\033[93m",   # yellow
    "noisy":    "\033[91m",   # red
}
RESET = "\033[0m"


def run():
    print("ContextAR - Noise Detector  (Ctrl+C to stop)\n")
    detector = NoiseDetector()
    detector.start()

    try:
        while True:
            s = detector.get_status()
            color = LEVEL_COLORS[s["level"]]
            print(
                f"\r{color}[{s['level'].upper():8}]{RESET} "
                f"{s['db']:6.1f} dBFS  |  centroid {s['centroid_hz']:7.1f} Hz   ",
                end="", flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    run()
