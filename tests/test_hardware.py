"""
Hardware integration tests for ContextAR.
These tests talk to real devices — camera and microphone.

Run with:
    python -m pytest test_hardware.py --hardware -v

Skipped automatically in CI and when running plain `python -m pytest`.
"""

import time
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Microphone — NoiseDetector
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestNoiseDetectorLive:

    def test_starts_and_stops_without_error(self):
        """NoiseDetector must start, run briefly, and stop cleanly."""
        from noise_detector import NoiseDetector
        detector = NoiseDetector()
        detector.start()
        time.sleep(0.5)   # let it capture one chunk
        detector.stop()   # should not raise

    def test_get_status_returns_real_reading_after_start(self):
        """After 1.5 s of capture, db should no longer be the default -80.
        We wait 1.5 s because CHUNK_DURATION=0.5 s, so we need at least one
        full chunk captured AND processed by the analysis thread."""
        from noise_detector import NoiseDetector
        detector = NoiseDetector()
        detector.start()
        time.sleep(1.5)
        status = detector.get_status()
        detector.stop()

        assert "db" in status
        assert "level" in status
        assert "centroid_hz" in status
        # In a quiet room this will be around -50 to -30, never exactly -80
        assert status["db"] > -80.0, \
            "db is still at the default — audio capture may not be working"

    def test_level_is_a_valid_string(self):
        from noise_detector import NoiseDetector
        detector = NoiseDetector()
        detector.start()
        time.sleep(0.5)
        level = detector.get_status()["level"]
        detector.stop()
        assert level in ("quiet", "moderate", "noisy")

    def test_centroid_is_non_negative(self):
        from noise_detector import NoiseDetector
        detector = NoiseDetector()
        detector.start()
        time.sleep(0.5)
        centroid = detector.get_status()["centroid_hz"]
        detector.stop()
        assert centroid >= 0.0


# ---------------------------------------------------------------------------
# Camera — one-shot frame capture
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCameraCapture:

    def test_camera_opens_and_returns_frame(self):
        """cv2 must be able to open camera index 0 and read one frame."""
        import cv2
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Camera index 0 could not be opened"
        ret, frame = cap.read()
        cap.release()
        assert ret, "cap.read() returned False — no frame captured"
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3          # H × W × C
        assert frame.shape[2] == 3      # BGR

    def test_frame_is_not_all_black(self):
        """A working camera should produce a frame with some variation."""
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        assert ret
        assert frame.std() > 1.0, \
            "Frame is all-black — lens cap on, or camera not working"


# ---------------------------------------------------------------------------
# Hand detector — live frame
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestHandDetectorLive:

    def test_pipeline_runs_on_real_frame(self):
        """MediaPipe hand detector must process a real camera frame without error."""
        import cv2
        import mediapipe as mp
        from hand_detector import is_hand_holding

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        assert ret

        mp_hands_mod = mp.solutions.hands
        with mp_hands_mod.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

        # No exception = pipeline works; hand presence is not guaranteed
        hand_list = results.multi_hand_landmarks or []
        for lm in hand_list:
            holding = is_hand_holding(lm)
            assert isinstance(holding, bool)


# ---------------------------------------------------------------------------
# Crowd detector — live frame
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestCrowdDetectorLive:

    def test_pipeline_runs_on_real_frame(self):
        """YOLO crowd detector must process a real camera frame without error."""
        import cv2
        from ultralytics import YOLO
        from crowd_detector import get_crowd_status, MODEL_PATH

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        assert ret

        model = YOLO(MODEL_PATH)
        status = get_crowd_status(frame, model)

        assert "count" in status
        assert "level" in status
        assert status["level"] in ("low", "moderate", "crowded")
        assert status["count"] >= 0


# ---------------------------------------------------------------------------
# Sensing loop smoke test — 2 seconds
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestSensingLoopSmoke:

    def test_sensing_loop_populates_state(self):
        """Run the full sensing loop for 2 s and verify _latest_state is filled."""
        import threading
        import server
        from noise_detector import NoiseDetector

        server._latest_state.clear()

        noise_det = NoiseDetector()
        noise_det.start()

        t = threading.Thread(
            target=server._sensing_loop,
            args=(noise_det,),
            daemon=True,
        )
        t.start()
        time.sleep(2.0)   # allow at least one sensing cycle
        noise_det.stop()

        assert server._latest_state, "_latest_state is still empty after 2 s"
        state = server._latest_state
        assert "hands" in state
        assert "crowd" in state
        assert "noise" in state
        assert "suggestion" in state
