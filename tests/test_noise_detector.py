"""
Tests for noise_detector DSP helpers and NoiseDetector.get_status().

Run with:
    python -m pytest test_noise_detector.py -v

No microphone needed — all tests use synthetic numpy arrays.
"""

import numpy as np
import pytest
from noise_detector import (
    _rms_db,
    _classify_level,
    _spectral_centroid,
    NoiseDetector,
    QUIET_DB,
    MODERATE_DB,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# _rms_db
# ---------------------------------------------------------------------------

class TestRmsDb:

    def test_silence_returns_minus80(self):
        chunk = np.zeros(1024, dtype=np.float32)
        assert _rms_db(chunk) == -80.0

    def test_near_zero_signal_returns_minus80(self):
        """Values below 1e-9 RMS are treated as silence."""
        chunk = np.full(1024, 1e-10, dtype=np.float32)
        assert _rms_db(chunk) == -80.0

    def test_full_scale_signal_returns_0_dbfs(self):
        """RMS of 1.0 → 0 dBFS."""
        chunk = np.ones(1024, dtype=np.float32)
        assert _rms_db(chunk) == pytest.approx(0.0, abs=0.01)

    def test_tenth_amplitude_returns_minus20_dbfs(self):
        """RMS of 0.1 → -20 dBFS."""
        chunk = np.full(1024, 0.1, dtype=np.float32)
        assert _rms_db(chunk) == pytest.approx(-20.0, abs=0.01)

    def test_returns_float(self):
        chunk = np.random.uniform(-0.5, 0.5, 1024).astype(np.float32)
        assert isinstance(_rms_db(chunk), float)


# ---------------------------------------------------------------------------
# _classify_level  (thresholds: QUIET_DB=-40, MODERATE_DB=-25)
# ---------------------------------------------------------------------------

class TestClassifyLevel:

    def test_below_quiet_threshold_is_quiet(self):
        assert _classify_level(QUIET_DB - 1) == "quiet"

    def test_well_below_quiet_is_quiet(self):
        assert _classify_level(-80.0) == "quiet"

    def test_at_quiet_boundary_is_moderate(self):
        """db == QUIET_DB is NOT quiet (condition is db < QUIET_DB)."""
        assert _classify_level(QUIET_DB) == "moderate"

    def test_between_thresholds_is_moderate(self):
        midpoint = (QUIET_DB + MODERATE_DB) / 2
        assert _classify_level(midpoint) == "moderate"

    def test_just_below_moderate_threshold_is_moderate(self):
        assert _classify_level(MODERATE_DB - 0.1) == "moderate"

    def test_at_moderate_boundary_is_noisy(self):
        """db == MODERATE_DB is noisy (condition is db < MODERATE_DB)."""
        assert _classify_level(MODERATE_DB) == "noisy"

    def test_above_moderate_threshold_is_noisy(self):
        assert _classify_level(MODERATE_DB + 10) == "noisy"

    def test_returns_string(self):
        assert isinstance(_classify_level(-50.0), str)


# ---------------------------------------------------------------------------
# _spectral_centroid
# ---------------------------------------------------------------------------

class TestSpectralCentroid:

    def test_silence_returns_zero(self):
        chunk = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        assert _spectral_centroid(chunk, SAMPLE_RATE) == 0.0

    def test_near_zero_signal_returns_zero(self):
        chunk = np.full(SAMPLE_RATE // 2, 1e-10, dtype=np.float32)
        assert _spectral_centroid(chunk, SAMPLE_RATE) == 0.0

    def test_sine_wave_centroid_near_frequency(self):
        """440 Hz pure tone → centroid should be close to 440 Hz."""
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        chunk = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        centroid = _spectral_centroid(chunk, SAMPLE_RATE)
        assert centroid == pytest.approx(440.0, rel=0.05)   # within 5 %

    def test_returns_float(self):
        chunk = np.random.uniform(-0.5, 0.5, 4096).astype(np.float32)
        assert isinstance(_spectral_centroid(chunk, SAMPLE_RATE), float)


# ---------------------------------------------------------------------------
# NoiseDetector.get_status() — state shape and defaults
# ---------------------------------------------------------------------------

class TestNoiseDetectorGetStatus:

    def test_default_state_before_start(self):
        """Before start(), get_status() should return safe quiet defaults."""
        detector = NoiseDetector()
        status = detector.get_status()
        assert status["db"] == -80.0
        assert status["level"] == "quiet"
        assert status["centroid_hz"] == 0.0

    def test_get_status_returns_correct_keys(self):
        detector = NoiseDetector()
        status = detector.get_status()
        assert set(status.keys()) == {"db", "level", "centroid_hz"}

    def test_get_status_reflects_internal_state(self):
        """Manually set internal state and verify get_status() reads it."""
        detector = NoiseDetector()
        with detector._lock:
            detector._db = -30.0
            detector._level = "moderate"
            detector._centroid = 1200.0

        status = detector.get_status()
        assert status["db"] == -30.0
        assert status["level"] == "moderate"
        assert status["centroid_hz"] == 1200.0

    def test_get_status_rounds_db_to_one_decimal(self):
        detector = NoiseDetector()
        with detector._lock:
            detector._db = -32.456
        assert detector.get_status()["db"] == -32.5

    def test_get_status_rounds_centroid_to_one_decimal(self):
        detector = NoiseDetector()
        with detector._lock:
            detector._centroid = 1234.567
        assert detector.get_status()["centroid_hz"] == 1234.6
