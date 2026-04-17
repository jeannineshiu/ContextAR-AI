"""
Tests for qa_pipeline._b64_to_frame(), _normalize_state(), and run().

Run with:
    python -m pytest test_qa_pipeline.py -v

All four downstream modules (exhibit_recognizer, rag_engine,
context_router, tts_engine) are mocked — no API calls, no camera.
"""

import base64
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import pytest

from qa_pipeline import _b64_to_frame, _normalize_state, run
from context_router import RouterDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_b64() -> str:
    """Encode a tiny black 10×10 JPEG as base64."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def make_mock_rag() -> MagicMock:
    return MagicMock()


def make_decision(
    mode: str = "FULL_VOICE",
    answer: str = "Test answer about the painting.",
    use_tts: bool = True,
) -> RouterDecision:
    return RouterDecision(
        mode=mode,
        answer=answer,
        use_tts=use_tts,
        tts_backend="gtts",
        xr_action=None,
        reason="test",
    )


# ---------------------------------------------------------------------------
# _b64_to_frame
# ---------------------------------------------------------------------------

class TestB64ToFrame:

    def test_valid_image_returns_numpy_array(self):
        frame = _b64_to_frame(make_valid_b64())
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (10, 10, 3)

    def test_invalid_base64_returns_none(self):
        assert _b64_to_frame("!!!not_base64!!!") is None

    def test_valid_base64_but_not_image_returns_none(self):
        not_an_image = base64.b64encode(b"this is not image data").decode()
        assert _b64_to_frame(not_an_image) is None


# ---------------------------------------------------------------------------
# _normalize_state
# ---------------------------------------------------------------------------

class TestNormalizeState:

    def test_maps_crowd_level(self):
        result = _normalize_state({"crowd": "crowded"})
        assert result["crowd"]["level"] == "crowded"

    def test_maps_noise_level(self):
        result = _normalize_state({"noise": "noisy"})
        assert result["noise"]["level"] == "noisy"

    def test_maps_detected(self):
        result = _normalize_state({"detected": True})
        assert result["hands"]["detected"] is True

    def test_maps_both_holding(self):
        result = _normalize_state({"both_holding": True})
        assert result["hands"]["both_holding"] is True

    def test_defaults_when_keys_missing(self):
        result = _normalize_state({})
        assert result["crowd"]["level"]      == "low"
        assert result["noise"]["level"]      == "quiet"
        assert result["hands"]["detected"]   is False
        assert result["hands"]["both_holding"] is False

    def test_output_has_nested_structure(self):
        result = _normalize_state({})
        assert "level" in result["crowd"]
        assert "level" in result["noise"]
        assert "both_holding" in result["hands"]

    def test_noise_includes_db_and_centroid(self):
        result = _normalize_state({})
        assert "db" in result["noise"]
        assert "centroid_hz" in result["noise"]


# ---------------------------------------------------------------------------
# run() — exhibit recognition branch
# ---------------------------------------------------------------------------

class TestRunExhibitRecognition:

    QUIET_STATE = {"crowd": "low", "noise": "quiet",
                   "detected": True, "both_holding": False}

    def test_no_image_skips_recognition(self):
        with patch("qa_pipeline.recognize_exhibit") as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64=None,
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_not_called()

    def test_empty_string_image_skips_recognition(self):
        with patch("qa_pipeline.recognize_exhibit") as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64="",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_not_called()

    def test_valid_image_calls_recognition(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "Madame X (Madame Pierre Gautreau)",
                 "confidence": "high",
             }) as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64="fake_b64",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_called_once()

    def test_recognised_exhibit_enriches_question(self):
        """Question sent to route() should include exhibit name."""
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "Madame X (Madame Pierre Gautreau)",
                 "confidence": "high",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Who is she?", image_b64="fake_b64",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert "Madame X" in called_question
            assert "Who is she?" in called_question

    def test_unknown_name_does_not_enrich_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "unknown", "confidence": "high",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Tell me more", image_b64="fake_b64",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"

    def test_low_confidence_does_not_enrich_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "The Harvesters", "confidence": "low",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Tell me more", image_b64="fake_b64",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"

    def test_recognition_error_does_not_enrich_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "error": "invalid_response",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Tell me more", image_b64="fake_b64",
                api_state=self.QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"


# ---------------------------------------------------------------------------
# run() — mode and TTS behaviour
# ---------------------------------------------------------------------------

class TestRunModes:

    def _run_with_mode(self, mode: str, use_tts: bool, answer: str = "Some answer."):
        with patch("qa_pipeline.recognize_exhibit"), \
             patch("qa_pipeline.route",
                   return_value=make_decision(mode=mode, answer=answer, use_tts=use_tts)), \
             patch("qa_pipeline.speak", return_value=b"audio") as mock_speak:
            result = run(
                question="test", image_b64=None,
                api_state={"crowd": "low", "noise": "quiet",
                           "detected": False, "both_holding": False},
                rag=make_mock_rag(),
            )
            return result, mock_speak

    def test_full_voice_returns_audio_bytes(self):
        result, _ = self._run_with_mode("FULL_VOICE", use_tts=True)
        assert result["mode"] == "FULL_VOICE"
        assert result["audio_bytes"] == b"audio"

    def test_full_voice_calls_speak(self):
        _, mock_speak = self._run_with_mode("FULL_VOICE", use_tts=True)
        mock_speak.assert_called_once()

    def test_brief_text_does_not_call_speak(self):
        _, mock_speak = self._run_with_mode("BRIEF_TEXT", use_tts=False)
        mock_speak.assert_not_called()

    def test_brief_text_audio_bytes_empty(self):
        result, _ = self._run_with_mode("BRIEF_TEXT", use_tts=False)
        assert result["audio_bytes"] == b""

    def test_xr_menu_does_not_call_speak(self):
        _, mock_speak = self._run_with_mode("XR_MENU", use_tts=False, answer="")
        mock_speak.assert_not_called()

    def test_xr_menu_audio_bytes_empty(self):
        result, _ = self._run_with_mode("XR_MENU", use_tts=False, answer="")
        assert result["audio_bytes"] == b""


# ---------------------------------------------------------------------------
# run() — return dict shape
# ---------------------------------------------------------------------------

class TestRunReturnShape:

    def test_returns_all_keys(self):
        with patch("qa_pipeline.route", return_value=make_decision()), \
             patch("qa_pipeline.speak", return_value=b"audio"):
            result = run(
                question="test", image_b64=None,
                api_state={"crowd": "low", "noise": "quiet",
                           "detected": False, "both_holding": False},
                rag=make_mock_rag(),
            )
        assert set(result.keys()) == {"mode", "answer", "audio_bytes", "exhibit"}

    def test_exhibit_empty_when_no_image(self):
        with patch("qa_pipeline.route", return_value=make_decision()), \
             patch("qa_pipeline.speak", return_value=b"audio"):
            result = run(
                question="test", image_b64=None,
                api_state={}, rag=make_mock_rag(),
            )
        assert result["exhibit"] == ""

    def test_exhibit_name_returned_when_recognised(self):
        with patch("qa_pipeline._b64_to_frame",
                   return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "The Card Players", "confidence": "high",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()), \
             patch("qa_pipeline.speak", return_value=b"audio"):
            result = run(
                question="test", image_b64="fake_b64",
                api_state={"crowd": "low", "noise": "quiet",
                           "detected": True, "both_holding": False},
                rag=make_mock_rag(),
            )
        assert result["exhibit"] == "The Card Players"
