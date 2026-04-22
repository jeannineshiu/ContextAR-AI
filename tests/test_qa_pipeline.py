"""
Tests for qa_pipeline._b64_to_frame() and run().

Run with:
    python -m pytest tests/test_qa_pipeline.py -v

All downstream modules (exhibit_recognizer, rag_engine, context_router)
are mocked — no API calls, no camera.
"""

import base64
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from qa_pipeline import _b64_to_frame, run
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
) -> RouterDecision:
    return RouterDecision(
        mode=mode,
        answer=answer,
        xr_action=None,
        reason="test",
    )


QUIET_STATE = {"crowd": "low", "noise": "quiet", "gaze_duration": 20.0}


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
# run() — exhibit recognition branch
# ---------------------------------------------------------------------------

class TestRunExhibitRecognition:

    def test_no_image_skips_recognition(self):
        with patch("qa_pipeline.recognize_exhibit") as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64=None,
                api_state=QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_not_called()

    def test_empty_string_image_skips_recognition(self):
        with patch("qa_pipeline.recognize_exhibit") as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64="",
                api_state=QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_not_called()

    def test_valid_image_calls_recognition(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "Madame X (Madame Pierre Gautreau)",
                 "confidence": "high",
             }) as mock_rec, \
             patch("qa_pipeline.route", return_value=make_decision()):
            run(question="test", image_b64="fake_b64",
                api_state=QUIET_STATE, rag=make_mock_rag())
            mock_rec.assert_called_once()

    def test_recognised_exhibit_enriches_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "Madame X (Madame Pierre Gautreau)",
                 "confidence": "high",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Who is she?", image_b64="fake_b64",
                api_state=QUIET_STATE, rag=make_mock_rag())
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
                api_state=QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"

    def test_low_confidence_does_not_enrich_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "name": "The Harvesters", "confidence": "low",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Tell me more", image_b64="fake_b64",
                api_state=QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"

    def test_recognition_error_does_not_enrich_question(self):
        with patch("qa_pipeline._b64_to_frame", return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             patch("qa_pipeline.recognize_exhibit", return_value={
                 "error": "invalid_response",
             }), \
             patch("qa_pipeline.route", return_value=make_decision()) as mock_route:
            run(question="Tell me more", image_b64="fake_b64",
                api_state=QUIET_STATE, rag=make_mock_rag())
            called_question = mock_route.call_args.kwargs["question"]
            assert called_question == "Tell me more"


# ---------------------------------------------------------------------------
# run() — return dict shape
# ---------------------------------------------------------------------------

class TestRunReturnShape:

    def test_returns_all_keys(self):
        with patch("qa_pipeline.route", return_value=make_decision()):
            result = run(
                question="test", image_b64=None,
                api_state=QUIET_STATE, rag=make_mock_rag(),
            )
        assert set(result.keys()) == {"mode", "answer", "exhibit"}

    def test_exhibit_empty_when_no_image(self):
        with patch("qa_pipeline.route", return_value=make_decision()):
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
             patch("qa_pipeline.route", return_value=make_decision()):
            result = run(
                question="test", image_b64="fake_b64",
                api_state=QUIET_STATE, rag=make_mock_rag(),
            )
        assert result["exhibit"] == "The Card Players"

    def test_mode_is_passed_through(self):
        with patch("qa_pipeline.route", return_value=make_decision(mode="BRIEF_TEXT")):
            result = run(
                question="test", image_b64=None,
                api_state={"crowd": "low", "noise": "quiet", "gaze_duration": 8.0},
                rag=make_mock_rag(),
            )
        assert result["mode"] == "BRIEF_TEXT"

    def test_no_response_returns_empty_answer(self):
        with patch("qa_pipeline.route", return_value=make_decision(mode="NO_RESPONSE", answer="")):
            result = run(
                question="test", image_b64=None,
                api_state={"crowd": "low", "noise": "quiet", "gaze_duration": 2.0},
                rag=make_mock_rag(),
            )
        assert result["mode"] == "NO_RESPONSE"
        assert result["answer"] == ""
