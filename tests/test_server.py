"""
Tests for server.py: _make_suggestion(), /health, /state, and /ask endpoints.

Run with:
    python -m pytest test_server.py -v

The lifespan (NoiseDetector + sensing thread + RAGEngine) is mocked so
no camera, microphone, or OpenAI API is needed.
"""

import os
import time
from unittest.mock import patch, MagicMock, mock_open
import pytest
from fastapi.testclient import TestClient

import server
from server import _make_suggestion


# ---------------------------------------------------------------------------
# Fixture: TestClient with lifespan mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Yield a TestClient that skips all real hardware and AI init.
    Patches: NoiseDetector, background Thread, RAGEngine, and the 1.5s sleep.
    Also ensures the static/audio directory exists for StaticFiles.
    """
    with patch("server.NoiseDetector") as mock_nd_cls, \
         patch("server.threading.Thread") as mock_thread_cls, \
         patch("server.time.sleep"), \
         patch("server.RAGEngine"):
        mock_nd_cls.return_value.start = MagicMock()
        mock_nd_cls.return_value.stop  = MagicMock()
        mock_thread_cls.return_value.start = MagicMock()

        server._latest_state.clear()

        with TestClient(server.app) as c:
            yield c


# ---------------------------------------------------------------------------
# _make_suggestion — pure function
# ---------------------------------------------------------------------------

class TestMakeSuggestion:

    def _hands(self, both_holding: bool) -> dict:
        return {"detected": both_holding, "both_holding": both_holding, "per_hand": []}

    def _crowd(self, level: str) -> dict:
        return {"count": 0, "level": level}

    def _noise(self, level: str) -> dict:
        return {"db": -50.0, "level": level, "centroid_hz": 0.0}

    def test_both_holding_returns_show_overlay(self):
        result = _make_suggestion(
            self._hands(True), self._crowd("low"), self._noise("quiet")
        )
        assert result == "show_overlay"

    def test_both_holding_beats_crowded_noisy(self):
        result = _make_suggestion(
            self._hands(True), self._crowd("crowded"), self._noise("noisy")
        )
        assert result == "show_overlay"

    def test_crowded_returns_minimal_ui(self):
        result = _make_suggestion(
            self._hands(False), self._crowd("crowded"), self._noise("quiet")
        )
        assert result == "minimal_ui"

    def test_noisy_returns_minimal_ui(self):
        result = _make_suggestion(
            self._hands(False), self._crowd("low"), self._noise("noisy")
        )
        assert result == "minimal_ui"

    def test_crowded_and_noisy_returns_minimal_ui(self):
        result = _make_suggestion(
            self._hands(False), self._crowd("crowded"), self._noise("noisy")
        )
        assert result == "minimal_ui"

    def test_low_quiet_hands_free_returns_full_ui(self):
        result = _make_suggestion(
            self._hands(False), self._crowd("low"), self._noise("quiet")
        )
        assert result == "full_ui"

    def test_moderate_crowd_quiet_returns_full_ui(self):
        result = _make_suggestion(
            self._hands(False), self._crowd("moderate"), self._noise("quiet")
        )
        assert result == "full_ui"


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_returns_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /state endpoint — fallback (empty _latest_state)
# ---------------------------------------------------------------------------

class TestStateEndpointFallback:

    def test_returns_200_when_state_empty(self, client):
        assert client.get("/state").status_code == 200

    def test_fallback_hands_are_not_detected(self, client):
        data = client.get("/state").json()
        assert data["hands"]["detected"] is False
        assert data["hands"]["both_holding"] is False
        assert data["hands"]["per_hand"] == []

    def test_fallback_crowd_is_low(self, client):
        data = client.get("/state").json()
        assert data["crowd"]["level"] == "low"
        assert data["crowd"]["count"] == 0

    def test_fallback_noise_is_quiet(self, client):
        data = client.get("/state").json()
        assert data["noise"]["level"] == "quiet"
        assert data["noise"]["db"] == -80.0

    def test_fallback_suggestion_is_full_ui(self, client):
        assert client.get("/state").json()["suggestion"] == "full_ui"

    def test_fallback_has_timestamp(self, client):
        data = client.get("/state").json()
        assert "timestamp" in data
        assert data["timestamp"] == pytest.approx(time.time(), abs=2.0)


# ---------------------------------------------------------------------------
# /state endpoint — live state (populated _latest_state)
# ---------------------------------------------------------------------------

class TestStateEndpointLive:

    SAMPLE_STATE = {
        "timestamp": 1700000000.0,
        "hands":  {"detected": True,  "both_holding": True,  "per_hand": ["holding", "holding"]},
        "crowd":  {"count": 6,        "level": "crowded"},
        "noise":  {"db": -20.0,       "level": "noisy",  "centroid_hz": 2500.0},
        "suggestion": "show_overlay",
    }

    def test_returns_live_state_when_available(self, client):
        server._latest_state.update(self.SAMPLE_STATE)
        data = client.get("/state").json()
        assert data["hands"]["both_holding"] is True
        assert data["crowd"]["level"] == "crowded"
        assert data["noise"]["level"] == "noisy"
        assert data["suggestion"] == "show_overlay"

    def test_live_state_crowd_count(self, client):
        server._latest_state.update(self.SAMPLE_STATE)
        assert client.get("/state").json()["crowd"]["count"] == 6

    def test_live_state_noise_db(self, client):
        server._latest_state.update(self.SAMPLE_STATE)
        assert client.get("/state").json()["noise"]["db"] == pytest.approx(-20.0)

    def test_live_state_per_hand(self, client):
        server._latest_state.update(self.SAMPLE_STATE)
        assert client.get("/state").json()["hands"]["per_hand"] == ["holding", "holding"]

    def test_response_schema_has_all_keys(self, client):
        data = client.get("/state").json()
        assert set(data.keys()) == {"timestamp", "hands", "crowd", "noise", "suggestion"}


# ---------------------------------------------------------------------------
# /ask endpoint — helpers
# ---------------------------------------------------------------------------

QUIET_STATE = {"crowd": "low", "noise": "quiet",
               "detected": True, "both_holding": False}

NOISY_STATE = {"crowd": "crowded", "noise": "noisy",
               "detected": False, "both_holding": False}

HOLDING_STATE = {"crowd": "low", "noise": "quiet",
                 "detected": True, "both_holding": True}


def pipeline_result(
    mode: str = "FULL_VOICE",
    answer: str = "This painting is...",
    audio_bytes: bytes = b"",
    exhibit: str = "",
) -> dict:
    return {"mode": mode, "answer": answer,
            "audio_bytes": audio_bytes, "exhibit": exhibit}


# ---------------------------------------------------------------------------
# /ask endpoint — availability
# ---------------------------------------------------------------------------

class TestAskEndpointAvailability:

    def test_returns_503_when_rag_not_ready(self, client):
        original = server._rag
        server._rag = None
        try:
            resp = client.post("/ask", json={"question": "test", "state": QUIET_STATE})
            assert resp.status_code == 503
        finally:
            server._rag = original

    def test_returns_200_when_rag_ready(self, client):
        with patch("server.qa_pipeline.run", return_value=pipeline_result()):
            resp = client.post("/ask", json={"question": "test", "state": QUIET_STATE})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /ask endpoint — response schema
# ---------------------------------------------------------------------------

class TestAskEndpointSchema:

    def test_response_has_all_keys(self, client):
        with patch("server.qa_pipeline.run", return_value=pipeline_result()):
            data = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()
        assert set(data.keys()) == {"mode", "answer", "audio_url", "exhibit"}

    def test_mode_is_returned(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(mode="BRIEF_TEXT")):
            data = client.post("/ask", json={
                "question": "test", "state": NOISY_STATE
            }).json()
        assert data["mode"] == "BRIEF_TEXT"

    def test_answer_is_returned(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(answer="Van Gogh painted this.")):
            data = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()
        assert data["answer"] == "Van Gogh painted this."

    def test_exhibit_is_returned(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(exhibit="Madame X (Madame Pierre Gautreau)")):
            data = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()
        assert data["exhibit"] == "Madame X (Madame Pierre Gautreau)"


# ---------------------------------------------------------------------------
# /ask endpoint — audio file handling
# ---------------------------------------------------------------------------

class TestAskEndpointAudio:

    def test_no_audio_bytes_gives_empty_audio_url(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(audio_bytes=b"")):
            data = client.post("/ask", json={
                "question": "test", "state": NOISY_STATE
            }).json()
        assert data["audio_url"] == ""

    def test_audio_bytes_gives_audio_url(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(audio_bytes=b"mp3data")), \
             patch("builtins.open", mock_open()):
            data = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()
        assert data["audio_url"].startswith("/audio/")
        assert data["audio_url"].endswith(".mp3")

    def test_audio_url_is_unique_per_request(self, client):
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(audio_bytes=b"mp3data")), \
             patch("builtins.open", mock_open()):
            url1 = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()["audio_url"]
            url2 = client.post("/ask", json={
                "question": "test", "state": QUIET_STATE
            }).json()["audio_url"]
        assert url1 != url2


# ---------------------------------------------------------------------------
# /ask endpoint — three demo scenes
# ---------------------------------------------------------------------------

class TestAskEndpointScenes:

    def test_scene_a_full_voice(self, client):
        """Low crowd + quiet + hands free → FULL_VOICE with audio."""
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(
                       mode="FULL_VOICE", audio_bytes=b"tts_audio"
                   )), \
             patch("builtins.open", mock_open()):
            data = client.post("/ask", json={
                "question": "Tell me about this painting",
                "state": QUIET_STATE,
            }).json()
        assert data["mode"] == "FULL_VOICE"
        assert data["audio_url"] != ""

    def test_scene_b_brief_text(self, client):
        """Crowded + noisy → BRIEF_TEXT, no audio."""
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(
                       mode="BRIEF_TEXT", audio_bytes=b""
                   )):
            data = client.post("/ask", json={
                "question": "Tell me about this painting",
                "state": NOISY_STATE,
            }).json()
        assert data["mode"] == "BRIEF_TEXT"
        assert data["audio_url"] == ""

    def test_scene_c_xr_menu(self, client):
        """Both hands holding → XR_MENU, no answer, no audio."""
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result(
                       mode="XR_MENU", answer="", audio_bytes=b""
                   )):
            data = client.post("/ask", json={
                "question": "Tell me about this painting",
                "state": HOLDING_STATE,
            }).json()
        assert data["mode"] == "XR_MENU"
        assert data["answer"] == ""
        assert data["audio_url"] == ""

    def test_state_defaults_used_when_state_omitted(self, client):
        """Omitting 'state' in request should use default AskStateInput values."""
        with patch("server.qa_pipeline.run",
                   return_value=pipeline_result()) as mock_run:
            client.post("/ask", json={"question": "test"})
            called_state = mock_run.call_args.kwargs["api_state"]
        assert called_state["crowd"] == "low"
        assert called_state["noise"] == "quiet"
        assert called_state["both_holding"] is False
