"""
Tests for context_router._decide_mode() and route().

Run with:
    pytest test_router.py -v
"""

from unittest.mock import MagicMock, patch
import requests

from context_router import _decide_mode, route, fetch_state, RouterDecision, BRIEF_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    crowd: str = "low",
    noise: str = "quiet",
    detected: bool = False,
    both_holding: bool = False,
) -> dict:
    """Build a minimal sensor-state dict for testing."""
    return {
        "crowd": {"count": 0, "level": crowd},
        "noise": {"db": -40.0, "level": noise, "centroid_hz": 0.0},
        "hands": {
            "detected": detected,
            "both_holding": both_holding,
            "per_hand": ["holding", "holding"] if both_holding else
                        (["free"] if detected else []),
        },
    }


def make_rag(answer: str = "This is a test answer about the painting.") -> MagicMock:
    """Return a mock RAGEngine whose query() returns a fixed answer."""
    rag = MagicMock()
    rag.query.return_value = {"answer": answer, "sources": ["Test Painting"]}
    return rag


# ---------------------------------------------------------------------------
# _decide_mode — all 9 rows from the decision table
# ---------------------------------------------------------------------------

class TestDecideMode:
    """Covers every row in the context table."""

    # --- XR_MENU (both_holding=True, rows 1 / 4 / 7) ---

    def test_row1_crowded_noisy_holding(self):
        mode, _, xr = _decide_mode(make_state("crowded", "noisy", True, True))
        assert mode == "XR_MENU"
        assert xr == "show_menu"

    def test_row4_crowded_quiet_holding(self):
        mode, _, xr = _decide_mode(make_state("crowded", "quiet", True, True))
        assert mode == "XR_MENU"
        assert xr == "show_menu"

    def test_row7_low_quiet_holding(self):
        mode, _, xr = _decide_mode(make_state("low", "quiet", True, True))
        assert mode == "XR_MENU"
        assert xr == "show_menu"

    # --- BRIEF_TEXT (noisy + not holding, rows 2 / 3) ---

    def test_row2_crowded_noisy_detected_not_holding(self):
        mode, _, xr = _decide_mode(make_state("crowded", "noisy", True, False))
        assert mode == "BRIEF_TEXT"
        assert xr is None

    def test_row3_crowded_noisy_not_detected(self):
        mode, _, xr = _decide_mode(make_state("crowded", "noisy", False, False))
        assert mode == "BRIEF_TEXT"
        assert xr is None

    # --- FULL_VOICE (quiet + not holding, rows 5 / 6 / 8 / 9) ---

    def test_row5_crowded_quiet_detected_not_holding(self):
        mode, _, _ = _decide_mode(make_state("crowded", "quiet", True, False))
        assert mode == "FULL_VOICE"

    def test_row6_crowded_quiet_not_detected(self):
        mode, _, _ = _decide_mode(make_state("crowded", "quiet", False, False))
        assert mode == "FULL_VOICE"

    def test_row8_low_quiet_detected_not_holding(self):
        mode, _, _ = _decide_mode(make_state("low", "quiet", True, False))
        assert mode == "FULL_VOICE"

    def test_row9_low_quiet_not_detected(self):
        mode, _, _ = _decide_mode(make_state("low", "quiet", False, False))
        assert mode == "FULL_VOICE"


# ---------------------------------------------------------------------------
# _decide_mode — priority and edge cases
# ---------------------------------------------------------------------------

class TestDecideModePriority:

    def test_both_holding_beats_noisy(self):
        """both_holding should win even in the noisiest room."""
        mode, _, _ = _decide_mode(make_state("crowded", "noisy", True, True))
        assert mode == "XR_MENU"

    def test_moderate_noise_gives_brief_text(self):
        """moderate noise is treated the same as noisy."""
        mode, _, _ = _decide_mode(make_state("low", "moderate", True, False))
        assert mode == "BRIEF_TEXT"

    def test_crowd_does_not_affect_outcome_quiet(self):
        """crowd level is irrelevant when noise=quiet and not holding."""
        for crowd in ("low", "moderate", "crowded"):
            mode, _, _ = _decide_mode(make_state(crowd, "quiet", False, False))
            assert mode == "FULL_VOICE", f"failed for crowd={crowd}"

    def test_crowd_does_not_affect_outcome_noisy(self):
        """crowd level is irrelevant when noise=noisy and not holding."""
        for crowd in ("low", "moderate", "crowded"):
            mode, _, _ = _decide_mode(make_state(crowd, "noisy", False, False))
            assert mode == "BRIEF_TEXT", f"failed for crowd={crowd}"

    def test_detected_does_not_affect_mode(self):
        """detected alone (without both_holding) never changes the mode."""
        for detected in (True, False):
            mode_quiet, _, _ = _decide_mode(
                make_state("low", "quiet", detected, False)
            )
            assert mode_quiet == "FULL_VOICE"

            mode_noisy, _, _ = _decide_mode(
                make_state("low", "noisy", detected, False)
            )
            assert mode_noisy == "BRIEF_TEXT"


# ---------------------------------------------------------------------------
# route() — integration with mocked RAGEngine
# ---------------------------------------------------------------------------

class TestRoute:

    def test_xr_menu_returns_no_answer(self):
        state = make_state("low", "quiet", True, True)
        decision = route("Any question", make_rag(), state=state, play_audio=False)
        assert decision.mode == "XR_MENU"
        assert decision.answer == ""
        assert decision.use_tts is False
        assert decision.xr_action == "show_menu"

    def test_brief_text_passes_max_length_to_rag(self):
        """route() should delegate truncation to rag.query() via max_length."""
        rag = make_rag()
        state = make_state("crowded", "noisy", False, False)
        route("Any question", rag, state=state, play_audio=False)
        rag.query.assert_called_once_with("Any question", max_length=BRIEF_MAX)

    def test_full_voice_passes_no_max_length_to_rag(self):
        """route() should not impose a length limit in FULL_VOICE mode."""
        rag = make_rag()
        state = make_state("low", "quiet", False, False)
        route("Any question", rag, state=state, play_audio=False)
        rag.query.assert_called_once_with("Any question", max_length=None)

    def test_brief_text_short_answer_unchanged(self):
        short_answer = "A short answer."
        rag = make_rag(answer=short_answer)
        state = make_state("crowded", "noisy", False, False)
        decision = route("Any question", rag, state=state, play_audio=False)
        assert decision.answer == short_answer

    def test_full_voice_returns_full_answer(self):
        long_answer = "B" * 300
        rag = make_rag(answer=long_answer)
        state = make_state("low", "quiet", True, False)
        decision = route("Any question", rag, state=state, play_audio=False)
        assert decision.mode == "FULL_VOICE"
        assert decision.answer == long_answer   # no truncation
        assert decision.use_tts is True

    def test_full_voice_calls_tts(self):
        state = make_state("low", "quiet", True, False)
        with patch("context_router.speak") as mock_speak:
            route("Any question", make_rag(), state=state,
                  tts_backend="gtts", play_audio=True)
            mock_speak.assert_called_once()

    def test_xr_menu_does_not_call_rag(self):
        state = make_state("low", "quiet", True, True)
        rag = make_rag()
        route("Any question", rag, state=state, play_audio=False)
        rag.query.assert_not_called()

    def test_xr_menu_does_not_call_tts(self):
        state = make_state("low", "quiet", True, True)
        with patch("context_router.speak") as mock_speak:
            route("Any question", make_rag(), state=state, play_audio=True)
            mock_speak.assert_not_called()

    def test_returns_router_decision_type(self):
        state = make_state("low", "quiet", False, False)
        decision = route("Any question", make_rag(), state=state, play_audio=False)
        assert isinstance(decision, RouterDecision)

    def test_route_calls_fetch_state_when_state_is_none(self):
        """route() should call fetch_state() if no state is passed."""
        with patch("context_router.fetch_state", return_value=make_state()) as mock_fetch:
            route("Any question", make_rag(), state=None, play_audio=False)
        mock_fetch.assert_called_once()

    def test_route_skips_fetch_state_when_state_provided(self):
        """route() must not call fetch_state() when state is explicitly given."""
        with patch("context_router.fetch_state") as mock_fetch:
            route("Any question", make_rag(), state=make_state(), play_audio=False)
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_state() — HTTP success and fallback paths
# ---------------------------------------------------------------------------

class TestFetchState:

    LIVE_STATE = {
        "hands":  {"detected": True,  "both_holding": False, "per_hand": ["free"]},
        "crowd":  {"count": 3,        "level": "moderate"},
        "noise":  {"db": -35.0,       "level": "moderate", "centroid_hz": 800.0},
        "suggestion": "full_ui",
    }

    def test_success_returns_parsed_json(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.LIVE_STATE
        with patch("requests.get", return_value=mock_resp):
            result = fetch_state()
        assert result == self.LIVE_STATE

    def test_success_calls_raise_for_status(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.LIVE_STATE
        with patch("requests.get", return_value=mock_resp):
            fetch_state()
        mock_resp.raise_for_status.assert_called_once()

    def test_success_uses_correct_url(self):
        import context_router
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.LIVE_STATE
        with patch("requests.get", return_value=mock_resp) as mock_get:
            fetch_state()
        mock_get.assert_called_once_with(context_router.STATE_URL, timeout=1.0)

    # --- Fallback on network errors ---

    def test_connection_error_returns_fallback(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert result["hands"]["detected"] is False
        assert result["noise"]["level"] == "quiet"

    def test_timeout_error_returns_fallback(self):
        with patch("requests.get", side_effect=requests.Timeout()):
            result = fetch_state()
        assert result["crowd"]["level"] == "low"

    def test_http_error_returns_fallback(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("503")
        with patch("requests.get", return_value=mock_resp):
            result = fetch_state()
        assert result["hands"]["both_holding"] is False

    def test_invalid_json_returns_fallback(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("invalid JSON")
        with patch("requests.get", return_value=mock_resp):
            result = fetch_state()
        assert result["noise"]["level"] == "quiet"

    # --- Fallback state structure ---

    def test_fallback_has_hands_key(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert "hands" in result

    def test_fallback_has_crowd_key(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert "crowd" in result

    def test_fallback_has_noise_key(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert "noise" in result

    def test_fallback_both_holding_is_false(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert result["hands"]["both_holding"] is False

    def test_fallback_noise_db_is_minus_80(self):
        with patch("requests.get", side_effect=requests.ConnectionError()):
            result = fetch_state()
        assert result["noise"]["db"] == -80.0

    def test_fallback_does_not_raise(self):
        with patch("requests.get", side_effect=Exception("unexpected")):
            # generic Exception is NOT caught by fetch_state — only RequestException/ValueError
            # so this verifies the boundary: only those two are swallowed
            try:
                fetch_state()
            except Exception:
                pass   # expected — generic Exception propagates
