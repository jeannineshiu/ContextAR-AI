"""
Tests for context_router._decide_mode() and route().

Run with:
    pytest tests/test_router.py -v
"""

from unittest.mock import MagicMock

from context_router import _decide_mode, route, RouterDecision, BRIEF_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    gaze_duration: float = 0.0,
    crowd: str = "low",
    noise: str = "quiet",
) -> dict:
    return {"gaze_duration": gaze_duration, "crowd": crowd, "noise": noise}


def make_rag(answer: str = "This is a test answer about the painting.") -> MagicMock:
    rag = MagicMock()
    rag.query.return_value = {"answer": answer, "sources": ["Test Painting"]}
    return rag


# ---------------------------------------------------------------------------
# _decide_mode — all rows from the decision table
# ---------------------------------------------------------------------------

class TestDecideMode:

    # --- NO_RESPONSE (gaze < 5s) ---

    def test_gaze_zero_is_no_response(self):
        mode, _, xr = _decide_mode(make_state(gaze_duration=0.0))
        assert mode == "NO_RESPONSE"
        assert xr is None

    def test_gaze_just_below_threshold_is_no_response(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=4.9))
        assert mode == "NO_RESPONSE"

    def test_no_response_regardless_of_crowd_and_noise(self):
        for crowd in ("low", "moderate", "crowded"):
            for noise in ("quiet", "moderate", "noisy"):
                mode, _, _ = _decide_mode(make_state(2.0, crowd, noise))
                assert mode == "NO_RESPONSE", f"failed for crowd={crowd} noise={noise}"

    # --- BRIEF_TEXT (5–15s, low/moderate crowd) ---

    def test_glancing_low_crowd_quiet_is_brief_text(self):
        mode, _, xr = _decide_mode(make_state(gaze_duration=8.0, crowd="low", noise="quiet"))
        assert mode == "BRIEF_TEXT"
        assert xr is None

    def test_glancing_low_crowd_noisy_is_brief_text(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=8.0, crowd="low", noise="noisy"))
        assert mode == "BRIEF_TEXT"

    def test_glancing_moderate_crowd_is_brief_text(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=8.0, crowd="moderate"))
        assert mode == "BRIEF_TEXT"

    # --- GLANCE_CARD (5–15s, crowded) ---

    def test_glancing_crowded_is_glance_card(self):
        mode, _, xr = _decide_mode(make_state(gaze_duration=10.0, crowd="crowded"))
        assert mode == "GLANCE_CARD"
        assert xr == "show_card"

    def test_glancing_crowded_noise_does_not_change_mode(self):
        for noise in ("quiet", "moderate", "noisy"):
            mode, _, _ = _decide_mode(make_state(10.0, "crowded", noise))
            assert mode == "GLANCE_CARD", f"failed for noise={noise}"

    # --- FULL_VOICE (> 15s, low/moderate crowd) ---

    def test_engaged_low_crowd_is_full_voice(self):
        mode, _, xr = _decide_mode(make_state(gaze_duration=20.0, crowd="low"))
        assert mode == "FULL_VOICE"
        assert xr is None

    def test_engaged_moderate_crowd_is_full_voice(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=20.0, crowd="moderate"))
        assert mode == "FULL_VOICE"

    def test_engaged_noise_does_not_affect_full_voice(self):
        for noise in ("quiet", "moderate", "noisy"):
            mode, _, _ = _decide_mode(make_state(20.0, "low", noise))
            assert mode == "FULL_VOICE", f"failed for noise={noise}"

    # --- BRIEF_TEXT_PROMPT (> 15s, crowded) ---

    def test_engaged_crowded_is_brief_text_prompt(self):
        mode, _, xr = _decide_mode(make_state(gaze_duration=20.0, crowd="crowded"))
        assert mode == "BRIEF_TEXT_PROMPT"
        assert xr == "show_quiet_prompt"

    def test_engaged_crowded_noise_does_not_change_mode(self):
        for noise in ("quiet", "moderate", "noisy"):
            mode, _, _ = _decide_mode(make_state(20.0, "crowded", noise))
            assert mode == "BRIEF_TEXT_PROMPT", f"failed for noise={noise}"

    # --- Boundary values ---

    def test_exactly_5s_is_not_no_response(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=5.0))
        assert mode != "NO_RESPONSE"

    def test_exactly_15s_is_full_voice_low_crowd(self):
        # threshold is >=15, so 15.0 enters the engaged range
        mode, _, _ = _decide_mode(make_state(gaze_duration=15.0, crowd="low"))
        assert mode == "FULL_VOICE"

    def test_just_above_15s_is_full_voice_low_crowd(self):
        mode, _, _ = _decide_mode(make_state(gaze_duration=15.1, crowd="low"))
        assert mode == "FULL_VOICE"


# ---------------------------------------------------------------------------
# route() — integration with mocked RAGEngine
# ---------------------------------------------------------------------------

class TestRoute:

    def test_no_response_returns_empty_answer(self):
        decision = route("Any question", make_rag(), state=make_state(2.0))
        assert decision.mode == "NO_RESPONSE"
        assert decision.answer == ""

    def test_no_response_does_not_call_rag(self):
        rag = make_rag()
        route("Any question", rag, state=make_state(2.0))
        rag.query.assert_not_called()

    def test_brief_text_passes_max_length_to_rag(self):
        rag = make_rag()
        route("Any question", rag, state=make_state(8.0, "low", "quiet"))
        rag.query.assert_called_once_with("Any question", max_length=BRIEF_MAX)

    def test_glance_card_passes_max_length_to_rag(self):
        rag = make_rag()
        route("Any question", rag, state=make_state(10.0, "crowded"))
        rag.query.assert_called_once_with("Any question", max_length=BRIEF_MAX)

    def test_full_voice_passes_no_max_length(self):
        rag = make_rag()
        route("Any question", rag, state=make_state(20.0, "low"))
        rag.query.assert_called_once_with("Any question", max_length=None)

    def test_brief_text_prompt_passes_max_length(self):
        rag = make_rag()
        route("Any question", rag, state=make_state(20.0, "crowded"))
        rag.query.assert_called_once_with("Any question", max_length=BRIEF_MAX)

    def test_answer_is_passed_through(self):
        rag = make_rag(answer="Specific answer.")
        decision = route("Any question", rag, state=make_state(8.0))
        assert decision.answer == "Specific answer."

    def test_returns_router_decision_type(self):
        decision = route("Any question", make_rag(), state=make_state(8.0))
        assert isinstance(decision, RouterDecision)

    def test_full_voice_returns_full_answer_without_truncation(self):
        long_answer = "B" * 300
        rag = make_rag(answer=long_answer)
        decision = route("Any question", rag, state=make_state(20.0, "low"))
        assert decision.mode == "FULL_VOICE"
        assert decision.answer == long_answer
