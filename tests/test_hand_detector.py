"""
Tests for hand_detector.is_hand_holding() and are_both_hands_holding().

Run with:
    python -m pytest test_hand_detector.py -v

No camera, no MediaPipe inference needed — landmarks are built manually.

Geometry used:
    Palm center = (0.5, 0.5)
    PIP joints  at (0.5, 0.0)  → distance 0.5 from palm center
    Curled tip  at (0.5, 0.1)  → distance 0.4  (<  0.5 * 1.1 = 0.55) → curled
    Extended tip at (0.5, -0.1) → distance 0.6  (>= 0.55)             → extended

Threshold in is_hand_holding: tip_dist < pip_dist * 1.1
Holding condition: 3 or more fingers curled
"""

from types import SimpleNamespace
from hand_detector import is_hand_holding, are_both_hands_holding, FINGER_TIPS, FINGER_PIPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_landmark(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def make_hand(curled_count: int) -> SimpleNamespace:
    """
    Build a fake hand_landmarks with `curled_count` fingers curled.

    Landmark indices that matter:
        0, 5, 9, 13, 17  — wrist + MCP joints (palm center calculation)
        FINGER_PIPS       — intermediate joints  [6, 10, 14, 18]
        FINGER_TIPS       — fingertips           [8, 12, 16, 20]
    """
    # Start with all 21 landmarks at palm center
    lm = [make_landmark(0.5, 0.5)] * 21
    lm = list(lm)

    # PIP joints: 0.5 above palm center → dist = 0.5
    for pip_idx in FINGER_PIPS:
        lm[pip_idx] = make_landmark(0.5, 0.0)

    # First `curled_count` tips curled; the rest extended
    for i, tip_idx in enumerate(FINGER_TIPS):
        if i < curled_count:
            lm[tip_idx] = make_landmark(0.5, 0.1)   # dist 0.4 → curled
        else:
            lm[tip_idx] = make_landmark(0.5, -0.1)  # dist 0.6 → extended

    return SimpleNamespace(landmark=lm)


# ---------------------------------------------------------------------------
# is_hand_holding — curl threshold tests
# ---------------------------------------------------------------------------

class TestIsHandHolding:

    def test_all_four_fingers_curled_is_holding(self):
        assert is_hand_holding(make_hand(curled_count=4)) is True

    def test_three_fingers_curled_is_holding(self):
        """Exactly at the threshold (>=3) — should still be holding."""
        assert is_hand_holding(make_hand(curled_count=3)) is True

    def test_two_fingers_curled_is_not_holding(self):
        """One below threshold — should be free."""
        assert is_hand_holding(make_hand(curled_count=2)) is False

    def test_one_finger_curled_is_not_holding(self):
        assert is_hand_holding(make_hand(curled_count=1)) is False

    def test_all_fingers_extended_is_not_holding(self):
        """Open palm — no fingers curled."""
        assert is_hand_holding(make_hand(curled_count=0)) is False


# ---------------------------------------------------------------------------
# are_both_hands_holding — multi-hand logic tests
# ---------------------------------------------------------------------------

class TestAreBothHandsHolding:

    def test_both_hands_holding_returns_true(self):
        hands = [make_hand(4), make_hand(4)]
        assert are_both_hands_holding(hands) is True

    def test_one_holding_one_free_returns_false(self):
        hands = [make_hand(4), make_hand(0)]
        assert are_both_hands_holding(hands) is False

    def test_both_hands_free_returns_false(self):
        hands = [make_hand(0), make_hand(0)]
        assert are_both_hands_holding(hands) is False

    def test_only_one_hand_detected_returns_false(self):
        """Requires exactly 2 hands — single hand should never trigger holding."""
        hands = [make_hand(4)]
        assert are_both_hands_holding(hands) is False

    def test_no_hands_detected_returns_false(self):
        assert are_both_hands_holding([]) is False

    def test_three_curled_each_counts_as_holding(self):
        """Minimum holding threshold (3 curled) should count for both-hands check."""
        hands = [make_hand(3), make_hand(3)]
        assert are_both_hands_holding(hands) is True

    def test_two_curled_each_does_not_count(self):
        hands = [make_hand(2), make_hand(2)]
        assert are_both_hands_holding(hands) is False
