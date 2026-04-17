"""
Tests for crowd_detector.classify_crowd() and get_crowd_status().

Run with:
    python -m pytest test_crowd_detector.py -v

No camera, no YOLO inference needed — model is mocked.
"""

from unittest.mock import MagicMock
import numpy as np
from crowd_detector import (
    classify_crowd,
    get_crowd_status,
    CROWD_THRESHOLD,
    NEAR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_box(x1=100, y1=200, x2=300, y2=400, conf=0.85) -> MagicMock:
    """Fake a single YOLO detection box."""
    box = MagicMock()
    box.xyxy = [[x1, y1, x2, y2]]
    box.conf = [conf]
    return box


def make_mock_model(num_persons: int) -> MagicMock:
    """
    Fake a YOLO model that always returns `num_persons` detections.
    model(frame, ...)[0].boxes  →  list of mock boxes
    """
    boxes = [make_mock_box() for _ in range(num_persons)]
    result = MagicMock()
    result.boxes = boxes
    model = MagicMock()
    model.return_value = [result]
    return model


def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# classify_crowd — label and color
# ---------------------------------------------------------------------------

class TestClassifyCrowd:

    def test_zero_people_is_low(self):
        label, color = classify_crowd(0)
        assert "LOW" in label
        assert color == (0, 255, 0)

    def test_below_near_threshold_is_low(self):
        label, color = classify_crowd(NEAR_THRESHOLD - 1)
        assert "LOW" in label

    def test_at_near_threshold_is_moderate(self):
        label, color = classify_crowd(NEAR_THRESHOLD)
        assert "MODERATE" in label
        assert color == (0, 165, 255)

    def test_between_thresholds_is_moderate(self):
        mid = (NEAR_THRESHOLD + CROWD_THRESHOLD) // 2
        label, _ = classify_crowd(mid)
        assert "MODERATE" in label

    def test_just_below_crowd_threshold_is_moderate(self):
        label, _ = classify_crowd(CROWD_THRESHOLD - 1)
        assert "MODERATE" in label

    def test_at_crowd_threshold_is_crowded(self):
        label, color = classify_crowd(CROWD_THRESHOLD)
        assert "CROWDED" in label
        assert color == (0, 0, 255)

    def test_well_above_crowd_threshold_is_crowded(self):
        label, _ = classify_crowd(CROWD_THRESHOLD + 10)
        assert "CROWDED" in label

    def test_label_contains_count(self):
        label, _ = classify_crowd(7)
        assert "7" in label


# ---------------------------------------------------------------------------
# get_crowd_status — level and dict structure
# ---------------------------------------------------------------------------

class TestGetCrowdStatus:

    def test_no_persons_returns_low(self):
        status = get_crowd_status(blank_frame(), make_mock_model(0))
        assert status["count"] == 0
        assert status["level"] == "low"

    def test_below_near_threshold_is_low(self):
        status = get_crowd_status(blank_frame(), make_mock_model(NEAR_THRESHOLD - 1))
        assert status["level"] == "low"

    def test_at_near_threshold_is_moderate(self):
        status = get_crowd_status(blank_frame(), make_mock_model(NEAR_THRESHOLD))
        assert status["level"] == "moderate"

    def test_just_below_crowd_threshold_is_moderate(self):
        status = get_crowd_status(blank_frame(), make_mock_model(CROWD_THRESHOLD - 1))
        assert status["level"] == "moderate"

    def test_at_crowd_threshold_is_crowded(self):
        status = get_crowd_status(blank_frame(), make_mock_model(CROWD_THRESHOLD))
        assert status["level"] == "crowded"

    def test_count_matches_number_of_detections(self):
        status = get_crowd_status(blank_frame(), make_mock_model(4))
        assert status["count"] == 4

    def test_returns_correct_keys(self):
        status = get_crowd_status(blank_frame(), make_mock_model(0))
        assert set(status.keys()) == {"count", "level", "boxes"}

    def test_boxes_length_matches_count(self):
        status = get_crowd_status(blank_frame(), make_mock_model(3))
        assert len(status["boxes"]) == status["count"]

    def test_each_box_has_five_values(self):
        """Each box entry should be (x1, y1, x2, y2, conf)."""
        status = get_crowd_status(blank_frame(), make_mock_model(2))
        for box in status["boxes"]:
            assert len(box) == 5

    def test_box_coordinates_are_ints(self):
        status = get_crowd_status(blank_frame(), make_mock_model(1))
        x1, y1, x2, y2, _ = status["boxes"][0]
        assert all(isinstance(v, int) for v in (x1, y1, x2, y2))

    def test_box_confidence_is_float(self):
        status = get_crowd_status(blank_frame(), make_mock_model(1))
        *_, conf = status["boxes"][0]
        assert isinstance(conf, float)

    def test_empty_boxes_when_no_detections(self):
        status = get_crowd_status(blank_frame(), make_mock_model(0))
        assert status["boxes"] == []
