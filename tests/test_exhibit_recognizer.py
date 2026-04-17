"""
Tests for exhibit_recognizer.py: _encode_image(), _encode_file(), and recognize_exhibit().

Run with:
    python -m pytest test_exhibit_recognizer.py -v

All OpenAI API calls are mocked — no network access or API key needed.
"""

import base64
import json
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, mock_open
import pytest

from exhibit_recognizer import _encode_image, _encode_file, recognize_exhibit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(h: int = 10, w: int = 10) -> np.ndarray:
    """Return a small black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_api_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like a ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


VALID_JSON = json.dumps({
    "name": "The Card Players",
    "type": "painting",
    "period": "Post-Impressionism, c. 1890–95",
    "brief": "Two peasants playing cards at a simple table.",
    "confidence": "high",
})

MARKDOWN_JSON = f"```json\n{VALID_JSON}\n```"
MARKDOWN_NO_LANG = f"```\n{VALID_JSON}\n```"


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------

class TestEncodeImage:

    def test_returns_string(self):
        result = _encode_image(make_frame())
        assert isinstance(result, str)

    def test_result_is_valid_base64(self):
        result = _encode_image(make_frame())
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_decoded_bytes_are_valid_jpeg(self):
        result = _encode_image(make_frame())
        buf = np.frombuffer(base64.b64decode(result), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert img is not None

    def test_shape_preserved_after_roundtrip(self):
        frame = make_frame(20, 30)
        result = _encode_image(frame)
        buf = np.frombuffer(base64.b64decode(result), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert img.shape == (20, 30, 3)


# ---------------------------------------------------------------------------
# _encode_file
# ---------------------------------------------------------------------------

class TestEncodeFile:

    def test_returns_string(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(b"fake_jpeg_bytes")
        assert isinstance(_encode_file(str(f)), str)

    def test_result_is_valid_base64(self, tmp_path):
        raw = b"some binary content"
        f = tmp_path / "img.jpg"
        f.write_bytes(raw)
        result = _encode_file(str(f))
        assert base64.b64decode(result) == raw

    def test_encodes_actual_jpeg(self, tmp_path):
        frame = make_frame()
        _, buf = cv2.imencode(".jpg", frame)
        f = tmp_path / "test.jpg"
        f.write_bytes(buf.tobytes())
        result = _encode_file(str(f))
        decoded = base64.b64decode(result)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None


# ---------------------------------------------------------------------------
# recognize_exhibit — input routing
# ---------------------------------------------------------------------------

class TestRecognizeExhibitRouting:

    def test_numpy_array_calls_encode_image_not_encode_file(self):
        with patch("exhibit_recognizer._encode_image", return_value="b64") as mock_enc_img, \
             patch("exhibit_recognizer._encode_file") as mock_enc_file, \
             patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame())
        mock_enc_img.assert_called_once()
        mock_enc_file.assert_not_called()

    def test_string_path_calls_encode_file_not_encode_image(self):
        with patch("exhibit_recognizer._encode_image") as mock_enc_img, \
             patch("exhibit_recognizer._encode_file", return_value="b64") as mock_enc_file, \
             patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit("/some/path/painting.jpg")
        mock_enc_file.assert_called_once_with("/some/path/painting.jpg")
        mock_enc_img.assert_not_called()


# ---------------------------------------------------------------------------
# recognize_exhibit — successful JSON parsing
# ---------------------------------------------------------------------------

class TestRecognizeExhibitSuccess:

    def _call(self, content: str) -> dict:
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(content)
            return recognize_exhibit(make_frame())

    def test_clean_json_returns_name(self):
        result = self._call(VALID_JSON)
        assert result["name"] == "The Card Players"

    def test_clean_json_returns_confidence(self):
        result = self._call(VALID_JSON)
        assert result["confidence"] == "high"

    def test_clean_json_returns_all_keys(self):
        result = self._call(VALID_JSON)
        assert set(result.keys()) == {"name", "type", "period", "brief", "confidence"}

    def test_clean_json_returns_correct_period(self):
        result = self._call(VALID_JSON)
        assert "Post-Impressionism" in result["period"]


# ---------------------------------------------------------------------------
# recognize_exhibit — markdown fence stripping (known fixed bug)
# ---------------------------------------------------------------------------

class TestRecognizeExhibitMarkdownStripping:

    def _call(self, content: str) -> dict:
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(content)
            return recognize_exhibit(make_frame())

    def test_markdown_fenced_json_is_parsed(self):
        """GPT-4o sometimes wraps response in ```json ... ``` — must be stripped."""
        result = self._call(MARKDOWN_JSON)
        assert result["name"] == "The Card Players"

    def test_markdown_fence_without_lang_is_parsed(self):
        """Plain ``` fences without 'json' label should also be stripped."""
        result = self._call(MARKDOWN_NO_LANG)
        assert result["name"] == "The Card Players"

    def test_markdown_stripped_response_has_all_keys(self):
        result = self._call(MARKDOWN_JSON)
        assert "confidence" in result
        assert "error" not in result

    def test_markdown_stripped_confidence_correct(self):
        result = self._call(MARKDOWN_JSON)
        assert result["confidence"] == "high"


# ---------------------------------------------------------------------------
# recognize_exhibit — error handling
# ---------------------------------------------------------------------------

class TestRecognizeExhibitErrors:

    def test_invalid_json_returns_error_key(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response("not valid json at all")
            result = recognize_exhibit(make_frame())
        assert "error" in result
        assert result["error"] == "invalid_response"

    def test_invalid_json_includes_raw_field(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response("not valid json at all")
            result = recognize_exhibit(make_frame())
        assert "raw" in result

    def test_api_exception_returns_error_key(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.side_effect = \
                Exception("connection timeout")
            result = recognize_exhibit(make_frame())
        assert "error" in result
        assert "connection timeout" in result["error"]

    def test_api_exception_does_not_raise(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.side_effect = \
                RuntimeError("network error")
            result = recognize_exhibit(make_frame())
        assert isinstance(result, dict)

    def test_nearly_valid_json_with_trailing_text_returns_error(self):
        malformed = VALID_JSON + " extra text"
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(malformed)
            result = recognize_exhibit(make_frame())
        assert "error" in result


# ---------------------------------------------------------------------------
# recognize_exhibit — API call structure
# ---------------------------------------------------------------------------

class TestRecognizeExhibitApiCall:

    def test_calls_gpt4o_model(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame())
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_uses_zero_temperature(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame())
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0

    def test_default_detail_is_low(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame())
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_block = next(b for b in user_content if b["type"] == "image_url")
        assert image_block["image_url"]["detail"] == "low"

    def test_high_detail_is_forwarded(self):
        with patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame(), detail="high")
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_block = next(b for b in user_content if b["type"] == "image_url")
        assert image_block["image_url"]["detail"] == "high"

    def test_b64_image_embedded_in_data_url(self):
        with patch("exhibit_recognizer._encode_image", return_value="TESTB64"), \
             patch("exhibit_recognizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = \
                make_api_response(VALID_JSON)
            recognize_exhibit(make_frame())
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        image_block = next(b for b in user_content if b["type"] == "image_url")
        assert "TESTB64" in image_block["image_url"]["url"]
