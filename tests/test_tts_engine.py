"""
Tests for tts_engine.py: speak(), save(), _play_bytes(), and backend routing.

Run with:
    python -m pytest test_tts_engine.py -v

No audio is played and no external APIs are called — all I/O is mocked.
"""

import os
from unittest.mock import patch, MagicMock, call
import pytest

from tts_engine import speak, save, _play_bytes, _play_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_gtts(write_bytes: bytes = b"fake_mp3_audio"):
    """
    Return a patched gTTS class whose write_to_fp() writes known bytes.
    Usage: with patch("gtts.gTTS", fake_gtts()): ...
    """
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.write_to_fp.side_effect = lambda buf: buf.write(write_bytes)
    mock_cls.return_value = mock_instance
    return mock_cls


# ---------------------------------------------------------------------------
# speak() — empty / whitespace input
# ---------------------------------------------------------------------------

class TestSpeakEmptyInput:

    def test_empty_string_returns_empty_bytes(self):
        assert speak("") == b""

    def test_whitespace_only_returns_empty_bytes(self):
        assert speak("   ") == b""

    def test_empty_input_does_not_call_gtts(self):
        with patch("gtts.gTTS") as mock_cls:
            speak("")
            mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# speak() — gtts backend
# ---------------------------------------------------------------------------

class TestSpeakGtts:

    def test_returns_audio_bytes(self):
        with patch("gtts.gTTS", fake_gtts(b"audio_data")):
            result = speak("Hello museum", backend="gtts", play=False)
        assert result == b"audio_data"

    def test_returns_bytes_type(self):
        with patch("gtts.gTTS", fake_gtts()):
            result = speak("Hello", backend="gtts", play=False)
        assert isinstance(result, bytes)

    def test_text_is_stripped_before_synthesis(self):
        """Leading/trailing whitespace should be stripped before passing to gTTS."""
        with patch("gtts.gTTS", fake_gtts()) as mock_cls:
            speak("  Hello  ", backend="gtts", play=False)
            mock_cls.assert_called_once_with(text="Hello", lang="en", slow=False)

    def test_play_false_does_not_call_afplay(self):
        with patch("gtts.gTTS", fake_gtts()), \
             patch("subprocess.run") as mock_run:
            speak("Hello", backend="gtts", play=False)
            mock_run.assert_not_called()

    def test_play_true_calls_afplay(self):
        with patch("gtts.gTTS", fake_gtts()), \
             patch("subprocess.run") as mock_run, \
             patch("os.unlink"):
            speak("Hello", backend="gtts", play=True)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "afplay"

    def test_unknown_backend_defaults_to_gtts(self):
        """Any unrecognised backend string should fall back to gtts."""
        with patch("gtts.gTTS", fake_gtts(b"gtts_bytes")):
            result = speak("Hello", backend="unknown_backend", play=False)
        assert result == b"gtts_bytes"


# ---------------------------------------------------------------------------
# speak() — elevenlabs backend
# ---------------------------------------------------------------------------

class TestSpeakElevenLabs:

    def test_raises_if_no_api_key(self):
        with patch("tts_engine.ELEVENLABS_API_KEY", ""):
            with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
                speak("Hello", backend="elevenlabs", play=False)

    def test_returns_audio_bytes_with_api_key(self):
        fake_audio = [b"chunk1", b"chunk2"]
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter(fake_audio)

        with patch("tts_engine.ELEVENLABS_API_KEY", "fake_key"), \
             patch("elevenlabs.ElevenLabs", return_value=mock_client):
            result = speak("Hello", backend="elevenlabs", play=False)

        assert result == b"chunk1chunk2"

    def test_elevenlabs_play_false_does_not_call_afplay(self):
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = iter([b"audio"])

        with patch("tts_engine.ELEVENLABS_API_KEY", "fake_key"), \
             patch("elevenlabs.ElevenLabs", return_value=mock_client), \
             patch("subprocess.run") as mock_run:
            speak("Hello", backend="elevenlabs", play=False)
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:

    def test_creates_file_with_audio_bytes(self, tmp_path):
        out = tmp_path / "output.mp3"
        with patch("gtts.gTTS", fake_gtts(b"saved_audio")):
            save("Hello museum", str(out), backend="gtts")
        assert out.read_bytes() == b"saved_audio"

    def test_save_does_not_play_audio(self, tmp_path):
        out = tmp_path / "output.mp3"
        with patch("gtts.gTTS", fake_gtts()), \
             patch("subprocess.run") as mock_run:
            save("Hello", str(out), backend="gtts")
            mock_run.assert_not_called()

    def test_file_is_written_even_for_long_text(self, tmp_path):
        out = tmp_path / "long.mp3"
        long_text = "This is a very long museum guide text. " * 20
        with patch("gtts.gTTS", fake_gtts(b"long_audio")):
            save(long_text, str(out), backend="gtts")
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# _play_bytes() — subprocess wiring
# ---------------------------------------------------------------------------

class TestPlayBytes:

    def test_calls_afplay_with_temp_file(self):
        with patch("subprocess.run") as mock_run, \
             patch("os.unlink"):
            _play_bytes(b"audio_data", fmt="mp3")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "afplay"
            assert cmd[1].endswith(".mp3")

    def test_temp_file_is_deleted_after_playback(self):
        deleted = []
        with patch("subprocess.run"), \
             patch("os.unlink", side_effect=lambda p: deleted.append(p)):
            _play_bytes(b"audio_data", fmt="mp3")
        assert len(deleted) == 1
        assert deleted[0].endswith(".mp3")

    def test_temp_file_deleted_even_if_afplay_fails(self):
        """os.unlink must run in the finally block even when afplay errors."""
        deleted = []
        with patch("subprocess.run", side_effect=Exception("afplay failed")), \
             patch("os.unlink", side_effect=lambda p: deleted.append(p)):
            with pytest.raises(Exception, match="afplay failed"):
                _play_bytes(b"audio_data")
        assert len(deleted) == 1


# ---------------------------------------------------------------------------
# _play_file()
# ---------------------------------------------------------------------------

class TestPlayFile:

    def test_calls_afplay_with_given_path(self):
        with patch("subprocess.run") as mock_run:
            _play_file("/tmp/test.mp3")
            mock_run.assert_called_once_with(["afplay", "/tmp/test.mp3"], check=True)
