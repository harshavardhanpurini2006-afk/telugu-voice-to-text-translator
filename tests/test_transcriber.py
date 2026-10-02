"""
Unit tests for Telugu Speech Transcriber and Code-Switching Detection.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from telugu_transcriber.audio_processor import AudioProcessor
from telugu_transcriber.transcriber import (
    GoogleSpeechTranscriber,
    MockSyntheticTranscriber,
    TeluguTranscriber,
    detect_code_switching,
    generate_accuracy_notes,
    get_transcriber,
)


@pytest.fixture
def sample_audio_path(tmp_path):
    proc = AudioProcessor(sample_rate=16000)
    audio = proc.generate_synthetic_speech_wave("నమస్కారం", duration_sec=2.0)
    target = tmp_path / "sample_te.wav"
    proc.save_wav(audio, target)
    return target


def test_detect_code_switching():
    # Pure Telugu
    assert not detect_code_switching("నమస్కారం ఎలా ఉన్నారు?")
    # Mixed Telugu and English (Tenglish)
    assert detect_code_switching("రేపు meeting ఉంది, please attend అవ్వండి")
    # Pure English
    assert not detect_code_switching("Hello world how are you")


def test_generate_accuracy_notes():
    # Low confidence
    notes_low_conf = generate_accuracy_notes("నమస్కారం", confidence=0.50, snr_db=25.0)
    assert "Low confidence" in notes_low_conf

    # High noise
    notes_noisy = generate_accuracy_notes("నమస్కారం", confidence=0.90, snr_db=5.0)
    assert "background noise" in notes_noisy

    # Code-switching
    notes_cs = generate_accuracy_notes("meeting ఉంది", confidence=0.90, snr_db=20.0)
    assert "Tenglish" in notes_cs


def test_mock_transcriber(sample_audio_path):
    transcriber = MockSyntheticTranscriber()
    result = transcriber.transcribe(sample_audio_path)

    assert result.telugu_text != ""
    assert result.confidence > 0.5
    assert result.provider == "offline_mock"
    assert not result.is_external_call
    assert result.duration_sec > 0.0


def test_google_transcriber_mocked(sample_audio_path):
    """Test Google Speech recognizer with mocked API response."""
    with patch("speech_recognition.Recognizer.recognize_google") as mock_rec:
        mock_rec.return_value = {
            "alternative": [
                {"transcript": "నమస్కారం, మీరు ఎలా ఉన్నారు?", "confidence": 0.95}
            ]
        }
        stt = GoogleSpeechTranscriber()
        result = stt.transcribe(sample_audio_path)

        assert result.telugu_text == "నమస్కారం, మీరు ఎలా ఉన్నారు?"
        assert result.confidence == 0.95
        assert result.provider == "google_speech"
        assert result.is_external_call is True


def test_telugu_transcriber_factory():
    # External allowed
    t1 = get_transcriber(provider="google", allow_external=True)
    assert isinstance(t1._engine, GoogleSpeechTranscriber)

    # Privacy / offline forced
    t2 = get_transcriber(provider="google", allow_external=False)
    assert isinstance(t2._engine, MockSyntheticTranscriber)

    # Explicit mock
    t3 = get_transcriber(provider="mock", allow_external=True)
    assert isinstance(t3._engine, MockSyntheticTranscriber)
