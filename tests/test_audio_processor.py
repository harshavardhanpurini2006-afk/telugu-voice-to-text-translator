"""
Unit tests for AudioProcessor and NumPy signal processing.
"""

import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from telugu_transcriber.audio_processor import AudioProcessor, AudioValidationError


@pytest.fixture
def processor():
    return AudioProcessor(sample_rate=16000, max_duration_sec=30.0)


@pytest.fixture
def sample_wav(tmp_path):
    """Create a valid 16-bit PCM mono WAV file using pure numpy and wave."""
    file_path = tmp_path / "test_audio.wav"
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    pcm16 = (audio * 32767).astype(np.int16)

    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    return file_path


def test_validate_file_success(processor, sample_wav):
    validated = processor.validate_file(sample_wav)
    assert validated.exists()
    assert validated.suffix == ".wav"


def test_validate_file_missing(processor):
    with pytest.raises(AudioValidationError, match="not found"):
        processor.validate_file("non_existent_file.wav")


def test_validate_file_invalid_extension(processor, tmp_path):
    bad_file = tmp_path / "song.mp3"
    bad_file.write_text("dummy")
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        processor.validate_file(bad_file)


def test_load_and_save_wav(processor, sample_wav, tmp_path):
    audio, sr = processor.load_wav(sample_wav)
    assert isinstance(audio, np.ndarray)
    assert sr == 16000
    assert len(audio) == 16000
    assert np.max(np.abs(audio)) <= 1.0

    # Save to a new file and reload
    save_path = tmp_path / "saved_copy.wav"
    saved = processor.save_wav(audio, save_path, sample_rate=sr)
    assert saved.exists()

    reloaded_audio, reloaded_sr = processor.load_wav(saved)
    assert reloaded_sr == 16000
    assert len(reloaded_audio) == len(audio)
    np.testing.assert_allclose(audio, reloaded_audio, atol=1e-3)


def test_numpy_normalization():
    sig = np.array([-0.2, 0.5, 0.1, -0.4], dtype=np.float32)
    norm = AudioProcessor.normalize(sig)
    assert np.max(np.abs(norm)) == pytest.approx(1.0)


def test_numpy_rms_energy():
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000))
    times, rms = AudioProcessor.compute_rms_energy(audio, frame_length=512, hop_length=256)
    assert len(times) == len(rms)
    assert len(rms) > 0
    assert np.all(rms >= 0.0)
    # RMS of unit sine wave is approx 1/sqrt(2) ~= 0.707
    assert np.mean(rms) == pytest.approx(0.707, rel=0.05)


def test_numpy_spectrogram():
    audio = np.random.randn(8000).astype(np.float32)
    freqs, times, spec_db = AudioProcessor.compute_spectrogram(audio, sample_rate=16000, n_fft=512)
    assert len(freqs) == 257  # n_fft // 2 + 1
    assert len(times) > 0
    assert spec_db.shape == (len(freqs), len(times))


def test_extract_audio_metrics(processor, sample_wav):
    audio, sr = processor.load_wav(sample_wav)
    metrics = processor.extract_audio_metrics(audio, sr)
    assert metrics["duration_sec"] == pytest.approx(1.0, rel=1e-2)
    assert metrics["samples_count"] == 16000
    assert not metrics["is_silent"]
    assert "snr_estimate_db" in metrics
    assert "zero_crossing_rate" in metrics


def test_generate_synthetic_speech_wave(processor):
    synth_audio = processor.generate_synthetic_speech_wave("నమస్కారం", duration_sec=2.0, sample_rate=16000)
    assert isinstance(synth_audio, np.ndarray)
    assert len(synth_audio) == 32000
    assert np.max(np.abs(synth_audio)) <= 1.0


def test_record_microphone_mocked(processor):
    """Verify microphone recording flow with mocked hardware."""
    mock_audio = np.zeros((16000, 1), dtype=np.float32)
    with patch("sounddevice.query_devices", return_value=[{"name": "Mock Mic"}]), \
         patch("sounddevice.rec", return_value=mock_audio):
        rec = processor.record_microphone(duration_sec=1.0)
        assert len(rec) == 16000
