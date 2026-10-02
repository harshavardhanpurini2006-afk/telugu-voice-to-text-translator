"""
Unit tests for AudioVisualizer and HistoryVisualizer with Matplotlib/Seaborn.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from telugu_transcriber.visualizer import AudioVisualizer, HistoryVisualizer


@pytest.fixture
def sample_audio():
    # 2 seconds synthetic tone
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 300 * t)


@pytest.fixture
def sample_history_df():
    return pd.DataFrame([
        {
            "id": "1",
            "timestamp": "2026-10-02 10:00:00",
            "audio_source": "microphone",
            "duration_sec": 3.5,
            "telugu_transcript": "నమస్కారం మీరు ఎలా ఉన్నారు",
            "english_translation": "Hello how are you",
            "confidence": 0.94,
            "stt_provider": "google",
            "translation_provider": "deep_translator",
            "has_code_switching": False,
            "warning_notes": "",
        },
        {
            "id": "2",
            "timestamp": "2026-10-02 10:05:00",
            "audio_source": "file (sample.wav)",
            "duration_sec": 4.2,
            "telugu_transcript": "meeting రేపు ఉంది",
            "english_translation": "meeting is tomorrow",
            "confidence": 0.75,
            "stt_provider": "offline_mock",
            "translation_provider": "mock",
            "has_code_switching": True,
            "warning_notes": "Mixed speech",
        },
    ])


def test_plot_waveform_and_rms(sample_audio, tmp_path):
    vis = AudioVisualizer()
    out_file = tmp_path / "waveform_test.png"
    fig = vis.plot_waveform_and_rms(sample_audio, save_path=out_file)

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    plt.close(fig)


def test_plot_spectrogram(sample_audio, tmp_path):
    vis = AudioVisualizer()
    out_file = tmp_path / "spec_test.png"
    fig = vis.plot_spectrogram(sample_audio, save_path=out_file)

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    plt.close(fig)


def test_plot_audio_dashboard(sample_audio, tmp_path):
    vis = AudioVisualizer()
    out_file = tmp_path / "dash_test.png"
    fig = vis.plot_audio_dashboard(sample_audio, save_path=out_file)

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    plt.close(fig)


def test_plot_usage_dashboard(sample_history_df, tmp_path):
    vis = HistoryVisualizer()
    out_file = tmp_path / "hist_test.png"
    fig = vis.plot_usage_dashboard(sample_history_df, save_path=out_file)

    assert isinstance(fig, plt.Figure)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    plt.close(fig)


def test_plot_usage_dashboard_empty():
    vis = HistoryVisualizer()
    fig = vis.plot_usage_dashboard(pd.DataFrame())
    assert fig is None
