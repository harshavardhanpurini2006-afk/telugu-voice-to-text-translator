"""
Unit tests for CLI commands and argument parsing.
"""

from unittest.mock import patch
import pytest

from telugu_transcriber.audio_processor import AudioProcessor
from telugu_transcriber.cli import main
from telugu_transcriber.config import Config


@pytest.fixture
def sample_wav(tmp_path):
    proc = AudioProcessor(sample_rate=16000)
    audio = proc.generate_synthetic_speech_wave("నమస్కారం", duration_sec=1.5)
    wav_file = tmp_path / "test_sample.wav"
    proc.save_wav(audio, wav_file)
    return wav_file


def test_cli_demo(tmp_path):
    custom_cfg = Config(data_dir=tmp_path / "data", samples_dir=tmp_path / "samples", history_dir=tmp_path / "history", plots_dir=tmp_path / "plots")
    with patch("sys.argv", ["cli.py", "demo"]), patch("telugu_transcriber.cli.DEFAULT_CONFIG", custom_cfg):
        # Should execute without errors
        main()
        assert len(list(custom_cfg.samples_dir.glob("*.wav"))) >= 1


def test_cli_transcribe_mock(sample_wav, tmp_path):
    custom_cfg = Config(data_dir=tmp_path / "data", history_dir=tmp_path / "history", plots_dir=tmp_path / "plots")
    with patch("sys.argv", ["cli.py", "transcribe", str(sample_wav), "--mock"]), patch("telugu_transcriber.cli.DEFAULT_CONFIG", custom_cfg):
        main()
        assert custom_cfg.history_file_csv.exists()


def test_cli_history_export(tmp_path):
    custom_cfg = Config(data_dir=tmp_path / "data", history_dir=tmp_path / "history", plots_dir=tmp_path / "plots")
    csv_out = tmp_path / "test_exp.csv"
    with patch("sys.argv", ["cli.py", "history", "--export-csv", str(csv_out)]), patch("telugu_transcriber.cli.DEFAULT_CONFIG", custom_cfg):
        main()
        assert csv_out.exists()


def test_cli_privacy():
    with patch("sys.argv", ["cli.py", "privacy"]):
        main()
