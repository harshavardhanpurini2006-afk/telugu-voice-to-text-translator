"""
Configuration module for Telugu Voice-to-Text Translator.
Manages provider settings, paths, privacy policies, and default parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
HISTORY_DIR = DATA_DIR / "history"
PLOTS_DIR = DATA_DIR / "plots"

# Accuracy & Limitation Disclaimer
ACCURACY_DISCLAIMER = (
    "NOTICE ON ACCURACY & LIMITATIONS:\n"
    "Telugu speech recognition and translation accuracy depends on audio quality, speaker accents "
    "(e.g., Coastal Andhra, Rayalaseema, Telangana dialects), background noise levels, and code-switching "
    "('Tenglish' - mixing Telugu and English words). False transcriptions or mistranslations may occur. "
    "Always review output for critical use cases."
)

PRIVACY_NOTICE = (
    "PRIVACY & EXTERNAL SERVICE POLICY:\n"
    "- When 'allow_external_services' is True, audio audio features or audio streams may be transmitted "
    "  to external cloud speech recognition APIs (e.g., Google Speech-to-Text) and text translation APIs.\n"
    "- When 'allow_external_services' is False or provider is set to 'mock', all processing remains 100% "
    "  local on your machine, using offline algorithms and local synthetic fixtures.\n"
    "- History logs are stored strictly locally on your filesystem and can be purged at any time."
)


@dataclass
class Config:
    """Application configuration container."""

    # Audio recording & processing defaults
    sample_rate: int = 16000  # 16 kHz is standard for speech STT
    channels: int = 1         # Mono audio
    recording_duration_sec: float = 5.0
    silence_threshold_db: float = -40.0
    max_audio_duration_sec: float = 120.0  # Guard against excessively large audio files

    # Transcription & Translation Providers
    # Options for stt_provider: 'google', 'mock'
    stt_provider: Literal["google", "mock"] = "google"
    # Options for translation_provider: 'google_direct', 'deep_translator', 'mock'
    translation_provider: Literal["google_direct", "deep_translator", "mock"] = "google_direct"

    # Languages
    speech_language: str = "te-IN"       # Telugu (India)
    target_language: str = "en"          # English

    # Privacy & external network switch
    allow_external_services: bool = True  # If False, forces local/mock mode

    # File paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    samples_dir: Path = SAMPLES_DIR
    history_dir: Path = HISTORY_DIR
    plots_dir: Path = PLOTS_DIR
    history_file_csv: Path = HISTORY_DIR / "transcription_history.csv"

    def __post_init__(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    def effective_stt_provider(self) -> str:
        """Return the active STT provider considering privacy flags."""
        if not self.allow_external_services:
            return "mock"
        return self.stt_provider

    @property
    def effective_translation_provider(self) -> str:
        """Return the active translation provider considering privacy flags."""
        if not self.allow_external_services:
            return "mock"
        return self.translation_provider


# Global default configuration instance
DEFAULT_CONFIG = Config()
