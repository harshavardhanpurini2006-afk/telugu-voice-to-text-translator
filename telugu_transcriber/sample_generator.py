"""
Sample Generator Module for Telugu Voice-to-Text Translator.
Generates synthetic acoustic WAV files and fictional session history data
for testing and demonstration purposes without requiring private recordings.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from telugu_transcriber.audio_processor import AudioProcessor
from telugu_transcriber.config import Config, DEFAULT_CONFIG
from telugu_transcriber.history_manager import HistoryManager

# Standard synthetic test cases
SYNTHETIC_SAMPLES_CATALOG: List[Tuple[str, str, str, float]] = [
    (
        "sample_greeting.wav",
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "Hello, how are you?",
        3.0,
    ),
    (
        "sample_weather.wav",
        "ఈ రోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది.",
        "The weather is very pleasant today.",
        3.5,
    ),
    (
        "sample_introduction.wav",
        "నా పేరు భారత్, నేను తెలుగు మాట్లాడుతున్నాను.",
        "My name is Bharat, I speak Telugu.",
        4.0,
    ),
    (
        "sample_tenglish_code_switch.wav",
        "రేపు ఉదయం ప్రాజెక్ట్ meeting ఉంది, please attend అవ్వండి.",
        "Tomorrow morning project meeting is there, please attend.",
        4.5,
    ),
    (
        "sample_gratitude.wav",
        "మీ సహాయానికి చాలా ధన్యవాదాలు.",
        "Thank you very much for your help.",
        2.5,
    ),
]


def generate_all_samples(config: Config = DEFAULT_CONFIG) -> List[Path]:
    """
    Generate synthetic WAV files in data/samples/ using NumPy formant synthesis.
    """
    processor = AudioProcessor(sample_rate=config.sample_rate)
    generated_paths = []

    for filename, telugu_text, _, duration in SYNTHETIC_SAMPLES_CATALOG:
        target_path = config.samples_dir / filename
        audio_array = processor.generate_synthetic_speech_wave(
            text=telugu_text,
            duration_sec=duration,
            sample_rate=config.sample_rate,
        )
        saved_file = processor.save_wav(audio_array, target_path, sample_rate=config.sample_rate)
        generated_paths.append(saved_file)

    return generated_paths


def seed_synthetic_history(history_mgr: HistoryManager) -> int:
    """
    Populate history manager with synthetic/fictional records if currently empty.
    Returns number of records added.
    """
    if len(history_mgr.get_all()) > 0:
        return 0

    records = [
        {
            "telugu": "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "english": "Hello, how are you?",
            "source": "microphone",
            "duration": 3.0,
            "conf": 0.94,
            "cs": False,
            "notes": "",
        },
        {
            "telugu": "ఈ రోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది.",
            "english": "The weather is very pleasant today.",
            "source": "file_upload (sample_weather.wav)",
            "duration": 3.5,
            "conf": 0.91,
            "cs": False,
            "notes": "",
        },
        {
            "telugu": "నా పేరు భారత్, నేను తెలుగు మాట్లాడుతున్నాను.",
            "english": "My name is Bharat, I speak Telugu.",
            "source": "microphone",
            "duration": 4.0,
            "conf": 0.88,
            "cs": False,
            "notes": "",
        },
        {
            "telugu": "రేపు ఉదయం ప్రాజెక్ట్ meeting ఉంది, please attend అవ్వండి.",
            "english": "Tomorrow morning project meeting is there, please attend.",
            "source": "file_upload (sample_tenglish.wav)",
            "duration": 4.5,
            "conf": 0.78,
            "cs": True,
            "notes": "Mixed Telugu-English ('Tenglish') detected; grammatical nuances may be lost.",
        },
        {
            "telugu": "మీ సహాయానికి చాలా ధన్యవాదాలు.",
            "english": "Thank you very much for your help.",
            "source": "microphone",
            "duration": 2.5,
            "conf": 0.96,
            "cs": False,
            "notes": "",
        },
        {
            "telugu": "కృత్రిమ మేధస్సు మరియు భాషా సాంకేతికత అభివృద్ధి చెందుతోంది.",
            "english": "Artificial intelligence and language technology are developing.",
            "source": "microphone",
            "duration": 5.2,
            "conf": 0.89,
            "cs": False,
            "notes": "",
        },
    ]

    for rec in records:
        history_mgr.add_record(
            telugu_transcript=rec["telugu"],
            english_translation=rec["english"],
            audio_source=rec["source"],
            duration_sec=rec["duration"],
            confidence=rec["conf"],
            stt_provider="google" if rec["conf"] > 0.85 else "offline_mock",
            translation_provider="deep_translator",
            has_code_switching=rec["cs"],
            warning_notes=rec["notes"],
        )

    return len(records)
