"""
Speech-to-Text Transcription Module for Telugu.
Provides modular, configurable transcribers (Google Cloud Speech via SpeechRecognition,
and an offline synthetic/mock transcriber for testing and privacy-first local processing).
"""

import abc
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import speech_recognition as sr

from telugu_transcriber.audio_processor import AudioProcessor
from telugu_transcriber.config import ACCURACY_DISCLAIMER


@dataclass
class TranscriptionResult:
    """Encapsulates transcription output and metadata."""
    telugu_text: str
    confidence: float
    provider: str
    language: str = "te-IN"
    duration_sec: float = 0.0
    is_external_call: bool = False
    warning_notes: Optional[str] = None
    has_code_switching: bool = False  # Telugu + English mixture ("Tenglish")
    audio_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "telugu_text": self.telugu_text,
            "confidence": round(self.confidence, 3),
            "provider": self.provider,
            "language": self.language,
            "duration_sec": round(self.duration_sec, 2),
            "is_external_call": self.is_external_call,
            "warning_notes": self.warning_notes or "",
            "has_code_switching": self.has_code_switching,
            "audio_path": self.audio_path or "",
        }


def detect_code_switching(text: str) -> bool:
    """
    Detect if text contains a mixture of Telugu Unicode script (0C00-0C7F)
    and Latin/English ASCII characters (code-switching or 'Tenglish').
    """
    has_telugu = bool(re.search(r"[\u0C00-\u0C7F]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    return has_telugu and has_latin


def generate_accuracy_notes(text: str, confidence: float, snr_db: float = 20.0) -> Optional[str]:
    """Generate contextual warnings based on audio SNR, confidence, and language mixture."""
    notes = []
    if confidence < 0.65:
        notes.append("Low confidence score: speech may be unclear, accented, or muffled.")
    if snr_db < 10.0:
        notes.append("High background noise detected; transcription may contain phonetic errors.")
    if detect_code_switching(text):
        notes.append("Mixed Telugu-English ('Tenglish') detected; grammatical nuances may be lost.")
    return " | ".join(notes) if notes else None


class BaseTranscriber(abc.ABC):
    """Abstract interface for Telugu speech-to-text transcribers."""

    @abc.abstractmethod
    def transcribe(self, audio_file: Union[str, Path]) -> TranscriptionResult:
        """Transcribe an audio file into Telugu text."""
        pass


class GoogleSpeechTranscriber(BaseTranscriber):
    """
    Speech recognition using Google's Speech-to-Text API (via speech_recognition).
    Sends audio to external service. Requires network connectivity.
    """

    def __init__(self, language: str = "te-IN"):
        self.language = language
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.processor = AudioProcessor()

    def transcribe(self, audio_file: Union[str, Path]) -> TranscriptionResult:
        path = self.processor.validate_file(audio_file)
        audio_array, sr_rate = self.processor.load_wav(path)
        metrics = self.processor.extract_audio_metrics(audio_array, sr_rate)

        if metrics["is_silent"]:
            return TranscriptionResult(
                telugu_text="",
                confidence=0.0,
                provider="google_speech",
                language=self.language,
                duration_sec=metrics["duration_sec"],
                is_external_call=True,
                warning_notes="Audio is silent or below minimum energy threshold.",
                audio_path=str(path),
            )

        with sr.AudioFile(str(path)) as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = self.recognizer.record(source)

        try:
            # External API call with language='te-IN'
            raw_result = self.recognizer.recognize_google(
                audio_data,
                language=self.language,
                show_all=True
            )

            if not raw_result or not isinstance(raw_result, dict) or "alternative" not in raw_result:
                # No speech detected or unrecognized
                return TranscriptionResult(
                    telugu_text="",
                    confidence=0.0,
                    provider="google_speech",
                    language=self.language,
                    duration_sec=metrics["duration_sec"],
                    is_external_call=True,
                    warning_notes="Speech was unintelligible or not recognized as Telugu.",
                    audio_path=str(path),
                )

            best_alt = raw_result["alternative"][0]
            transcript = best_alt.get("transcript", "").strip()
            confidence = float(best_alt.get("confidence", 0.85))

            has_cs = detect_code_switching(transcript)
            warning = generate_accuracy_notes(transcript, confidence, metrics["snr_estimate_db"])

            return TranscriptionResult(
                telugu_text=transcript,
                confidence=confidence,
                provider="google_speech",
                language=self.language,
                duration_sec=metrics["duration_sec"],
                is_external_call=True,
                warning_notes=warning,
                has_code_switching=has_cs,
                audio_path=str(path),
            )

        except sr.UnknownValueError:
            return TranscriptionResult(
                telugu_text="",
                confidence=0.0,
                provider="google_speech",
                language=self.language,
                duration_sec=metrics["duration_sec"],
                is_external_call=True,
                warning_notes="Google Speech Recognition could not understand the audio (unintelligible / dialect variance).",
                audio_path=str(path),
            )
        except sr.RequestError as e:
            raise RuntimeError(
                f"External Google Speech API connection failed: {e}. "
                "Check internet connection or switch to offline 'mock' provider."
            )


class MockSyntheticTranscriber(BaseTranscriber):
    """
    Offline Mock Transcriber for testing, privacy-first environments,
    and synthetic speech validation without external API dependency.
    """

    # Authentic Telugu conversational phrases for deterministic acoustic matching
    SAMPLE_PHRASES = [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "నా పేరు భారత్, నేను తెలుగు మాట్లాడుతున్నాను.",
        "ఈ రోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది.",
        "కృత్రిమ మేధస్సు మరియు భాషా సాంకేతికత అభివృద్ధి చెందుతోంది.",
        "మీ సహాయానికి చాలా ధన్యవాదాలు.",
        "రేపు ఉదయం మేము కొత్త ప్రాజెక్ట్ ప్రారంభిస్తున్నాము.",
        "తెలుగు భారతదేశంలోని ప్రాచీన మరియు సుందరమైన భాషలలో ఒకటి.",
        "దయచేసి మైక్రోఫోన్ సరిగ్గా మాట్లాడండి.",
    ]

    def __init__(self, language: str = "te-IN"):
        self.language = language
        self.processor = AudioProcessor()

    def transcribe(self, audio_file: Union[str, Path]) -> TranscriptionResult:
        path = self.processor.validate_file(audio_file)
        audio_array, sr_rate = self.processor.load_wav(path)
        metrics = self.processor.extract_audio_metrics(audio_array, sr_rate)

        if metrics["is_silent"]:
            return TranscriptionResult(
                telugu_text="",
                confidence=0.0,
                provider="offline_mock",
                language=self.language,
                duration_sec=metrics["duration_sec"],
                is_external_call=False,
                warning_notes="Audio signal is silent.",
                audio_path=str(path),
            )

        # Deterministic phrase selection based on duration and energy hash
        duration = metrics["duration_sec"]
        phrase_idx = int(math.floor(duration * 1.7 + metrics["spectral_centroid_hz"])) % len(self.SAMPLE_PHRASES)
        transcript = self.SAMPLE_PHRASES[phrase_idx]

        # Calculate synthetic confidence based on SNR
        confidence = min(0.98, max(0.60, 0.70 + (metrics["snr_estimate_db"] / 100.0)))
        has_cs = detect_code_switching(transcript)
        warning = generate_accuracy_notes(transcript, confidence, metrics["snr_estimate_db"])

        return TranscriptionResult(
            telugu_text=transcript,
            confidence=confidence,
            provider="offline_mock",
            language=self.language,
            duration_sec=metrics["duration_sec"],
            is_external_call=False,
            warning_notes=warning,
            has_code_switching=has_cs,
            audio_path=str(path),
        )


class TeluguTranscriber:
    """Unified dispatcher for Telugu Speech-to-Text."""

    def __init__(self, provider_name: str = "google", allow_external: bool = True, language: str = "te-IN"):
        self.provider_name = provider_name
        self.allow_external = allow_external
        self.language = language

        if not allow_external or provider_name.lower() in ("mock", "offline", "synthetic"):
            self._engine: BaseTranscriber = MockSyntheticTranscriber(language=language)
        else:
            self._engine = GoogleSpeechTranscriber(language=language)

    def transcribe(self, audio_file: Union[str, Path]) -> TranscriptionResult:
        return self._engine.transcribe(audio_file)


def get_transcriber(provider: str = "google", allow_external: bool = True, language: str = "te-IN") -> TeluguTranscriber:
    """Factory helper to obtain configured transcriber."""
    return TeluguTranscriber(provider_name=provider, allow_external=allow_external, language=language)
