"""
Telugu Voice-to-Text Translator Package
=======================================
A modular Python framework for transcribing spoken Telugu into Telugu script,
optionally translating to English, managing history with pandas, and visualizing
audio signals and usage patterns with NumPy, Matplotlib, and Seaborn.
"""

__version__ = "1.0.0"
__author__ = "Telugu Voice-to-Text Project Team"

from telugu_transcriber.config import Config
from telugu_transcriber.audio_processor import AudioProcessor
from telugu_transcriber.transcriber import get_transcriber, TeluguTranscriber
from telugu_transcriber.translator import get_translator, TeluguTranslator
from telugu_transcriber.history_manager import HistoryManager
from telugu_transcriber.visualizer import AudioVisualizer, HistoryVisualizer

__all__ = [
    "Config",
    "AudioProcessor",
    "TeluguTranscriber",
    "get_transcriber",
    "TeluguTranslator",
    "get_translator",
    "HistoryManager",
    "AudioVisualizer",
    "HistoryVisualizer",
]
