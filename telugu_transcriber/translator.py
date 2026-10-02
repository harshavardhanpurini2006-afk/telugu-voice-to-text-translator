"""
Telugu-to-English Translation Module.
Provides high-resilience, multi-tier translation engines:
1. Direct Google Translate API (fast, reliable, free from scraping bot blocks)
2. DeepTranslator (secondary cloud fallback)
3. Offline Mock & Dictionary Engine (zero-network privacy fallback)
"""

import abc
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False


@dataclass
class TranslationResult:
    """Encapsulates translation results and metadata."""
    original_text: str
    translated_text: str
    source_language: str = "te"
    target_language: str = "en"
    provider: str = "google_direct"
    is_external_call: bool = False
    warning_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "provider": self.provider,
            "is_external_call": self.is_external_call,
            "warning_notes": self.warning_notes or "",
        }


class BaseTranslator(abc.ABC):
    """Abstract interface for text translation."""

    @abc.abstractmethod
    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        pass


class GoogleDirectTranslatorEngine(BaseTranslator):
    """
    Direct HTTPS Google Translate service.
    Fast, reliable, and avoids the scraping bot-detection throttles of web scrapers.
    """

    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        clean_text = text.strip()
        if not clean_text:
            return TranslationResult(
                original_text="",
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                provider="google_direct",
                is_external_call=True,
            )

        encoded_text = urllib.parse.quote(clean_text)
        url = (
            f"https://translate.googleapis.com/translate_a/single?"
            f"client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded_text}"
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6.0) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP Status {response.status}")
            data = json.loads(response.read().decode("utf-8"))
            # Format: data[0] contains array of [translated_part, original_part, ...]
            if not data or not isinstance(data, list) or not data[0]:
                raise RuntimeError("Empty response from translation endpoint")

            translated_parts = []
            for item in data[0]:
                if item and isinstance(item, list) and len(item) > 0 and item[0]:
                    translated_parts.append(str(item[0]))

            translated_text = "".join(translated_parts).strip()
            return TranslationResult(
                original_text=clean_text,
                translated_text=translated_text,
                source_language=source_lang,
                target_language=target_lang,
                provider="google_direct",
                is_external_call=True,
                warning_notes=None,
            )


class DeepTranslatorEngine(BaseTranslator):
    """Secondary external translation engine powered by deep-translator."""

    def __init__(self):
        if not DEEP_TRANSLATOR_AVAILABLE:
            raise RuntimeError("deep-translator is not installed.")

    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        clean_text = text.strip()
        if not clean_text:
            return TranslationResult(
                original_text="",
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                provider="deep_translator",
                is_external_call=True,
            )

        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(clean_text)
        return TranslationResult(
            original_text=clean_text,
            translated_text=translated or "",
            source_language=source_lang,
            target_language=target_lang,
            provider="deep_translator",
            is_external_call=True,
            warning_notes=None,
        )


class MockTranslatorEngine(BaseTranslator):
    """
    Offline Mock Translator for testing and privacy-first local translation.
    Contains built-in Telugu -> English dictionary pairs and word-level fallback.
    """

    EXACT_PHRASES: Dict[str, str] = {
        "నమస్కారం": "Hello / Greetings",
        "నమస్కారం, మీరు ఎలా ఉన్నారు?": "Hello, how are you?",
        "నా పేరు భారత్, నేను తెలుగు మాట్లాడుతున్నాను.": "My name is Bharat, I speak Telugu.",
        "ఈ రోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది.": "The weather is very pleasant today.",
        "కృత్రిమ మేధస్సు మరియు భాషా సాంకేతికత అభివృద్ధి చెందుతోంది.": "Artificial intelligence and language technology are developing.",
        "మీ సహాయానికి చాలా ధన్యవాదాలు.": "Thank you very much for your help.",
        "రేపు ఉదయం మేము కొత్త ప్రాజెక్ట్ ప్రారంభిస్తున్నాము.": "Tomorrow morning we are starting a new project.",
        "తెలుగు భారతదేశంలోని ప్రాచీన మరియు సుందరమైన భాషలలో ఒకటి.": "Telugu is one of the ancient and beautiful languages of India.",
        "దయచేసి మైక్రోఫోన్ సరిగ్గా మాట్లాడండి.": "Please speak into the microphone properly.",
        "శుభోదయం": "Good morning",
        "ధన్యవాదాలు": "Thank you",
        "మీ పేరు ఏమిటి?": "What is your name?",
        "నేను బాగున్నాను": "I am fine",
    }

    WORD_DICT: Dict[str, str] = {
        "నమస్కారం": "hello",
        "మీరు": "you",
        "ఎలా": "how",
        "ఉన్నారు": "are",
        "నా": "my",
        "పేరు": "name",
        "నేను": "I",
        "తెలుగు": "Telugu",
        "మాట్లాడుతున్నాను": "speaking",
        "ఈ": "this",
        "రోజు": "day",
        "వాతావరణం": "weather",
        "చాలా": "very",
        "బాగుంది": "good",
        "ధన్యవాదాలు": "thank you",
        "సహాయం": "help",
        "ప్రాజెక్ట్": "project",
        "భాష": "language",
        "మంచిది": "good",
        "ఉంది": "is there",
        "లేకపోతే": "otherwise",
        "మరియు": "and",
    }

    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult(
                original_text="",
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                provider="offline_mock",
                is_external_call=False,
            )

        # Check exact phrase match first
        if clean in self.EXACT_PHRASES:
            return TranslationResult(
                original_text=clean,
                translated_text=self.EXACT_PHRASES[clean],
                source_language=source_lang,
                target_language=target_lang,
                provider="offline_mock",
                is_external_call=False,
            )

        # Word-by-word fallback lookup
        words = clean.replace(",", "").replace(".", "").replace("?", "").split()
        translated_words = []
        for w in words:
            translated_words.append(self.WORD_DICT.get(w, f"[{w}]"))

        fallback_trans = " ".join(translated_words)
        return TranslationResult(
            original_text=clean,
            translated_text=fallback_trans,
            source_language=source_lang,
            target_language=target_lang,
            provider="offline_mock",
            is_external_call=False,
            warning_notes="Offline translation used (local dictionary approximations).",
        )


class TeluguTranslator:
    """
    Unified, highly resilient dispatcher for Telugu-to-English translation.
    Gracefully tries direct cloud API -> deep-translator -> offline dictionary without crashing.
    """

    def __init__(self, provider_name: str = "google_direct", allow_external: bool = True):
        self.provider_name = provider_name
        self.allow_external = allow_external
        self.mock_engine = MockTranslatorEngine()
        self.direct_engine = GoogleDirectTranslatorEngine()
        self.deep_engine = DeepTranslatorEngine() if DEEP_TRANSLATOR_AVAILABLE else None

    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        clean_text = text.strip()
        if not clean_text:
            return TranslationResult(
                original_text="",
                translated_text="",
                source_language=source_lang,
                target_language=target_lang,
                provider="none",
                is_external_call=False,
            )

        # 1. Offline Mode explicitly selected
        if not self.allow_external or self.provider_name.lower() in ("mock", "offline"):
            return self.mock_engine.translate(clean_text, source_lang=source_lang, target_lang=target_lang)

        # 2. Try Direct Google API Engine (Tier 1)
        try:
            return self.direct_engine.translate(clean_text, source_lang=source_lang, target_lang=target_lang)
        except Exception as e_direct:
            pass

        # 3. Try DeepTranslator Engine (Tier 2)
        if self.deep_engine:
            try:
                return self.deep_engine.translate(clean_text, source_lang=source_lang, target_lang=target_lang)
            except Exception as e_deep:
                pass

        # 4. Graceful Offline Fallback (Tier 3)
        mock_res = self.mock_engine.translate(clean_text, source_lang=source_lang, target_lang=target_lang)
        mock_res.warning_notes = "External translation service busy or rate-limited; fallback translation applied."
        return mock_res


def get_translator(provider: str = "google_direct", allow_external: bool = True) -> TeluguTranslator:
    """Factory helper to obtain configured translator."""
    return TeluguTranslator(provider_name=provider, allow_external=allow_external)
