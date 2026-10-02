"""
Unit tests for Telugu-to-English Translator.
"""

import io
import json
from unittest.mock import MagicMock, patch
import pytest

from telugu_transcriber.translator import (
    DeepTranslatorEngine,
    GoogleDirectTranslatorEngine,
    MockTranslatorEngine,
    TeluguTranslator,
    get_translator,
)


def test_mock_translator_exact_phrase():
    translator = MockTranslatorEngine()
    res = translator.translate("నమస్కారం, మీరు ఎలా ఉన్నారు?")
    assert res.translated_text == "Hello, how are you?"
    assert not res.is_external_call
    assert res.provider == "offline_mock"


def test_mock_translator_word_fallback():
    translator = MockTranslatorEngine()
    res = translator.translate("ఈ రోజు వాతావరణం బాగుంది")
    assert "day" in res.translated_text or "weather" in res.translated_text
    assert not res.is_external_call


def test_mock_translator_empty():
    translator = MockTranslatorEngine()
    res = translator.translate("   ")
    assert res.translated_text == ""


def test_google_direct_translator_mocked():
    fake_response = [[["Good morning", "శుభోదయం", None, None]]]
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        engine = GoogleDirectTranslatorEngine()
        res = engine.translate("శుభోదయం")
        assert res.translated_text == "Good morning"
        assert res.is_external_call is True
        assert res.provider == "google_direct"


def test_deep_translator_mocked():
    with patch("deep_translator.GoogleTranslator.translate", return_value="Good morning"):
        engine = DeepTranslatorEngine()
        res = engine.translate("శుభోదయం")
        assert res.translated_text == "Good morning"
        assert res.is_external_call is True
        assert res.provider == "deep_translator"


def test_translator_resilient_fallback():
    # When external fails, falls back gracefully to mock without throwing exception
    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        translator = TeluguTranslator(provider_name="google_direct", allow_external=True)
        res = translator.translate("నమస్కారం")
        assert "Hello" in res.translated_text
        assert res.warning_notes is not None


def test_translator_factory():
    t1 = get_translator(provider="google_direct", allow_external=True)
    assert t1.allow_external is True

    t2 = get_translator(provider="google_direct", allow_external=False)
    assert t2.allow_external is False
