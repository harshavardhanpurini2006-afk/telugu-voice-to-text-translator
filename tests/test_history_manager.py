"""
Unit tests for HistoryManager and pandas data analytics.
"""

from pathlib import Path
import pandas as pd
import pytest

from telugu_transcriber.history_manager import HistoryManager


@pytest.fixture
def temp_history_csv(tmp_path):
    return tmp_path / "test_history.csv"


@pytest.fixture
def history_mgr(temp_history_csv):
    return HistoryManager(temp_history_csv)


def test_add_and_retrieve_record(history_mgr):
    rec = history_mgr.add_record(
        telugu_transcript="నమస్కారం",
        english_translation="Hello",
        audio_source="microphone",
        duration_sec=3.2,
        confidence=0.95,
        stt_provider="google",
        translation_provider="deep_translator",
    )
    assert rec["id"] != ""
    assert rec["telugu_transcript"] == "నమస్కారం"

    df = history_mgr.get_all()
    assert len(df) == 1
    assert df.iloc[0]["telugu_transcript"] == "నమస్కారం"
    assert df.iloc[0]["english_translation"] == "Hello"


def test_search_history(history_mgr):
    history_mgr.add_record(telugu_transcript="వాతావరణం బాగుంది", english_translation="Weather is good")
    history_mgr.add_record(telugu_transcript="శుభోదయం", english_translation="Good morning")

    res_te = history_mgr.search("వాతావరణం")
    assert len(res_te) == 1
    assert res_te.iloc[0]["english_translation"] == "Weather is good"

    res_en = history_mgr.search("morning")
    assert len(res_en) == 1
    assert res_en.iloc[0]["telugu_transcript"] == "శుభోదయం"


def test_delete_by_id(history_mgr):
    r1 = history_mgr.add_record(telugu_transcript="Phrase 1")
    r2 = history_mgr.add_record(telugu_transcript="Phrase 2")
    assert len(history_mgr.get_all()) == 2

    deleted = history_mgr.delete_by_id(r1["id"])
    assert deleted is True
    assert len(history_mgr.get_all()) == 1
    assert history_mgr.get_all().iloc[0]["id"] == r2["id"]


def test_clear_all_history(history_mgr):
    history_mgr.add_record(telugu_transcript="Test 1")
    history_mgr.add_record(telugu_transcript="Test 2")
    assert len(history_mgr.get_all()) == 2

    count = history_mgr.clear_all_history()
    assert count == 2
    assert len(history_mgr.get_all()) == 0


def test_exports(history_mgr, tmp_path):
    history_mgr.add_record(
        telugu_transcript="నమస్కారం",
        english_translation="Hello",
        duration_sec=2.5,
        confidence=0.92,
    )
    csv_out = tmp_path / "exported.csv"
    txt_out = tmp_path / "exported.txt"

    history_mgr.export_to_csv(csv_out)
    assert csv_out.exists()
    assert "నమస్కారం" in csv_out.read_text(encoding="utf-8")

    history_mgr.export_to_txt(txt_out)
    assert txt_out.exists()
    assert "TELUGU VOICE-TO-TEXT" in txt_out.read_text(encoding="utf-8")
    assert "నమస్కారం" in txt_out.read_text(encoding="utf-8")


def test_summary_statistics(history_mgr):
    history_mgr.add_record(
        telugu_transcript="నమస్కారం ఎలా ఉన్నారు",
        english_translation="Hello how are you",
        duration_sec=4.0,
        confidence=0.90,
        has_code_switching=False,
    )
    history_mgr.add_record(
        telugu_transcript="meeting ఉంది",
        english_translation="meeting is there",
        duration_sec=2.0,
        confidence=0.80,
        has_code_switching=True,
    )

    stats = history_mgr.get_summary_statistics()
    assert stats["total_records"] == 2
    assert stats["total_duration_sec"] == 6.0
    assert stats["avg_confidence"] == pytest.approx(0.85)
    assert stats["code_switching_count"] == 1
    assert stats["code_switching_pct"] == 50.0
