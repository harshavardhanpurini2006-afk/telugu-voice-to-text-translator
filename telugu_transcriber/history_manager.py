"""
History Manager Module using pandas.
Handles persistent transcription and translation session history,
filtering, analytics aggregations, CSV/TXT exports, and privacy-compliant deletions.
"""

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import pandas as pd


class HistoryManager:
    """Manages transcription history using pandas DataFrames with local CSV/TXT storage."""

    COLUMNS = [
        "id",
        "timestamp",
        "audio_source",
        "duration_sec",
        "telugu_transcript",
        "english_translation",
        "confidence",
        "stt_provider",
        "translation_provider",
        "has_code_switching",
        "warning_notes",
    ]

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        if storage_path is None:
            self.storage_path = Path(__file__).resolve().parent.parent / "data" / "history" / "transcription_history.csv"
        else:
            self.storage_path = Path(storage_path).resolve()

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame = self._load()

    def _load(self) -> pd.DataFrame:
        """Load history from CSV or return an empty structured DataFrame."""
        if self.storage_path.exists() and self.storage_path.stat().st_size > 0:
            try:
                df = pd.read_csv(self.storage_path, dtype={"id": str})
                # Ensure all columns exist
                for col in self.COLUMNS:
                    if col not in df.columns:
                        df[col] = "" if col not in ["duration_sec", "confidence", "has_code_switching"] else 0
                df["warning_notes"] = df["warning_notes"].fillna("").astype(str)
                df["english_translation"] = df["english_translation"].fillna("").astype(str)
                df["telugu_transcript"] = df["telugu_transcript"].fillna("").astype(str)
                return df[self.COLUMNS]
            except Exception:
                pass
        return pd.DataFrame(columns=self.COLUMNS)

    def save(self) -> None:
        """Persist current DataFrame to local CSV."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.storage_path, index=False, encoding="utf-8")

    def add_record(
        self,
        telugu_transcript: str,
        english_translation: str = "",
        audio_source: str = "microphone",
        duration_sec: float = 0.0,
        confidence: float = 1.0,
        stt_provider: str = "google",
        translation_provider: str = "deep_translator",
        has_code_switching: bool = False,
        warning_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Add a new transcription entry to history using pandas and save to CSV.
        """
        record_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_row = {
            "id": record_id,
            "timestamp": timestamp_str,
            "audio_source": str(audio_source),
            "duration_sec": round(float(duration_sec), 2),
            "telugu_transcript": str(telugu_transcript).strip(),
            "english_translation": str(english_translation).strip(),
            "confidence": round(float(confidence), 3),
            "stt_provider": str(stt_provider),
            "translation_provider": str(translation_provider),
            "has_code_switching": bool(has_code_switching),
            "warning_notes": str(warning_notes).strip(),
        }

        new_df = pd.DataFrame([new_row])
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self.save()
        return new_row

    def get_all(self) -> pd.DataFrame:
        """Return the complete history as a pandas DataFrame."""
        return self._df.copy()

    def get_recent(self, n: int = 10) -> pd.DataFrame:
        """Return the most recent N records."""
        return self._df.tail(n).copy()

    def search(self, query: str) -> pd.DataFrame:
        """Search across Telugu transcripts and English translations."""
        if self._df.empty or not query:
            return self._df.copy()
        q = query.lower()
        mask = (
            self._df["telugu_transcript"].str.lower().str.contains(q, na=False)
            | self._df["english_translation"].str.lower().str.contains(q, na=False)
        )
        return self._df[mask].copy()

    def delete_by_id(self, record_id: str) -> bool:
        """Delete a single record by its unique ID."""
        initial_len = len(self._df)
        self._df = self._df[self._df["id"] != str(record_id)].reset_index(drop=True)
        if len(self._df) < initial_len:
            self.save()
            return True
        return False

    def clear_all_history(self) -> int:
        """
        Delete all recorded history for user privacy.
        Returns the number of deleted records.
        """
        count = len(self._df)
        self._df = pd.DataFrame(columns=self.COLUMNS)
        self.save()
        return count

    # -------------------------------------------------------------------------
    # Export Functions (CSV / TXT)
    # -------------------------------------------------------------------------
    def export_to_csv(self, output_path: Union[str, Path]) -> Path:
        """Export history records to a custom CSV file."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(target, index=False, encoding="utf-8")
        return target

    def export_to_txt(self, output_path: Union[str, Path]) -> Path:
        """Export a clean, human-readable text report of transcription history."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 70,
            "           TELUGU VOICE-TO-TEXT TRANSLATION HISTORY REPORT",
            f"           Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"           Total Records: {len(self._df)}",
            "=" * 70,
            "",
        ]

        if self._df.empty:
            lines.append("No transcription records found.")
        else:
            for idx, row in self._df.iterrows():
                lines.extend([
                    f"Record ID      : {row['id']}",
                    f"Timestamp      : {row['timestamp']}",
                    f"Source         : {row['audio_source']} (Duration: {row['duration_sec']}s)",
                    f"Confidence     : {row['confidence']:.1%} | Code-Switching: {row['has_code_switching']}",
                    f"Providers      : STT={row['stt_provider']}, Trans={row['translation_provider']}",
                    f"Telugu (తెలుగు) : {row['telugu_transcript']}",
                    f"English        : {row['english_translation'] if row['english_translation'] else '[Not Translated]'}",
                ])
                if row['warning_notes']:
                    lines.append(f"Notes          : {row['warning_notes']}")
                lines.append("-" * 70)

        with open(target, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return target

    # -------------------------------------------------------------------------
    # Statistical Aggregations with pandas
    # -------------------------------------------------------------------------
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregated statistical metrics over the dataset using pandas."""
        if self._df.empty:
            return {
                "total_records": 0,
                "total_duration_sec": 0.0,
                "avg_duration_sec": 0.0,
                "avg_confidence": 0.0,
                "code_switching_count": 0,
                "code_switching_pct": 0.0,
                "unique_words_telugu": 0,
                "unique_words_english": 0,
            }

        total_records = len(self._df)
        total_duration = float(self._df["duration_sec"].sum())
        avg_duration = float(self._df["duration_sec"].mean())
        avg_conf = float(self._df["confidence"].mean())

        cs_count = int(self._df["has_code_switching"].astype(bool).sum())
        cs_pct = (cs_count / total_records) * 100.0 if total_records > 0 else 0.0

        all_te_words = " ".join(self._df["telugu_transcript"].fillna("")).split()
        all_en_words = " ".join(self._df["english_translation"].fillna("")).split()

        return {
            "total_records": total_records,
            "total_duration_sec": round(total_duration, 2),
            "avg_duration_sec": round(avg_duration, 2),
            "avg_confidence": round(avg_conf, 3),
            "code_switching_count": cs_count,
            "code_switching_pct": round(cs_pct, 1),
            "unique_words_telugu": len(set(all_te_words)),
            "unique_words_english": len(set(all_en_words)),
        }
