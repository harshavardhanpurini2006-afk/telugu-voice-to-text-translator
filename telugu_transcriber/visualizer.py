"""
Visualization Module for Telugu Voice-to-Text Translator.
Uses Matplotlib and Seaborn to generate publication-grade visual representations of
audio acoustic features (waveforms, STFT spectrograms, RMS envelopes, spectral power)
and pandas-driven transcription usage statistics.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
# Safe headless rendering backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from telugu_transcriber.audio_processor import AudioProcessor


class AudioVisualizer:
    """Visualizes raw audio signals, spectrograms, and acoustic characteristics."""

    def __init__(self):
        # Set harmonious Seaborn theme
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
        })

    def plot_waveform_and_rms(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Audio Waveform & RMS Energy Envelope",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot the time-domain waveform alongside its Root Mean Square (RMS) energy envelope.
        """
        duration = len(audio) / sample_rate
        time_axis = np.linspace(0, duration, len(audio), endpoint=False)
        rms_times, rms_values = AudioProcessor.compute_rms_energy(audio)
        rms_times = rms_times * (duration / max(1e-6, rms_times[-1] if len(rms_times) > 0 else 1.0))

        fig, ax1 = plt.subplots(figsize=(10, 4), dpi=150)

        # Raw audio waveform (Time Domain)
        ax1.plot(time_axis, audio, color="#2b5c8f", alpha=0.6, linewidth=0.7, label="Acoustic Waveform")
        ax1.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Amplitude (normalized)", color="#2b5c8f", fontsize=10, fontweight="bold")
        ax1.set_ylim(-1.05, 1.05)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # RMS Energy overlay on secondary axis
        if len(rms_values) > 0:
            ax2 = ax1.twinx()
            ax2.plot(rms_times, rms_values, color="#e65100", linewidth=2.0, label="RMS Energy Envelope")
            ax2.set_ylabel("RMS Energy", color="#e65100", fontsize=10, fontweight="bold")
            ax2.set_ylim(0, max(0.6, np.max(rms_values) * 1.25))
            ax2.grid(False)

        ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()

        if save_path:
            p = Path(save_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")

        return fig

    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Telugu Speech STFT Spectrogram (dB)",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot the Short-Time Fourier Transform (STFT) frequency spectrogram.
        """
        freqs, times, spec_db = AudioProcessor.compute_spectrogram(audio, sample_rate=sample_rate)

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        c = ax.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="magma", vmin=-60, vmax=0)
        cbar = fig.colorbar(c, ax=ax, pad=0.02)
        cbar.set_label("Magnitude (dB)", fontsize=9, fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency (Hz)", fontsize=10, fontweight="bold")
        ax.set_ylim(0, min(8000, sample_rate // 2))

        fig.tight_layout()

        if save_path:
            p = Path(save_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")

        return fig

    def plot_audio_dashboard(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        title: str = "Telugu Audio Signal Analysis Dashboard",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Generate a multi-panel acoustic diagnostic dashboard:
        1. Waveform & RMS Envelope
        2. STFT Spectrogram
        3. Power Spectral Density (FFT)
        4. Cumulative Energy Distribution
        """
        duration = len(audio) / sample_rate
        time_axis = np.linspace(0, duration, len(audio), endpoint=False)
        rms_times, rms_values = AudioProcessor.compute_rms_energy(audio)
        if len(rms_times) > 0 and rms_times[-1] > 0:
            rms_times = rms_times * (duration / rms_times[-1])
        freqs, times, spec_db = AudioProcessor.compute_spectrogram(audio, sample_rate=sample_rate)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # 1. Waveform & RMS
        ax1.plot(time_axis, audio, color="#1976d2", alpha=0.5, linewidth=0.6, label="Waveform")
        if len(rms_values) > 0:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(rms_times, rms_values, color="#d84315", linewidth=1.8, label="RMS")
            ax1_twin.set_ylabel("RMS", color="#d84315", fontsize=9)
            ax1_twin.grid(False)
        ax1.set_title("1. Time Domain Waveform & RMS Envelope", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Amplitude", fontsize=9)
        ax1.set_ylim(-1.05, 1.05)

        # 2. Spectrogram
        c2 = ax2.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="viridis", vmin=-60, vmax=0)
        fig.colorbar(c2, ax=ax2, pad=0.02).set_label("dB", fontsize=8)
        ax2.set_title("2. Short-Time Fourier Transform (STFT)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (s)", fontsize=9)
        ax2.set_ylabel("Frequency (Hz)", fontsize=9)
        ax2.set_ylim(0, min(8000, sample_rate // 2))

        # 3. FFT Power Spectrum (NumPy FFT)
        n_fft = min(4096, len(audio))
        fft_vals = np.abs(np.fft.rfft(audio[:n_fft]))
        fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        fft_db = 20.0 * np.log10(np.maximum(fft_vals, 1e-6))
        ax3.plot(fft_freqs, fft_db, color="#388e3c", linewidth=1.0)
        ax3.set_title("3. Power Spectral Density (FFT Spectrum)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Frequency (Hz)", fontsize=9)
        ax3.set_ylabel("Power (dB)", fontsize=9)
        ax3.set_xlim(0, min(6000, sample_rate // 2))

        # 4. Cumulative Energy over time
        cum_energy = np.cumsum(audio**2)
        total_energy = cum_energy[-1] if len(cum_energy) > 0 and cum_energy[-1] > 0 else 1.0
        norm_cum_energy = cum_energy / total_energy
        ax4.plot(time_axis, norm_cum_energy, color="#7b1fa2", linewidth=2.0)
        ax4.set_title("4. Normalized Cumulative Acoustic Energy", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Time (s)", fontsize=9)
        ax4.set_ylabel("Cumulative Fraction", fontsize=9)
        ax4.set_ylim(0, 1.05)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            p = Path(save_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")

        return fig


class HistoryVisualizer:
    """Visualizes session history and transcription analytics with Seaborn and pandas."""

    def __init__(self):
        sns.set_theme(style="whitegrid", palette="tab10")

    def plot_usage_dashboard(
        self,
        df: pd.DataFrame,
        title: str = "Telugu Voice-to-Text Usage Analytics",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[plt.Figure]:
        """
        Generate a 4-panel Seaborn analytical summary:
        1. Audio Duration vs Telugu Word Count (Scatter & Trend)
        2. Transcription Confidence Distribution (KDE / Histogram)
        3. Audio Input Source Distribution (Bar Chart)
        4. Code-Switching ('Tenglish') Breakdown (Pie / Count)
        """
        if df.empty:
            return None

        # Prepare features for plotting
        plot_df = df.copy()
        plot_df["word_count_telugu"] = plot_df["telugu_transcript"].fillna("").apply(lambda s: len(str(s).split()))
        plot_df["word_count_english"] = plot_df["english_translation"].fillna("").apply(lambda s: len(str(s).split()))
        plot_df["code_switch_label"] = plot_df["has_code_switching"].apply(lambda x: "Tenglish / Mixed" if x else "Pure Telugu")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # 1. Duration vs Word Count
        if len(plot_df) > 1:
            sns.regplot(
                data=plot_df,
                x="duration_sec",
                y="word_count_telugu",
                ax=ax1,
                color="#1976d2",
                scatter_kws={"alpha": 0.7, "s": 50},
                line_kws={"color": "#d32f2f", "linewidth": 1.5},
            )
        else:
            sns.scatterplot(
                data=plot_df,
                x="duration_sec",
                y="word_count_telugu",
                ax=ax1,
                color="#1976d2",
                s=60
            )
        ax1.set_title("1. Audio Duration vs Transcribed Word Count", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Audio Duration (seconds)", fontsize=9)
        ax1.set_ylabel("Telugu Words Count", fontsize=9)

        # 2. Confidence Distribution
        sns.histplot(
            data=plot_df,
            x="confidence",
            hue="code_switch_label",
            multiple="stack",
            bins=8,
            kde=True if len(plot_df) >= 5 else False,
            ax=ax2,
            palette={"Pure Telugu": "#2e7d32", "Tenglish / Mixed": "#f57c00"}
        )
        ax2.set_title("2. Confidence Score Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Confidence Score (0.0 - 1.0)", fontsize=9)
        ax2.set_ylabel("Number of Sessions", fontsize=9)

        # 3. Audio Source Breakdown
        source_counts = plot_df["audio_source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        sns.barplot(
            data=source_counts,
            x="Source",
            y="Count",
            hue="Source",
            legend=False,
            ax=ax3,
            palette="Blues_d"
        )
        ax3.set_title("3. Audio Input Source Distribution", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Input Source", fontsize=9)
        ax3.set_ylabel("Count", fontsize=9)

        # 4. Code-Switching Prevalence
        cs_counts = plot_df["code_switch_label"].value_counts()
        colors = ["#43a047", "#fb8c00"]
        ax4.pie(
            cs_counts.values,
            labels=cs_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[:len(cs_counts)],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        ax4.set_title("4. Code-Switching ('Tenglish') Prevalence", fontsize=11, fontweight="bold")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            p = Path(save_path).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")

        return fig
