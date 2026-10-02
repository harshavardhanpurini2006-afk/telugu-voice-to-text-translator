"""
Audio Processor Module for Telugu Voice-to-Text Translator.
Uses NumPy for low-level audio signal processing, feature extraction,
validation, silence trimming, RMS energy envelopes, FFT spectrograms,
and synthetic speech generation.
"""

import math
import os
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


class AudioValidationError(Exception):
    """Raised when an audio file or signal fails validation."""
    pass


class AudioProcessor:
    """Handles audio capture, validation, file I/O, and NumPy-based DSP analysis."""

    def __init__(self, sample_rate: int = 16000, max_duration_sec: float = 120.0):
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec

    # -------------------------------------------------------------------------
    # Audio Validation & File Loading
    # -------------------------------------------------------------------------
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate that the audio file exists, has a valid extension, and is readable."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise AudioValidationError(f"Audio file not found: '{path}'")
        if not path.is_file():
            raise AudioValidationError(f"Path is not a regular file: '{path}'")
        if path.stat().st_size == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): '{path}'")

        allowed_extensions = {".wav", ".wave", ".raw"}
        if path.suffix.lower() not in allowed_extensions:
            # We also note that .wav is standard for speech STT engines
            # but allow user to be aware
            raise AudioValidationError(
                f"Unsupported audio format '{path.suffix}'. Supported formats: {', '.join(allowed_extensions)}. "
                "For MP3/OGG/FLAC, please convert to standard 16-bit PCM WAV."
            )
        return path

    def load_wav(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file into a normalized NumPy array in range [-1.0, 1.0].
        Returns (audio_array, sample_rate).
        """
        path = self.validate_file(file_path)
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                if n_frames == 0:
                    raise AudioValidationError(f"WAV file '{path.name}' contains 0 audio frames.")

                duration = n_frames / framerate
                if duration > self.max_duration_sec:
                    raise AudioValidationError(
                        f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({self.max_duration_sec:.1f}s)."
                    )

                raw_bytes = wf.readframes(n_frames)

            # Decode based on bit depth
            if sampwidth == 1:
                # 8-bit unsigned PCM
                data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
                data = (data - 128.0) / 128.0
            elif sampwidth == 2:
                # 16-bit signed PCM
                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                data = data / 32768.0
            elif sampwidth == 3:
                # 24-bit signed PCM (pack into 32-bit int)
                raw_bytes_padded = bytearray()
                for i in range(0, len(raw_bytes), 3):
                    raw_bytes_padded.extend(b"\x00" + raw_bytes[i : i + 3])
                data = np.frombuffer(raw_bytes_padded, dtype=np.int32).astype(np.float32)
                data = data / 2147483648.0
            elif sampwidth == 4:
                # 32-bit signed PCM or float
                data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
                data = data / 2147483648.0
            else:
                raise AudioValidationError(f"Unsupported sample width: {sampwidth * 8} bits.")

            # Multichannel to Mono downmixing using NumPy
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
                data = np.mean(data, axis=1)

            # Resample or warning if needed (standard speech uses 16kHz)
            return data, framerate

        except wave.Error as e:
            raise AudioValidationError(f"Corrupted or invalid WAV file: {e}")
        except Exception as e:
            if isinstance(e, AudioValidationError):
                raise
            raise AudioValidationError(f"Failed to read audio file '{path}': {e}")

    def save_wav(self, audio: np.ndarray, file_path: Union[str, Path], sample_rate: Optional[int] = None) -> Path:
        """Save a NumPy audio float array [-1.0, 1.0] to a 16-bit PCM WAV file."""
        target = Path(file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        sr = sample_rate or self.sample_rate

        # Normalize and clip to avoid wrap-around distortion
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(target), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())

        return target

    # -------------------------------------------------------------------------
    # Microphone Recording
    # -------------------------------------------------------------------------
    def record_microphone(self, duration_sec: float = 5.0, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Record audio from the default input device (microphone) into a NumPy array.
        Handles missing microphone or permission issues gracefully.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "Microphone recording library 'sounddevice' is not available. "
                "Please verify sounddevice is installed."
            )

        sr = sample_rate or self.sample_rate
        total_samples = int(duration_sec * sr)

        try:
            # Query default device to check existence
            devices = sd.query_devices()
            if not devices:
                raise RuntimeError("No audio input hardware devices detected.")

            recording = sd.rec(
                total_samples,
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocking=True
            )
            audio = recording.flatten()
            return audio

        except Exception as e:
            raise RuntimeError(
                f"Microphone recording error: {e}. "
                "Ensure microphone permissions are granted and an active input device is connected."
            )

    # -------------------------------------------------------------------------
    # NumPy Signal Processing & Feature Extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """Peak normalize an audio signal to range [-1.0, 1.0]."""
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            return audio / peak
        return audio

    @staticmethod
    def compute_rms_energy(audio: np.ndarray, frame_length: int = 512, hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Root Mean Square (RMS) energy envelope over sliding frames using NumPy.
        Returns (times_array, rms_energy_array).
        """
        if len(audio) == 0:
            return np.array([]), np.array([])

        num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
        rms = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            if len(frame) > 0:
                rms[i] = np.sqrt(np.mean(frame**2))

        times = np.arange(num_frames) * hop_length / 16000.0
        return times, rms

    @staticmethod
    def compute_spectrogram(
        audio: np.ndarray,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        window: str = "hann"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT) Spectrogram using pure NumPy.
        Returns:
            frequencies: 1D array of frequency bins (Hz)
            times: 1D array of time stamps (seconds)
            spectrogram_db: 2D array of magnitude spectrogram in decibels (dB)
        """
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        # Create window
        if window == "hann":
            win = np.hanning(n_fft)
        else:
            win = np.ones(n_fft)

        num_frames = (len(audio) - n_fft) // hop_length + 1
        spec_matrix = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start : start + n_fft] * win
            # Real FFT
            fft_res = np.fft.rfft(frame, n=n_fft)
            mag = np.abs(fft_res)
            spec_matrix[:, i] = mag

        # Convert to dB with floor
        spec_db = 20.0 * np.log10(np.maximum(spec_matrix, 1e-6))

        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        times = np.arange(num_frames) * (hop_length / sample_rate)

        return frequencies, times, spec_db

    @staticmethod
    def extract_audio_metrics(audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Union[float, int]]:
        """
        Extract numerical signal metrics using NumPy for speech audio characterization.
        """
        if len(audio) == 0:
            return {
                "duration_sec": 0.0,
                "samples_count": 0,
                "peak_amplitude": 0.0,
                "rms_level": 0.0,
                "snr_estimate_db": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid_hz": 0.0,
                "is_silent": True,
            }

        duration = len(audio) / sample_rate
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))

        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio))))
        zcr = float(zero_crossings / max(1, len(audio) - 1))

        # Spectral Centroid (NumPy FFT)
        n_fft = min(2048, len(audio))
        fft_vals = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        sum_fft = np.sum(fft_vals)
        centroid = float(np.sum(freqs * fft_vals) / sum_fft) if sum_fft > 1e-6 else 0.0

        # Simple SNR estimate (ratio of 90th percentile power to 10th percentile power)
        frame_size = 512
        num_f = len(audio) // frame_size
        if num_f > 2:
            frame_powers = np.array([np.mean(audio[i*frame_size:(i+1)*frame_size]**2) for i in range(num_f)])
            high_p = np.percentile(frame_powers, 90)
            low_p = max(np.percentile(frame_powers, 10), 1e-9)
            snr_est = float(10.0 * np.log10(high_p / low_p))
        else:
            snr_est = 0.0

        is_silent = rms < 0.005

        return {
            "duration_sec": round(duration, 3),
            "samples_count": int(len(audio)),
            "peak_amplitude": round(peak, 4),
            "rms_level": round(rms, 4),
            "snr_estimate_db": round(snr_est, 2),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_centroid_hz": round(centroid, 1),
            "is_silent": is_silent,
        }

    # -------------------------------------------------------------------------
    # Synthetic Speech Audio Synthesis (NumPy Based)
    # -------------------------------------------------------------------------
    @staticmethod
    def generate_synthetic_speech_wave(
        text: str = "నమస్కారం",
        duration_sec: float = 3.0,
        sample_rate: int = 16000,
        pitch_f0: float = 140.0
    ) -> np.ndarray:
        """
        Generate a synthetic acoustic waveform using NumPy with vowel formant synthesis,
        syllabic rhythm envelopes, and harmonic overtones.
        Enables 100% offline realistic audio testing and demo without external recordings.
        """
        total_samples = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)

        # Base glottal pulse train with harmonics
        glottal_wave = np.zeros(total_samples, dtype=np.float32)
        for harmonic in range(1, 12):
            amp = 1.0 / (harmonic ** 0.9)
            # Add slight vibrato
            vibrato = 3.0 * np.sin(2 * np.pi * 5.0 * t)
            freq = harmonic * (pitch_f0 + vibrato)
            glottal_wave += amp * np.sin(2 * np.pi * freq * t)

        # Formant filters (Simulate typical Telugu vowels: /a/, /u/, /i/, /e/, /o/)
        # Resonant frequencies: F1 (500-800Hz), F2 (1100-2200Hz), F3 (2500-3000Hz)
        formant1 = np.sin(2 * np.pi * 700.0 * t) * np.exp(-t % 0.15 * 20.0)
        formant2 = np.sin(2 * np.pi * 1700.0 * t) * np.exp(-t % 0.15 * 30.0)
        formant3 = np.sin(2 * np.pi * 2600.0 * t) * np.exp(-t % 0.15 * 40.0)

        raw_speech = 0.5 * glottal_wave + 0.3 * formant1 + 0.2 * formant2 + 0.1 * formant3

        # Syllabic envelope modulation based on word length
        num_syllables = max(2, len(text.replace(" ", "")))
        syllable_rate = num_syllables / duration_sec
        envelope = np.abs(np.sin(np.pi * syllable_rate * t)) ** 1.5

        # Attack and release decay
        fade_samples = int(0.08 * sample_rate)
        if total_samples > 2 * fade_samples:
            envelope[:fade_samples] *= np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Add subtle acoustic air noise
        noise = 0.02 * np.random.randn(total_samples)

        signal = (raw_speech * envelope) + noise
        return AudioProcessor.normalize(signal.astype(np.float32))
