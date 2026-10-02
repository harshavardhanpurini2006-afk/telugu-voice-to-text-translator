# 🎙️ Telugu Voice-to-Text Translator

> **తెలుగు ప్రసంగం నుండి వచనం మరియు ఆంగ్ల అనువాదం**  
> An end-to-end Python framework for Telugu speech recognition, English translation, acoustic signal feature extraction, session history management, and publication-quality visual analytics.

---

## 📌 Features & Highlights

- 🗣️ **Telugu Speech-to-Text (STT)**: Transcribes spoken Telugu into native Telugu script (తెలుగు లిపి) with confidence scores and code-switching detection.
- 🌐 **Telugu-to-English Translation**: Optionally translates the transcribed Telugu into clear English text.
- 🔢 **NumPy Signal Processing**:
  - Audio array normalization, mono downmixing, and silence detection.
  - Root Mean Square (RMS) short-time energy envelopes.
  - Short-Time Fourier Transform (STFT) spectrogram calculation via pure NumPy FFT.
  - Formant-based harmonic acoustic waveform synthesis for offline testing.
- 📊 **pandas History Management**:
  - Structured storage with unique session IDs, timestamps, audio sources, durations, confidence scores, and code-switching tags.
  - Search across transcripts and translations.
  - One-click export to CSV and formatted TXT reports.
  - Privacy-first data deletion (single record or complete history purge).
- 📈 **Matplotlib & Seaborn Visualizations**:
  - **Acoustic Diagnostic Dashboard**: Time-domain waveform, RMS energy, STFT Spectrogram (dB), and Power Spectral Density (PSD).
  - **Usage Analytics Dashboard**: Duration vs word count regression, confidence distribution with code-switching hue, source breakdowns, and Tenglish prevalence.
- 💻 **Multiple Interfaces**:
  - **Command-Line Interface (CLI)**: Fast, scriptable, and rich terminal output.
  - **Interactive Web Interface**: Streamlit web app with browser recording, file uploader, live plots, and history tables.
- 🔒 **Privacy & Configurable Providers**:
  - Supports Google Cloud Speech & DeepTranslator for online high-accuracy transcription.
  - Supports 100% offline local/mock processing mode that guarantees audio and text never leave your machine.
- 🧪 **Comprehensive Test Suite**: 35 automated unit and integration tests with mocked microphone and external APIs.

---

## ⚠️ Important Notice on Accuracy & Limitations

> **Accuracy & Dialect Nuances**:  
> Telugu speech recognition and translation accuracy depends heavily on audio clarity, speaker dialect/accent (e.g., **Coastal Andhra**, **Rayalaseema**, **Telangana**), ambient background noise, and code-switching (**"Tenglish"** — mixing Telugu and English vocabulary).  
> Transcriptions and translations may occasionally contain phonetic or grammatical errors. Always verify critical outputs.

---

## 📂 Proposed Project Structure

```
project 6/
├── telugu_transcriber/
│   ├── __init__.py             # Package exports and versioning
│   ├── config.py               # Paths, privacy toggles, language codes, and disclaimers
│   ├── audio_processor.py      # NumPy audio arrays, validation, RMS energy, STFT spectrogram, synthesis
│   ├── transcriber.py          # STT provider abstraction (Google Speech & Offline Mock engine)
│   ├── translator.py           # Translation provider abstraction (DeepTranslator & Offline Dictionary)
│   ├── history_manager.py      # pandas DataFrame history manager, CSV/TXT export, and purge
│   ├── visualizer.py           # Matplotlib & Seaborn acoustic and analytics charts
│   ├── sample_generator.py     # NumPy synthetic WAV generator & fictional history seeding
│   └── cli.py                  # Full-featured Command-Line Interface
├── web_app.py                  # Streamlit Web UI dashboard
├── tests/
│   ├── __init__.py
│   ├── test_audio_processor.py # NumPy DSP, validation, and synthesis tests
│   ├── test_transcriber.py     # STT provider and code-switching tests
│   ├── test_translator.py      # Translation and fallback tests
│   ├── test_history_manager.py # pandas storage, search, and export tests
│   ├── test_visualizer.py      # Matplotlib and Seaborn plotting tests
│   └── test_cli.py             # CLI argument parsing and execution tests
├── data/
│   ├── samples/                # Synthetic/fictional Telugu audio WAV files
│   ├── history/                # Local history CSV and TXT reports
│   └── plots/                  # Generated diagnostic and analytics PNG figures
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation and guide
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.9 or higher (Tested on Python 3.10, 3.11, 3.12, 3.13)
- Operating System: Windows, macOS, or Linux

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Microphone Permissions
- **Windows**: Settings → Privacy & Security → Microphone → Enable "Let desktop apps access your microphone".
- **macOS**: System Settings → Privacy & Security → Microphone → Grant Terminal / Python access.
- **Linux**: Ensure ALSA / PulseAudio / PipeWire is configured (`sudo apt-get install portaudio19-dev`).

---

## 🖥️ Command-Line Interface (CLI) Usage

Run the CLI using `py -m telugu_transcriber.cli` (or `python -m telugu_transcriber.cli`):

### 1. Run the Zero-Config Synthetic Demo
Generates synthetic Telugu audio files, runs transcription, translation, and produces visual plots:
```bash
py -m telugu_transcriber.cli demo
```

### 2. Record from Microphone
Record live Telugu speech for 5 seconds, transcribe, translate to English, and plot audio diagnostics:
```bash
py -m telugu_transcriber.cli record --duration 5.0 --translate --plot
```
*Force 100% offline local processing (Privacy mode):*
```bash
py -m telugu_transcriber.cli record --duration 5.0 --mock
```

### 3. Transcribe an Audio File
Transcribe any 16-bit PCM WAV audio file:
```bash
py -m telugu_transcriber.cli transcribe data/samples/sample_greeting.wav --translate --plot
```

### 4. Plot Acoustic Audio Diagnostics
Generate Waveform, RMS envelope, STFT Spectrogram, and FFT Spectrum:
```bash
py -m telugu_transcriber.cli visualize-audio data/samples/sample_weather.wav --output data/plots/weather_analysis.png
```

### 5. Plot History Analytics Dashboard
Generate session volume, confidence distributions, and code-switching statistics:
```bash
py -m telugu_transcriber.cli visualize-stats --output data/plots/usage_stats.png
```

### 6. Manage Session History & Privacy
```bash
# View recent history records and summary metrics
py -m telugu_transcriber.cli history

# Search history by Telugu or English keyword
py -m telugu_transcriber.cli history --search "వాతావరణం"

# Export history to a custom CSV file
py -m telugu_transcriber.cli history --export-csv my_history.csv

# Export human-readable formatted TXT report
py -m telugu_transcriber.cli history --export-txt transcription_report.txt

# Delete all recorded history (Privacy purge)
py -m telugu_transcriber.cli history --clear
```

### 7. View Privacy Disclosure
```bash
py -m telugu_transcriber.cli privacy
```

---

## 🌐 Interactive Web Interface

Launch the interactive Streamlit dashboard:
```bash
py -m streamlit run web_app.py
```

### Web App Capabilities:
1. **Live Audio Recording & Upload**: Record live audio directly in the browser or upload `.wav` files.
2. **Instant Bilingual Cards**: Displays Telugu script transcription alongside English translation.
3. **Interactive Signal Visualizer**: Generates STFT spectrograms, RMS curves, and frequency distributions.
4. **Synthetic Sample Library**: Listen to and test pre-generated synthetic Telugu audio phrases.
5. **Session History Manager**: Search, filter, download CSV/TXT, or purge history.
6. **Analytics Tab**: Seaborn metrics on word counts, code-switching rates, and confidence distributions.

---

## 🔒 Privacy & Data Governance

| Mode | Audio Destination | Transcript Storage | Network Activity |
| :--- | :--- | :--- | :--- |
| **External Cloud Services** | Transmitted over TLS to Google Speech Recognition & DeepTranslator | Stored locally in `data/history/transcription_history.csv` | Requires Internet connection |
| **Offline Local (Privacy / Mock)** | Processed 100% locally via NumPy acoustic formant matching & offline dictionary | Stored locally in `data/history/transcription_history.csv` | Zero external network calls |

- **No audio is sent externally without user consent**: Select "Offline Local" in the UI or use `--mock` on the CLI to guarantee zero outbound network requests.
- **Easy Deletion**: Users can delete individual records or wipe the entire history dataset at any time via `history --clear` or the web UI button.

---

## 🧪 Running the Automated Test Suite

Run all 35 tests using `pytest`:
```bash
py -m pytest tests/ -v
```

All external APIs and microphone hardware are mocked in unit tests, allowing 100% offline test execution.

---

## 📜 Supported Audio Formats

- **Format**: WAV (16-bit PCM / 8-bit / 24-bit / 32-bit mono or stereo).
- **Sampling Rates**: 16,000 Hz recommended for speech recognition; 44,100 Hz / 48,000 Hz are automatically downmixed and analyzed.
- **Converting other formats (MP3 / FLAC / OGG)**:
  ```bash
  ffmpeg -i input.mp3 -ac 1 -ar 16000 output.wav
  ```

---

## 🤝 License & Disclaimer
This project is built for educational, research, and assistive language processing purposes. For high-stakes medical, legal, or emergency scenarios, always have a fluent Telugu speaker review transcripts and translations.
