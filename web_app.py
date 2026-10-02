"""
Telugu Voice-to-Text Translator - Interactive Web Interface.
Built with Streamlit, NumPy, pandas, Matplotlib, and Seaborn.
"""

import io
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from telugu_transcriber.audio_processor import AudioProcessor, AudioValidationError
from telugu_transcriber.config import (
    ACCURACY_DISCLAIMER,
    Config,
    DEFAULT_CONFIG,
    PRIVACY_NOTICE,
)
from telugu_transcriber.history_manager import HistoryManager
from telugu_transcriber.sample_generator import (
    SYNTHETIC_SAMPLES_CATALOG,
    generate_all_samples,
    seed_synthetic_history,
)
from telugu_transcriber.transcriber import TeluguTranscriber
from telugu_transcriber.translator import TeluguTranslator
from telugu_transcriber.visualizer import AudioVisualizer, HistoryVisualizer

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Telugu Voice-to-Text Translator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.9rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #78350f;
    }
    .privacy-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 0.9rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #14532d;
    }
    .output-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .telugu-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.6;
    }
    .english-text {
        font-size: 1.2rem;
        color: #1e40af;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session managers
config = DEFAULT_CONFIG
history_mgr = HistoryManager(config.history_file_csv)
audio_proc = AudioProcessor(sample_rate=config.sample_rate)
audio_vis = AudioVisualizer()
hist_vis = HistoryVisualizer()

# Ensure synthetic sample audio files exist
samples_dir = config.samples_dir
if not any(samples_dir.glob("*.wav")):
    generate_all_samples(config)

# -----------------------------------------------------------------------------
# Sidebar Settings & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Speech & Translation Engines")
    provider_mode = st.radio(
        "Processing Mode",
        options=["External Cloud Services", "Offline Local (Privacy / Mock)"],
        index=0,
        help="External uses Google Speech & DeepTranslator. Offline uses local acoustic pattern matching and dictionary."
    )
    is_offline = (provider_mode == "Offline Local (Privacy / Mock)")
    
    st.markdown("---")
    st.subheader("Options")
    enable_translation = st.checkbox("Translate Telugu to English", value=True)
    show_spectrogram = st.checkbox("Generate Audio Spectrogram & FFT", value=True)
    
    st.markdown("---")
    st.subheader("Privacy & History")
    st.caption("All transcripts are saved locally in your project's CSV file.")
    if st.button("🗑️ Clear All History", help="Permanently delete all session history records"):
        count = history_mgr.clear_all_history()
        st.success(f"Deleted {count} records from local history.")
        st.rerun()

    if st.button("🌱 Load Synthetic Demo Dataset", help="Seed fictional Telugu transcription records"):
        added = seed_synthetic_history(history_mgr)
        st.success(f"Added sample records to history.")
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🎙️ Telugu Voice-to-Text Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">తెలుగు ప్రసంగం నుండి వచనం మరియు ఆంగ్ల అనువాదం (NumPy • pandas • Matplotlib • Seaborn)</div>', unsafe_allow_html=True)

# Accuracy & Limitation Disclaimer
st.markdown(f"""
<div class="disclaimer-box">
    <strong>⚠️ Notice on Accuracy & Nuances:</strong><br>
    Telugu speech recognition and translation accuracy depends on audio clarity, regional accents 
    (e.g., Coastal Andhra, Rayalaseema, Telangana dialects), background noise, and code-switching 
    (<strong>"Tenglish"</strong> - mixing Telugu and English). Always review transcripts for critical usage.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_transcribe, tab_samples, tab_history, tab_analytics, tab_privacy = st.tabs([
    "🎙️ Transcribe & Translate",
    "📁 Sample Audio Library",
    "📜 Session History",
    "📊 Usage Analytics",
    "🔒 Privacy & Architecture",
])

# -----------------------------------------------------------------------------
# TAB 1: Transcribe & Translate
# -----------------------------------------------------------------------------
with tab_transcribe:
    col_input, col_results = st.columns([1, 1], gap="large")
    
    audio_bytes = None
    input_source_name = ""
    
    with col_input:
        st.subheader("1. Audio Input")
        input_choice = st.radio("Choose Input Method:", ["Upload Audio File (WAV)", "Record with Microphone", "Use Pre-generated Sample"], horizontal=True)
        
        if input_choice == "Upload Audio File (WAV)":
            uploaded_file = st.file_uploader("Upload Telugu WAV Audio File", type=["wav", "wave"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                input_source_name = f"upload: {uploaded_file.name}"
                st.audio(audio_bytes, format="audio/wav")
                
        elif input_choice == "Record with Microphone":
            st.info("Click the microphone button below to record your voice in Telugu.")
            # Streamlit built-in audio_input (supports direct browser microphone recording)
            mic_audio = st.audio_input("Record Telugu Speech")
            if mic_audio is not None:
                audio_bytes = mic_audio.getvalue()
                input_source_name = "microphone (live recording)"
                st.audio(audio_bytes, format="audio/wav")
                
        elif input_choice == "Use Pre-generated Sample":
            sample_files = list(samples_dir.glob("*.wav"))
            if sample_files:
                sample_map = {f.name: f for f in sample_files}
                selected_sample = st.selectbox("Select a synthetic sample file:", list(sample_map.keys()))
                sample_path = sample_map[selected_sample]
                with open(sample_path, "rb") as f:
                    audio_bytes = f.read()
                input_source_name = f"sample: {selected_sample}"
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning("No sample files found. Use the sidebar button to generate samples.")

    with col_results:
        st.subheader("2. Transcription & Translation")
        
        if audio_bytes is not None:
            if st.button("🚀 Process Telugu Audio", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio with NumPy and transcribing..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    try:
                        # Load & Validate with NumPy
                        audio_array, sr_rate = audio_proc.load_wav(tmp_path)
                        metrics = audio_proc.extract_audio_metrics(audio_array, sr_rate)

                        # Transcriber
                        stt_engine = TeluguTranscriber(
                            provider_name="mock" if is_offline else "google",
                            allow_external=not is_offline
                        )
                        trans_result = stt_engine.transcribe(tmp_path)

                        # Robust Translator
                        eng_text = ""
                        trans_warning = None
                        if enable_translation and trans_result.telugu_text:
                            try:
                                trans_engine = TeluguTranslator(
                                    provider_name="google_direct" if not is_offline else "mock",
                                    allow_external=not is_offline
                                )
                                t_res = trans_engine.translate(trans_result.telugu_text)
                                eng_text = t_res.translated_text
                                trans_warning = t_res.warning_notes
                            except Exception as e_trans:
                                eng_text = "[Translation unavailable: Service busy]"
                                trans_warning = f"Translation notice: {e_trans}"

                        # Display Results
                        st.markdown(f"""
                        <div class="output-card">
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">TELUGU TRANSCRIPT (తెలుగు లిపి)</div>
                            <div class="telugu-text">{trans_result.telugu_text if trans_result.telugu_text else "<em>[No speech detected or audio silent]</em>"}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if enable_translation:
                            st.markdown(f"""
                            <div class="output-card">
                                <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 4px;">ENGLISH TRANSLATION</div>
                                <div class="english-text">{eng_text if eng_text else "<em>[No translation available]</em>"}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Metadata Metrics Row
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        mcol1.metric("Duration", f"{metrics['duration_sec']}s")
                        mcol2.metric("Confidence", f"{trans_result.confidence:.0%}")
                        mcol3.metric("SNR Estimate", f"{metrics['snr_estimate_db']} dB")
                        mcol4.metric("STT Engine", trans_result.provider)

                        if trans_result.has_code_switching:
                            st.warning("⚠️ **Code-Switching Detected**: Input contains mixed Telugu-English ('Tenglish').")
                        if trans_result.warning_notes:
                            st.info(f"ℹ️ **Audio Diagnostic**: {trans_result.warning_notes}")
                        if trans_warning:
                            st.caption(f"ℹ️ {trans_warning}")

                        # Combined notes for history
                        all_notes = []
                        if trans_result.warning_notes:
                            all_notes.append(trans_result.warning_notes)
                        if trans_warning:
                            all_notes.append(trans_warning)
                        combined_notes = " | ".join(all_notes)

                        # Save to history
                        history_mgr.add_record(
                            telugu_transcript=trans_result.telugu_text,
                            english_translation=eng_text,
                            audio_source=input_source_name,
                            duration_sec=metrics["duration_sec"],
                            confidence=trans_result.confidence,
                            stt_provider=trans_result.provider,
                            translation_provider="google_direct" if (enable_translation and not is_offline) else "mock/none",
                            has_code_switching=trans_result.has_code_switching,
                            warning_notes=combined_notes,
                        )
                        st.success("✓ Saved session to local history.")

                        # Signal Visualizer
                        if show_spectrogram:
                            st.markdown("---")
                            st.subheader("3. Acoustic Signal Analysis (NumPy & Matplotlib)")
                            fig = audio_vis.plot_audio_dashboard(audio_array, sample_rate=sr_rate)
                            st.pyplot(fig, clear_figure=True)

                    except AudioValidationError as ave:
                        st.error(f"Audio Validation Error: {ave}")
                    except Exception as ex:
                        st.error(f"Processing Error: {ex}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        else:
            st.info("👈 Please select or record an audio file to begin.")

# -----------------------------------------------------------------------------
# TAB 2: Sample Library
# -----------------------------------------------------------------------------
with tab_samples:
    st.subheader("Synthetic Telugu Audio Samples (No Private Recordings Needed)")
    st.caption("These samples use NumPy harmonic formant synthesis to simulate Telugu vowels and phrases.")

    for fname, te_text, en_text, dur in SYNTHETIC_SAMPLES_CATALOG:
        sample_p = samples_dir / fname
        with st.expander(f"🎵 {fname} ({dur}s) - {te_text[:30]}...", expanded=True):
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                if sample_p.exists():
                    st.audio(str(sample_p), format="audio/wav")
            with sc2:
                st.markdown(f"**Telugu Phrase**: {te_text}")
                st.markdown(f"**English Meaning**: {en_text}")

# -----------------------------------------------------------------------------
# TAB 3: History Management
# -----------------------------------------------------------------------------
with tab_history:
    st.subheader("Session Transcription History (pandas)")
    df = history_mgr.get_all()

    if df.empty:
        st.info("No transcription history available yet. Record audio or click 'Load Synthetic Demo Dataset' in the sidebar.")
    else:
        # Search & Filter
        search_q = st.text_input("🔍 Search History (Telugu or English keyword):", "")
        filtered_df = history_mgr.search(search_q) if search_q else df

        st.dataframe(
            filtered_df[[
                "id", "timestamp", "audio_source", "duration_sec", "confidence",
                "telugu_transcript", "english_translation", "has_code_switching"
            ]],
            use_container_width=True,
            height=320,
        )

        # Export Actions
        ecol1, ecol2, ecol3 = st.columns([1, 1, 2])
        with ecol1:
            csv_data = filtered_df.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="📥 Download History as CSV",
                data=csv_data,
                file_name="telugu_transcription_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ecol2:
            txt_path = config.data_dir / "temp_history.txt"
            history_mgr.export_to_txt(txt_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                txt_content = f.read()
            st.download_button(
                label="📄 Download Summary TXT",
                data=txt_content,
                file_name="telugu_transcription_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# TAB 4: Usage Analytics (Seaborn & Matplotlib)
# -----------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Transcription Analytics & Usage Visualizations (Seaborn & Matplotlib)")
    df_all = history_mgr.get_all()

    if df_all.empty:
        st.info("History is currently empty. Seed demo data from sidebar to view charts.")
    else:
        stats = history_mgr.get_summary_statistics()
        
        # Metric indicators
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        kcol1.metric("Total Sessions", stats["total_records"])
        kcol2.metric("Total Audio Recorded", f"{stats['total_duration_sec']} s")
        kcol3.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
        kcol4.metric("Tenglish / Mixed Rate", f"{stats['code_switching_pct']}%")

        st.markdown("---")
        fig_stats = hist_vis.plot_usage_dashboard(df_all)
        if fig_stats:
            st.pyplot(fig_stats, clear_figure=True)

# -----------------------------------------------------------------------------
# TAB 5: Privacy & Architecture
# -----------------------------------------------------------------------------
with tab_privacy:
    st.subheader("🔒 Privacy Policy & System Architecture")
    
    st.markdown(f"""
    <div class="privacy-box">
        <strong>Data Governance & Transparency:</strong><br>
        {PRIVACY_NOTICE.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ Technical Architecture & Modules
    - **NumPy**: Low-level audio array handling, downmixing, RMS energy calculations, Short-Time Fourier Transform (STFT) spectrogram calculation, and vowel formant audio synthesis.
    - **pandas**: Session tracking, search filtering, statistical aggregations, and local CSV/TXT export/purge.
    - **Matplotlib & Seaborn**: Acoustic signal diagnostic charts (Time-domain waveforms, STFT Spectrograms, PSD, Cumulative energy) and session analytics (Duration vs word count, Confidence distribution, Code-switching breakdown).
    - **Transcriber**: Decoupled Speech-to-Text abstraction with Google Cloud API and 100% offline acoustic mock engines.
    - **Translator**: Configurable translation module with DeepTranslator and offline dictionary fallbacks.
    """)
