"""
Command-Line Interface (CLI) for Telugu Voice-to-Text Translator.
Provides robust commands for audio recording, file transcription, translation,
history viewing/export/deletion, and Matplotlib/Seaborn visualization generation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from telugu_transcriber.audio_processor import AudioProcessor, AudioValidationError
from telugu_transcriber.config import ACCURACY_DISCLAIMER, Config, DEFAULT_CONFIG, PRIVACY_NOTICE
from telugu_transcriber.history_manager import HistoryManager
from telugu_transcriber.sample_generator import generate_all_samples, seed_synthetic_history
from telugu_transcriber.transcriber import TeluguTranscriber
from telugu_transcriber.translator import TeluguTranslator
from telugu_transcriber.visualizer import AudioVisualizer, HistoryVisualizer


def print_banner():
    """Print ASCII header banner and disclaimers."""
    print("=" * 75)
    print("   🎙️  TELUGU VOICE-TO-TEXT TRANSLATOR & AUDIO ANALYZER  🎙️")
    print("      తెలుగు ప్రసంగం నుండి వచనం మరియు ఆంగ్ల అనువాదం")
    print("=" * 75)


def print_accuracy_disclaimer():
    """Print the required accuracy and dialect notice."""
    print("\n" + "-" * 75)
    print(ACCURACY_DISCLAIMER)
    print("-" * 75 + "\n")


def print_privacy_notice():
    """Print privacy and external service disclosure."""
    print("\n" + "-" * 75)
    print(PRIVACY_NOTICE)
    print("-" * 75 + "\n")


def cmd_record(args, config: Config):
    """Record speech from microphone, transcribe to Telugu, and optionally translate."""
    print_banner()
    if not args.quiet:
        print_accuracy_disclaimer()

    processor = AudioProcessor(sample_rate=config.sample_rate)
    duration = args.duration or config.recording_duration_sec

    print(f"[*] Preparing microphone recording for {duration} seconds...")
    print("[*] Speak clearly in Telugu into your microphone...")
    print("    (ఉదాహరణ: 'నమస్కారం, మీరు ఎలా ఉన్నారు?')")

    try:
        audio = processor.record_microphone(duration_sec=duration)
    except Exception as e:
        print(f"\n[!] Microphone Error: {e}", file=sys.stderr)
        print("[!] Tip: You can test with audio files using: py -m telugu_transcriber.cli transcribe <file.wav>")
        print("[!] Or use synthetic offline demo: py -m telugu_transcriber.cli demo")
        sys.exit(1)

    # Save temporary recording
    temp_wav = config.data_dir / "temp_recording.wav"
    processor.save_wav(audio, temp_wav)
    print(f"[✓] Recording captured and saved temporarily to {temp_wav.name}")

    # Process and transcribe
    allow_ext = not args.mock and config.allow_external_services
    provider = "mock" if args.mock else config.stt_provider
    transcriber = TeluguTranscriber(provider_name=provider, allow_external=allow_ext)

    print(f"[*] Transcribing using '{provider}' provider (External: {allow_ext})...")
    trans_res = transcriber.transcribe(temp_wav)

    print("\n" + "=" * 50)
    print("📝 TRANSCRIPTION RESULT (తెలుగు)")
    print("=" * 50)
    print(f"Telugu Text : {trans_res.telugu_text if trans_res.telugu_text else '[No speech detected]'}")
    print(f"Confidence  : {trans_res.confidence:.1%}")
    print(f"Provider    : {trans_res.provider}")
    print(f"Duration    : {trans_res.duration_sec:.2f}s")
    if trans_res.has_code_switching:
        print("Notice      : ⚠️ Mixed Telugu-English ('Tenglish') detected.")
    if trans_res.warning_notes:
        print(f"Warning     : {trans_res.warning_notes}")

    # Optional Translation
    eng_trans = ""
    if args.translate and trans_res.telugu_text:
        trans_prov = "mock" if args.mock else config.translation_provider
        translator = TeluguTranslator(provider_name=trans_prov, allow_external=allow_ext)
        t_res = translator.translate(trans_res.telugu_text)
        eng_trans = t_res.translated_text
        print("\n" + "=" * 50)
        print("🌐 ENGLISH TRANSLATION")
        print("=" * 50)
        print(f"English     : {eng_trans}")
        print(f"Provider    : {t_res.provider}")

    # Save to history
    history_mgr = HistoryManager(config.history_file_csv)
    history_mgr.add_record(
        telugu_transcript=trans_res.telugu_text,
        english_translation=eng_trans,
        audio_source="microphone",
        duration_sec=trans_res.duration_sec,
        confidence=trans_res.confidence,
        stt_provider=trans_res.provider,
        translation_provider="deep_translator" if (args.translate and not args.mock) else "none",
        has_code_switching=trans_res.has_code_switching,
        warning_notes=trans_res.warning_notes or "",
    )
    print(f"\n[✓] Session saved to local history ({config.history_file_csv.name})")

    # Optional Audio Visualization
    if args.plot:
        vis = AudioVisualizer()
        plot_out = config.plots_dir / "latest_microphone_analysis.png"
        vis.plot_audio_dashboard(audio, sample_rate=config.sample_rate, save_path=plot_out)
        print(f"[✓] Audio diagnostic plot saved to: {plot_out}")


def cmd_transcribe(args, config: Config):
    """Transcribe an audio file into Telugu text."""
    print_banner()
    if not args.quiet:
        print_accuracy_disclaimer()

    audio_path = Path(args.file)
    processor = AudioProcessor(sample_rate=config.sample_rate)

    try:
        audio, sr_rate = processor.load_wav(audio_path)
    except AudioValidationError as e:
        print(f"\n[!] Audio Validation Error: {e}", file=sys.stderr)
        sys.exit(1)

    allow_ext = not args.mock and config.allow_external_services
    provider = "mock" if args.mock else config.stt_provider
    transcriber = TeluguTranscriber(provider_name=provider, allow_external=allow_ext)

    print(f"[*] Processing audio file: {audio_path.name} ({len(audio)/sr_rate:.2f}s, {sr_rate}Hz)...")
    print(f"[*] Transcribing using '{provider}' provider...")
    trans_res = transcriber.transcribe(audio_path)

    print("\n" + "=" * 50)
    print("📝 TRANSCRIPTION RESULT (తెలుగు)")
    print("=" * 50)
    print(f"Telugu Text : {trans_res.telugu_text if trans_res.telugu_text else '[No speech detected]'}")
    print(f"Confidence  : {trans_res.confidence:.1%}")
    print(f"Duration    : {trans_res.duration_sec:.2f}s")
    if trans_res.warning_notes:
        print(f"Notes       : {trans_res.warning_notes}")

    eng_trans = ""
    if args.translate and trans_res.telugu_text:
        trans_prov = "mock" if args.mock else config.translation_provider
        translator = TeluguTranslator(provider_name=trans_prov, allow_external=allow_ext)
        t_res = translator.translate(trans_res.telugu_text)
        eng_trans = t_res.translated_text
        print("\n" + "=" * 50)
        print("🌐 ENGLISH TRANSLATION")
        print("=" * 50)
        print(f"English     : {eng_trans}")

    # History update
    history_mgr = HistoryManager(config.history_file_csv)
    history_mgr.add_record(
        telugu_transcript=trans_res.telugu_text,
        english_translation=eng_trans,
        audio_source=f"file ({audio_path.name})",
        duration_sec=trans_res.duration_sec,
        confidence=trans_res.confidence,
        stt_provider=trans_res.provider,
        translation_provider="deep_translator" if (args.translate and not args.mock) else "none",
        has_code_switching=trans_res.has_code_switching,
        warning_notes=trans_res.warning_notes or "",
    )

    if args.plot:
        vis = AudioVisualizer()
        plot_out = config.plots_dir / f"{audio_path.stem}_analysis.png"
        vis.plot_audio_dashboard(audio, sample_rate=sr_rate, save_path=plot_out)
        print(f"[✓] Audio diagnostic plot saved to: {plot_out}")


def cmd_visualize_audio(args, config: Config):
    """Generate audio signal analysis plots using Matplotlib and Seaborn."""
    audio_path = Path(args.file)
    processor = AudioProcessor(sample_rate=config.sample_rate)
    try:
        audio, sr_rate = processor.load_wav(audio_path)
    except AudioValidationError as e:
        print(f"[!] Validation Error: {e}", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else config.plots_dir / f"{audio_path.stem}_analysis.png"
    vis = AudioVisualizer()
    vis.plot_audio_dashboard(audio, sample_rate=sr_rate, save_path=output)
    print(f"[✓] Audio feature dashboard plotted and saved to: {output}")


def cmd_visualize_stats(args, config: Config):
    """Generate session history analytics plots using Seaborn and Matplotlib."""
    history_mgr = HistoryManager(config.history_file_csv)
    df = history_mgr.get_all()
    if df.empty:
        print("[!] History is empty. Seeding sample records for demonstration...")
        seed_synthetic_history(history_mgr)
        df = history_mgr.get_all()

    output = Path(args.output) if args.output else config.plots_dir / "history_usage_analytics.png"
    vis = HistoryVisualizer()
    vis.plot_usage_dashboard(df, save_path=output)
    print(f"[✓] Usage statistics dashboard plotted and saved to: {output}")


def cmd_history(args, config: Config):
    """Inspect, export, search, or purge transcription history."""
    history_mgr = HistoryManager(config.history_file_csv)

    if args.clear:
        deleted = history_mgr.clear_all_history()
        print(f"[✓] Privacy Action: Cleared all {deleted} records from local history.")
        return

    if args.export_csv:
        out = history_mgr.export_to_csv(args.export_csv)
        print(f"[✓] History successfully exported to CSV: {out}")
        return

    if args.export_txt:
        out = history_mgr.export_to_txt(args.export_txt)
        print(f"[✓] Human-readable report exported to TXT: {out}")
        return

    if args.search:
        results = history_mgr.search(args.search)
        print(f"\n[*] Found {len(results)} matches for query '{args.search}':")
        for _, r in results.iterrows():
            print(f"- [{r['timestamp']}] Telugu: {r['telugu_transcript']} | English: {r['english_translation']}")
        return

    # Default: List summary and recent records
    stats = history_mgr.get_summary_statistics()
    print_banner()
    print("📊 SESSION USAGE SUMMARY:")
    print(f"Total Sessions       : {stats['total_records']}")
    print(f"Total Audio Duration : {stats['total_duration_sec']}s (Avg: {stats['avg_duration_sec']}s)")
    print(f"Average Confidence   : {stats['avg_confidence']:.1%}")
    print(f"Code-Switching Rate  : {stats['code_switching_pct']}% ({stats['code_switching_count']} sessions)")
    print(f"Unique Telugu Words  : {stats['unique_words_telugu']}")
    print("-" * 75)

    recent = history_mgr.get_recent(args.limit or 5)
    if recent.empty:
        print("No history records yet. Try recording or transcribing an audio file.")
    else:
        print(f"🕒 MOST RECENT {len(recent)} RECORDS:")
        for _, r in recent.iterrows():
            print(f"ID: {r['id']} | {r['timestamp']} | Dur: {r['duration_sec']}s | Conf: {r['confidence']:.0%}")
            print(f"  Telugu (తెలుగు) : {r['telugu_transcript']}")
            if r['english_translation']:
                print(f"  English        : {r['english_translation']}")
            if r['warning_notes'] and str(r['warning_notes']).strip() and str(r['warning_notes']).lower() != 'nan':
                print(f"  Notes          : {r['warning_notes']}")
            print("-" * 40)


def cmd_demo(args, config: Config):
    """Run an automated synthetic demonstration of the entire pipeline."""
    print_banner()
    print_accuracy_disclaimer()
    print("[*] Generating synthetic Telugu speech sample files...")
    samples = generate_all_samples(config)
    print(f"[✓] Generated {len(samples)} synthetic WAV files in: {config.samples_dir}")

    print("\n[*] Seeding synthetic session history...")
    history_mgr = HistoryManager(config.history_file_csv)
    seed_synthetic_history(history_mgr)

    # Transcribe first sample
    test_sample = samples[0]
    print(f"\n[*] Running offline transcription & translation on sample: {test_sample.name}")
    transcriber = TeluguTranscriber(provider_name="mock", allow_external=False)
    translator = TeluguTranslator(provider_name="mock", allow_external=False)

    trans_res = transcriber.transcribe(test_sample)
    t_res = translator.translate(trans_res.telugu_text)

    print("\n" + "=" * 50)
    print("DEMO TRANSCRIPTION & TRANSLATION OUTPUT")
    print("=" * 50)
    print(f"Sample File : {test_sample.name}")
    print(f"Telugu Text : {trans_res.telugu_text}")
    print(f"English     : {t_res.translated_text}")
    print(f"Confidence  : {trans_res.confidence:.1%}")

    print("\n[*] Generating demonstration diagnostic plots with Matplotlib and Seaborn...")
    processor = AudioProcessor(sample_rate=config.sample_rate)
    audio, sr = processor.load_wav(test_sample)

    audio_vis = AudioVisualizer()
    plot_audio = config.plots_dir / "demo_audio_analysis.png"
    audio_vis.plot_audio_dashboard(audio, sample_rate=sr, save_path=plot_audio)
    print(f"[✓] Audio Analysis Plot saved: {plot_audio}")

    hist_vis = HistoryVisualizer()
    plot_stats = config.plots_dir / "demo_history_analytics.png"
    hist_vis.plot_usage_dashboard(history_mgr.get_all(), save_path=plot_stats)
    print(f"[✓] History Usage Analytics Plot saved: {plot_stats}")

    print("\n✨ Demo completed successfully! All pipeline stages verified.")


def main():
    """Main CLI entrypoint."""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--mock", action="store_true", help="Force offline mock mode (no external network calls).")
    common_parser.add_argument("--quiet", action="store_true", help="Suppress introductory disclaimers.")

    parser = argparse.ArgumentParser(
        description="Telugu Voice-to-Text Translator: Acoustic analysis, transcription, and translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: record
    rec_parser = subparsers.add_parser("record", parents=[common_parser], help="Record from microphone and transcribe.")
    rec_parser.add_argument("-d", "--duration", type=float, default=5.0, help="Duration in seconds (default: 5.0)")
    rec_parser.add_argument("-t", "--translate", action="store_true", default=True, help="Translate transcript to English.")
    rec_parser.add_argument("-p", "--plot", action="store_true", help="Generate acoustic feature plot.")

    # Command: transcribe
    trans_parser = subparsers.add_parser("transcribe", parents=[common_parser], help="Transcribe a WAV audio file.")
    trans_parser.add_argument("file", type=str, help="Path to the WAV audio file.")
    trans_parser.add_argument("-t", "--translate", action="store_true", default=True, help="Translate transcript to English.")
    trans_parser.add_argument("-p", "--plot", action="store_true", help="Generate acoustic feature plot.")

    # Command: visualize-audio
    va_parser = subparsers.add_parser("visualize-audio", parents=[common_parser], help="Plot audio waveform, RMS, and spectrogram.")
    va_parser.add_argument("file", type=str, help="Path to WAV audio file.")
    va_parser.add_argument("-o", "--output", type=str, help="Output PNG path.")

    # Command: visualize-stats
    vs_parser = subparsers.add_parser("visualize-stats", parents=[common_parser], help="Plot history analytics with Seaborn.")
    vs_parser.add_argument("-o", "--output", type=str, help="Output PNG path.")

    # Command: history
    hist_parser = subparsers.add_parser("history", parents=[common_parser], help="Manage session history and privacy.")
    hist_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of records to show.")
    hist_parser.add_argument("--export-csv", type=str, help="Export history to specified CSV path.")
    hist_parser.add_argument("--export-txt", type=str, help="Export human-readable TXT report.")
    hist_parser.add_argument("--search", type=str, help="Search history by keyword.")
    hist_parser.add_argument("--clear", action="store_true", help="Delete all stored history (Privacy action).")

    # Command: demo
    subparsers.add_parser("demo", parents=[common_parser], help="Generate synthetic samples, test full pipeline, and save plots.")

    # Command: privacy
    subparsers.add_parser("privacy", parents=[common_parser], help="Display privacy policy and data governance information.")

    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.mock:
        config.allow_external_services = False

    if args.command == "record":
        cmd_record(args, config)
    elif args.command == "transcribe":
        cmd_transcribe(args, config)
    elif args.command == "visualize-audio":
        cmd_visualize_audio(args, config)
    elif args.command == "visualize-stats":
        cmd_visualize_stats(args, config)
    elif args.command == "history":
        cmd_history(args, config)
    elif args.command == "demo":
        cmd_demo(args, config)
    elif args.command == "privacy":
        print_privacy_notice()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
