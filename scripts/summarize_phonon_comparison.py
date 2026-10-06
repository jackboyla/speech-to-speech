"""Show paired Phonon/Parakeet reports from the same recorded audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text


def validate_pair(candidate, baseline):
    if candidate.get("backend") != "phonon" or baseline.get("backend") != "parakeet-tdt":
        raise ValueError("Supply Phonon first, then the Parakeet TDT baseline.")
    for field in ["machine", "manifest_sha256", "chunk_ms", "partial_interval_s", "threads", "benchmark_sha256"]:
        if candidate.get(field) is None or candidate[field] != baseline.get(field):
            raise ValueError(f"Reports differ in {field}; rerun with the same audio and settings.")
    if len(candidate.get("warmup", [])) != len(baseline.get("warmup", [])):
        raise ValueError("Warmup counts differ.")
    clips = candidate.get("clips", [])
    other = baseline.get("clips", [])
    if not clips or len(clips) != len(other):
        raise ValueError("Measured clip counts differ or are empty.")
    for a, b in zip(clips, other):
        for field in ["id", "audio_sha256", "text", "duration_s"]:
            if a.get(field) is None or a[field] != b.get(field):
                raise ValueError(f"Clips differ in {field}; use the same saved recording and reference.")
    for report in [candidate, baseline]:
        if report.get("error") or report.get("failed_clips") or report.get("warmup_failures"):
            raise ValueError(f"{report['backend']} has failures; inspect its report before comparing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phonon", type=Path)
    parser.add_argument("parakeet", type=Path)
    args = parser.parse_args()
    console = Console()
    try:
        candidate, baseline = [json.loads(path.read_text()) for path in [args.phonon, args.parakeet]]
        validate_pair(candidate, baseline)
    except (OSError, ValueError) as exc:
        console.print(Text(f"Cannot compare: {exc}", style="bold red"))
        return 1
    table = Table(title="Same recorded speech · paced playback · warmup excluded")
    for heading in ["Backend", "First text (median)", "Final wait (median)", "Word errors", "WER"]:
        table.add_column(heading)
    for label, report in [("Parakeet TDT v3 (baseline)", baseline), ("Phonon-2", candidate)]:

        def seconds(value):
            return "none" if value is None else f"{value:.3f} s"

        table.add_row(
            label,
            seconds(report.get("median_first_partial_s")),
            seconds(report.get("median_final_latency_s")),
            f"{report['word_errors_total']} / {report['reference_words_total']}",
            f"{report['wer']:.1%}" if report.get("wer") is not None else "n/a",
        )
    console.print(table)
    for label, report in [("Baseline", baseline), ("Phonon", candidate)]:
        for clip in report["clips"]:
            console.print(Text(f"{label} [{clip['id']}]: {clip['transcript']}"))
    console.print("First text may change. Final wait starts at capture end; it excludes VAD, LLM and TTS.")
    console.print("One recording is a demo, not a general accuracy or speed benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
