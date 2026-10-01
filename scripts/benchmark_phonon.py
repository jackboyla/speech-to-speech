"""Paced, handler-level Phonon / Parakeet comparison. See docs/phonon-benchmark.md."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import resource
import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import perf_counter, process_time
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio

RATE = 16000


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.casefold()).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for row, expected in enumerate(ref, 1):
        current = [row]
        for column, actual in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1], len(ref)


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("audio"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"Manifest line {line_number} needs string audio and text fields")
        if not words(row["text"]):
            raise ValueError(f"Manifest line {line_number} needs a nonempty reference")
        audio = Path(row["audio"])
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            raise ValueError(f"Missing audio at manifest line {line_number}: {audio}")
        rows.append({"id": str(row.get("id", line_number)), "audio": str(audio.resolve()), "text": row["text"]})
    if not rows:
        raise ValueError("Manifest must contain at least one labeled file")
    return rows


def load_audio(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        audio = soxr.resample(audio, rate, RATE)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite samples and must not be empty")
    return audio.astype(np.float32)


def pcm16(audio: np.ndarray) -> bytes:
    return np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ["numpy", "soundfile", "soxr", "websockets", "nano-parakeet", "mlx-audio"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_handler(args: argparse.Namespace):
    if args.backend == "phonon":
        from speech_to_speech.STT.streaming_handler import PhononSTTHandler

        cls = PhononSTTHandler
        kwargs = {"base_url": args.base_url, "api_key": args.api_key, "final_timeout": args.timeout}
    else:
        from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler

        cls = ParakeetTDTSTTHandler
        kwargs = {
            "device": args.device,
            "enable_live_transcription": True,
            "live_transcription_update_interval": args.partial_interval,
        }
    return cls(Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs=kwargs)


def run_clip(
    handler: Any,
    audio: np.ndarray,
    *,
    native: bool,
    chunk_ms: float,
    partial_interval: float,
    timeout: float,
    turn_id: str,
) -> dict[str, Any]:
    """Feed once at capture cadence; keep at most one pending offline window."""
    output = handler.queue_out
    while not output.empty():
        output.get_nowait()
    incoming: Queue[VADAudio | None] = Queue(maxsize=1)
    errors: list[str] = []
    audio_end: list[float] = []
    stop = Event()
    started = perf_counter()
    cpu_started = process_time()
    if native:
        handler.start_turn(turn_id, 0)

    def consume():
        try:
            while not stop.is_set():
                source = incoming.get()
                if source is None:
                    return
                for result in handler.process(source):
                    output.put(result)
                if source.mode == "final":
                    return
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            output.put(TranscriptionFailure(message="benchmark handler failed", turn_id=turn_id, turn_revision=0))

    def enqueue_latest(source: VADAudio):
        # The pipeline skips obsolete growing windows; it never drops the final.
        try:
            incoming.get_nowait()
        except Empty:
            pass
        incoming.put(source)

    def produce():
        chunk_samples = max(1, round(chunk_ms * RATE / 1000))
        next_partial = partial_interval
        try:
            for offset in range(0, len(audio), chunk_samples):
                end = min(len(audio), offset + chunk_samples)
                if stop.wait(max(0, started + end / RATE - perf_counter())):
                    return
                if native:
                    handler.append_audio(pcm16(audio[offset:end]))
                elif end / RATE >= next_partial and end < len(audio):
                    enqueue_latest(VADAudio(audio=audio[:end], mode="progressive", turn_id=turn_id, turn_revision=0))
                    next_partial = end / RATE + partial_interval
            audio_end.append(perf_counter())
            if native:
                handler.commit_boundary(turn_id, 0)
            enqueue_latest(VADAudio(audio=audio, mode="final", turn_id=turn_id, turn_revision=0))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            output.put(TranscriptionFailure(message="benchmark audio feed failed", turn_id=turn_id, turn_revision=0))

    consumer = Thread(target=consume, daemon=True)
    producer = Thread(target=produce, daemon=True)
    consumer.start()
    producer.start()
    first_partial = None
    partial_count = 0
    transcript = None
    finished = None
    try:
        capture_deadline = started + len(audio) / RATE + timeout
        while True:
            deadline = audio_end[0] + timeout if audio_end else capture_deadline
            if perf_counter() >= deadline:
                break
            try:
                result = output.get(timeout=min(0.05, max(0.001, deadline - perf_counter())))
            except Empty:
                continue
            now = perf_counter()
            if isinstance(result, PartialTranscription) and result.text:
                partial_count += 1
                if first_partial is None:
                    first_partial = now - started
            elif isinstance(result, Transcription):
                transcript, finished = result.text, now
                break
            elif isinstance(result, TranscriptionFailure):
                errors.append(result.message)
                break
        if transcript is None and not errors:
            errors.append("benchmark final timeout")
    finally:
        stop.set()
        producer.join(timeout=1)
        if consumer.is_alive():
            enqueue_latest(None)  # type: ignore[arg-type]
        consumer.join(timeout=1)
    if producer.is_alive() or consumer.is_alive():
        # Inference cannot be safely interrupted in a thread. Stop the benchmark;
        # don't start another clip sharing a live decoder.
        raise RuntimeError("benchmark worker did not stop; rerun in a fresh process")
    return {
        "duration_s": len(audio) / RATE,
        "first_partial_s": first_partial,
        "partial_count": partial_count,
        "final_latency_s": finished - audio_end[0] if finished is not None and audio_end else None,
        "audio_feed_s": audio_end[0] - started if audio_end else None,
        "elapsed_s": perf_counter() - started,
        "client_cpu_s": process_time() - cpu_started,
        "transcript": transcript,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["phonon", "parakeet-tdt"], required=True)
    parser.add_argument("--base-url", default="ws://127.0.0.1:18090/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-ms", type=float, default=32)
    parser.add_argument("--partial-interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", choices=["info", "debug"], default="info")
    args = parser.parse_args(argv)
    if (
        any(value <= 0 for value in [args.threads, args.chunk_ms, args.partial_interval, args.timeout])
        or args.warmup < 0
    ):
        parser.error("threads, chunk-ms, partial-interval and timeout must be positive; warmup must be nonnegative")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    rows = load_manifest(args.manifest)
    import torch

    torch.set_num_threads(args.threads)
    report: dict[str, Any] = {
        "backend": args.backend,
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": args.device,
        "threads": args.threads,
        "chunk_ms": args.chunk_ms,
        "partial_interval_s": args.partial_interval,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "accuracy_normalization": "casefold, remove punctuation, split whitespace; word edit distance",
        "package_versions": package_versions(),
        "warmup": [],
        "clips": [],
        "scope": "handler only; no VAD/LLM/TTS; CPU/RSS exclude external Phonon server",
    }
    handler = None
    try:
        started = perf_counter()
        handler = make_handler(args)
        report["handler_startup_s"] = perf_counter() - started
        for index in range(args.warmup):
            report["warmup"].append(
                run_clip(
                    handler,
                    load_audio(rows[0]["audio"]),
                    native=args.backend == "phonon",
                    chunk_ms=args.chunk_ms,
                    partial_interval=args.partial_interval,
                    timeout=args.timeout,
                    turn_id=f"warmup_{index}",
                )
            )
        report["warmup_failures"] = sum(bool(row["errors"]) for row in report["warmup"])
        for index, row in enumerate(rows):
            result = run_clip(
                handler,
                load_audio(row["audio"]),
                native=args.backend == "phonon",
                chunk_ms=args.chunk_ms,
                partial_interval=args.partial_interval,
                timeout=args.timeout,
                turn_id=f"clip_{index}",
            )
            result.update(row)
            if result["transcript"] is not None:
                count, total = word_errors(row["text"], result["transcript"])
                result.update(word_errors=count, reference_words=total, wer=count / total)
            report["clips"].append(result)
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "first_partial_s": result["first_partial_s"],
                        "final_latency_s": result["final_latency_s"],
                        "wer": result.get("wer"),
                        "errors": result["errors"],
                    }
                ),
                flush=True,
            )
        successful = [row for row in report["clips"] if row["transcript"] is not None]
        report["successful_clips"] = len(successful)
        report["failed_clips"] = len(rows) - len(successful)
        total_words = sum(row["reference_words"] for row in successful)
        report["reference_words_total"] = total_words
        report["word_errors_total"] = sum(row["word_errors"] for row in successful)
        report["wer"] = report["word_errors_total"] / total_words if total_words else None
        for key in ["first_partial_s", "final_latency_s"]:
            values = [row[key] for row in successful if row[key] is not None]
            report[f"median_{key}"] = float(np.median(values)) if values else None
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["client_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if handler is not None:
            cleanup = getattr(handler, "cleanup", None)
            if cleanup:
                cleanup()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if report.get("error") or report.get("failed_clips") or report.get("warmup_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
