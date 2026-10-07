#!/usr/bin/env python3
"""Measure real HTTP uploads using synthetic audio and a controlled ASR server.

This measures request bounds, work, and transcript assembly, not ASR accuracy.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import logging
import sys
import time
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech_to_speech.pipeline.messages import Transcription, TranscriptionFailure, VADAudio  # noqa: E402
from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler  # noqa: E402

SAMPLE_RATE = 16000


class ControlledServer(ThreadingHTTPServer):
    def __init__(self, max_seconds: float, latency_per_second: float):
        super().__init__(("127.0.0.1", 0), TranscriptionEndpoint)
        self.max_seconds = max_seconds
        self.latency_per_second = latency_per_second
        self.requests: list[dict] = []


class TranscriptionEndpoint(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["content-length"]))
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {self.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode() + raw
        )
        audio = next(part.get_payload(decode=True) for part in message.iter_parts() if part.get_filename())
        with wave.open(io.BytesIO(audio)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        # Each synthetic second has a unique sample value. Consecutive duplicates
        # collapse, while tokens shared by overlapping windows remain identifiable.
        edges = np.flatnonzero(np.r_[True, samples[1:] != samples[:-1]]) if len(samples) else np.array([], dtype=int)
        tokens = samples[edges]
        words = [
            {
                "word": f"word{int(token) - 1000:04d}",
                "start": int(edge) / SAMPLE_RATE,
                "end": int(edges[index + 1] if index + 1 < len(edges) else len(samples)) / SAMPLE_RATE,
            }
            for index, (edge, token) in enumerate(zip(edges, tokens))
            if token > 1000
        ]
        text = " ".join(word["word"] for word in words)
        rejected = self.server.max_seconds > 0 and duration > self.server.max_seconds
        self.server.requests.append({"duration_seconds": duration, "rejected": rejected, "text": text})
        time.sleep(duration * self.server.latency_per_second)
        body = json.dumps(
            {"error": "audio duration limit"} if rejected else {"text": text, "language": "en", "words": words}
        ).encode()
        self.send_response(413 if rejected else 200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def await_idle(handler, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with handler._request_lock:
            if not handler._workers_running:
                return
        time.sleep(0.002)
    raise TimeoutError("STT worker did not finish")


def drain(queue):
    outputs = []
    while True:
        try:
            outputs.append(queue.get_nowait())
        except Empty:
            return outputs


def run_case(server, seconds, window_seconds, args):
    setup = {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "controlled-asr", "timeout": 120}
    if "window_seconds" in inspect.signature(OpenAICompatibleSTTHandler.setup).parameters:
        setup.update(window_seconds=window_seconds, overlap_seconds=args.overlap_seconds)
        if "boundary_mode" in inspect.signature(OpenAICompatibleSTTHandler.setup).parameters:
            setup["boundary_mode"] = args.boundary_mode
    elif window_seconds:
        raise RuntimeError("This checkout does not yet support bounded windows; run with --windows 0 for the baseline")
    outputs = Queue()
    handler = OpenAICompatibleSTTHandler(Event(), Queue(), outputs, setup_kwargs=setup)
    server.requests.clear()  # Exclude the endpoint warmup from work metrics.
    audio = np.repeat(np.arange(1001, 1001 + seconds, dtype=np.int16), SAMPLE_RATE)
    started = time.perf_counter()
    try:
        for end in [] if args.final_only else range(args.step_seconds, seconds + 1, args.step_seconds):
            list(
                handler.process(
                    VADAudio(audio=audio[: end * SAMPLE_RATE], mode="progressive", turn_id="benchmark", turn_revision=0)
                )
            )
            await_idle(handler)
            drain(outputs)
        final_started = time.perf_counter()
        list(handler.process(VADAudio(audio=audio, mode="final", turn_id="benchmark", turn_revision=0)))
        await_idle(handler)
        final_latency = time.perf_counter() - final_started
        final = next((item for item in drain(outputs) if isinstance(item, (Transcription, TranscriptionFailure))), None)
        expected = " ".join(f"word{token:04d}" for token in range(1, seconds + 1))
        actual = getattr(final, "text", "")
        requests = list(server.requests)
        reopened = None
        if args.reopen_seconds:
            reopened_audio = np.repeat(
                np.arange(1001, 1001 + seconds + args.reopen_seconds, dtype=np.int16), SAMPLE_RATE
            )
            reopen_started = time.perf_counter()
            list(handler.process(VADAudio(audio=reopened_audio, mode="final", turn_id="benchmark", turn_revision=1)))
            await_idle(handler)
            reopened_final = next(
                (item for item in drain(outputs) if isinstance(item, (Transcription, TranscriptionFailure))), None
            )
            reopened_expected = " ".join(f"word{token:04d}" for token in range(1, seconds + args.reopen_seconds + 1))
            reopened = {
                "speech_seconds": seconds + args.reopen_seconds,
                "final_succeeded": isinstance(reopened_final, Transcription),
                "synthetic_transcript_exact": getattr(reopened_final, "text", "") == reopened_expected,
                "latency_seconds": time.perf_counter() - reopen_started,
                "requests": list(server.requests[len(requests) :]),
            }
        return {
            "speech_seconds": seconds,
            "final_only": args.final_only,
            "reopened_revision": reopened,
            "window_seconds": window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "request_count": len(requests),
            "submitted_audio_seconds": sum(item["duration_seconds"] for item in requests),
            "largest_request_seconds": max(item["duration_seconds"] for item in requests),
            "rejected_requests": sum(item["rejected"] for item in requests),
            "final_succeeded": isinstance(final, Transcription),
            "synthetic_transcript_exact": actual == expected,
            "final_token_count": len(actual.split()),
            "expected_token_count": seconds,
            "final_latency_seconds": final_latency,
            "runtime_seconds": time.perf_counter() - started,
            "requests": requests,
        }
    finally:
        handler.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", nargs="+", type=int, default=[30, 120, 600])
    parser.add_argument("--windows", nargs="+", type=float, default=[0, 30])
    parser.add_argument("--overlap-seconds", type=float, default=4)
    parser.add_argument("--boundary-mode", choices=["aligned", "text"], default="aligned")
    parser.add_argument("--step-seconds", type=int, default=5)
    parser.add_argument("--final-only", action="store_true", help="Skip progressive requests; exercise final catch-up")
    parser.add_argument(
        "--reopen-seconds", type=int, default=0, help="Append audio and submit a reopened final revision"
    )
    parser.add_argument("--max-request-seconds", type=float, default=60, help="0 disables the controlled server limit")
    parser.add_argument(
        "--latency-per-audio-second", type=float, default=0.0002, help="Synthetic backend delay; not model latency"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step_seconds <= 0 or any(seconds <= 0 or seconds > 31767 for seconds in args.durations):
        parser.error("positive step and durations in 1..31767 are required")
    if args.reopen_seconds < 0 or max(args.durations) + args.reopen_seconds > 31767:
        parser.error("reopened duration must be in 1..31767")
    if not args.final_only and any(seconds % args.step_seconds for seconds in args.durations):
        parser.error("durations must be multiples of the progressive update step")
    logging.basicConfig(level=logging.CRITICAL)
    server = ControlledServer(args.max_request_seconds, args.latency_per_audio_second)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        results = []
        for seconds in args.durations:
            for window in args.windows:
                result = run_case(server, seconds, window, args)
                results.append(result)
                summary = {key: value for key, value in result.items() if key != "requests"}
                if summary.get("reopened_revision"):
                    summary["reopened_revision"] = {
                        key: value for key, value in summary["reopened_revision"].items() if key != "requests"
                    }
                print(json.dumps(summary), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "backend": "controlled synthetic HTTP ASR; no model or WER measurement",
                    "max_request_seconds": args.max_request_seconds,
                    "latency_per_audio_second": args.latency_per_audio_second,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    main()
