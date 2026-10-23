"""Exercise the packaged Windows audio path without loading inference models."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

from speech_to_speech.api.openai_realtime.audio_client import (
    PlaybackBuffer,
    RealtimeAudioClientConfig,
    _AudioWorker,
    _open_audio_streams,
)

RATE = 16000
SOURCE_REVISION = "64c6bb64c0ef947488e816e6c12edbd7d3f55c4e"


def level(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(audio.astype(float) ** 2) / 32768**2 + 1e-12))


def record_phase(results: Path, name: str, *, aec: bool, played: np.ndarray) -> dict:
    config = RealtimeAudioClientConfig(echo_cancellation=aec)
    worker = _AudioWorker(config) if aec else None
    queue = worker.mic_queue if worker else Queue(maxsize=128)
    playback = worker.playback if worker else PlaybackBuffer(RATE)
    context = worker if worker else _open_audio_streams(config, playback, queue)
    captured = bytearray()
    duration = len(played) / RATE
    with context:
        # Receiving the first callback confirms the spawned worker opened devices.
        ready_deadline = time.monotonic() + 20
        while True:
            if worker and not worker._process.is_alive():
                asyncio.run(worker.wait_for_failure())
            try:
                queue.get(timeout=0.1)
                break
            except Empty:
                if time.monotonic() >= ready_deadline:
                    raise RuntimeError(
                        "No microphone samples arrived. Check Windows microphone permission and input device."
                    )
        playback.append(played.tobytes())
        playback.finish()
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if worker and not worker._process.is_alive():
                asyncio.run(worker.wait_for_failure())
            try:
                captured.extend(queue.get(timeout=0.1))
            except Empty:
                pass
    if worker and worker._process.exitcode != 0:
        raise RuntimeError(f"{name}: audio worker did not shut down cleanly (exit {worker._process.exitcode})")
    audio = np.frombuffer(bytes(captured), dtype=np.int16)
    sf.write(results / f"{name}.wav", audio, RATE, subtype="PCM_16")
    if len(audio) < (duration - 1) * RATE:
        raise RuntimeError(f"{name}: received only {len(audio) / RATE:.1f}s of microphone audio in {duration:.1f}s")
    # Skip initial device buffering and filter convergence when comparing phases.
    measured = audio[4 * RATE : -RATE] if duration >= 10 else audio
    return {
        "aec": aec,
        "seconds": len(audio) / RATE,
        "nonzero_samples": int(np.count_nonzero(audio)),
        "level_dbfs": level(measured),
        "clipped_fraction": float(np.mean(np.abs(audio.astype(float)) >= 32760)),
        "file": f"{name}.wav",
    }


def physical_check(results: Path, report: dict) -> None:
    print("\nUse the laptop's BUILT-IN speakers and microphone, with no headphones.", flush=True)
    print("Set speaker volume around 50%. Keep the laptop still in a quiet room.", flush=True)
    print("This records the microphone locally into this results folder. Nothing is uploaded.", flush=True)
    print("If Windows asks, allow desktop apps to use the microphone.", flush=True)
    input("Press Enter when ready: ")
    reference = Path(__file__).resolve().parents[1] / "src/speech_to_speech/TTS/ref_audio.wav"
    speech, source_rate = sf.read(reference, dtype="float32", always_2d=True)
    speech = soxr.resample(speech.mean(axis=1), source_rate, RATE)
    speech *= 0.35 / max(float(np.max(np.abs(speech))), 1e-6)
    played = (np.tile(speech, 2)[: 16 * RATE] * 32767).astype(np.int16)
    sf.write(results / "speaker-reference.wav", played, RATE, subtype="PCM_16")
    phases = report["phases"] = {}
    print("\n1/4: Stay silent for 4 seconds while we measure background sound.", flush=True)
    phases["background"] = record_phase(results, "background", aec=False, played=np.zeros(4 * RATE, dtype=np.int16))
    print("2/4: Stay silent during 16 seconds of speaker playback. Cancellation is OFF.", flush=True)
    phases["echo-off"] = record_phase(results, "echo-off", aec=False, played=played)
    print("3/4: Stay silent during the same playback. Cancellation is ON in the PR's audio process.", flush=True)
    phases["echo-on"] = record_phase(results, "echo-on", aec=True, played=played)
    print("4/4: Now SPEAK OVER the playback. Repeat: 'Please stop, I would like to ask another question.'", flush=True)
    phases["voice-overlap"] = record_phase(results, "voice-overlap", aec=True, played=played)
    heard = input("Did you hear the recorded speaker voice in ALL three playback phases? [y/N] ").strip().lower()
    report["speaker_playback_confirmed"] = heard in {"y", "yes"}
    background = phases["background"]["level_dbfs"]
    off = phases["echo-off"]["level_dbfs"]
    on = phases["echo-on"]["level_dbfs"]
    reduction = off - on
    report["measured_reduction_db"] = reduction
    report["echo_above_background_db"] = off - background
    invalid = []
    if not report["speaker_playback_confirmed"]:
        invalid.append(
            "Speaker playback was not confirmed in every phase; reduced microphone level could mean missing output."
        )
    if phases["echo-off"]["nonzero_samples"] == 0:
        invalid.append("Microphone recording is silent; capture was not verified.")
    if off - background < 6:
        invalid.append("Speaker echo is too close to background sound to judge cancellation.")
    if any(phase["clipped_fraction"] > 0.01 for phase in phases.values()):
        invalid.append("Microphone clipping makes the comparison unreliable; lower speaker volume.")
    report["measurement_issues"] = invalid
    report["acoustic_result"] = "inconclusive" if invalid else "recorded; inspect measurements and voice-overlap.wav"
    print(f"\nMeasured off/on level reduction: {reduction:.1f} dB.", flush=True)
    for issue in invalid:
        print(f"INCONCLUSIVE: {issue}", flush=True)
    print("Listen to voice-overlap.wav to check that your own words survived.", flush=True)
    print("These recordings do not verify server speech detection or a full local model pipeline.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--no-audio", action="store_true", help="Record environment only, for hosted CI without audio hardware."
    )
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.results / "audio.log", encoding="utf-8"), logging.StreamHandler()],
    )
    report = {
        "source_base_revision": SOURCE_REVISION,
        "bundle_revision": os.environ.get("AEC_BUNDLE_REVISION"),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "openai", "jsonschema", "pywebrtc-audio", "sounddevice", "soundfile", "soxr")
        },
        "scope": "Client-only environment; full project installation and model inference are not exercised.",
        "physical_audio_tested": False,
    }
    exit_code = 0
    try:
        report["portaudio"] = sd.get_portaudio_version()
        report["devices"] = list(sd.query_devices())
        report["host_apis"] = list(sd.query_hostapis())
        report["default_devices"] = list(sd.default.device)
        if args.no_audio:
            print("Hosted check finished. Physical mic/speaker testing was skipped.", flush=True)
        else:
            print(f"Microphone: {sd.query_devices(kind='input')['name']}", flush=True)
            print(f"Speakers: {sd.query_devices(kind='output')['name']}", flush=True)
            physical_check(args.results, report)
            report["physical_audio_tested"] = True
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        logging.error("Audio check stopped", exc_info=True)
        (args.results / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 1
    finally:
        (args.results / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
