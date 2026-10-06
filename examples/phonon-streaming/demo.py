"""Show microphone or paced-file STT through the repo's Phonon handler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from time import monotonic, sleep

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from speech_to_speech.pipeline.messages import PartialTranscription, TranscriptionFailure, VADAudio
from speech_to_speech.STT.streaming_handler import PhononSTTHandler

RATE = 16000
CHUNK = 512


def microphone(seconds, device):
    import sounddevice as sd

    pending = Queue(maxsize=32)
    failed = Event()

    def callback(data, frames, timing, status):
        if status:
            failed.set()
        try:
            pending.put_nowait(bytes(data))
        except Full:
            failed.set()

    with sd.RawInputStream(
        samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, device=device, callback=callback
    ):
        deadline = monotonic() + seconds
        while monotonic() < deadline:
            if failed.is_set():
                raise RuntimeError("Microphone overflow or input error; try a shorter recording.")
            try:
                yield pending.get(timeout=0.05)
            except Empty:
                yield None


def recording(path):
    import soxr

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        audio = soxr.resample(audio, rate, RATE)
    started = monotonic()
    for offset in range(0, len(audio), CHUNK):
        end = min(offset + CHUNK, len(audio))
        sleep(max(0, started + end / RATE - monotonic()))
        yield (np.clip(audio[offset:end], -1, 1) * 32767).astype("<i2").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="ws://127.0.0.1:18090/v1")
    parser.add_argument("--seconds", type=float, default=20, help="Microphone capture length (1–120 seconds)")
    parser.add_argument("--device", help="Microphone name or index; use --list-devices to find it")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--audio", type=Path, help="Replay a recording at capture speed instead of using the microphone"
    )
    parser.add_argument("--save-audio", type=Path, help="Save the audio sent to the handler as a WAV")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if not 1 <= args.seconds <= 120:
        parser.error("--seconds must be between 1 and 120")
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    output = Queue()
    handler = PhononSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=output,
        setup_kwargs={"base_url": args.base_url, "api_key": os.getenv("PHONON_API_KEY"), "final_timeout": 60.0},
    )
    console = Console()
    sent = bytearray()
    text = "Waiting for speech…"
    status = "REPLAY" if args.audio else "MICROPHONE"
    started = monotonic()

    def panel():
        content = Text(f"{status}  •  {monotonic() - started:.1f} seconds\n\n", style="cyan")
        content.append(text, style="bold white")
        return Panel(content, title="Phonon-2 · native streaming STT", border_style="cyan")

    try:
        handler.start_turn("demo", 0)
        console.print("Speak now. Capture ends automatically; Ctrl+C cancels.")
        with Live(panel(), console=console, refresh_per_second=10) as live:
            source = recording(args.audio) if args.audio else microphone(args.seconds, device)
            try:
                for chunk in source:
                    if chunk:
                        handler.append_audio(chunk)
                        sent.extend(chunk)
                    while True:
                        try:
                            event = output.get_nowait()
                        except Empty:
                            break
                        if isinstance(event, PartialTranscription):
                            text = event.text
                    live.update(panel())
            finally:
                source.close()
            if not sent:
                raise RuntimeError("No audio captured.")
            status = "FINISHING"
            live.update(panel(), refresh=True)
            handler.commit_boundary("demo", 0)
            result = next(
                handler.process(
                    VADAudio(
                        audio=np.frombuffer(sent, dtype="<i2").astype(np.float32) / 32768,
                        mode="final",
                        turn_id="demo",
                        turn_revision=0,
                    )
                ),
                None,
            )
            if result is None or isinstance(result, TranscriptionFailure):
                raise RuntimeError(result.message if result is not None else "No final transcript received.")
            text = result.text
            if not text.strip():
                raise RuntimeError("No speech recognized; check your microphone and speak during capture.")
            status = "FINAL"
            live.update(panel(), refresh=True)
        console.print()
        console.print(Text(f"Final transcript: {text}", style="bold green"))
        return 0
    except KeyboardInterrupt:
        console.print("Cancelled.")
        return 130
    except Exception as exc:
        console.print(Text(f"Demo failed: {exc}", style="bold red"))
        return 1
    finally:
        handler.cleanup()
        if args.save_audio and sent:
            args.save_audio.parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.save_audio, np.frombuffer(sent, dtype="<i2"), RATE, format="WAV", subtype="PCM_16")
            console.print(f"Audio saved: {args.save_audio}")


if __name__ == "__main__":
    raise SystemExit(main())
