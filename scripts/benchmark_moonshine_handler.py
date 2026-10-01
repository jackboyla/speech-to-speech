"""Benchmark-only Moonshine bridge. Not a production speech-to-speech backend."""

from __future__ import annotations

import hashlib
from pathlib import Path
from queue import Queue

import numpy as np

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription


class MoonshineBenchmarkHandler:
    def __init__(self, model: str, interval: float):
        from moonshine_voice import ModelArch, Transcriber, TranscriptEventListener, get_model_for_language

        arch = {
            "tiny-streaming": ModelArch.TINY_STREAMING,
            "small-streaming": ModelArch.SMALL_STREAMING,
            "medium-streaming": ModelArch.MEDIUM_STREAMING,
        }[model]
        path, arch = get_model_for_language("en", arch)
        self.model_metadata = {
            "path": str(path),
            "architecture": arch.name,
            "runtime_options": {"identify_speakers": "false"},
            "files": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(Path(path).iterdir())
                if p.is_file()
            ],
        }
        self.transcriber = Transcriber(path, arch, update_interval=interval, options={"identify_speakers": "false"})
        self.queue_out = Queue()
        self.stream = None
        self.lines = {}
        self.last_text = ""
        self.final_text = None
        owner = self

        class Listener(TranscriptEventListener):
            def on_line_text_changed(self, event):
                owner.lines[event.line.line_id] = event.line
                text = " ".join(line.text.strip() for line in sorted(owner.lines.values(), key=lambda x: x.start_time))
                if text and text != owner.last_text:
                    owner.last_text = text
                    owner.queue_out.put(PartialTranscription(text=text, turn_id=owner.turn_id, turn_revision=0))

            def on_line_completed(self, event):
                self.on_line_text_changed(event)

            def on_error(self, event):
                owner.error = str(event)

        self.listener = Listener()

    def start_turn(self, turn_id, revision):
        if self.stream is not None:
            self.stream.close()
        self.turn_id = turn_id
        self.lines = {}
        self.last_text = ""
        self.final_text = None
        self.error = None
        self.stream = self.transcriber.create_stream()
        self.stream.add_listener(self.listener)
        self.stream.start()

    def append_audio(self, audio):
        samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
        self.stream.add_audio(samples.tolist(), 16000)

    def commit_boundary(self, turn_id, revision):
        transcript = self.stream.stop()
        if self.error or transcript is None:
            raise RuntimeError(self.error or "Moonshine stop returned no transcript")
        self.final_text = " ".join(line.text.strip() for line in sorted(transcript.lines, key=lambda x: x.start_time))

    def process(self, source):
        if self.final_text is None:
            raise RuntimeError("Moonshine has no committed transcript")
        yield Transcription(text=self.final_text, language_code="en", turn_id=source.turn_id, turn_revision=0)

    def cleanup(self):
        if self.stream is not None:
            self.stream.close()
        self.transcriber.close()
