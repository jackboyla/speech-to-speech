from __future__ import annotations

import json
from queue import Queue

import numpy as np
import pytest
from scripts.benchmark_phonon import load_manifest, pcm16, run_clip, word_errors

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure


def test_word_error_counts_and_shared_normalization():
    assert word_errors("Don't ASK, your country!", "dont ask your country") == (0, 4)
    assert word_errors("one two three", "one too") == (2, 3)
    assert word_errors("one", "one extra") == (1, 1)


def test_manifest_resolves_paths_and_requires_real_labeled_audio(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "clip", "audio": "clip.wav", "text": "hello"}) + "\n")
    assert load_manifest(manifest)[0]["audio"] == str(audio.resolve())
    manifest.write_text(json.dumps({"audio": "missing.wav", "text": "hello"}) + "\n")
    with pytest.raises(ValueError, match="Missing audio"):
        load_manifest(manifest)
    manifest.write_text(json.dumps({"audio": "clip.wav", "text": "..."}) + "\n")
    with pytest.raises(ValueError, match="nonempty reference"):
        load_manifest(manifest)


class NativeHandler:
    def __init__(self):
        self.queue_out = Queue()
        self.sent = []
        self.started = None
        self.committed = None

    def start_turn(self, turn, revision):
        self.started = (turn, revision)

    def append_audio(self, audio):
        self.sent.append(audio)
        self.queue_out.put(PartialTranscription(text="hello", turn_id=self.started[0], turn_revision=0))

    def commit_boundary(self, turn, revision):
        self.committed = (turn, revision)

    def process(self, source):
        assert source.mode == "final"
        assert self.committed == (source.turn_id, source.turn_revision)
        yield Transcription(text="hello", turn_id=source.turn_id, turn_revision=0)


def test_native_benchmark_paces_and_sends_pcm_once_before_final():
    handler = NativeHandler()
    audio = np.linspace(-1.1, 1.1, 1024, dtype=np.float32)
    result = run_clip(handler, audio, native=True, chunk_ms=16, partial_interval=0.02, timeout=1, turn_id="clip")
    assert b"".join(handler.sent) == pcm16(audio)
    assert len(handler.sent) == 4
    assert result["audio_feed_s"] >= len(audio) / 16000 - 0.002
    assert result["first_partial_s"] is not None
    assert result["final_latency_s"] >= 0
    assert result["transcript"] == "hello"
    assert result["errors"] == []


def test_offline_benchmark_feeds_growing_windows_then_whole_final():
    class Offline:
        queue_out = Queue()
        inputs = []

        def process(self, source):
            self.inputs.append((source.mode, len(source.audio)))
            if source.mode == "progressive":
                yield PartialTranscription(text="hello", turn_id=source.turn_id, turn_revision=0)
            else:
                yield Transcription(text="hello", turn_id=source.turn_id, turn_revision=0)

    handler = Offline()
    result = run_clip(
        handler,
        np.ones(1024, dtype=np.float32),
        native=False,
        chunk_ms=16,
        partial_interval=0.016,
        timeout=1,
        turn_id="clip",
    )
    assert handler.inputs[-1] == ("final", 1024)
    windows = [length for mode, length in handler.inputs if mode == "progressive"]
    assert windows and windows == sorted(windows)
    assert all(length < 1024 for length in windows)
    assert result["transcript"] == "hello"


def test_failed_final_does_not_turn_into_successful_empty_transcript():
    class Failed(NativeHandler):
        def process(self, source):
            yield TranscriptionFailure(message="engine busy", turn_id=source.turn_id, turn_revision=0)

    result = run_clip(
        Failed(),
        np.zeros(256, dtype=np.float32),
        native=True,
        chunk_ms=16,
        partial_interval=0.02,
        timeout=1,
        turn_id="clip",
    )
    assert result["transcript"] is None
    assert result["final_latency_s"] is None
    assert result["errors"] == ["engine busy"]
