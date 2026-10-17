"""Tyger turn contracts with only neural inference replaced by deterministic scores."""

from queue import Queue
from threading import Event

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.messages import DuplexAudio, Transcription, TranscriptionFailure
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.tyger_handler import HOP, SR, TygerInputHandler


@pytest.fixture
def handler(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            self.endpoint = self.barge = 0.0
            self.blocks = []
            self.modes = []
            self.resets = 0
            self.error = False

        def reset(self):
            self.resets += 1

        def reset_controller(self):
            pass

        def step(self, user, playback, speaking):
            assert user.shape == playback.shape == (HOP,)
            self.modes.append(speaking)
            result = np.zeros((4, 8), dtype=np.float32)
            result[:, 5], result[:, 7] = self.barge, self.endpoint
            return result

        def transcribe(self, audio):
            if self.error:
                raise RuntimeError("inference failure")
            self.blocks.append(audio.copy())
            return f"segment {len(self.blocks)}"

    monkeypatch.setattr("speech_to_speech.STT.tyger_model.TygerModel", Model)
    return TygerInputHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={"text_output_queue": Queue(), "speculative_turns": SpeculativeTurnTracker()},
    )


def feed(handler, *, value=1000, playing=False, samples=HOP, config=None):
    audio = np.full(samples, value, dtype="<i2").tobytes()
    return list(
        handler.process(
            DuplexAudio(
                audio=audio,
                playback=bytes(len(audio)),
                playback_active=playing,
                runtime_config=config or RuntimeConfig(),
            )
        )
    )


def events(handler):
    result = []
    while not handler.text_output_queue.empty():
        result.append(handler.text_output_queue.get_nowait())
    return result


def test_endpoint_orders_one_input_pair_and_transcription_and_next_turn(handler):
    # Rechunked appends must preserve the same 320 ms model hop.
    assert feed(handler, samples=HOP - 1) == []
    assert feed(handler, samples=1) == []
    assert events(handler) == []
    feed(handler)
    [start] = events(handler)
    assert isinstance(start, SpeechStartedEvent)
    assert start.audio_start_ms == 0
    handler.model.endpoint = 0.95
    [transcript] = feed(handler, value=0)
    [stop] = events(handler)
    assert isinstance(stop, SpeechStoppedEvent)
    assert stop.turn_id == start.turn_id == transcript.turn_id
    assert isinstance(transcript, Transcription)
    assert transcript.text == "segment 1"
    assert len(handler.model.blocks[0]) == 3 * HOP
    handler.model.endpoint = 0
    feed(handler)
    feed(handler)
    [next_start] = events(handler)
    assert next_start.turn_id != start.turn_id
    assert not next_start.reopened


def test_playback_needs_learned_barge_and_preserves_user_onset(handler):
    # Zero-valued playback PCM still represents a speaking interval.
    for _ in range(6):
        feed(handler, playing=True)
    assert events(handler) == []
    handler.model.barge = 0.8
    feed(handler, playing=True)
    [start] = events(handler)
    assert isinstance(start, SpeechStartedEvent)
    assert len(handler._audio) == 2 * SR
    # Client cancellation takes a round trip; do not issue another barge-in.
    feed(handler, playing=True)
    assert events(handler) == []
    assert handler.model.modes[-1] is False
    handler.model.endpoint = 0.95
    [transcript] = feed(handler, playing=False)
    assert transcript.turn_id == start.turn_id
    assert len(handler.model.blocks[-1]) > 2 * SR


def test_long_turn_segments_preserve_all_audio_and_close_failure(handler):
    for _ in range(55):
        feed(handler)
    assert len(handler.model.blocks) == 1
    assert len(handler.model.blocks[0]) == 15 * SR
    handler.model.endpoint = 0.95
    [transcript] = feed(handler, value=0)
    assert transcript.text == "segment 1 segment 2"
    assert sum(len(block) for block in handler.model.blocks) == 56 * HOP
    handler.model.endpoint = 0
    feed(handler)
    feed(handler)
    handler.model.error = True
    handler.model.endpoint = 0.95
    [failure] = feed(handler)
    assert isinstance(failure, TranscriptionFailure)
    assert failure.turn_id == handler.speculative_turns.current_turn()[0]
    assert handler._turn is None


def test_session_reset_discards_partial_hop_and_prior_turn(handler):
    feed(handler)
    feed(handler)
    events(handler)
    feed(handler, samples=111)
    handler.on_session_end()
    assert handler.model.resets == 2
    assert handler.speculative_turns.current_turn() == (None, None)
    feed(handler)
    assert events(handler) == []
    feed(handler)
    [start] = events(handler)
    assert start.audio_start_ms == 0
    assert len(handler._audio) == 2 * HOP


def test_speaker_echo_cannot_start_a_user_turn_after_playback_finishes(handler):
    for _ in range(10):
        feed(handler, playing=True)
    for _ in range(5):
        feed(handler, playing=False, value=0)
    assert events(handler) == []


def test_noninterrupting_speech_keeps_transcribing_during_playback(handler):
    config = RuntimeConfig()
    config.session.audio.input.turn_detection = {"type": "server_vad", "interrupt_response": False}
    handler.model.barge = 0.99
    handler.model.endpoint = 0.99
    result = []
    for _ in range(4):
        result.extend(feed(handler, playing=True, config=config))
    assert any(isinstance(item, Transcription) for item in result)
    assert any(isinstance(item, SpeechStartedEvent) for item in events(handler))
    handler.model.barge = handler.model.endpoint = 0
    for _ in range(3):
        feed(handler, playing=True, config=config)
    assert events(handler) == []
    assert handler.model.modes[-1] is True


def test_user_onset_crossing_playback_end_survives_without_a_yield(handler):
    for _ in range(2):
        feed(handler, playing=True, value=1000)
    for _ in range(2):
        feed(handler, playing=False, value=2000)
    [start] = events(handler)
    assert start.audio_start_ms == 0
    assert len(handler._audio) == 4 * HOP
    assert np.allclose(handler._audio[: 2 * HOP], 1000 / 32768)
