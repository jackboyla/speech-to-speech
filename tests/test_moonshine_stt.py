from __future__ import annotations

from queue import Queue
from threading import Event
from typing import Any

import numpy as np
import pytest
import torch

from speech_to_speech import s2s_pipeline
from speech_to_speech.backend_registry import BackendSelection, HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.STT import moonshine_handler
from speech_to_speech.STT.moonshine_handler import MoonshineSTTHandler, language_from_model_name


class _FakeInputs(dict):
    def to(self, *args: Any, **kwargs: Any) -> "_FakeInputs":
        return self


class _FakeProcessor:
    def __call__(self, audio: Any, sampling_rate: int, return_tensors: str) -> _FakeInputs:
        assert sampling_rate == 16000
        return _FakeInputs(input_values=torch.zeros((1, len(audio))))

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens
        return " hello world "


class _FakeModel:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        return torch.zeros((1, 4), dtype=torch.long)


class _FakeTransformers:
    """Stands in for the ``transformers`` module inside the handler."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.loaded: list[tuple[str, Any]] = []
        outer = self

        class AutoProcessor:
            @staticmethod
            def from_pretrained(name: str) -> _FakeProcessor:
                outer.loaded.append(("processor", name))
                return _FakeProcessor()

        class AutoModelForSpeechSeq2Seq:
            @staticmethod
            def from_pretrained(name: str, **kwargs: Any) -> _FakeModel:
                outer.loaded.append(("model", kwargs.get("dtype")))
                return outer.model

        self.AutoProcessor = AutoProcessor
        self.AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq


@pytest.fixture
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> _FakeTransformers:
    fake = _FakeTransformers()
    monkeypatch.setattr(moonshine_handler, "transformers", fake)
    monkeypatch.setattr(moonshine_handler.console, "print", lambda *args, **kwargs: None)
    return fake


def _build(argv: list[str]) -> tuple[MoonshineSTTHandler, HandlerContext, BackendSelection]:
    args = parse_arguments(["--stt", "moonshine", "--moonshine_device", "cpu", *argv])
    context = HandlerContext(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        text_output_queue=Queue(),
        should_listen=Event(),
        cancel_scope=CancelScope(),
        speculative_turns=SpeculativeTurnTracker(),
        pipeline_index=0,
        sample_rate=16000,
        enable_live_transcription=False,
        live_transcription_update_interval=0.5,
    )
    handler = create_backend_handler(args.stt_backend, context)
    assert isinstance(handler, MoonshineSTTHandler)
    return handler, context, args.stt_backend


def _vad_audio(mode: str, seconds: float) -> VADAudio:
    return VADAudio(
        audio=np.zeros(int(16000 * seconds), dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=2,
        speech_end_at_s=123.0,
    )


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("moonshine-ai/moonshine-streaming-medium", "en"),
        ("moonshine-ai/moonshine-base", "en"),
        ("moonshine-ai/moonshine-tiny-ja", "ja"),
        ("moonshine-ai/moonshine-streaming-tiny-es", "es"),
        ("moonshine-ai/moonshine-streaming-small-de", "de"),
        ("/models/moonshine-base-zh/", "zh"),
    ],
)
def test_language_from_model_name(model_name: str, expected: str) -> None:
    assert language_from_model_name(model_name) == expected


def test_cli_builds_a_moonshine_handler_that_warms_up(fake_transformers: _FakeTransformers) -> None:
    handler, context, selection = _build(["--moonshine_model_name", "moonshine-ai/moonshine-tiny-ko"])

    assert handler.speculative_turns is context.speculative_turns
    assert ("processor", "moonshine-ai/moonshine-tiny-ko") in fake_transformers.loaded
    assert ("model", torch.float32) in fake_transformers.loaded
    assert handler.start_language == handler.last_language == "ko"
    assert s2s_pipeline._stt_session_languages(selection, handler) == {"ko"}
    assert len(fake_transformers.model.generate_calls) == 1, "setup must run one warmup generation"


def test_transcriptions_cap_new_tokens_by_audio_length(fake_transformers: _FakeTransformers) -> None:
    handler, _, _ = _build(["--moonshine_language", "FR"])
    handler.turn_latency_store = store = TurnLatencyStore()

    partial = list(handler.process(_vad_audio("progressive", seconds=0.1)))
    assert store.get_or_create_for_turn("turn_1", 2).stt_s is None
    final = list(handler.process(_vad_audio("final", seconds=2.0)))
    assert store.get_or_create_for_turn("turn_1", 2).stt_s is not None

    assert final == [
        Transcription(
            text="hello world", language_code="fr", turn_id="turn_1", turn_revision=2, speech_stopped_at_s=123.0
        )
    ]
    assert partial == [PartialTranscription(text="hello world", turn_id="turn_1", turn_revision=2)]
    assert [call["max_length"] for call in fake_transformers.model.generate_calls] == [8, 2, 14]
