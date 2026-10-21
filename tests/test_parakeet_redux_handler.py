import sys
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from speech_to_speech.backend_registry import HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import _stt_session_languages, parse_arguments
from speech_to_speech.STT import parakeet_redux_handler as redux


@pytest.fixture
def photon(monkeypatch):
    model = Mock()
    model.transcribe.return_value = {"text": " A sufficiently long sentence in French. "}
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "moondream", SimpleNamespace(photon=factory))
    monkeypatch.setattr(redux, "warm_language_detector", lambda _languages: object())
    monkeypatch.setattr(redux, "detect_language_from_text", lambda _text, _detector: "fr")
    return factory, model


def build_handler():
    args = parse_arguments(["--stt", "parakeet-redux", "--parakeet_redux_device", "cpu"])
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
        enable_live_transcription=True,
        live_transcription_update_interval=0.5,
    )
    return args.stt_backend, context, create_backend_handler(args.stt_backend, context)


@pytest.mark.parametrize("mode", ["progressive", "final"])
def test_registered_handler_preserves_audio_and_turn_contract(photon, mode):
    factory, model = photon
    selection, context, handler = build_handler()
    assert selection.spec.required_extra == "parakeet-redux"
    assert handler.speculative_turns is context.speculative_turns
    assert _stt_session_languages(selection, handler) == set()
    factory.assert_called_once_with(
        "moondream/parakeet-redux",
        device="cpu",
        cpu_threads=torch.get_num_threads(),
        single_pass_batch_capacity=1,
    )
    audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    item = VADAudio(audio=audio, mode=mode, turn_id="turn", turn_revision=3, speech_end_at_s=12.0)
    outputs = list(handler.process(item))
    assert len(outputs) == 1
    output = outputs[0]
    assert output.text == "A sufficiently long sentence in French."
    assert (output.turn_id, output.turn_revision) == ("turn", 3)
    assert model.transcribe.call_args.kwargs["sample_rate"] == 16000
    np.testing.assert_array_equal(model.transcribe.call_args.kwargs["audio"], audio)
    if mode == "progressive":
        assert isinstance(output, PartialTranscription)
    else:
        assert isinstance(output, Transcription)
        assert output.language_code == "fr-auto"
        assert output.speech_stopped_at_s == 12.0
    handler.cleanup()
    handler.cleanup()
    model.close.assert_called_once()


def test_empty_final_is_emitted_without_inventing_a_language(photon, monkeypatch):
    _, model = photon
    _, _, handler = build_handler()
    model.transcribe.return_value = {"text": ""}
    monkeypatch.setattr(redux, "detect_language_from_text", lambda _text, _detector: None)
    (output,) = handler.process(VADAudio(audio=np.zeros(1600, dtype=np.float32), mode="final", turn_id="empty"))
    assert isinstance(output, Transcription)
    assert output.text == ""
    assert output.language_code is None
    assert output.turn_id == "empty"
    handler.cleanup()


def test_warmup_failure_releases_photon_client(photon):
    _, model = photon
    model.transcribe.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"):
        build_handler()
    model.close.assert_called_once()


def test_auto_uses_cpu_on_apple_silicon(photon, monkeypatch):
    factory, _ = photon
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    handler = redux.ParakeetReduxSTTHandler(Event(), Queue(), Queue())
    assert factory.call_args.kwargs["device"] == "cpu"
    handler.cleanup()
