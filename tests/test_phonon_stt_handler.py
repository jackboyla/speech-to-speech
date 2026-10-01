from __future__ import annotations

import json
from queue import Queue
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest
from websockets.sync.server import serve

from speech_to_speech.backend_registry import STT_BACKENDS
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.STT.streaming_handler import PhononSTTHandler


@pytest.fixture
def phonon_server():
    sessions: list[list[Any]] = []
    replies: Queue[list[dict[str, Any]]] = Queue()

    def connection(socket):
        received: list[Any] = []
        sessions.append(received)
        received.append(json.loads(socket.recv()))
        events = replies.get(timeout=2)
        for raw in socket:
            received.append(raw if isinstance(raw, bytes) else json.loads(raw))
            if isinstance(raw, bytes):
                for event in events:
                    socket.send(json.dumps(event))
                events = []
            elif json.loads(raw) == {"type": "end"}:
                socket.send(json.dumps({"type": "done", "text": "The final transcript."}))
                return

    server = serve(connection, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}/v1", sessions, replies
    finally:
        server.shutdown()
        thread.join(timeout=2)


def handler(url, **kwargs):
    return PhononSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"base_url": url, "final_timeout": 1.0, **kwargs},
    )


def final(turn_id="turn_1", revision=0):
    return VADAudio(audio=np.zeros(160, dtype=np.float32), mode="final", turn_id=turn_id, turn_revision=revision)


def test_native_binary_pcm_replacement_partials_segments_and_reconnect(phonon_server):
    url, sessions, replies = phonon_server
    replies.put(
        [
            {"type": "partial", "text": "A wrong"},
            {"type": "partial", "text": "A correct phrase."},
            {"type": "final", "text": "A correct phrase.", "segment": 1},
            {"type": "partial", "text": "Next phrase."},
        ]
    )
    stt = handler(url)
    pcm = np.arange(160, dtype="<i2").tobytes()
    try:
        stt.start_turn("turn_1", 0)
        stt.append_audio(pcm)
        partials = [stt.queue_out.get(timeout=2) for _ in range(4)]
        assert all(isinstance(item, PartialTranscription) for item in partials)
        assert [item.text for item in partials] == [
            "A wrong",
            "A correct phrase.",
            "A correct phrase.",
            "A correct phrase. Next phrase.",
        ]
        assert list(stt.process(final()))[0].text == "The final transcript."
        assert sessions[0] == [
            {"sample_rate": 16000, "format": "pcm_s16le"},
            pcm,
            {"type": "end"},
        ]
        replies.put([])
        stt.start_turn("turn_2", 0)
        stt.append_audio(pcm)
        output = list(stt.process(final("turn_2")))
        assert isinstance(output[0], Transcription)
        assert output[0].language_code == "en"
        assert len(sessions) == 2
    finally:
        stt.cleanup()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "message": "engine busy; secret token"},
        {"type": "partial", "text": 123},
    ],
)
def test_busy_or_invalid_response_fails_without_leaking_server_message(phonon_server, event):
    url, _, replies = phonon_server
    replies.put([event])
    stt = handler(url)
    try:
        stt.start_turn("turn_1", 0)
        stt.append_audio(b"\x00\x00" * 160)
        output = list(stt.process(final()))
        assert isinstance(output[0], TranscriptionFailure)
        assert "secret" not in output[0].message
    finally:
        stt.cleanup()


def test_cancel_and_discard_reconnect_without_reusing_audio(phonon_server):
    url, sessions, replies = phonon_server
    stt = handler(url)
    try:
        for index, action in enumerate([stt.discard_utterance, stt.cancel_session]):
            replies.put([{"type": "partial", "text": "discard me"}])
            stt.start_turn(f"discard_{index}", 0)
            stt.append_audio(b"\x01\x00" * 160)
            stt.queue_out.get(timeout=2)
            action()
        replies.put([])
        stt.start_turn("turn_1", 0)
        stt.append_audio(b"\x02\x00" * 160)
        assert list(stt.process(final()))[0].text == "The final transcript."
        assert len(sessions) == 3
        assert sessions[-1][1] == b"\x02\x00" * 160
    finally:
        stt.cleanup()


def test_reopened_revision_keeps_committed_prefix(phonon_server):
    url, _, replies = phonon_server
    stt = handler(url)
    try:
        replies.put([])
        stt.start_turn("turn_1", 0)
        stt.append_audio(b"\x00\x00" * 160)
        list(stt.process(final()))
        replies.put([{"type": "partial", "text": "Continuation"}])
        stt.start_turn("turn_1", 1)
        stt.append_audio(b"\x00\x00" * 160)
        assert stt.queue_out.get(timeout=2).text == "The final transcript. Continuation"
        assert list(stt.process(final(revision=1)))[0].text == "The final transcript. The final transcript."
    finally:
        stt.cleanup()


def test_phonon_is_registered_as_native_streaming():
    spec = STT_BACKENDS["phonon"]
    assert spec.capabilities.streams_audio_chunks
    config = spec.normalize(spec.config_type())
    assert config["base_url"] == "ws://localhost:8000/v1"
    assert "model" not in config  # The server selects its model, not the wire config.


def test_full_endpoint_and_authentication_header():
    calls = []

    def connect(url, **kwargs):
        calls.append((url, kwargs))
        raise ConnectionError()

    stt = handler("https://example.com/v1/audio/stream?tenant=test", api_key="test-key", connect_factory=connect)
    try:
        stt.start_turn("turn_1", 0)
        stt.append_audio(b"\x00\x00")
        assert isinstance(list(stt.process(final()))[0], TranscriptionFailure)
        assert calls[0][0] == "wss://example.com/v1/audio/stream?tenant=test"
        assert calls[0][1]["headers"] == {"Authorization": "Bearer test-key"}
    finally:
        stt.cleanup()


@pytest.mark.parametrize("kwargs", [{"audio_sample_rate": 24000}, {"language": "fr"}])
def test_phonon_rejects_unsupported_audio_and_language(kwargs):
    with pytest.raises(ValueError):
        handler("ws://example.com/v1", **kwargs)


def test_no_done_times_out_and_does_not_publish_final():
    from queue import Empty

    class Socket:
        def send(self, _message):
            pass

        def recv(self, timeout=None):
            raise Empty()

        def close(self):
            pass

    stt = PhononSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={
            "base_url": "ws://example.com/v1",
            "final_timeout": 0.1,
            "connect_factory": lambda *a, **k: Socket(),
        },
    )
    try:
        stt.start_turn("turn_1", 0)
        stt.append_audio(b"\x00\x00")
        result = list(stt.process(final()))
        assert len(result) == 1
        assert isinstance(result[0], TranscriptionFailure)
        assert "timed out" in result[0].message
    finally:
        stt.cleanup()
