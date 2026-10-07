from __future__ import annotations

import io
import json
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Barrier, Event, Thread

import numpy as np
import pytest
from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
    RealtimeSessionCreateRequest,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import SpeechStartedEvent
from speech_to_speech.pipeline.messages import (
    PIPELINE_END,
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.openai_compatible_handler import (
    PIPELINE_SAMPLE_RATE,
    HttpTranscriptionOperation,
    HttpTranscriptionResult,
    OpenAICompatibleSTTHandler,
    TranscriptionRequestError,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


class _TranscriptionServer(BaseHTTPRequestHandler):
    received_path = ""
    received_body = b""

    def do_POST(self) -> None:
        type(self).received_path = self.path
        length = int(self.headers["content-length"])
        type(self).received_body = self.rfile.read(length)
        body = json.dumps({"text": "hello", "language": "en"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


def test_http_transcription_operation_uploads_wav_multipart():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="test-model",
            wav_bytes=OpenAICompatibleSTTHandler._encode_wav(np.zeros(160, dtype=np.float32)),
            language="en",
            response_format="json",
            timeout_s=2,
        )
        result = operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result == HttpTranscriptionResult(text="hello", language="en")
    assert _TranscriptionServer.received_path == "/v1/audio/transcriptions"
    assert b'form-data; name="model"' in _TranscriptionServer.received_body
    assert b"test-model" in _TranscriptionServer.received_body
    assert b'filename="audio.wav"' in _TranscriptionServer.received_body
    assert b"RIFF" in _TranscriptionServer.received_body


def test_http_transcription_operation_can_select_model_by_language():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model=None,
            wav_bytes=b"RIFF-test-wave",
            language="en-US",
            response_format="json",
            timeout_s=2,
        )
        operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert b'form-data; name="model"' not in _TranscriptionServer.received_body
    assert b'form-data; name="language"' in _TranscriptionServer.received_body
    assert b"en-US" in _TranscriptionServer.received_body


def test_http_transcription_operation_uses_gpt_transcribe_language_contract():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="gpt-transcribe",
            wav_bytes=b"RIFF-test-wave",
            language="fr",
            response_format="json",
            timeout_s=2,
        )
        operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert b'form-data; name="languages[]"' in _TranscriptionServer.received_body
    assert b'form-data; name="language"' not in _TranscriptionServer.received_body
    assert b"fr" in _TranscriptionServer.received_body


def test_http_transcription_operation_parses_gpt_transcribe_languages():
    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="gpt-transcribe",
        wav_bytes=b"RIFF-test-wave",
        language=None,
        response_format="json",
        timeout_s=2,
    )

    result = operation._parse_response(
        json.dumps({"text": "bonjour", "languages": [{"code": "fr"}]}).encode(),
        "application/json",
    )

    assert result == HttpTranscriptionResult(text="bonjour", language="fr")


def test_http_transcription_operation_parses_plain_text():
    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language="en",
        response_format="text",
        timeout_s=2,
    )

    result = operation._parse_response(b" hello world\n", "text/plain; charset=utf-8")

    assert result == HttpTranscriptionResult(text="hello world", language="en")


def test_openai_stt_encodes_mono_pcm16_16khz_wav():
    encoded = OpenAICompatibleSTTHandler._encode_wav(np.array([-1.0, 0.0, 1.0], dtype=np.float32))

    with wave.open(io.BytesIO(encoded), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 3


class _FakeOperation:
    results: list[HttpTranscriptionResult] = []
    error: Exception | None = None
    instances: list[_FakeOperation] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)

    def cancel(self, reason="superseded"):
        self.cancel_reason = reason

    def run(self, cancel_check=lambda: False):
        if type(self).error is not None:
            raise type(self).error
        return type(self).results.pop(0)


def _handler(
    monkeypatch,
    *,
    tracker: SpeculativeTurnTracker | None = None,
    **setup_overrides,
) -> OpenAICompatibleSTTHandler:
    _FakeOperation.results = [HttpTranscriptionResult(text="")]
    _FakeOperation.error = None
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)
    handler = OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"speculative_turns": tracker, "boundary_mode": "text", **setup_overrides},
    )
    _FakeOperation.results = []
    return handler


def _audio(mode: str = "final", *, revision: int = 0, samples: int = 160) -> VADAudio:
    return VADAudio(
        audio=np.zeros(samples, dtype=np.float32),
        mode=mode,
        turn_id="turn-1",
        turn_revision=revision,
    )


def _run_final(handler: OpenAICompatibleSTTHandler, source: VADAudio | None = None) -> list:
    assert list(handler.process(source if source is not None else _audio())) == []
    thread = handler._final_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    return outputs


def _run_progressive(handler: OpenAICompatibleSTTHandler, source: VADAudio | None = None) -> list[PartialTranscription]:
    assert list(handler.process(source if source is not None else _audio("progressive"))) == []
    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()

    outputs = []
    while not handler.queue_out.empty():
        output = handler.queue_out.get_nowait()
        assert isinstance(output, PartialTranscription)
        outputs.append(output)
    return outputs


def test_openai_stt_warmup_uses_configured_operation_before_readiness(monkeypatch):
    handler = _handler(
        monkeypatch,
        base_url="https://transcription.example/v1/",
        api_key="endpoint-secret",
        model="test-model",
        language="en",
        response_format="json",
        timeout=2,
    )

    assert len(_FakeOperation.instances) == 1
    operation = _FakeOperation.instances[0].kwargs
    assert operation["endpoint_url"] == "https://transcription.example/v1/audio/transcriptions"
    assert operation["api_key"] == "endpoint-secret"
    assert operation["model"] == "test-model"
    assert operation["language"] == "en"
    assert operation["response_format"] == "json"
    assert operation["timeout_s"] == 2
    with wave.open(io.BytesIO(operation["wav_bytes"]), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == PIPELINE_SAMPLE_RATE
        assert wav.getnframes() == PIPELINE_SAMPLE_RATE
    assert handler.queue_out.empty()


def test_openai_stt_warmup_failure_prevents_handler_construction(monkeypatch):
    _FakeOperation.results = []
    _FakeOperation.error = TranscriptionRequestError("transcription server returned HTTP 404")
    _FakeOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)

    with pytest.raises(TranscriptionRequestError, match="transcription server returned HTTP 404"):
        OpenAICompatibleSTTHandler(
            Event(),
            queue_in=Queue(),
            queue_out=Queue(),
            setup_kwargs={"model": "missing-model"},
        )


def test_openai_stt_returns_final_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [HttpTranscriptionResult(text="hello", language="en")]

    outputs = _run_final(handler)

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "hello"
    assert outputs[0].language_code == "en"
    assert _FakeOperation.instances[-1].kwargs["endpoint_url"].endswith("/v1/audio/transcriptions")
    assert _FakeOperation.instances[-1].kwargs["wav_bytes"].startswith(b"RIFF")


def test_first_turn_uses_session_selected_language(monkeypatch):
    service = RealtimeService()
    conn_id = service.register()
    update = SessionUpdateEvent.model_validate(
        {
            "type": "session.update",
            "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": "es"}}}},
        }
    )
    assert service.handle_session_update(conn_id, update) is None

    handler = _handler(monkeypatch, language="en")
    _FakeOperation.results = [HttpTranscriptionResult(text="hola", language="es")]
    source = _audio()
    source.runtime_config = service._state(conn_id).runtime_config
    assert list(handler.process(source)) == []
    assert handler._final_thread is not None
    handler._final_thread.join(timeout=1)

    assert _FakeOperation.instances[-1].kwargs["language"] == "es"


def test_explicit_auto_removes_setup_language_from_stt_request(monkeypatch):
    handler = _handler(monkeypatch, language="en")
    _FakeOperation.results = [HttpTranscriptionResult(text="hola", language="es")]
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime",
            audio={"input": {"transcription": {"language": "auto"}}},
        )
    )
    source = _audio()
    source.runtime_config = config

    assert list(handler.process(source)) == []
    assert handler._final_thread is not None
    handler._final_thread.join(timeout=1)

    assert _FakeOperation.instances[-1].kwargs["language"] is None
    assert handler.language == "en"


def test_mid_turn_session_update_applies_to_pending_stt_request(monkeypatch):
    handler = _handler(monkeypatch, language="en")
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "es"}}})
    )
    config.session.audio.input.transcription.language = "de"
    _FakeOperation.results = [HttpTranscriptionResult(text="hallo", language="de")]
    first = _audio()
    first.runtime_config = config

    assert list(handler.process(first)) == []
    assert handler._final_thread is not None
    handler._final_thread.join(timeout=1)

    assert _FakeOperation.instances[-1].kwargs["language"] == "de"


def test_open_sessions_and_reused_worker_keep_selections_isolated(monkeypatch):
    service = RealtimeService()
    first_conn = service.register()
    second_conn = service.register()
    handler = _handler(monkeypatch, language="en")
    turn_number = 0

    def select(conn_id: str, language: str) -> None:
        update = SessionUpdateEvent.model_validate(
            {
                "type": "session.update",
                "session": {"type": "realtime", "audio": {"input": {"transcription": {"language": language}}}},
            }
        )
        assert service.handle_session_update(conn_id, update) is None
        assert service.build_session_updated(conn_id).session.audio.input.transcription.language == language

    def transcribe(conn_id: str) -> None:
        nonlocal turn_number
        turn_number += 1
        source = _audio()
        source.turn_id = f"turn-{turn_number}"
        source.runtime_config = service._state(conn_id).runtime_config
        _FakeOperation.results.append(HttpTranscriptionResult(text="hello", language="en"))
        assert list(handler.process(source)) == []
        assert handler._final_thread is not None
        handler._final_thread.join(timeout=1)
        assert not handler._final_thread.is_alive()

    select(first_conn, "auto")
    select(second_conn, "fr")
    transcribe(first_conn)
    transcribe(second_conn)
    select(first_conn, "es")
    transcribe(first_conn)
    select(first_conn, "de")
    transcribe(first_conn)
    select(first_conn, "auto")
    transcribe(first_conn)
    service.unregister(first_conn)
    next_conn = service.register()
    transcribe(next_conn)

    assert [operation.kwargs["language"] for operation in _FakeOperation.instances[1:]] == [
        None,
        "fr",
        "es",
        "de",
        None,
        "en",
    ]
    assert handler.language == "en"


def test_remote_progressive_hypotheses_remain_cumulative(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
    ]

    first = _run_progressive(handler)
    second = _run_progressive(handler)

    assert first == [PartialTranscription(text="hello", turn_id="turn-1", turn_revision=0)]
    assert second == [PartialTranscription(text="hello world", turn_id="turn-1", turn_revision=0)]


def test_remote_progressive_hypothesis_corrections_reach_the_router(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello there"),
        HttpTranscriptionResult(text="hello their"),
    ]

    assert _run_progressive(handler) == [PartialTranscription(text="hello there", turn_id="turn-1", turn_revision=0)]
    assert _run_progressive(handler) == [PartialTranscription(text="hello their", turn_id="turn-1", turn_revision=0)]


def test_remote_progressive_hypotheses_emit_realtime_deltas(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello"),
        HttpTranscriptionResult(text="hello world"),
        HttpTranscriptionResult(text="hello world again"),
        HttpTranscriptionResult(text="hello world again today"),
    ]
    text_output_queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=text_output_queue)
    service = RealtimeService()
    conn_id = service.register()
    service.dispatch_pipeline_event(
        conn_id,
        SpeechStartedEvent(turn_id="turn-1", turn_revision=0),
    )

    wire_events = []
    for _ in range(4):
        for partial in _run_progressive(handler):
            assert list(notifier.process(partial)) == []
            wire_events.extend(service.dispatch_pipeline_event(conn_id, text_output_queue.get_nowait()))

    assert all(isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent) for event in wire_events)
    assert [event.delta for event in wire_events] == ["hello", " world"]
    service.unregister(conn_id)


def test_final_transport_failure_does_not_create_a_transcription(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    outputs = _run_final(handler)

    assert len(outputs) == 1
    assert isinstance(outputs[0], TranscriptionFailure)
    assert outputs[0].message == "transcription request timed out"
    assert outputs[0].turn_id == "turn-1"


def test_progressive_transport_failure_is_discarded(monkeypatch):
    handler = _handler(monkeypatch)
    _FakeOperation.error = TranscriptionRequestError("transcription request timed out")

    assert _run_progressive(handler) == []


def test_final_request_does_not_wait_for_in_flight_progressive(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()
    final_started = Event()

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="partial")

    class _FinalOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            final_started.set()
            return HttpTranscriptionResult(text="final", language="en")

    operations = iter([_BlockingProgressiveOperation(), _FinalOperation()])
    monkeypatch.setattr(handler, "_make_operation", lambda _audio: next(operations))
    handler_thread = Thread(target=handler.run, daemon=True)
    handler_thread.start()

    try:
        handler.queue_in.put(_audio("progressive"))
        assert progressive_started.wait(timeout=1)
        handler.queue_in.put(_audio())

        assert final_started.wait(timeout=1)
        output = handler.queue_out.get(timeout=1)
        assert isinstance(output, Transcription)
        assert output.text == "final"
        assert output.language_code == "en"
        assert output.turn_id == "turn-1"
        assert output.turn_revision == 0
        assert not release_progressive.is_set()

        release_progressive.set()
        thread = handler._progressive_thread
        assert thread is not None
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert handler.queue_out.empty()
    finally:
        release_progressive.set()
        handler.stop_event.set()
        handler.queue_in.put(PIPELINE_END)
        handler_thread.join(timeout=1)
    assert not handler_thread.is_alive()


def test_final_requests_from_pipelines_using_the_same_endpoint_can_overlap(monkeypatch):
    handlers = [_handler(monkeypatch, api_key="shared-endpoint-key") for _ in range(2)]
    both_requests_started = Barrier(2)

    class _ConcurrentOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            both_requests_started.wait(timeout=2)
            return HttpTranscriptionResult(text="final", language="en")

    for handler in handlers:
        monkeypatch.setattr(handler, "_make_operation", lambda _audio: _ConcurrentOperation())

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lambda handler=handler: _run_final(handler)) for handler in handlers]
            outputs = [future.result(timeout=3) for future in futures]
        for output in outputs:
            assert len(output) == 1
            assert isinstance(output[0], Transcription)
            assert output[0].text == "final"
    finally:
        for handler in handlers:
            handler.cleanup()


def test_pending_progressive_requests_keep_only_the_latest_window(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()
    dispatched_samples = []

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="partial")

    def make_operation(_audio):
        dispatched_samples.append(len(_audio))
        return _BlockingProgressiveOperation()

    monkeypatch.setattr(handler, "_make_operation", make_operation)

    assert list(handler.process(_audio("progressive"))) == []
    assert progressive_started.wait(timeout=1)
    assert list(handler.process(_audio("progressive", samples=320))) == []
    assert list(handler.process(_audio("progressive", samples=480))) == []
    assert dispatched_samples == [160]

    release_progressive.set()
    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert dispatched_samples == [160, 480]
    assert handler.queue_out.qsize() == 2
    assert handler.queue_out.get_nowait() == PartialTranscription(
        text="partial",
        turn_id="turn-1",
        turn_revision=0,
    )


def test_session_end_suppresses_in_flight_progressive_result(monkeypatch):
    handler = _handler(monkeypatch)
    progressive_started = Event()
    release_progressive = Event()

    class _BlockingProgressiveOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            progressive_started.set()
            assert release_progressive.wait(timeout=2)
            return HttpTranscriptionResult(text="old session")

    monkeypatch.setattr(handler, "_make_operation", lambda _audio: _BlockingProgressiveOperation())

    assert list(handler.process(_audio("progressive"))) == []
    assert progressive_started.wait(timeout=1)
    handler.on_session_end()
    release_progressive.set()

    thread = handler._progressive_thread
    assert thread is not None
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert handler.queue_out.empty()


@pytest.mark.parametrize("superseded_by", ["final", "new_revision", "session_end", "shutdown", "cleanup"])
@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_obsolete_progressive_request_is_not_sent_before_worker_starts(monkeypatch, superseded_by):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)
    worker_started = Event()
    release_worker = Event()
    relevance_checked = Event()
    cancellation_done = Event()
    run_request = handler._run_request
    cancel_request = handler._cancel_request

    def delayed_worker(request):
        if request.source.mode == "progressive":
            worker_started.set()
            release_worker.wait()
        run_request(request)
        if request.source.mode == "progressive":
            relevance_checked.set()

    def observe_cancel(request, reason):
        cancel_request(request, reason)
        if reason == "shutdown":
            cancellation_done.set()

    monkeypatch.setattr(handler, "_run_request", delayed_worker)
    monkeypatch.setattr(handler, "_cancel_request", observe_cancel)
    _FakeOperation.results = [HttpTranscriptionResult(text="final")]
    _FakeOperation.instances = []
    thread = None
    cleanup_thread = None
    try:
        assert list(handler.process(_audio("progressive"))) == []
        thread = handler._progressive_thread
        assert thread is not None
        assert worker_started.wait(timeout=1)

        if superseded_by == "final":
            outputs = _run_final(handler)
            assert len(outputs) == 1
            assert isinstance(outputs[0], Transcription)
            assert outputs[0].text == "final"
        elif superseded_by == "new_revision":
            tracker.observe("turn-1", 1)
        elif superseded_by == "session_end":
            handler.on_session_end()
        elif superseded_by == "shutdown":
            handler.stop_event.set()
        else:
            cleanup_thread = Thread(target=handler.cleanup, daemon=True)
            cleanup_thread.start()
            assert cancellation_done.wait(timeout=1)

        release_worker.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert relevance_checked.is_set()
        if cleanup_thread is not None:
            cleanup_thread.join(timeout=1)
            assert not cleanup_thread.is_alive()
        assert len(_FakeOperation.instances) == (1 if superseded_by == "final" else 0)
        assert handler.queue_out.empty()
    finally:
        release_worker.set()
        if thread is not None:
            thread.join(timeout=1)
        if cleanup_thread is not None:
            cleanup_thread.join(timeout=1)
        handler.cleanup()


def test_stale_revision_is_dropped_after_request(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _handler(monkeypatch, tracker=tracker)

    class _ReopeningOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            tracker.observe("turn-1", 1)
            return HttpTranscriptionResult(text="stale")

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _ReopeningOperation)

    assert _run_final(handler) == []


def test_openai_api_key_is_not_sent_to_other_endpoints(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "official-secret")

    local_handler = _handler(monkeypatch, base_url="http://localhost:8000/v1")
    official_handler = _handler(monkeypatch, base_url="https://api.openai.com/v1/")
    explicit_handler = _handler(
        monkeypatch,
        base_url="https://transcription.example/v1",
        api_key="endpoint-secret",
    )

    assert local_handler.api_key is None
    assert official_handler.api_key == "official-secret"
    assert explicit_handler.api_key == "endpoint-secret"


@pytest.mark.parametrize(
    ("prefix", "tail", "expected", "matched"),
    [
        ("Start HELLO, WORLD!", "hello world again", "Start hello world again", True),
        ("你好世界", "世界和平", "你好世界和平", True),
        ("the old boundary", "a revised boundary", "the old boundary a revised boundary", False),
        ("yes", "yes again", "yes yes again", False),
        ("go go go go", "go go go go", "go go go go go go go go", False),
        ("say it say it say it", "say it say it again", "say it say it say it say it say it again", False),
        (
            "in the season of the year, when.",
            "Of season of the year, with other words",
            "in the season of the year, with other words",
            True,
        ),
        (
            "Mason’s exquisite idles are as",
            "Its exquisite idles are as good as ever",
            "Mason’s exquisite idles are as good as ever",
            True,
        ),
        (
            "keep these three words",
            "one two extra these three words continued",
            "keep these three words one two extra these three words continued",
            False,
        ),
        ("", "new words", "new words", True),
        ("keep these words", "", "keep these words", True),
    ],
)
def test_window_reconciliation_retains_unmatched_speech(prefix, tail, expected, matched):
    from speech_to_speech.STT.audio_windows import reconcile_overlap

    assert reconcile_overlap(prefix, tail) == (expected, matched)


def test_bounded_requests_reuse_prefix_across_final_and_reopened_revision(monkeypatch):
    handler = _handler(monkeypatch, window_seconds=1, overlap_seconds=0.25)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [
        HttpTranscriptionResult(text="first hello world"),
        HttpTranscriptionResult(text="hello world partial"),
    ]
    partial = _run_progressive(handler, _audio("progressive", samples=24000))
    assert partial[0].text == "first hello world partial"
    assert len(_FakeOperation.instances) == 2

    _FakeOperation.results = [HttpTranscriptionResult(text="hello world final")]
    final = _run_final(handler, _audio(samples=24000))
    assert final[0].text == "first hello world final"
    assert len(_FakeOperation.instances) == 3

    _FakeOperation.results = [
        HttpTranscriptionResult(text="hello world final extra words"),
        HttpTranscriptionResult(text="extra words continued"),
    ]
    reopened = _run_final(handler, _audio(revision=1, samples=40000))
    assert reopened[0].text == "first hello world final extra words continued"
    assert len(_FakeOperation.instances) == 5
    for operation in _FakeOperation.instances:
        with wave.open(io.BytesIO(operation.kwargs["wav_bytes"]), "rb") as wav:
            assert wav.getnframes() <= PIPELINE_SAMPLE_RATE
    assert not _FakeOperation.results


@pytest.mark.parametrize("change", ["audio", "session", "language"])
def test_completed_windows_are_invalidated_when_input_context_changes(monkeypatch, change):
    handler = _handler(monkeypatch, window_seconds=1, overlap_seconds=0.25)
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "es"}}})
    )
    first = _audio("progressive", samples=24000)
    first.runtime_config = config
    _FakeOperation.results = [HttpTranscriptionResult(text="old first"), HttpTranscriptionResult(text="old last")]
    _run_progressive(handler, first)
    source = _audio("progressive", samples=24000)
    source.runtime_config = config
    if change == "audio":
        source.audio[0] = 0.5
    elif change == "session":
        handler.on_session_end()
    else:
        config.session.audio.input.transcription.language = "de"
    _FakeOperation.instances.clear()
    _FakeOperation.results = [HttpTranscriptionResult(text="new first"), HttpTranscriptionResult(text="new last")]

    outputs = _run_progressive(handler, source)

    assert outputs[0].text == "new first new last"
    assert len(_FakeOperation.instances) == 2
    if change == "language":
        assert all(operation.kwargs["language"] == "de" for operation in _FakeOperation.instances)


def test_cancelled_window_does_not_launch_more_requests_or_cache_result(monkeypatch):
    handler = _handler(monkeypatch, window_seconds=1, overlap_seconds=0.25)
    started = Event()
    release = Event()

    class _BlockedWindowOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            started.set()
            assert release.wait(timeout=2)
            return HttpTranscriptionResult(text="cancelled text")

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _BlockedWindowOperation)
    _BlockedWindowOperation.instances = []
    assert list(handler.process(_audio("progressive", samples=40000))) == []
    try:
        assert started.wait(timeout=1)
        handler.on_session_end()
    finally:
        release.set()
        assert handler._progressive_thread is not None
        handler._progressive_thread.join(timeout=1)
    assert not handler._progressive_thread.is_alive()
    assert len(_BlockedWindowOperation.instances) == 1
    assert handler.queue_out.empty()

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _FakeOperation)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [HttpTranscriptionResult(text="fresh first"), HttpTranscriptionResult(text="fresh last")]
    outputs = _run_progressive(handler, _audio("progressive", samples=24000))
    assert outputs[0].text == "fresh first fresh last"
    assert len(_FakeOperation.instances) == 2


def test_zero_audio_overlap_keeps_repeated_words(monkeypatch):
    handler = _handler(monkeypatch, window_seconds=1, overlap_seconds=0)
    _FakeOperation.results = [HttpTranscriptionResult(text="yes yes"), HttpTranscriptionResult(text="yes yes")]

    outputs = _run_final(handler, _audio(samples=24000))

    assert outputs[0].text == "yes yes yes yes"


@pytest.mark.parametrize(
    ("window_seconds", "overlap_seconds"),
    [(float("nan"), 0), (float("inf"), 0), (-1, 0), (1, -1), (1, float("nan")), (1, 1), (0.00001, 0)],
)
def test_invalid_audio_window_settings_fail_before_warmup(monkeypatch, window_seconds, overlap_seconds):
    with pytest.raises(ValueError):
        _handler(monkeypatch, window_seconds=window_seconds, overlap_seconds=overlap_seconds)
    assert not _FakeOperation.instances


def test_audio_window_cli_arguments():
    from transformers import HfArgumentParser

    from speech_to_speech.arguments_classes.openai_stt_arguments import OpenAICompatibleSTTHandlerArguments

    parser = HfArgumentParser(OpenAICompatibleSTTHandlerArguments)
    defaults = parser.parse_args_into_dataclasses([])[0]
    assert defaults.openai_stt_window_seconds == 0
    assert defaults.openai_stt_overlap_seconds == 2
    configured = parser.parse_args_into_dataclasses(
        ["--openai_stt_window_seconds", "30", "--openai_stt_overlap_seconds", "3"]
    )[0]
    assert configured.openai_stt_window_seconds == 30
    assert configured.openai_stt_overlap_seconds == 3


@pytest.mark.parametrize(
    ("status", "body", "hint"),
    [
        (400, "Maximum allowed duration exceeded. secret-credential", "audio duration limit exceeded"),
        (400, "VLLM_MAX_AUDIO_DECODE_BYTES exceeded. secret-credential", "decoded audio limit exceeded"),
        (413, "secret-credential", "upload size limit exceeded"),
        (400, "secret-credential", None),
    ],
)
def test_http_audio_limit_errors_give_safe_window_hint(monkeypatch, status, body, hint):
    import httpx

    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    monkeypatch.setattr(stt_module.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    operation = HttpTranscriptionOperation(
        endpoint_url="http://transcription.example/v1/audio/transcriptions",
        api_key=None,
        model="Qwen/Qwen3-ASR-0.6B",
        wav_bytes=b"RIFF-test-wave",
        language=None,
        response_format="json",
        timeout_s=2,
    )

    with pytest.raises(TranscriptionRequestError) as error:
        operation.run()

    message = str(error.value)
    assert f"HTTP {status}" in message
    assert "secret-credential" not in message
    if hint is not None:
        assert hint in message
        assert "openai_stt_window_seconds" in message
    else:
        assert message == f"transcription server returned HTTP {status}"


def test_language_hint_is_fixed_for_batch_and_new_revision_uses_update(monkeypatch):
    handler = _handler(monkeypatch, window_seconds=1, overlap_seconds=0.25)
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "es"}}})
    )

    class _UpdatingLanguageOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            result = super().run(cancel_check)
            config.session.audio.input.transcription.language = "de"
            return result

    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", _UpdatingLanguageOperation)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [HttpTranscriptionResult(text="hola mundo"), HttpTranscriptionResult(text="mundo nuevo")]
    source = _audio("progressive", samples=24000)
    source.runtime_config = config

    outputs = _run_progressive(handler, source)

    assert outputs[0].text == "hola mundo mundo nuevo"
    assert len(_FakeOperation.instances) == 2
    assert all(operation.kwargs["language"] == "es" for operation in _FakeOperation.instances)

    _FakeOperation.instances.clear()
    _FakeOperation.results = [HttpTranscriptionResult(text="hallo welt"), HttpTranscriptionResult(text="welt neu")]
    revised = _audio(revision=1, samples=24000)
    revised.runtime_config = config

    outputs = _run_final(handler, revised)

    assert outputs[0].text == "hallo welt welt neu"
    assert len(_FakeOperation.instances) == 2
    assert all(operation.kwargs["language"] == "de" for operation in _FakeOperation.instances)


def test_bounded_runtime_auto_does_not_restore_setup_language(monkeypatch):
    handler = _handler(monkeypatch, language="en", window_seconds=1, overlap_seconds=0.25)
    source = _audio(samples=24000)
    source.runtime_config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "auto"}}})
    )
    _FakeOperation.results = [HttpTranscriptionResult(text="first words"), HttpTranscriptionResult(text="last words")]

    result = _run_final(handler, source)

    assert result[0].language_code is None
    assert all(operation.kwargs["language"] is None for operation in _FakeOperation.instances[1:])


def _timed_repetition_audio(seconds, *, mode="final", revision=0):
    source = _audio(mode, revision=revision)
    # Encode a source second in each PCM value so the fake backend knows which
    # physical audio it received. Every second contains one spoken "very".
    source.audio = np.repeat(np.arange(1000, 1000 + seconds, dtype=np.int16), 16000)
    return source


def _install_timed_repetition_backend(monkeypatch):
    from speech_to_speech.STT.word_alignment import WordTiming

    class TimedOperation(_FakeOperation):
        def run(self, cancel_check=lambda: False):
            with wave.open(io.BytesIO(self.kwargs["wav_bytes"]), "rb") as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            start = int(pcm[0]) - 1000
            seconds = len(pcm) // 16000
            self.start = start
            self.seconds = seconds
            return HttpTranscriptionResult(
                text="very " * seconds,
                language="en",
                words=tuple(WordTiming("very", i + 0.1, i + 0.4) for i in range(seconds)),
            )

    TimedOperation.instances = []
    monkeypatch.setattr(stt_module, "HttpTranscriptionOperation", TimedOperation)
    return TimedOperation


@pytest.mark.parametrize("seconds", [30, 120, 600])
def test_aligned_windows_preserve_every_repetition_in_long_speech(monkeypatch, seconds):
    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=30, overlap_seconds=4)
    backend = _install_timed_repetition_backend(monkeypatch)
    outputs = _run_final(handler, _timed_repetition_audio(seconds))
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text.split() == ["very"] * seconds
    assert all(operation.seconds <= 30 for operation in backend.instances)
    assert len(backend.instances) == max(1, (seconds - 4 + 25) // 26)


@pytest.mark.parametrize("change", ["reopen", "audio", "session", "language"])
def test_aligned_window_cache_reuses_only_unchanged_context(monkeypatch, change):
    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=30, overlap_seconds=4)
    backend = _install_timed_repetition_backend(monkeypatch)
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "en"}}})
    )
    first = _timed_repetition_audio(40, mode="progressive")
    first.runtime_config = config
    assert _run_progressive(handler, first)[0].text.split() == ["very"] * 40
    backend.instances.clear()
    source = _timed_repetition_audio(66, revision=1)
    source.runtime_config = config
    if change == "audio":
        source.audio[0] += 1  # A changed sample invalidates the digest, without changing the word.
    elif change == "session":
        handler.on_session_end()
    elif change == "language":
        config.session.audio.input.transcription.language = "de"
    output = _run_final(handler, source)[0]
    assert isinstance(output, Transcription) and output.text.split() == ["very"] * 66
    assert len(backend.instances) == (2 if change == "reopen" else 3)
    if change == "language":
        assert all(operation.kwargs["language"] == "de" for operation in backend.instances)


def test_cancel_during_word_alignment_discards_result_and_stops_http(monkeypatch):
    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    entered, release = Event(), Event()

    class BlockedAligner:
        def align(self, audio, text, language, *, cancel_check):
            from speech_to_speech.STT.word_alignment import WordTiming

            entered.set()
            assert release.wait(timeout=2)
            return [WordTiming("very", 0.1, 0.4)]

    handler._word_aligner = BlockedAligner()
    _FakeOperation.results = [HttpTranscriptionResult("very", "en")]
    _FakeOperation.instances.clear()
    source = _audio("progressive", samples=16000 * 6)
    source.audio[:] = 0.1
    assert list(handler.process(source)) == []
    try:
        assert entered.wait(timeout=1)
        handler.on_session_end()
    finally:
        release.set()
        handler._progressive_thread.join(timeout=1)
    assert not handler._progressive_thread.is_alive()
    assert len(_FakeOperation.instances) == 1
    assert not handler._aligned_windows
    assert handler.queue_out.empty()


@pytest.mark.parametrize("corruption", ["incomplete", "shift_forward", "shift_backward", "nonfinite"])
def test_invalid_backend_word_metadata_never_falls_back_to_text_join(monkeypatch, corruption):
    from speech_to_speech.STT.word_alignment import WordTiming

    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    _FakeOperation.instances.clear()
    words = tuple(WordTiming(word, i + 0.1, i + 0.4) for i, word in enumerate(("one", "two", "three")))
    if corruption == "incomplete":
        words = words[:1]
    elif corruption == "nonfinite":
        words = (WordTiming("one", float("nan"), 0.4),) + words[1:]
    else:
        shift = 1 if corruption == "shift_forward" else -1
        words = tuple(WordTiming(word.text, word.start + shift, word.end + shift) for word in words)
    _FakeOperation.results = [HttpTranscriptionResult("one two three", "en", words)]
    source = _audio(samples=16000 * 6)
    source.audio[:] = 0.1
    outputs = _run_final(handler, source)
    assert len(outputs) == 1 and isinstance(outputs[0], TranscriptionFailure)
    assert len(_FakeOperation.instances) == 1
    assert not handler._aligned_windows


def test_failed_alignment_join_retries_only_one_bounded_bridge(monkeypatch):
    from speech_to_speech.STT.word_alignment import WordTiming

    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [
        HttpTranscriptionResult("one two", "en", (WordTiming("one", 0.1, 0.4), WordTiming("two", 1.1, 1.4))),
        HttpTranscriptionResult("three four", "en", (WordTiming("three", 0.1, 0.4), WordTiming("four", 1.1, 1.4))),
        HttpTranscriptionResult("five six", "en", (WordTiming("five", 0.1, 0.4), WordTiming("six", 1.1, 1.4))),
    ]
    source = _audio(samples=16000 * 5)
    source.audio[:] = 0.1
    outputs = _run_final(handler, source)
    assert len(outputs) == 1 and isinstance(outputs[0], TranscriptionFailure)
    assert len(_FakeOperation.instances) == 3
    for operation in _FakeOperation.instances:
        with wave.open(io.BytesIO(operation.kwargs["wav_bytes"]), "rb") as wav:
            assert wav.getnframes() <= 3 * 16000


@pytest.mark.parametrize("change_audio", [False, True])
def test_bridge_repair_keeps_source_offsets_correct_for_the_next_window(monkeypatch, change_audio):
    from speech_to_speech.STT.word_alignment import WordTiming

    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=30, overlap_seconds=4)
    backend = _install_timed_repetition_backend(monkeypatch)
    original_run = backend.run

    def inconsistent_boundary(operation, cancel_check=lambda: False):
        result = original_run(operation, cancel_check)
        words = list(result.words)
        # The two decodes disagree only near the cut. The bridge's wider
        # context recognizes those words; later joins must still find offsets.
        if operation.start == 0:
            words[26:] = [WordTiming("left", w.start, w.end) for w in words[26:]]
        elif operation.start == 26:
            words[:4] = [WordTiming("right", w.start, w.end) for w in words[:4]]
        return HttpTranscriptionResult(" ".join(w.text for w in words) + " ", "en", tuple(words))

    monkeypatch.setattr(backend, "run", inconsistent_boundary)
    outputs = _run_final(handler, _timed_repetition_audio(66))
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text.split() == ["very"] * 66
    assert [operation.start for operation in backend.instances] == [0, 26, 18, 52]
    assert all(operation.seconds <= 30 for operation in backend.instances)

    backend.instances.clear()
    reopened = _timed_repetition_audio(92, revision=1)
    if change_audio:
        reopened.audio[100] += 1
    outputs = _run_final(handler, reopened)
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text.split() == ["very"] * 92
    # Verified joins of immutable completed windows should not re-run a bridge.
    expected_starts = [0, 26, 18, 52, 78] if change_audio else [52, 78]
    assert [operation.start for operation in backend.instances] == expected_starts


@pytest.mark.parametrize(
    "language,first,last,expected",
    [
        ("en", "first section", "last section", "first section last section"),
        ("zh", "你好。", "再见。", "你好。再见。"),
        ("ja", "こんにちは。", "さようなら。", "こんにちは。さようなら。"),
        ("yue", "你好。", "再見。", "你好。再見。"),
    ],
)
def test_aligned_windows_cut_at_pause_without_loading_word_aligner(monkeypatch, language, first, last, expected):
    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [
        HttpTranscriptionResult(first, language),
        HttpTranscriptionResult(last, language),
    ]
    source = _audio(samples=16000 * 4)
    source.audio[:] = 0.1
    source.audio[40000:44000] = 0
    outputs = _run_final(handler, source)
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text == expected
    sizes = []
    for operation in _FakeOperation.instances:
        with wave.open(io.BytesIO(operation.kwargs["wav_bytes"]), "rb") as wav:
            sizes.append(wav.getnframes())
    assert len(sizes) == 2 and sum(sizes) == len(source.audio)
    assert 40000 <= sizes[0] <= 44000


def test_concurrent_pipeline_setup_shares_one_aligner_instance(monkeypatch):
    from time import sleep

    constructed = []
    start = Barrier(2)

    class FakeAligner:
        def __init__(self, *, model_name, device):
            constructed.append((model_name, device))
            sleep(0.05)  # Make simultaneous first-use requests overlap.

    monkeypatch.setattr(stt_module, "QwenWordAligner", FakeAligner)
    stt_module._cached_aligner.cache_clear()

    def create():
        start.wait(timeout=1)
        return stt_module._shared_aligner("test-concurrent-aligner", "cpu")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(create) for _ in range(2)]
            first, second = [future.result(timeout=2) for future in futures]
        assert first is second
        assert constructed == [("test-concurrent-aligner", "cpu")]
    finally:
        stt_module._cached_aligner.cache_clear()


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_silent_aligned_tail_never_calls_backend_or_adds_hallucinated_text(monkeypatch, dtype):
    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [
        HttpTranscriptionResult("actual speech", "en"),
        HttpTranscriptionResult("hallucinated words", "en"),
    ]
    source = _audio(samples=16000 * 4)
    source.audio = np.zeros(16000 * 4, dtype=dtype)
    source.audio[:40000] = 0.1 if dtype == np.float32 else 1000
    outputs = _run_final(handler, source)
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text == "actual speech"
    assert len(_FakeOperation.instances) == 1
    assert _FakeOperation.results == [HttpTranscriptionResult("hallucinated words", "en")]


def test_verbose_json_requests_word_timestamps_in_actual_multipart(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        handler = _handler(
            monkeypatch,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            response_format="verbose_json",
        )
        request = _FakeOperation.instances[0].kwargs
        assert request["extra_fields"]["timestamp_granularities[]"] == "word"
        HttpTranscriptionOperation(**request).run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert b'form-data; name="timestamp_granularities[]"' in _TranscriptionServer.received_body
    assert b"word" in _TranscriptionServer.received_body
    assert b"verbose_json" in _TranscriptionServer.received_body
    assert handler.queue_out.empty()


@pytest.mark.parametrize("response_format", ["json", "verbose_json"])
def test_http_transcription_parses_backend_word_timings(response_format):
    from speech_to_speech.STT.word_alignment import WordTiming

    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language="en",
        response_format=response_format,
        timeout_s=2,
    )
    payload = {
        "text": "Hello, world!",
        "words": [{"word": "Hello", "start": 0.1, "end": 0.4}, {"word": "world", "start": 0.5, "end": 0.8}],
    }
    result = operation._parse_response(json.dumps(payload).encode(), "application/json")
    assert result == HttpTranscriptionResult(
        "Hello, world!", "en", (WordTiming("Hello", 0.1, 0.4), WordTiming("world", 0.5, 0.8))
    )


@pytest.mark.parametrize("words", [{}, ["word"], [{"word": "one", "start": 0}], [{"word": 1, "start": 0, "end": 1}]])
def test_http_transcription_rejects_malformed_word_metadata(words):
    operation = HttpTranscriptionOperation(
        endpoint_url="http://127.0.0.1:1/v1/audio/transcriptions",
        api_key=None,
        model="test-model",
        wav_bytes=b"RIFF-test-wave",
        language="en",
        response_format="verbose_json",
        timeout_s=2,
    )
    with pytest.raises(TranscriptionRequestError, match="invalid word timings"):
        operation._parse_response(json.dumps({"text": "one", "words": words}).encode(), "application/json")


def test_full_negative_pcm_amplitude_is_speech_not_silence(monkeypatch):
    from speech_to_speech.STT.word_alignment import WordTiming

    handler = _handler(monkeypatch, boundary_mode="aligned", window_seconds=3, overlap_seconds=1)
    _FakeOperation.instances.clear()
    _FakeOperation.results = [
        HttpTranscriptionResult("one two", "en", (WordTiming("one", 2.1, 2.3), WordTiming("two", 2.4, 2.7))),
        HttpTranscriptionResult(
            "one two three",
            "en",
            (WordTiming("one", 0.1, 0.3), WordTiming("two", 0.4, 0.7), WordTiming("three", 1.1, 1.4)),
        ),
    ]
    source = _audio(samples=16000 * 4)
    source.audio = np.full(16000 * 4, -32768, dtype=np.int16)
    outputs = _run_final(handler, source)
    assert len(outputs) == 1 and isinstance(outputs[0], Transcription)
    assert outputs[0].text == "one two three"
    assert len(_FakeOperation.instances) == 2
