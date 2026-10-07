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
        setup_kwargs={"speculative_turns": tracker, **setup_overrides},
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
