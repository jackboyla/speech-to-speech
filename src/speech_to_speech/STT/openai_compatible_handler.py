from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import wave
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Event, Lock, RLock, Thread, current_thread
from time import perf_counter
from typing import Any, Iterator, Literal

import anyio
import httpx
import numpy as np

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.aligned_audio_windows import (
    AlignedAudioWindow,
    RollingTranscript,
    pause_cut,
    timed_boundary,
    validate_words,
)
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.word_alignment import AlignmentCancelled, AlignmentError, QwenWordAligner, WordTiming

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
OPENAI_BASE_URL = "https://api.openai.com/v1"


class TranscriptionRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to surface to a client."""


class TranscriptionRequestCancelled(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"transcription request cancelled: {reason}")


@dataclass(frozen=True)
class HttpTranscriptionResult:
    text: str
    language: str | None = None
    words: tuple[WordTiming, ...] | None = None


_aligner_factory_lock = Lock()


@lru_cache(maxsize=2)
def _cached_aligner(model_name: str, device: str) -> QwenWordAligner:
    return QwenWordAligner(model_name=model_name, device=device)


def _shared_aligner(model_name: str, device: str) -> QwenWordAligner:
    # lru_cache alone permits duplicate constructors during concurrent misses.
    with _aligner_factory_lock:
        return _cached_aligner(model_name, device)


@dataclass
class HttpTranscriptionOperation:
    """One cancellable HTTP request, independent of endpoint capacity."""

    endpoint_url: str
    api_key: str | None
    model: str | None
    wav_bytes: bytes
    language: str | None
    response_format: str
    timeout_s: float
    extra_fields: dict[str, Any] | None = None
    _cancelled: Event = field(default_factory=Event, init=False, repr=False)
    _transport_lock: Any = field(default_factory=Lock, init=False, repr=False)
    _cancel_reason: str = field(default="superseded", init=False, repr=False)
    _worker_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _worker_task: asyncio.Task[HttpTranscriptionResult] | None = field(default=None, init=False, repr=False)
    _worker_cancel_scope: anyio.CancelScope | None = field(default=None, init=False, repr=False)

    def run(self, cancel_check: Callable[[], bool] = lambda: False) -> HttpTranscriptionResult:
        self._raise_if_cancelled(cancel_check)
        result: Future[HttpTranscriptionResult] = Future()
        done = Event()
        worker = Thread(target=self._run_http, args=(result, done), name="stt-http-reader", daemon=True)
        worker.start()
        try:
            while not done.wait(0.05):
                self._raise_if_cancelled(cancel_check)
            self._raise_if_cancelled(cancel_check)
            return result.result()
        finally:
            if not done.is_set():
                self.cancel("shutdown")
            worker.join()

    def _run_http(self, result: Future[HttpTranscriptionResult], done: Event) -> None:
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.new_event_loop()
            task = loop.create_task(self._request_with_cancel_scope())
            with self._transport_lock:
                self._worker_loop = loop
                self._worker_task = task
            result.set_result(loop.run_until_complete(task))
        except asyncio.CancelledError:
            result.set_exception(TranscriptionRequestCancelled(self._cancel_reason))
        except Exception as exc:
            result.set_exception(exc)
        finally:
            with self._transport_lock:
                self._worker_loop = None
                self._worker_task = None
            try:
                if loop is not None:
                    loop.close()
            finally:
                done.set()

    def cancel(self, reason: str = "superseded") -> None:
        with self._transport_lock:
            if self._cancelled.is_set():
                return
            self._cancel_reason = reason
            self._cancelled.set()
            loop, scope = self._worker_loop, self._worker_cancel_scope
        if loop is not None and scope is not None:
            try:
                loop.call_soon_threadsafe(scope.cancel)
            except RuntimeError:
                # Completion may close the loop after the snapshot above.
                pass

    def _raise_if_cancelled(self, cancel_check: Callable[[], bool]) -> None:
        if cancel_check():
            self.cancel("superseded")
        if self._cancelled.is_set():
            raise TranscriptionRequestCancelled(self._cancel_reason)

    async def _request_with_cancel_scope(self) -> HttpTranscriptionResult:
        # A connection handoff can consume a single Task.cancel(). Keep cancellation
        # effective at later upload/read checkpoints until the request has settled.
        with anyio.CancelScope() as scope:
            with self._transport_lock:
                self._worker_cancel_scope = scope
                if self._cancelled.is_set():
                    scope.cancel()
            try:
                return await self._request_async()
            finally:
                with self._transport_lock:
                    self._worker_cancel_scope = None
        raise TranscriptionRequestCancelled(self._cancel_reason)

    async def _request_async(self) -> HttpTranscriptionResult:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data: dict[str, Any] = {
            "response_format": self.response_format,
            **(self.extra_fields or {}),
        }
        if self.model:
            data["model"] = self.model
        if self.language:
            if self.model == "gpt-transcribe":
                data["languages[]"] = self.language
            else:
                data["language"] = self.language

        client = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            response = await client.post(
                self.endpoint_url,
                headers=headers,
                data=data,
                files={"file": ("audio.wav", self.wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            return self._parse_response(response.content, response.headers.get("content-type", ""))
        except httpx.HTTPStatusError as exc:
            # Never echo arbitrary server text (which can contain prompts, paths or
            # credentials). Recognize known audio-limit errors and emit our own hint.
            message = f"transcription server returned HTTP {exc.response.status_code}"
            body = exc.response.text[:8192].lower()
            if exc.response.status_code in {400, 413}:
                if "maximum allowed duration" in body or "vllm_max_audio_decode_duration_s" in body:
                    message += ": audio duration limit exceeded; enable or reduce openai_stt_window_seconds"
                elif exc.response.status_code == 413 or "vllm_max_audio_clip_filesize_mb" in body:
                    message += ": upload size limit exceeded; enable or reduce openai_stt_window_seconds"
                elif "vllm_max_audio_decode_bytes" in body:
                    message += ": decoded audio limit exceeded; enable or reduce openai_stt_window_seconds"
            raise TranscriptionRequestError(message) from exc
        except httpx.TimeoutException as exc:
            raise TranscriptionRequestError("transcription request timed out") from exc
        except httpx.HTTPError as exc:
            raise TranscriptionRequestError(f"transcription transport failed: {type(exc).__name__}") from exc
        finally:
            # Repeated cancellation must not interrupt release of the HTTP transport.
            with anyio.CancelScope(shield=True):
                await client.aclose()

    def _parse_response(self, body: bytes, content_type: str) -> HttpTranscriptionResult:
        if self.response_format == "text" or "text/plain" in content_type:
            return HttpTranscriptionResult(text=body.decode("utf-8").strip(), language=self.language)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionRequestError("transcription server returned an invalid JSON response") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise TranscriptionRequestError("transcription response is missing a string 'text' field")
        language = None
        languages = payload.get("languages")
        if isinstance(languages, list):
            for detected_language in languages:
                if isinstance(detected_language, dict) and isinstance(detected_language.get("code"), str):
                    language = detected_language["code"]
                    break
        if language is None and isinstance(payload.get("language"), str):
            language = payload["language"]
        words = None
        if "words" in payload:
            if not isinstance(payload["words"], list):
                raise TranscriptionRequestError("transcription response has invalid word timings")
            try:
                words = tuple(
                    WordTiming(word["word"], float(word["start"]), float(word["end"])) for word in payload["words"]
                )
                if any(not isinstance(word.text, str) for word in words):
                    raise ValueError
            except (TypeError, KeyError, ValueError, OverflowError) as exc:
                raise TranscriptionRequestError("transcription response has invalid word timings") from exc
        return HttpTranscriptionResult(text=text, language=language or self.language, words=words)


@dataclass(eq=False)
class _TranscriptionRequest:
    source: VADAudio
    session_generation: int
    operation: HttpTranscriptionOperation | None = None
    cancelled: bool = False
    elapsed_s: float | None = None
    selected_language: str | None = field(init=False)

    def __post_init__(self) -> None:
        # A session update must not change the hint halfway through a window batch.
        self.selected_language = self.source.runtime_config.selected_language if self.source.runtime_config else None


class OpenAICompatibleSTTHandler(BaseSTTHandler):
    """Per-pipeline asynchronous STT with turn and session lifecycle ownership."""

    # Bound retained utterances behind a stalled request, independently of server capacity.
    _MAX_PENDING_FINAL_REQUESTS = 8

    def setup(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str | None = "nvidia/parakeet-tdt-0.6b-v3",
        language: str | None = None,
        response_format: str = "json",
        timeout: float = 60.0,
        speculative_turns: SpeculativeTurnTracker | None = None,
        final_revision_settle_s: float = 0.0,
        gen_kwargs: dict[str, Any] | None = None,
        window_seconds: float = 0.0,
        overlap_seconds: float = 2.0,
        aligner_model: str | None = None,
        aligner_device: str = "cpu",
    ) -> None:
        if response_format not in {"json", "text", "verbose_json"}:
            raise ValueError("OpenAI-compatible STT response_format must be json, text or verbose_json")
        if timeout <= 0:
            raise ValueError("OpenAI-compatible STT timeout must be > 0")
        if not math.isfinite(window_seconds) or window_seconds < 0:
            raise ValueError("STT window_seconds must be finite and >= 0")
        if not math.isfinite(overlap_seconds) or overlap_seconds < 0:
            raise ValueError("STT overlap_seconds must be finite and >= 0")
        self._window_samples = int(window_seconds * PIPELINE_SAMPLE_RATE)
        self._overlap_samples = int(overlap_seconds * PIPELINE_SAMPLE_RATE)
        if window_seconds and self._window_samples <= self._overlap_samples:
            raise ValueError("STT window_seconds must exceed overlap_seconds by at least one sample")
        if window_seconds and self._overlap_samples < int(0.5 * PIPELINE_SAMPLE_RATE):
            raise ValueError("STT windows need at least 0.5 seconds of overlap")
        self._word_aligner = _shared_aligner(aligner_model, aligner_device) if aligner_model else None
        model = model.strip() if model else None
        language = language.strip() if language else None
        if model is None and language is None:
            raise ValueError("OpenAI-compatible STT requires either a model or language")

        self.base_url = base_url.rstrip("/")
        self.endpoint_url = f"{self.base_url}/audio/transcriptions"
        self.api_key = api_key
        if self.api_key is None and self.base_url == OPENAI_BASE_URL:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.response_format = response_format
        self.timeout = timeout
        self.speculative_turns = speculative_turns
        self.final_revision_settle_s = final_revision_settle_s
        self.gen_kwargs = gen_kwargs or {}
        self._request_lock = RLock()
        self._session_generation = 0
        self._closed = False
        self._pending_finals: deque[_TranscriptionRequest] = deque()
        self._pending_progressive: _TranscriptionRequest | None = None
        self._active: dict[str, _TranscriptionRequest] = {}
        self._workers_running: set[str] = set()
        self._progressive_thread: Thread | None = None
        self._final_thread: Thread | None = None
        self._window_states: OrderedDict[tuple[str, str | None], RollingTranscript] = OrderedDict()
        self.warmup()

    def warmup(self) -> None:
        """Validate the configured transcription endpoint before accepting sessions."""
        logger.info("Warming up %s", self.__class__.__name__)
        started_at_s = perf_counter()
        self._make_operation(np.zeros(PIPELINE_SAMPLE_RATE, dtype=np.float32)).run()
        logger.info(
            "%s warmed up in %.3fs",
            self.__class__.__name__,
            perf_counter() - started_at_s,
        )

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        mode: Literal["progressive", "final"] = "progressive" if vad_audio.mode == "progressive" else "final"
        failed_requests: list[_TranscriptionRequest] = []
        failure_message = "transcription worker could not start"
        with self._request_lock:
            request = _TranscriptionRequest(vad_audio, self._session_generation)
            if not self._request_is_current(request):
                return
            for active in self._active.values():
                if not self._request_is_current(active):
                    self._cancel_request(active, "superseded")
            self._pending_finals = deque(r for r in self._pending_finals if self._request_is_current(r))
            final_requests = [*self._pending_finals]
            if "final" in self._active:
                final_requests.append(self._active["final"])
            key = self._revision_key(vad_audio)
            if key is not None and any(
                self._revision_key(r.source) == key and self._request_is_current(r) for r in final_requests
            ):
                return
            if mode == "progressive":
                # Only the latest cumulative window is useful once this lane is free.
                self._pending_progressive = request
            else:
                for progressive in (self._pending_progressive, self._active.get("progressive")):
                    if progressive is not None and progressive.source.turn_id == vad_audio.turn_id:
                        self._cancel_request(progressive, "final_received")
                if len(self._pending_finals) >= self._MAX_PENDING_FINAL_REQUESTS:
                    failed_requests = [request]
                    failure_message = "transcription queue is full"
                else:
                    self._pending_finals.append(request)
            if not failed_requests:
                try:
                    self._start_worker(mode)
                except Exception:
                    if mode == "final":
                        failed_requests = list(self._pending_finals)
                        self._pending_finals.clear()
                    else:
                        failed_requests = [request]
                        self._pending_progressive = None
        for failed in failed_requests:
            self._publish_failure(failed, failure_message)
        # The handler must remain free to process SESSION_END and new revisions.
        yield from ()

    def _start_worker(self, mode: Literal["progressive", "final"]) -> None:
        if mode in self._workers_running:
            return
        thread = Thread(target=self._worker, args=(mode,), name=f"openai-stt-{mode}", daemon=True)
        setattr(self, f"_{mode}_thread", thread)
        self._workers_running.add(mode)
        try:
            thread.start()
        except Exception:
            self._workers_running.discard(mode)
            setattr(self, f"_{mode}_thread", None)
            raise

    def _worker(self, mode: str) -> None:
        if self.pipeline_index is not None:
            pipeline_log_ctx.set(self.pipeline_index)
        try:
            while True:
                with self._request_lock:
                    if mode == "final":
                        request = self._pending_finals.popleft() if self._pending_finals else None
                    else:
                        request, self._pending_progressive = self._pending_progressive, None
                    if request is None or self._closed:
                        self._workers_running.discard(mode)
                        return
                    if not self._request_is_current(request):
                        continue
                    self._active[mode] = request
                try:
                    self._run_request(request)
                finally:
                    with self._request_lock:
                        self._active.pop(mode, None)
        finally:
            with self._request_lock:
                if getattr(self, f"_{mode}_thread") is current_thread():
                    self._workers_running.discard(mode)

    def _run_request(self, request: _TranscriptionRequest) -> None:
        source = request.source
        started_at_s = perf_counter()
        try:
            if not self._request_is_current(request):
                return
            result = self._transcribe_windows(request)
        except (TranscriptionRequestCancelled, AlignmentCancelled):
            return
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (TranscriptionRequestError, AlignmentError))
                else "transcription request failed"
            )
            self._publish_failure(request, message)
            return

        output: STTOut
        if source.mode == "progressive":
            output = PartialTranscription(text=result.text, turn_id=source.turn_id, turn_revision=source.turn_revision)
        else:
            output = Transcription(
                text=result.text,
                language_code=result.language,
                turn_id=source.turn_id,
                turn_revision=source.turn_revision,
                speech_stopped_at_s=source.speech_end_at_s,
            )
        elapsed = perf_counter() - started_at_s
        request.elapsed_s = elapsed
        if self._publish_output(request, output):
            self._times.append(elapsed)
            logger.info(
                "OpenAI-compatible STT request completed turn=%s rev=%s mode=%s in %.3fs",
                source.turn_id,
                source.turn_revision,
                source.mode,
                elapsed,
            )

    def _run_audio_window(self, request: _TranscriptionRequest, audio: np.ndarray) -> HttpTranscriptionResult:
        # Check between windows as well as during each HTTP request.
        if not self._request_is_current(request):
            raise TranscriptionRequestCancelled("superseded")
        selected = request.selected_language
        if selected is None:
            operation = self._make_operation(audio)
        else:
            operation = self._make_operation(
                audio, language=None if selected == "auto" else selected, use_setup_language=False
            )
        with self._request_lock:
            request.operation = operation
            if not self._request_is_current(request):
                operation.cancel("superseded")
                raise TranscriptionRequestCancelled("superseded")
        result = operation.run(cancel_check=lambda: not self._request_is_current(request))
        if not self._request_is_current(request):
            raise TranscriptionRequestCancelled("superseded")
        return result

    def _timed_window(
        self,
        request: _TranscriptionRequest,
        audio: np.ndarray,
        start: int,
        end: int,
        next_start: int,
        needs_times: bool,
    ) -> AlignedAudioWindow:
        chunk = audio[start:end]
        if not self._request_is_current(request):
            raise TranscriptionRequestCancelled("superseded")
        # Do not decode trailing digital silence: ASR can hallucinate words on it.
        silent = not np.any(
            np.abs(chunk.astype(np.float64, copy=False)) > (0 if np.issubdtype(chunk.dtype, np.integer) else 1e-7)
        )
        if silent:
            selected = request.selected_language
            language = self.language if selected is None else (None if selected == "auto" else selected)
            return AlignedAudioWindow(start, end, next_start, "", language)
        result = self._run_audio_window(request, chunk)
        words: tuple[WordTiming, ...] = ()
        if needs_times and result.text.strip():
            if result.words is not None:
                words = result.words
            elif self._word_aligner is not None:
                words = tuple(
                    self._word_aligner.align(
                        chunk,
                        result.text,
                        result.language,
                        cancel_check=lambda: not self._request_is_current(request),
                    )
                )
            else:
                raise AlignmentError("Continuous speech needs word timestamps or openai_stt_aligner_model")
            validate_words(result.text, words, len(chunk) / PIPELINE_SAMPLE_RATE)
        if not self._request_is_current(request):
            raise TranscriptionRequestCancelled("superseded")
        return AlignedAudioWindow(start, end, next_start, result.text, result.language, words)

    @staticmethod
    def _absolute_words(window: AlignedAudioWindow) -> tuple[WordTiming, ...]:
        offset = window.start / PIPELINE_SAMPLE_RATE
        return tuple(WordTiming(word.text, word.start + offset, word.end + offset) for word in window.words)

    def _aligned_boundary(
        self, request: _TranscriptionRequest, audio: np.ndarray, left: AlignedAudioWindow, right: AlignedAudioWindow
    ) -> tuple[int, str, int]:
        old, new = self._absolute_words(left), self._absolute_words(right)
        try:
            left_cut, right_cut = timed_boundary(
                left.text, old, right.text, new, right.start / PIPELINE_SAMPLE_RATE, left.end / PIPELINE_SAMPLE_RATE
            )
            return left_cut, right.text[right_cut:], right_cut
        except AlignmentError:
            # One bounded bridge decode supplies more context around the join.
            # Join it at two independently checked physical audio anchors.
            start = max(left.start, right.start - 2 * self._overlap_samples)
            end = min(right.end, left.end + 2 * self._overlap_samples, start + self._window_samples)
            bridge = self._timed_window(request, audio, start, end, end, True)
            bridge_words = self._absolute_words(bridge)
            left_cut, bridge_start = timed_boundary(
                left.text, old, bridge.text, bridge_words, start / PIPELINE_SAMPLE_RATE, left.end / PIPELINE_SAMPLE_RATE
            )
            bridge_end, right_cut = timed_boundary(
                bridge.text,
                bridge_words,
                right.text,
                new,
                right.start / PIPELINE_SAMPLE_RATE,
                end / PIPELINE_SAMPLE_RATE,
            )
            if bridge_end < bridge_start:
                raise AlignmentError("Audio-timed retry crossed its boundary anchors")
            return left_cut, bridge.text[bridge_start:bridge_end] + right.text[right_cut:], right_cut

    def _transcribe_windows(self, request: _TranscriptionRequest) -> HttpTranscriptionResult:
        audio = np.asarray(request.source.audio).squeeze()
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise TranscriptionRequestError("STT audio must be finite and mono")
        key = (request.source.turn_id, request.selected_language) if request.source.turn_id is not None else None
        with self._request_lock:
            state = self._window_states.get(key) if key is not None else None
        digest = hashlib.sha256()
        hashed_end = 0
        if state is not None:
            if state.window.end < len(audio):
                hashed_end = state.window.end
                digest.update(audio[:hashed_end].tobytes())
            if not hashed_end or digest.digest() != state.digest:
                with self._request_lock:
                    if key is not None and self._window_states.get(key) is state:
                        self._window_states.pop(key)
                state, digest, hashed_end = None, hashlib.sha256(), 0
        if not self._window_samples or len(audio) <= self._window_samples:
            return self._run_audio_window(request, audio)
        previous = state.window if state else None
        text, offset = (state.text, state.offset) if state else ("", 0)
        start = previous.next_start if previous else 0
        language = (
            state.language
            if state
            else (
                self.language
                if request.selected_language is None
                else (None if request.selected_language == "auto" else request.selected_language)
            )
        )
        while start < len(audio):
            end = min(start + self._window_samples, len(audio))
            stable = end < len(audio)
            cut = pause_cut(audio, start, end, self._overlap_samples) if stable else None
            if cut is not None and (previous is None or cut > previous.end):
                end, next_start = cut, cut
            else:
                next_start = end - self._overlap_samples if stable else end
            incoming = previous is not None and start < previous.end
            window = self._timed_window(request, audio, start, end, next_start, incoming or next_start < end)
            language = window.language or language
            if previous is None:
                text = window.text
            elif not incoming or not previous.text.strip() or not window.text.strip():
                no_spaces = (language or "").lower() in {"zh", "yue", "ja", "chinese", "cantonese", "japanese"}
                separator = " " if text and window.text and not no_spaces else ""
                offset = len(text) + len(separator)
                text += separator + window.text
            else:
                left_cut, suffix, right_cut = self._aligned_boundary(request, audio, previous, window)
                splice = offset + left_cut
                if splice < 0 or splice > len(text):
                    raise AlignmentError("Audio-timed join tried to revise a completed prefix")
                text = text[:splice] + suffix
                # A bridge can insert text before the right-window suffix.
                offset = splice + len(suffix) - len(window.text[right_cut:]) - right_cut
            if window.end >= len(audio):
                break
            digest.update(audio[hashed_end : window.end].tobytes())
            hashed_end = window.end
            completed = RollingTranscript(window, text, offset, language, digest.digest())
            with self._request_lock:
                if not self._request_is_current(request):
                    raise TranscriptionRequestCancelled("superseded")
                # Another worker may already have advanced this turn. Never
                # replace its state with a result based on an older snapshot.
                if key is not None and self._window_states.get(key) is state:
                    self._window_states[key] = completed
                    self._window_states.move_to_end(key)
                    state = completed
                    while len(self._window_states) > self._MAX_PENDING_FINAL_REQUESTS:
                        self._window_states.popitem(last=False)
            previous, start = window, window.next_start
        return HttpTranscriptionResult(text, language)

    def _publish_failure(self, request: _TranscriptionRequest, message: str) -> None:
        if not self._request_is_current(request):
            return
        source = request.source
        if source.mode == "progressive":
            logger.warning(
                "OpenAI-compatible progressive STT failed turn=%s rev=%s: %s",
                source.turn_id,
                source.turn_revision,
                message,
            )
            return
        logger.error(
            "OpenAI-compatible STT failed turn=%s rev=%s: %s",
            source.turn_id,
            source.turn_revision,
            message,
        )
        self._publish_output(
            request,
            TranscriptionFailure(
                message=message,
                turn_id=source.turn_id,
                turn_revision=source.turn_revision,
                speech_stopped_at_s=source.speech_end_at_s,
            ),
        )

    def _publish_output(self, request: _TranscriptionRequest, output: STTOut) -> bool:
        # Waiting for a speculative reopen must not hold the teardown lock.
        if not self._request_is_current(request) or not self.should_emit_output(output):
            return False
        with self._request_lock:
            if not self._request_is_current(request):
                return False
            self.before_emit_output(output)
            if isinstance(output, Transcription) and request.elapsed_s is not None:
                store = getattr(self, "turn_latency_store", None)
                tracker = store.get_or_create_for_turn(output.turn_id, output.turn_revision) if store else None
                if tracker is not None:
                    tracker.record_stt(request.elapsed_s)
            self.queue_out.put(self.output_for_queue(output, request.source))
            return True

    def _request_is_current(self, request: _TranscriptionRequest) -> bool:
        with self._request_lock:
            if self._closed or self.stop_event.is_set() or request.cancelled:
                return False
            if request.session_generation != self._session_generation:
                return False
            source = request.source
            if self._is_completed_final_revision(source):
                return False
            tracker = self.speculative_turns
            return tracker is None or tracker.is_latest(source.turn_id, source.turn_revision)

    @staticmethod
    def _cancel_request(request: _TranscriptionRequest, reason: str) -> None:
        request.cancelled = True
        if request.operation is not None:
            request.operation.cancel(reason)

    def on_session_end(self) -> None:
        with self._request_lock:
            self._session_generation += 1
            self._window_states.clear()
            self._pending_finals.clear()
            self._pending_progressive = None
            for request in self._active.values():
                self._cancel_request(request, "session_end")
            super().on_session_end()

    def cleanup(self) -> None:
        with self._request_lock:
            self._closed = True
            self._session_generation += 1
            self._window_states.clear()
            self._pending_finals.clear()
            self._pending_progressive = None
            for request in self._active.values():
                self._cancel_request(request, "shutdown")
            workers = (self._final_thread, self._progressive_thread)
        for worker in workers:
            if worker is not None and worker is not current_thread():
                worker.join(timeout=2)

    def _make_operation(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> HttpTranscriptionOperation:
        extra_fields = self.gen_kwargs.copy()
        if self.response_format == "verbose_json":
            extra_fields.setdefault("timestamp_granularities[]", "word")
        if not use_setup_language:
            extra_fields.pop("language", None)
            extra_fields.pop("languages[]", None)
        return HttpTranscriptionOperation(
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            model=self.model,
            wav_bytes=self._encode_wav(audio),
            language=self.language if use_setup_language else language,
            response_format=self.response_format,
            timeout_s=self.timeout,
            extra_fields=extra_fields,
        )

    @staticmethod
    def _encode_wav(audio: np.ndarray) -> bytes:
        waveform = np.asarray(audio).squeeze()
        if waveform.ndim != 1:
            raise ValueError(f"STT audio must be mono, got shape {waveform.shape}")
        if np.issubdtype(waveform.dtype, np.floating):
            pcm = np.clip(waveform, -1.0, 1.0)
            pcm = np.round(pcm * 32767.0).astype("<i2")
        else:
            pcm = np.clip(waveform, -32768, 32767).astype("<i2")

        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(PIPELINE_SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return output.getvalue()
