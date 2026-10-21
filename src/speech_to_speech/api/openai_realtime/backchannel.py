"""Classify overlapping speech before exposing it as an interruption.

Only a final transcript can suppress a turn. Errors and expired holds release
input through the usual interruption path. Suppression respects the same
speculative speech gate as assistant output, so resumed speech can revise it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from speech_to_speech.pipeline.events import (
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker, TurnGateAction
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore

logger = logging.getLogger(__name__)

PCM16_BYTES_PER_SECOND = 16000 * 2
ASSISTANT_CONTEXT_CHARS = 400
_DROPPED_REVISIONS_KEPT = 64

_USER_TURN_EVENTS = (
    SpeechStartedEvent,
    SpeechStoppedEvent,
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
    AudioInputCompletedEvent,
)

_STATE_TEMPLATE = (
    'A voice assistant is talking to a user. Recent assistant response text: "{assistant}"\n'
    'While the assistant was still talking, the user said: "{user}"'
)

_QUESTION = {
    "type": "choice",
    "instructions": "What is the user doing?",
    "criteria": {
        "backchannel": (
            "a short acknowledgement or reaction such as mm-hmm, yeah, okay, or wow, "
            "which lets the assistant keep talking"
        ),
        "interruption": (
            "wants the assistant to stop talking: to stop it, correct it, ask something, or change the topic"
        ),
    },
}


@dataclass(frozen=True)
class BackchannelConfig:
    url: str
    model: str | None = None
    threshold: float = 0.5
    max_hold_ms: int = 1500
    timeout_ms: int = 300

    def __post_init__(self) -> None:
        url = httpx.URL(self.url)
        if not url.is_absolute_url or url.scheme not in {"http", "https"}:
            raise ValueError("backchannel_url must be an absolute HTTP or HTTPS URL")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("backchannel_threshold must be between 0 and 1")
        if not math.isfinite(self.max_hold_ms) or self.max_hold_ms <= 0:
            raise ValueError("backchannel_max_hold_ms must be positive")
        if not math.isfinite(self.timeout_ms) or self.timeout_ms <= 0:
            raise ValueError("backchannel_timeout_ms must be positive")


class InterruptClassifier(Protocol):
    async def interrupt_probability(self, assistant_text: str, user_text: str) -> float: ...


class SystemOneClassifier:
    """Scores overlapping speech through a ``/v1/systemone`` endpoint."""

    def __init__(self, config: BackchannelConfig) -> None:
        self._endpoint = config.url.rstrip("/") + "/v1/systemone"
        self._model = config.model
        self._timeout_s = config.timeout_ms / 1000.0
        self._client: httpx.AsyncClient | None = None

    async def interrupt_probability(self, assistant_text: str, user_text: str) -> float:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        body: dict[str, object] = {
            "state": _STATE_TEMPLATE.format(assistant=assistant_text, user=user_text),
            "questions": {"turn": _QUESTION},
        }
        if self._model:
            body["model"] = self._model
        response = await self._client.post(self._endpoint, json=body)
        response.raise_for_status()
        probability = response.json()["answers"]["turn"]["probabilities"]["interruption"]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Invalid interruption probability from decision model")
        return float(probability)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


Verdict = Literal["interrupt", "backchannel"]


@dataclass
class _HeldTurn:
    turn_id: str | None
    started_at: float
    latest_revision: int | None
    assistant_text: str
    events: list[PipelineEvent] = field(default_factory=list)
    revisions: set[int | None] = field(default_factory=set)
    verdict: Verdict | None = None
    reason: str = ""
    task: asyncio.Task[float] | None = None
    task_text: tuple[str, bool, int | None] | None = None
    queued_text: tuple[str, bool, int | None] | None = None


class BackchannelGate:
    """Per-session hold on user turns that start during assistant playback.

    ``offer`` takes the events it holds; ``poll`` returns the events to
    dispatch now, in their original order. Both run on the send loop, so the
    gate needs no locking.
    """

    def __init__(
        self,
        classifier: InterruptClassifier,
        config: BackchannelConfig,
        *,
        speculative_turns: SpeculativeTurnTracker | None = None,
        turn_latency_store: TurnLatencyStore | None = None,
    ) -> None:
        self._classifier = classifier
        self._config = config
        self._turns = speculative_turns
        self._latency_store = turn_latency_store
        self._held: list[_HeldTurn] = []
        self._dropped: deque[tuple[str | None, int | None]] = deque(maxlen=_DROPPED_REVISIONS_KEPT)
        self._playback_until = 0.0
        self._assistant_text = ""

    def note_assistant_audio(self, num_bytes: int, now: float | None = None) -> None:
        """Extend the estimated end of client playback by a PCM16 chunk."""
        now = time.monotonic() if now is None else now
        self._playback_until = max(self._playback_until, now) + num_bytes / PCM16_BYTES_PER_SECOND

    def note_assistant_text(self, text: str) -> None:
        if text:
            self._assistant_text = (self._assistant_text + text)[-ASSISTANT_CONTEXT_CHARS:]

    def stop_playback(self, *, release_pending: bool = False) -> None:
        """End playback; explicit client stops also preserve undecided input."""
        self._playback_until = 0.0
        if release_pending:
            for held in self._held:
                self._decide(held, "interrupt", "client stopped playback")
                self._cancel_task(held)

    def assistant_audible(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self._playback_until

    def offer(self, event: PipelineEvent, now: float | None = None, *, enabled: bool = True) -> bool:
        """Return True when the gate holds or drops ``event``."""
        turn_id = getattr(event, "turn_id", None)
        # Without a turn id, later events of the turn cannot be matched to it.
        if not isinstance(event, _USER_TURN_EVENTS) or turn_id is None:
            return False
        now = time.monotonic() if now is None else now
        revision = getattr(event, "turn_revision", None)
        held = self._find(turn_id)
        if held is None:
            if isinstance(event, SpeechStartedEvent):
                if not enabled or not event.interrupt_response:
                    return False
                # A later turn waits behind an undecided one so events keep
                # their order, whether or not the assistant is still audible.
                if not self._held and not self.assistant_audible(now):
                    return False
                held = _HeldTurn(
                    turn_id=turn_id,
                    started_at=now,
                    latest_revision=revision,
                    assistant_text=self._assistant_text,
                )
                if not self.assistant_audible(now):
                    self._decide(held, "interrupt", "speech began after playback")
                self._held.append(held)
                logger.info("Backchannel gate: holding turn %s rev %s during assistant playback", turn_id, revision)
            elif (turn_id, revision) in self._dropped:
                return True
            else:
                return False
        held.events.append(event)
        held.revisions.add(revision)
        self._observe(held, event)
        return True

    def poll(self, now: float | None = None, *, enabled: bool = True) -> list[PipelineEvent]:
        """Advance classifications and return released events, oldest first."""
        now = time.monotonic() if now is None else now
        for held in self._held:
            if not enabled:
                self._decide(held, "interrupt", "interrupt_response disabled")
            elif held.verdict != "interrupt":
                self._advance(held, now)
        released: list[PipelineEvent] = []
        while self._held and self._held[0].verdict is not None:
            held = self._held[0]
            if held.verdict == "backchannel" and self._turns is not None:
                decision = self._turns.gate(held.turn_id, held.latest_revision, commit=True)
                if decision.action is TurnGateAction.HOLD:
                    break
                if decision.action is TurnGateAction.DROP and self._turns.is_current_turn(held.turn_id):
                    # VAD can advance before its speech_started reaches this loop.
                    held.verdict = None
                    break
                if decision.action is TurnGateAction.ACCEPT:
                    self._turns.close(held.turn_id, held.latest_revision)
            self._held.pop(0)
            self._cancel_task(held)
            if held.verdict == "interrupt":
                logger.info("Backchannel gate: releasing turn %s as an interruption (%s)", held.turn_id, held.reason)
                released.extend(held.events)
            else:
                logger.info("Backchannel gate: dropping turn %s as a backchannel (%s)", held.turn_id, held.reason)
                self._dropped.extend((held.turn_id, revision) for revision in held.revisions)
                if self._latency_store is not None:
                    for revision in held.revisions:
                        self._latency_store.discard_pending_turn(held.turn_id, revision)
        return released

    def has_held_turns(self) -> bool:
        return bool(self._held)

    def close(self) -> None:
        for held in self._held:
            self._cancel_task(held)
        self._held.clear()

    def _find(self, turn_id: str) -> _HeldTurn | None:
        for held in self._held:
            if held.turn_id == turn_id:
                return held
        return None

    def _observe(self, held: _HeldTurn, event: PipelineEvent) -> None:
        revision = getattr(event, "turn_revision", None)
        if isinstance(event, SpeechStartedEvent):
            # Resumed speech supersedes any earlier transcript of this turn.
            if revision != held.latest_revision:
                self._cancel_task(held)
                held.verdict = None
            held.latest_revision = revision
            held.queued_text = None
            return
        if not self._is_current_revision(held, revision) or held.verdict is not None:
            return
        if isinstance(event, SpeechStoppedEvent):
            # The VAD reports no duration when it discards the segment as too
            # short or empty; no transcript will follow.
            if event.duration_s <= 0 and held.queued_text is None and held.task is None:
                self._decide(held, "backchannel", "segment discarded by VAD")
        elif isinstance(event, PartialTranscriptionEvent):
            if event.delta.strip():
                held.queued_text = (event.delta.strip(), False, revision)
        elif isinstance(event, TranscriptionCompletedEvent):
            transcript = event.transcript.strip()
            if transcript:
                held.queued_text = (transcript, True, revision)
            else:
                self._decide(held, "interrupt", "empty transcript")
        elif isinstance(event, (TranscriptionFailedEvent, AudioInputCompletedEvent)):
            self._decide(held, "interrupt", f"no transcript to classify ({event.type})")

    def _advance(self, held: _HeldTurn, now: float) -> None:
        if held.task is not None and held.task.done():
            self._apply_result(held)
        if (now - held.started_at) * 1000 >= self._config.max_hold_ms:
            self._decide(held, "interrupt", f"held longer than {self._config.max_hold_ms}ms")
        if held.verdict is not None:
            return
        if held.task is None and held.queued_text is not None:
            held.task_text, held.queued_text = held.queued_text, None
            if not self._is_current_revision(held, held.task_text[2]):
                held.task_text = None
                return
            held.task = asyncio.ensure_future(
                self._classifier.interrupt_probability(held.assistant_text, held.task_text[0])
            )

    def _apply_result(self, held: _HeldTurn) -> None:
        task, task_text = held.task, held.task_text
        held.task, held.task_text = None, None
        assert task is not None and task_text is not None
        text, is_final, revision = task_text
        try:
            probability = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if self._is_current_revision(held, revision):
                self._decide(held, "interrupt", f"classifier error: {exc!r}")
            return
        if not self._is_current_revision(held, revision):
            return
        summary = f"p_interrupt={probability:.2f} {'final' if is_final else 'partial'} {transcript_for_log(text)}"
        if probability >= self._config.threshold:
            self._decide(held, "interrupt", summary)
        elif is_final and revision == held.latest_revision and held.queued_text is None:
            self._decide(held, "backchannel", summary)
        else:
            logger.debug("Backchannel gate: turn %s still held (%s)", held.turn_id, summary)

    @staticmethod
    def _decide(held: _HeldTurn, verdict: Verdict, reason: str) -> None:
        held.verdict = verdict
        held.reason = reason

    def _is_current_revision(self, held: _HeldTurn, revision: int | None) -> bool:
        return revision == held.latest_revision and (
            self._turns is None or self._turns.is_latest(held.turn_id, revision)
        )

    @staticmethod
    def _cancel_task(held: _HeldTurn) -> None:
        if held.task is not None:
            held.task.cancel()
            held.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
            held.task = None
            held.task_text = None
