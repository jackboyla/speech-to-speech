"""Hold barge-ins until a decision model says they are real interruptions.

While assistant audio is playing, a short "mm-hmm" or "yeah" from the user
should not cancel the response. ``BackchannelGate`` holds a user turn that
starts during playback: its pipeline events are kept back from the realtime
service, so no input item is created and nothing is cancelled. The turn's
transcript is then classified by a llama.cpp decision model
(``/v1/systemone``):

- interruption: the held events are released in order and take the normal
  barge-in path, which cancels the response;
- backchannel (final transcript only): the held events are dropped and the
  assistant keeps talking.

A partial transcript can only release a turn, never drop it: "Okay so" reads
as a backchannel until the user finishes the sentence. Classifier errors,
transcription failures, and holds longer than ``max_hold_ms`` release the turn,
which is the behavior without the gate.
"""

from __future__ import annotations

import asyncio
import logging
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
from speech_to_speech.pipeline.transcript_logging import transcript_for_log

logger = logging.getLogger(__name__)

PCM16_BYTES_PER_SECOND = 16000 * 2
# Text that reached TTS ahead of playback is close enough to what the user has
# heard; keep only the tail so the classifier prompt stays short.
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
    'A voice assistant is talking to a user. So far the assistant has said: "{assistant}"\n'
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


class InterruptClassifier(Protocol):
    async def interrupt_probability(self, assistant_text: str, user_text: str) -> float: ...


class SystemOneClassifier:
    """Scores an overlap utterance with a llama.cpp ``/v1/systemone`` model."""

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
        return float(response.json()["answers"]["turn"]["probabilities"]["interruption"])

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

    def __init__(self, classifier: InterruptClassifier, config: BackchannelConfig) -> None:
        self._classifier = classifier
        self._config = config
        self._held: list[_HeldTurn] = []
        self._dropped: deque[tuple[str | None, int | None]] = deque(maxlen=_DROPPED_REVISIONS_KEPT)
        self._playback_until = 0.0
        self._assistant_text = ""

    # ── Assistant side ────────────────────────────────────────────────

    def note_assistant_audio(self, num_bytes: int, now: float | None = None) -> None:
        """Extend the estimated end of client playback by a PCM16 chunk."""
        now = time.monotonic() if now is None else now
        self._playback_until = max(self._playback_until, now) + num_bytes / PCM16_BYTES_PER_SECOND

    def note_assistant_text(self, text: str) -> None:
        if text:
            self._assistant_text = (self._assistant_text + text)[-ASSISTANT_CONTEXT_CHARS:]

    def stop_playback(self) -> None:
        """The client stopped playing assistant audio (barge-in, clear, cancel)."""
        self._playback_until = 0.0

    def assistant_audible(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self._playback_until

    # ── User side ─────────────────────────────────────────────────────

    def offer(self, event: PipelineEvent, now: float | None = None) -> bool:
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
                # A later turn waits behind an undecided one so events keep
                # their order, whether or not the assistant is still audible.
                if not self._held and not self.assistant_audible(now):
                    return False
                held = _HeldTurn(turn_id=turn_id, started_at=now, latest_revision=revision)
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

    def poll(self, now: float | None = None) -> list[PipelineEvent]:
        """Advance classifications and return released events, oldest first."""
        now = time.monotonic() if now is None else now
        for held in self._held:
            if held.verdict is None:
                self._advance(held, now)
        released: list[PipelineEvent] = []
        while self._held and self._held[0].verdict is not None:
            held = self._held.pop(0)
            if held.task is not None and not held.task.done():
                held.task.cancel()
            if held.verdict == "interrupt":
                logger.info("Backchannel gate: releasing turn %s as an interruption (%s)", held.turn_id, held.reason)
                released.extend(held.events)
            else:
                logger.info("Backchannel gate: dropping turn %s as a backchannel (%s)", held.turn_id, held.reason)
                self._dropped.extend((held.turn_id, revision) for revision in held.revisions)
        return released

    def has_held_turns(self) -> bool:
        return bool(self._held)

    def close(self) -> None:
        for held in self._held:
            if held.task is not None and not held.task.done():
                held.task.cancel()
        self._held.clear()

    # ── Internals ─────────────────────────────────────────────────────

    def _find(self, turn_id: str) -> _HeldTurn | None:
        for held in self._held:
            if held.turn_id == turn_id:
                return held
        return None

    def _observe(self, held: _HeldTurn, event: PipelineEvent) -> None:
        revision = getattr(event, "turn_revision", None)
        if isinstance(event, SpeechStartedEvent):
            # Resumed speech supersedes any earlier transcript of this turn.
            held.latest_revision = revision
            held.queued_text = None
            return
        if revision != held.latest_revision or held.verdict is not None:
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
                self._decide(held, "backchannel", "empty transcript")
        elif isinstance(event, (TranscriptionFailedEvent, AudioInputCompletedEvent)):
            self._decide(held, "interrupt", f"no transcript to classify ({event.type})")

    def _advance(self, held: _HeldTurn, now: float) -> None:
        if held.task is not None and held.task.done():
            self._apply_result(held)
        if held.verdict is not None:
            return
        if held.task is None and held.queued_text is not None:
            held.task_text, held.queued_text = held.queued_text, None
            held.task = asyncio.ensure_future(
                self._classifier.interrupt_probability(self._assistant_text, held.task_text[0])
            )
        if (now - held.started_at) * 1000 >= self._config.max_hold_ms:
            self._decide(held, "interrupt", f"held longer than {self._config.max_hold_ms}ms")

    def _apply_result(self, held: _HeldTurn) -> None:
        task, (text, is_final, revision) = held.task, held.task_text  # type: ignore[misc]
        held.task, held.task_text = None, None
        assert task is not None
        try:
            probability = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._decide(held, "interrupt", f"classifier error: {exc!r}")
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
