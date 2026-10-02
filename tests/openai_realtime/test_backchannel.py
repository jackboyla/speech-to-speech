"""Tests for the backchannel gate that holds barge-ins during assistant playback."""

import asyncio
import json

import httpx
import pytest

from speech_to_speech.api.openai_realtime.backchannel import (
    PCM16_BYTES_PER_SECOND,
    BackchannelConfig,
    BackchannelGate,
    SystemOneClassifier,
)
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)


class FakeClassifier:
    """Interrupt probability per user text; unknown text raises."""

    def __init__(self, scores: dict[str, float], delay_s: float = 0.0) -> None:
        self.scores = scores
        self.delay_s = delay_s
        self.calls: list[tuple[str, str]] = []

    async def interrupt_probability(self, assistant_text: str, user_text: str) -> float:
        self.calls.append((assistant_text, user_text))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self.scores[user_text]


def _gate(scores: dict[str, float], **config) -> tuple[BackchannelGate, FakeClassifier]:
    classifier = FakeClassifier(scores)
    gate = BackchannelGate(classifier, BackchannelConfig(url="http://unused", **config))
    gate.note_assistant_audio(PCM16_BYTES_PER_SECOND * 10)  # ten seconds of playback
    return gate, classifier


def _turn(turn_id: str, transcript: str, revision: int = 0):
    return [
        SpeechStartedEvent(turn_id=turn_id, turn_revision=revision),
        SpeechStoppedEvent(duration_s=0.5, turn_id=turn_id, turn_revision=revision),
        TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=revision),
    ]


async def _settle(gate: BackchannelGate) -> list:
    released: list = []
    for _ in range(20):
        await asyncio.sleep(0)
        released.extend(gate.poll())
    return released


async def test_turn_passes_through_when_assistant_is_silent():
    gate, classifier = _gate({})
    gate.stop_playback()
    assert not gate.offer(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    assert classifier.calls == []


async def test_untracked_turn_passes_through():
    gate, _ = _gate({})
    assert not gate.offer(SpeechStartedEvent())


async def test_backchannel_is_dropped_and_its_late_events_are_swallowed():
    gate, _ = _gate({"Mm-hmm.": 0.1})
    for event in _turn("t1", "Mm-hmm."):
        assert gate.offer(event)
    assert await _settle(gate) == []
    assert not gate.has_held_turns()
    assert gate.offer(SpeechStoppedEvent(turn_id="t1", turn_revision=0))


async def test_interruption_releases_held_events_in_order():
    gate, _ = _gate({"Wait.": 0.9})
    events = _turn("t1", "Wait.")
    for event in events:
        gate.offer(event)
    assert await _settle(gate) == events
    # Later events of a released turn are no longer held.
    assert not gate.offer(SpeechStoppedEvent(turn_id="t1", turn_revision=0))


async def test_partial_transcript_can_release_but_not_drop():
    gate, _ = _gate({"Okay": 0.2, "Okay, stop.": 0.95})
    gate.offer(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    gate.offer(PartialTranscriptionEvent(delta="Okay", turn_id="t1", turn_revision=0))
    assert await _settle(gate) == []
    assert gate.has_held_turns()

    gate.offer(PartialTranscriptionEvent(delta="Okay, stop.", turn_id="t1", turn_revision=0))
    released = await _settle(gate)
    assert [type(e) for e in released] == [SpeechStartedEvent, PartialTranscriptionEvent, PartialTranscriptionEvent]


async def test_resumed_speech_overrides_backchannel_verdict_of_earlier_revision():
    gate, _ = _gate({"Okay so": 0.2, "Okay so what about the price?": 0.9})
    for event in _turn("t1", "Okay so", revision=0):
        gate.offer(event)
    gate.offer(SpeechStartedEvent(turn_id="t1", turn_revision=1, reopened=True))
    assert await _settle(gate) == []
    assert gate.has_held_turns()

    resumed = _turn("t1", "Okay so what about the price?", revision=1)[1:]
    for event in resumed:
        gate.offer(event)
    released = await _settle(gate)
    assert [(e.type, e.turn_revision) for e in released] == [
        ("speech_started", 0),
        ("speech_stopped", 0),
        ("transcription_completed", 0),
        ("speech_started", 1),
        ("speech_stopped", 1),
        ("transcription_completed", 1),
    ]


async def test_empty_transcript_is_dropped_without_classification():
    gate, classifier = _gate({})
    for event in _turn("t1", ""):
        gate.offer(event)
    assert await _settle(gate) == []
    assert classifier.calls == []


async def test_vad_discarded_segment_is_dropped():
    gate, _ = _gate({})
    gate.offer(SpeechStartedEvent(turn_id="t1", turn_revision=0))
    gate.offer(SpeechStoppedEvent(turn_id="t1", turn_revision=0))
    assert await _settle(gate) == []
    assert not gate.has_held_turns()


async def test_classifier_error_releases_turn():
    gate, _ = _gate({})  # every lookup raises KeyError
    events = _turn("t1", "Hmm, I wonder.")
    for event in events:
        gate.offer(event)
    assert await _settle(gate) == events


async def test_transcription_failure_releases_turn():
    gate, classifier = _gate({})
    started = SpeechStartedEvent(turn_id="t1", turn_revision=0)
    failed = TranscriptionFailedEvent(message="boom", turn_id="t1", turn_revision=0)
    gate.offer(started)
    gate.offer(failed)
    assert await _settle(gate) == [started, failed]
    assert classifier.calls == []


async def test_long_hold_releases_turn():
    gate, _ = _gate({}, max_hold_ms=100)
    started = SpeechStartedEvent(turn_id="t1", turn_revision=0)
    gate.offer(started, now=0.0)
    assert gate.poll(now=0.05) == []
    assert gate.poll(now=0.2) == [started]


async def test_later_turn_waits_behind_undecided_turn():
    gate, classifier = _gate({"Mm-hmm.": 0.1, "Wait.": 0.9})
    classifier.delay_s = 0.01
    first = _turn("t1", "Mm-hmm.")
    second = _turn("t2", "Wait.")
    for event in first:
        gate.offer(event)
    gate.stop_playback()
    # Held even though playback ended, so it cannot overtake t1.
    assert gate.offer(second[0])
    for event in second[1:]:
        gate.offer(event)
    released: list = []
    for _ in range(50):
        await asyncio.sleep(0.005)
        released.extend(gate.poll())
    assert released == second


async def test_playback_estimate_follows_audio_sent():
    gate = BackchannelGate(FakeClassifier({}), BackchannelConfig(url="http://unused"))
    gate.note_assistant_audio(PCM16_BYTES_PER_SECOND, now=100.0)
    gate.note_assistant_audio(PCM16_BYTES_PER_SECOND, now=100.1)
    assert gate.assistant_audible(now=101.9)
    assert not gate.assistant_audible(now=102.1)


async def test_classifier_sees_recent_assistant_text():
    gate, classifier = _gate({"Yeah.": 0.1})
    gate.note_assistant_text("x" * 500)
    gate.note_assistant_text("The second flight leaves at noon.")
    for event in _turn("t1", "Yeah."):
        gate.offer(event)
    await _settle(gate)
    assistant_text, _ = classifier.calls[0]
    assert assistant_text.endswith("The second flight leaves at noon.")
    assert len(assistant_text) == 400


async def test_systemone_classifier_request_and_response():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "turn": {
                        "type": "choice",
                        "choice": "interruption",
                        "probabilities": {"backchannel": 0.2, "interruption": 0.8},
                    }
                }
            },
        )

    classifier = SystemOneClassifier(BackchannelConfig(url="http://dm:8093/", model="ggml-org/Kev-4B-GGUF"))
    classifier._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await classifier.interrupt_probability("I found three flights", "Wait.") == pytest.approx(0.8)
    await classifier.aclose()

    assert seen["url"] == "http://dm:8093/v1/systemone"
    assert seen["body"]["model"] == "ggml-org/Kev-4B-GGUF"
    assert '"I found three flights"' in seen["body"]["state"]
    assert '"Wait."' in seen["body"]["state"]
    assert set(seen["body"]["questions"]["turn"]["criteria"]) == {"backchannel", "interruption"}
