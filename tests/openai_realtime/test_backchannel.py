"""Tests for the backchannel gate that holds barge-ins during assistant playback."""

import asyncio
import json
from types import SimpleNamespace

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
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker, TurnPhase
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore


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


def _gate(scores: dict[str, float], *, turns=None, latency_store=None, **config):
    classifier = FakeClassifier(scores)
    gate = BackchannelGate(
        classifier,
        BackchannelConfig(url="http://unused", **config),
        speculative_turns=turns,
        turn_latency_store=latency_store,
    )
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


async def test_empty_transcript_releases_without_classification():
    gate, classifier = _gate({})
    events = _turn("t1", "")
    for event in events:
        gate.offer(event)
    assert await _settle(gate) == events
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
    gate, classifier = _gate({"Mm-hmm.": 0.1, "Yeah.": 0.1})
    classifier.delay_s = 0.01
    first = _turn("t1", "Mm-hmm.")
    second = _turn("t2", "Yeah.")
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
    assert [text for _, text in classifier.calls] == ["Mm-hmm."]


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
    gate.note_assistant_text("This text arrived after the user started speaking.")
    await _settle(gate)
    assistant_text, _ = classifier.calls[0]
    assert assistant_text.endswith("The second flight leaves at noon.")
    assert len(assistant_text) == 400


async def test_backchannel_waits_for_reopen_grace_and_closes_each_dropped_turn(monkeypatch):
    now = 0.0
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time", SimpleNamespace(monotonic=lambda: now))
    turns = SpeculativeTurnTracker()
    latency_store = TurnLatencyStore()
    gate, _ = _gate({"Mm-hmm.": 0.1}, turns=turns, latency_store=latency_store)
    previous_turn = None
    for audio_ms in (0, 1000):
        turn_id, revision, reopened = turns.speech_started(audio_ms)
        assert not reopened
        assert turn_id != previous_turn
        original_latency = latency_store.get_or_create_for_turn(turn_id, revision)
        turns.segment_finalized(audio_ms + 400, output_hold_ms=40)
        for event in _turn(turn_id, "Mm-hmm.", revision):
            gate.offer(event)
        assert await _settle(gate) == []
        assert gate.has_held_turns()
        assert turns.phase is TurnPhase.SOFT_ENDED
        now += 0.06
        assert gate.poll() == []
        assert not gate.has_held_turns()
        assert turns.phase is TurnPhase.CLOSED
        assert latency_store.get_or_create_for_turn(turn_id, revision) is not original_latency
        latency_store.discard_pending_turn(turn_id, revision)
        previous_turn = turn_id


@pytest.mark.parametrize("old_transcript", ["Wait.", "classifier fails"])
async def test_stale_result_does_not_interrupt_before_reopen_event_arrives(old_transcript):
    turns = SpeculativeTurnTracker()
    gate, classifier = _gate({"Wait.": 0.9, "Mm-hmm.": 0.1}, turns=turns)
    classifier.delay_s = 0.01
    turn_id, revision, _ = turns.speech_started(0)
    turns.segment_finalized(400)
    for event in _turn(turn_id, old_transcript, revision):
        gate.offer(event)
    gate.poll()
    await asyncio.sleep(0)
    _, resumed_revision, reopened = turns.speech_started(500)
    assert reopened
    await asyncio.sleep(0.02)
    assert gate.poll() == []
    assert gate.has_held_turns()
    gate.offer(SpeechStartedEvent(turn_id=turn_id, turn_revision=resumed_revision, reopened=True))
    turns.segment_finalized(900)
    for event in _turn(turn_id, "Mm-hmm.", resumed_revision)[1:]:
        gate.offer(event)
    gate.poll()
    await asyncio.sleep(0.02)
    assert gate.poll() == []
    assert not gate.has_held_turns()
    assert turns.phase is TurnPhase.CLOSED


async def test_resumed_speech_replaces_pending_backchannel_before_commit():
    turns = SpeculativeTurnTracker()
    gate, _ = _gate({"Okay": 0.1, "Okay, stop.": 0.9}, turns=turns)
    turn_id, revision, _ = turns.speech_started(0)
    turns.segment_finalized(400, output_hold_ms=100)
    for event in _turn(turn_id, "Okay", revision):
        gate.offer(event)
    assert await _settle(gate) == []
    assert gate.has_held_turns()
    _, resumed_revision, _ = turns.speech_started(500)
    # Poll before the queued speech event: a stale backchannel must not commit.
    assert gate.poll() == []
    gate.offer(SpeechStartedEvent(turn_id=turn_id, turn_revision=resumed_revision, reopened=True))
    turns.segment_finalized(900)
    for event in _turn(turn_id, "Okay, stop.", resumed_revision)[1:]:
        gate.offer(event)
    released = await _settle(gate)
    assert released[-1].transcript == "Okay, stop."
    assert not turns.is_committed(turn_id, resumed_revision)


async def test_disabled_interruption_bypasses_gate_and_releases_pending_input():
    gate, classifier = _gate({})
    assert not gate.offer(SpeechStartedEvent(turn_id="t1", turn_revision=0), enabled=False)
    assert not gate.offer(SpeechStartedEvent(turn_id="t2", turn_revision=0, interrupt_response=False))
    events = _turn("t3", "Yeah.")
    for event in events:
        gate.offer(event)
    assert gate.poll(enabled=False) == events
    assert classifier.calls == []


async def test_close_cancels_outstanding_classification():
    gate, classifier = _gate({"Yeah.": 0.1})
    classifier.delay_s = 10
    for event in _turn("t1", "Yeah."):
        gate.offer(event)
    previous_tasks = asyncio.all_tasks()
    gate.poll()
    await asyncio.sleep(0)
    tasks = asyncio.all_tasks() - previous_tasks
    assert tasks
    gate.close()
    await asyncio.sleep(0)
    assert not gate.has_held_turns()
    assert all(task.done() for task in tasks)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "localhost:8093"},
        {"url": "ftp://localhost"},
        {"threshold": float("nan")},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"max_hold_ms": 0},
        {"timeout_ms": -1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BackchannelConfig(**({"url": "http://unused"} | kwargs))


@pytest.mark.parametrize("probability", ["NaN", "Infinity", -0.1, 1.1, None, False])
async def test_invalid_probability_releases_turn(probability):
    classifier = SystemOneClassifier(BackchannelConfig(url="http://unused"))
    classifier._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"answers": {"turn": {"probabilities": {"interruption": probability}}}})
        )
    )
    gate = BackchannelGate(classifier, BackchannelConfig(url="http://unused"))
    gate.note_assistant_audio(PCM16_BYTES_PER_SECOND)
    events = _turn("t1", "Wait.")
    try:
        for event in events:
            gate.offer(event)
        assert await _settle(gate) == events
    finally:
        await classifier.aclose()


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


async def test_explicit_playback_stop_releases_pending_speech():
    gate, classifier = _gate({"okay": 0.01})
    classifier.delay_s = 0.1
    events = _turn("t1", "okay")
    for event in events:
        assert gate.offer(event)
    gate.poll()
    await asyncio.sleep(0)
    gate.stop_playback(release_pending=True)
    assert gate.poll() == events
    assert not gate.assistant_audible()
    assert not gate.has_held_turns()
