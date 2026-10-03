"""Client/service tests for tool follow-ups that finish around user speech.

The packaged ``talk`` / ``local`` client runs local tools and asks for a
follow-up response once their outputs are delivered. These tests connect that
client coordinator to the real Realtime service: client events go through the
same service handlers the WebSocket router uses, and server events come back to
the coordinator in order. Pipeline events stand in for the VAD, STT and model.
"""

import asyncio
from queue import Queue
from typing import Any

import pytest
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig, _ToolCallCoordinator
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.messages import AssistantToolCallPart, GenerateResponseRequest
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

TOOL_DEFINITION = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a value.",
    "parameters": {"type": "object", "properties": {}},
}


class _ClientServiceSession:
    """One packaged-client connection to an in-process Realtime service."""

    def __init__(self, runtime_config, should_listen) -> None:
        self.tracker = SpeculativeTurnTracker()
        self.prompts: Queue = Queue()
        self.service = RealtimeService(
            text_prompt_queue=self.prompts,
            should_listen=should_listen,
            speculative_turns=self.tracker,
        )
        self.conn_id = self.service.register()
        self.service._state(self.conn_id).runtime_config = runtime_config
        self.chat = runtime_config.chat
        self.release_tool = asyncio.Event()
        self.client_events: list[dict[str, Any]] = []
        self.server_events: list[Any] = []
        self.tool = _ToolCallCoordinator(
            self,
            RealtimeAudioClientConfig(tools=[TOOL_DEFINITION], tool_executor=self._executor),
        )

    async def _executor(self, _name, _arguments):
        await self.release_tool.wait()
        return {"forecast": "sunny"}

    # ── Transport ────────────────────────────────

    async def send(self, raw: dict[str, Any]) -> None:
        """Handle one client event as the WebSocket router does."""
        self.client_events.append(raw)
        event = self.service.parse_client_event(raw)
        assert event is not None, raw
        if event.type == "conversation.item.create":
            events = self.service.handle_conversation_item_create(self.conn_id, event)
        elif event.type == "response.create":
            result = self.service.handle_response_create(self.conn_id, event)
            events = [result] if result is not None else []
            for outgoing in events:
                if outgoing.type == "error":
                    outgoing.error.event_id = raw.get("event_id")
            if result is not None and result.type == "response.created":
                self.service.response.mark_response_created_sent(
                    self.conn_id, self.service._state(self.conn_id).current_response_key
                )
        else:
            raise AssertionError(f"unexpected client event {raw['type']}")
        self.receive(events)

    def receive(self, events: list[Any]) -> None:
        self.server_events.extend(events)
        for event in events:
            self.tool.handle_event(event)

    def dispatch(self, event) -> list[Any]:
        events = self.service.dispatch_pipeline_event(self.conn_id, event)
        self.receive(events)
        return events

    async def settle(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0)

    # ── Conversation steps ───────────────────────

    def user_turn(self, turn_id: str, transcript: str) -> GenerateResponseRequest | None:
        self.start_speech(turn_id)
        return self.finish_speech(turn_id, transcript)

    def start_speech(self, turn_id: str) -> None:
        assert self.tracker.start_turn() == (turn_id, 0)
        self.dispatch(SpeechStartedEvent(turn_id=turn_id, turn_revision=0, audio_start_ms=0))

    def finish_speech(self, turn_id: str, transcript: str) -> GenerateResponseRequest | None:
        self.dispatch(SpeechStoppedEvent(turn_id=turn_id, turn_revision=0, duration_s=1.0, audio_end_ms=1000))
        self.dispatch(TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=0))
        return self.prompts.get_nowait() if not self.prompts.empty() else None

    def call_tool(self, turn_id: str, request: GenerateResponseRequest, call_id: str = "call_1") -> None:
        """Answer a user turn with one tool call and end that response."""
        call = RealtimeConversationItemFunctionCall(
            type="function_call", id=f"fc_{call_id}", call_id=call_id, name="lookup", arguments="{}"
        )
        self.chat.add_provisional_generation_items(request.response_key, [call])
        self.dispatch(
            AssistantOutputEvent(
                response_key=request.response_key,
                turn_id=turn_id,
                turn_revision=0,
                parts=[
                    AssistantToolCallPart(
                        tool={
                            "type": "function_call",
                            **call.model_dump(include={"id", "call_id", "name", "arguments"}),
                        }
                    )
                ],
            )
        )
        self.dispatch(ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[call_id]))
        self.receive(self.service.finish_response(self.conn_id, response_key=request.response_key))

    def answer(self, turn_id: str, request: GenerateResponseRequest, text: str = "Sure.") -> None:
        self.chat.finalize_provisional_generation(request.response_key)
        self.dispatch(
            AssistantOutputEvent(response_key=request.response_key, text=text, turn_id=turn_id, turn_revision=0)
        )
        self.dispatch(ResponseGenerationDoneEvent(response_key=request.response_key))
        self.receive(self.service.finish_response(self.conn_id, response_key=request.response_key))

    # ── Views ────────────────────────────────────

    @property
    def client_event_types(self) -> list[str]:
        return [event["type"] for event in self.client_events]

    @property
    def server_event_types(self) -> list[str]:
        return [event.type for event in self.server_events]

    def responses_created_during_speech(self) -> list[str]:
        """Return responses the server opened between a speech start and its stop."""
        speaking = False
        opened = []
        for event in self.server_events:
            if event.type == "input_audio_buffer.speech_started":
                speaking = True
            elif event.type == "input_audio_buffer.speech_stopped":
                speaking = False
            elif event.type == "response.created" and speaking:
                opened.append(event.response.id)
        return opened


@pytest.fixture
async def session(runtime_config, should_listen):
    active = _ClientServiceSession(runtime_config, should_listen)
    yield active
    await active.tool.close()
    active.service.unregister(active.conn_id)


async def test_tool_finishing_while_user_is_idle_requests_a_follow_up(session):
    request = session.user_turn("turn_1", "What is the weather?")
    session.call_tool("turn_1", request)
    session.release_tool.set()
    await session.settle()

    assert session.client_event_types == ["conversation.item.create", "response.create"]
    assert session.server_event_types[-1] == "response.created"
    assert session.prompts.qsize() == 1


async def test_tool_finishing_during_user_speech_waits_for_the_turn(session):
    request = session.user_turn("turn_1", "What is the weather?")
    session.call_tool("turn_1", request)

    session.start_speech("turn_2")
    session.release_tool.set()
    await session.settle()

    # The result reaches the conversation at once, but nothing speaks over
    # the user.
    assert session.client_event_types == ["conversation.item.create"]
    assert session.responses_created_during_speech() == []
    assert session.prompts.empty()

    request = session.finish_speech("turn_2", "And tomorrow?")
    assert request is not None
    # The turn's own response sees the tool output before the new question.
    roles = [getattr(item, "type", None) for item in session.chat.buffer][-3:]
    assert roles == ["function_call", "function_call_output", "message"]

    session.answer("turn_2", request, "Sunny today, and tomorrow too.")
    await session.settle()

    # That response already used the result, so no separate follow-up follows.
    assert session.client_event_types == ["conversation.item.create"]
    assert session.prompts.empty()
    assert session.server_event_types.count("response.created") == 2


async def test_tool_finishing_during_unanswered_speech_follows_up_once_it_settles(session):
    request = session.user_turn("turn_1", "What is the weather?")
    session.call_tool("turn_1", request)

    session.start_speech("turn_2")
    session.release_tool.set()
    await session.settle()
    assert session.client_event_types == ["conversation.item.create"]

    session.dispatch(SpeechStoppedEvent(turn_id="turn_2", turn_revision=0, duration_s=1.0, audio_end_ms=1000))
    session.dispatch(TranscriptionFailedEvent(message="STT failed", turn_id="turn_2", turn_revision=0))
    assert "conversation.item.input_audio_transcription.failed" in session.server_event_types
    await session.settle()

    # Nothing answered the turn, so the tool result gets its own follow-up.
    assert session.client_event_types == ["conversation.item.create", "response.create"]
    assert session.server_event_types[-1] == "response.created"
    assert session.responses_created_during_speech() == []


async def test_barge_in_on_the_answer_that_used_the_result_keeps_the_hold(session):
    request = session.user_turn("turn_1", "What is the weather?")
    session.call_tool("turn_1", request)
    session.start_speech("turn_2")
    session.release_tool.set()
    await session.settle()

    request = session.finish_speech("turn_2", "And tomorrow?")
    session.chat.finalize_provisional_generation(request.response_key)
    session.dispatch(
        AssistantOutputEvent(response_key=request.response_key, text="Sunny", turn_id="turn_2", turn_revision=0)
    )

    # The user cuts that answer off. Server VAD cancels it, then announces
    # the new speech.
    session.start_speech("turn_3")
    assert session.server_event_types[-2:] == ["response.done", "input_audio_buffer.speech_started"]
    assert session.server_events[-2].response.status_details.reason == "turn_detected"
    await session.settle()
    assert session.client_event_types == ["conversation.item.create"]

    request = session.finish_speech("turn_3", "Sorry, go on.")
    session.answer("turn_3", request, "Sunny today and tomorrow.")
    await session.settle()

    assert session.client_event_types == ["conversation.item.create"]
    assert session.responses_created_during_speech() == []
