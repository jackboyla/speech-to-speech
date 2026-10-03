import asyncio
from queue import Queue

import httpx
import pytest

from speech_to_speech.experimental.llama_prefill import LlamaPrefill
from speech_to_speech.experimental.transcription_speculation import TranscriptionSpeculation
from speech_to_speech.experimental.turn_speculation import Action, RepeatedPrefixPolicy, Tool, ToolCall, TurnSpeculation
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def controller(**kwargs):
    tracker = SpeculativeTurnTracker()
    owner = tracker.start_turn()
    return TurnSpeculation(tracker, **kwargs), owner


def test_prefix_handles_replacement_tail_and_empty_hypothesis():
    policy = RepeatedPrefixPolicy()
    assert (
        policy.stable_prefix(["What is the difference between IPv", "What is the difference between IPv4"])
        == "What is the difference between"
    )
    assert policy.stable_prefix(["weather in Dublin today", "weather in Milan today"]) == ""
    assert policy.stable_prefix(["What is the difference", ""]) == ""


async def test_notifier_events_drive_private_prefill_without_llm_or_protocol_output():
    calls = []

    class Backend:
        async def prefill(self, text):
            calls.append(text)

    c, owner = controller(backend=Backend())
    await c.begin(*owner)
    adapter = TranscriptionSpeculation(c)
    events = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=events)
    for text in ("What is the difference between IPv", "What is the difference between IPv4"):
        assert list(notifier.process(PartialTranscription(text=text, turn_id=owner[0], turn_revision=owner[1]))) == []
        event = events.get_nowait()
        assert isinstance(event, PartialTranscriptionEvent)
        await adapter.observe(event)
    await asyncio.sleep(0)
    assert (
        list(
            notifier.process(
                Transcription(
                    text="What is the difference between IPv4 and IPv6", turn_id=owner[0], turn_revision=owner[1]
                )
            )
        )
        == []
    )
    event = events.get_nowait()
    assert isinstance(event, TranscriptionCompletedEvent)
    result = await adapter.finalize(event)
    assert result.action == Action.COMMIT
    assert calls == ["What is the difference between"]
    assert events.empty()
    assert not c.tracker.is_committed(*owner)


async def test_corrected_destination_and_late_constraint_reject_old_tool_results():
    launched = []

    async def search(arguments):
        launched.append(arguments)
        return arguments

    def plan(text, final):
        city = "Milan" if "Milan" in text else "Dublin"
        date = "tomorrow" if "tomorrow" in text else "today"
        return [ToolCall.create("weather", {"city": city, "date": date})]

    c, owner = controller(tools={"weather": Tool(search, True)}, planner=plan)
    await c.begin(*owner)
    await c.update("Tell me the weather in Dublin")
    await c.update("Tell me the weather in Dublin today")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert launched == [{"city": "Dublin", "date": "today"}]
    result = await c.finalize("Tell me the weather in Milan tomorrow")
    assert result.tools == {}
    assert result.action == Action.COMMIT


async def test_valid_result_is_private_until_final_and_reused_once():
    count = 0
    call = ToolCall.create("search", {"query": "speech agents"})

    async def search(arguments):
        nonlocal count
        count += 1
        return ["paper"]

    c, owner = controller(tools={"search": Tool(search, True)}, planner=lambda text, final: [call])
    await c.begin(*owner)
    await c.update("Search for speech agents")
    await c.update("Search for speech agents now")
    await c.update("Search for speech agents now")
    result = await c.finalize("Search for speech agents now")
    assert result.tools == {call: ["paper"]}
    assert count == 1
    assert (await c.finalize("again")).action == Action.CANCEL


async def test_side_effect_tools_are_not_speculated_and_budget_is_bounded():
    calls = []

    async def run(arguments):
        calls.append(arguments)
        return 1

    def planner(text, final):
        return [ToolCall.create("read", {"text": text}), ToolCall.create("write", {})]

    c, owner = controller(tools={"read": Tool(run, True), "write": Tool(run)}, planner=planner, max_tool_launches=1)
    await c.begin(*owner)
    for text in ("Search for speech agents", "Search for speech agents now", "Search for speech agents today"):
        await c.update(text)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    await c.finalize("Search for speech agents today")
    assert len(calls) == 1
    assert "text" in calls[0]


async def test_new_turn_during_tool_wait_drops_result():
    started, release = asyncio.Event(), asyncio.Event()
    call = ToolCall.create("search", {})

    async def search(arguments):
        started.set()
        await release.wait()
        return "obsolete"

    c, owner = controller(tools={"search": Tool(search, True)}, planner=lambda text, final: [call])
    await c.begin(*owner)
    await c.update("Search for speech agents")
    await c.update("Search for speech agents now")
    await started.wait()
    final = asyncio.create_task(c.finalize("Search for speech agents now"))
    await asyncio.sleep(0)
    c.tracker.start_turn()
    release.set()
    result = await final
    assert result.action == Action.CANCEL
    assert result.tools == {}


async def test_stale_revision_failure_and_empty_final_cancel_work():
    c, owner = controller()
    await c.begin(*owner)
    c.tracker.observe(owner[0], owner[1] + 1)
    assert await c.update("What is the weather today") == Action.CANCEL
    with pytest.raises(ValueError):
        await c.begin(*owner)
    await c.begin(owner[0], owner[1] + 1)
    assert (await c.finalize("")).action == Action.CANCEL


async def test_prefill_coalesces_rapid_partials_and_failure_falls_back():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Backend:
        async def prefill(self, text):
            calls.append(text)
            entered.set()
            await release.wait()
            raise RuntimeError("backend unavailable")

    c, owner = controller(backend=Backend())
    await c.begin(*owner)
    await c.update("What is the weather in Dublin today")
    await c.update("What is the weather in Dublin today please")
    await entered.wait()
    for suffix in ("this morning", "this morning please", "this morning please thanks"):
        await c.update("What is the weather in Dublin today " + suffix)
    final = asyncio.create_task(c.finalize("What is the weather in Dublin today this morning please thanks"))
    await asyncio.sleep(0)
    release.set()
    result = await final
    assert result.action == Action.COMMIT
    assert result.errors == ["RuntimeError"]
    assert len(calls) == 1


async def test_tool_timeout_falls_back_without_speculative_result():
    async def slow(arguments):
        await asyncio.Event().wait()

    call = ToolCall.create("search", {})
    c, owner = controller(tools={"search": Tool(slow, True)}, planner=lambda text, final: [call], timeout_s=0.01)
    await c.begin(*owner)
    await c.update("Search for speech agents")
    await c.update("Search for speech agents now")
    result = await c.finalize("Search for speech agents now")
    assert result.tools == {}
    assert result.errors == ["TimeoutError"]


async def test_llama_adapter_tokenizes_whole_prompt_and_never_decodes_partial():
    requests = []

    def transport(request):
        import json

        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, len(body["content"]), 3]})
        return httpx.Response(200, json={"content": "" if body["n_predict"] == 0 else "answer", "tokens_cached": 2})

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
        backend = LlamaPrefill(client, lambda text: "frozen history\nuser:" + text + "\nassistant:")
        await backend.prefill("IPv")
        await backend.complete("IPv4 and IPv6")
    completions = [body for path, body in requests if path == "/completion"]
    assert [body["n_predict"] for body in completions] == [0, 1]
    assert all(isinstance(body["prompt"][0], int) for body in completions)
    assert all(body["id_slot"] == 0 and body["cache_prompt"] for body in completions)


async def test_cancel_then_new_begin_does_not_let_old_finalize_commit_new_results():
    entered = asyncio.Event()
    call = ToolCall.create("search", {})

    async def search(arguments):
        entered.set()
        await asyncio.Event().wait()

    c, owner = controller(tools={"search": Tool(search, True)}, planner=lambda text, final: [call])
    await c.begin(*owner)
    await c.update("Search for speech agents")
    await c.update("Search for speech agents now")
    await entered.wait()
    old_final = asyncio.create_task(c.finalize("Search for speech agents now"))
    await asyncio.sleep(0)
    next_owner = c.tracker.start_turn()
    await c.begin(*next_owner)
    assert (await old_final).action == Action.CANCEL
    assert c.owner == next_owner
    assert await c.update("New unrelated request") == Action.WAIT
    await c.cancel()


async def test_pending_reopen_never_releases_private_results():
    c, owner = controller()
    await c.begin(*owner)
    assert c.tracker.begin_reopen_candidate(*owner) == 1
    assert (await c.finalize("What is the weather today")).action == Action.CANCEL


async def test_llama_discards_compatibility_sample_and_rejects_excess_budget():
    import json

    sampled = 1

    def transport(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2]})
        body = json.loads(request.content)
        assert body["n_predict"] == 0
        return httpx.Response(
            200, json={"content": "private", "tokens": [42], "prompt": "private", "tokens_predicted": sampled}
        )

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
        backend = LlamaPrefill(client, lambda text: text)
        result = await backend.prefill("partial")
        assert result["discarded_samples"] == 1
        assert not {"content", "tokens", "prompt"} & result.keys()
        sampled = 2
        with pytest.raises(RuntimeError, match="budget"):
            await backend.prefill("partial")
