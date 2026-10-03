"""Paired cache and tool-overlap measurements; run with --help for options."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from time import perf_counter

import httpx

from speech_to_speech.experimental.llama_prefill import LlamaPrefill
from speech_to_speech.experimental.transcription_speculation import TranscriptionSpeculation
from speech_to_speech.experimental.turn_speculation import RepeatedPrefixPolicy, Tool, ToolCall, TurnSpeculation
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

CASES = [
    {
        "name": "ipv4",
        "partials": [
            "What is the difference",
            "What is the difference between",
            "What is the difference between IPv",
            "What is the difference between IPv4 and",
            "What is the difference between IPv4 and IPv6",
        ],
        "final": "What is the difference between IPv4 and IPv6?",
    },
    {
        "name": "weather_late_date",
        "partials": [
            "Can you tell me the weather",
            "Can you tell me the weather in",
            "Can you tell me the weather in Dublin",
            "Can you tell me the weather in Dublin tomorrow",
        ],
        "final": "Can you tell me the weather in Dublin tomorrow morning?",
    },
    {
        "name": "destination_revision",
        "partials": [
            "Find me a flight from Dublin",
            "Find me a flight from Dublin to Rome",
            "Find me a flight from Dublin to Rome next week",
            "Find me a flight from Dublin to Milan next week",
        ],
        "final": "Find me a flight from Dublin to Milan next week.",
    },
    {
        "name": "asr_correction",
        "partials": [
            "Explain how cash invalidation works",
            "Explain how cache invalidation works",
            "Explain how cache invalidation works in",
            "Explain how cache invalidation works in a distributed system",
        ],
        "final": "Explain how cache invalidation works in a distributed system.",
    },
    {
        "name": "long_request",
        "partials": [
            "Compare the benefits and costs of running",
            "Compare the benefits and costs of running a speech agent",
            "Compare the benefits and costs of running a speech agent on local hardware rather than",
            "Compare the benefits and costs of running a speech agent on local hardware rather than a hosted service with particular attention to latency",
            "Compare the benefits and costs of running a speech agent on local hardware rather than a hosted service with particular attention to latency privacy memory use cancellation and maintenance",
        ],
        "final": "Compare the benefits and costs of running a speech agent on local hardware rather than a hosted service with particular attention to latency privacy memory use cancellation and maintenance.",
    },
]


async def prefill_benchmark(args):
    rows = []
    async with httpx.AsyncClient(base_url=args.url, timeout=30) as client:
        props = (await client.get("/props")).json()
        for repeat in range(args.repeats):
            for case in CASES:
                # Identical conversation in both arms; warm history in both.
                history = "User prefers clear short answers. " * args.history_repeats
                messages = [
                    {"role": "system", "content": "Answer briefly. " + history},
                    {"role": "user", "content": "SPECULATION_INPUT_MARKER"},
                ]
                response = await client.post("/apply-template", json={"messages": messages})
                response.raise_for_status()
                template = response.json()["prompt"]
                before, marker, after = template.partition("SPECULATION_INPUT_MARKER")
                if not marker:
                    raise RuntimeError("server template dropped the user marker")
                backend = LlamaPrefill(client, lambda text: before + text + after)
                arms = {}
                # Alternate order to limit order and thermal bias.
                for candidate in [False, True] if repeat % 2 == 0 else [True, False]:
                    await backend.complete("", n_predict=0, cache_prompt=False)
                    prefix_work = []
                    runtime = None
                    adapter = None
                    if candidate and args.partial_interval > 0:
                        tracker = SpeculativeTurnTracker()
                        owner = tracker.start_turn()
                        runtime = TurnSpeculation(tracker, backend=backend)
                        await runtime.begin(*owner)
                        adapter = TranscriptionSpeculation(runtime)
                        first_report = len(backend.reports)
                        for partial in case["partials"]:
                            await adapter.observe(
                                PartialTranscriptionEvent(delta=partial, turn_id=owner[0], turn_revision=owner[1])
                            )
                            await asyncio.sleep(args.partial_interval)
                        prefix_work = backend.reports[first_report:]
                    elif candidate:
                        policy = RepeatedPrefixPolicy()
                        hypotheses = []
                        previous = ""
                        for partial in case["partials"]:
                            hypotheses.append(partial)
                            stable = policy.stable_prefix(hypotheses)
                            if stable and stable != previous:
                                t0 = perf_counter()
                                result = await backend.prefill(stable)
                                prefix_work.append(
                                    {
                                        "wall_ms": (perf_counter() - t0) * 1000,
                                        "metrics": result.get("timings"),
                                        "tokens_cached": result.get("tokens_cached"),
                                        "prefix_words": len(stable.split()),
                                        "discarded_samples": result.get("discarded_samples"),
                                    }
                                )
                                previous = stable
                    t0 = perf_counter()
                    if adapter is not None:
                        await adapter.finalize(
                            TranscriptionCompletedEvent(
                                transcript=case["final"], turn_id=owner[0], turn_revision=owner[1]
                            )
                        )
                    result = await backend.complete(case["final"], n_predict=1)
                    wall_ms = (perf_counter() - t0) * 1000
                    arms["candidate" if candidate else "baseline"] = {
                        "final_one_token_wall_ms": wall_ms,
                        "metrics": result.get("timings"),
                        "tokens_cached": result.get("tokens_cached"),
                        "tokens_evaluated": result.get("tokens_evaluated"),
                        "first_token": result.get("content"),
                        "prefix_work": prefix_work,
                        "controller_actions": [action.value for action in runtime.actions]
                        if runtime is not None
                        else [],
                    }
                rows.append({"case": case["name"], "repeat": repeat, **arms})
    baseline = [row["baseline"]["final_one_token_wall_ms"] for row in rows]
    candidate = [row["candidate"]["final_one_token_wall_ms"] for row in rows]
    return {
        "kind": "real_llama_cpp_cpu_timed_partials" if args.partial_interval > 0 else "real_llama_cpp_cpu_prefill",
        "partial_interval_s": args.partial_interval,
        "limits": "Handwritten replacement transcripts, no audio/STT/TTS, one-token nonstream wall time includes tokenize/HTTP. Both arms warm identical conversation history. With partial_interval=0 prefix work completes before final timing (overlap upper bound). With a positive interval the actual controller consumes timed synthetic notifier events and final timing includes pending prefill. Neither measures end-to-end voice latency.",
        "server_props": props,
        "history_repeats": args.history_repeats,
        "median_baseline_ms": statistics.median(baseline),
        "median_candidate_ms": statistics.median(candidate),
        "median_paired_saving_ms": statistics.median(a - b for a, b in zip(baseline, candidate)),
        "matching_first_tokens": sum(row["baseline"]["first_token"] == row["candidate"]["first_token"] for row in rows),
        "rows": rows,
    }


async def tool_benchmark(args):
    rows = []
    for scenario in ("unchanged", "late_filter", "corrected_destination"):
        for candidate in (False, True):
            launched = []
            cancelled = []

            async def search(arguments):
                launched.append(arguments)
                try:
                    await asyncio.sleep(args.tool_delay)
                except asyncio.CancelledError:
                    cancelled.append(arguments)
                    raise
                return [
                    {"title": "Speech agents", "year": 2026, **arguments},
                    {"title": "Older paper", "year": 2025, **arguments},
                ]

            def plan(text, final):
                # Example-only planner: broad retrieval is valid for the late
                # year filter, but a changed destination requires a new call.
                return [ToolCall.create("search", {"query": "Milan" if "Milan" in text else "speech agents"})]

            tracker = SpeculativeTurnTracker()
            owner = tracker.start_turn()
            c = TurnSpeculation(tracker, tools={"search": Tool(search, True)}, planner=plan)
            await c.begin(*owner)
            if candidate:
                await c.update("Search for recent papers about speech agents")
                await c.update("Search for recent papers about speech agents and")
            # Controlled synthetic time remaining in user speech.
            await asyncio.sleep(args.speaking_tail)
            final = "Search for recent papers about speech agents"
            if scenario == "late_filter":
                final += " only from this year"
            if scenario == "corrected_destination":
                final = "Actually search Milan"
            t0 = perf_counter()
            ready = await c.finalize(final)
            call = plan(final, True)[0]
            results = ready.tools.get(call)
            reused = results is not None
            if results is None:
                results = await search(call.arguments)
            if scenario == "late_filter":
                results = [row for row in results if row["year"] == 2026]
            rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate,
                    "post_final_ms": (perf_counter() - t0) * 1000,
                    "reused": reused,
                    "launches": launched,
                    "cancelled": cancelled,
                    "results": results,
                }
            )
            await c.cancel()
    return {
        "kind": "synthetic_tool_overlap",
        "tool_delay_s": args.tool_delay,
        "speaking_tail_s": args.speaking_tail,
        "limits": "Handwritten planner and asyncio.sleep tool/speech costs. Demonstrates scheduling and rejection, not real search quality or learned decisions.",
        "rows": rows,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prefill", "tools"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18093")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--partial-interval", type=float, default=0.0)
    parser.add_argument("--history-repeats", type=int, default=32)
    parser.add_argument("--tool-delay", type=float, default=1)
    parser.add_argument("--speaking-tail", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.history_repeats < 0
        or args.partial_interval < 0
        or args.tool_delay <= 0
        or args.speaking_tail < 0
    ):
        parser.error("invalid timing or repeat count")
    report = await (prefill_benchmark(args) if args.mode == "prefill" else tool_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "server_props")}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
