"""Move private prefill and read-only tool work before final transcription.

The caller owns turn lifecycle and final tool planning. This controller never
accepts assistant output, commits SpeculativeTurnTracker, or alters chat history.
All methods run on one asyncio loop; one controller belongs to one connection.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


class Action(str, Enum):
    WAIT = "WAIT"
    SPECULATE = "SPECULATE"
    COMMIT = "COMMIT"
    REVISE = "REVISE"
    CANCEL = "CANCEL"


class PrefixPolicy(Protocol):
    def stable_prefix(self, hypotheses: Sequence[str]) -> str: ...


@dataclass(frozen=True)
class RepeatedPrefixPolicy:
    """Untrained baseline: repeated words, excluding the current last word."""

    observations: int = 2
    min_words: int = 3

    def __post_init__(self) -> None:
        if self.observations < 2 or self.min_words < 1:
            raise ValueError("require at least two observations and one word")

    def stable_prefix(self, hypotheses: Sequence[str]) -> str:
        if len(hypotheses) < self.observations:
            return ""
        words = [text.split() for text in hypotheses[-self.observations :]]
        stable = []
        for group in zip(*words):
            if len(set(group)) != 1:
                break
            stable.append(group[0])
        # Even a repeated last word may be an incomplete ASR word.
        stable = stable[: min(len(stable), max(0, len(words[-1]) - 1))]
        return " ".join(stable) if len(stable) >= self.min_words else ""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str

    @classmethod
    def create(cls, name: str, arguments: Mapping[str, Any]) -> ToolCall:
        return cls(name, json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"), allow_nan=False))

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)


@dataclass(frozen=True)
class Tool:
    execute: Callable[[dict[str, Any]], Awaitable[Any]]
    allow_speculation: bool = False


class PrefillBackend(Protocol):
    async def prefill(self, text: str) -> Any: ...


@dataclass
class Finalized:
    """Private results, usable only by the caller's validated final plan."""

    action: Action
    tools: dict[ToolCall, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


ToolPlanner = Callable[[str, bool], Sequence[ToolCall]]


class TurnSpeculation:
    def __init__(
        self,
        tracker: SpeculativeTurnTracker,
        *,
        backend: PrefillBackend | None = None,
        policy: PrefixPolicy | None = None,
        tools: Mapping[str, Tool] | None = None,
        planner: ToolPlanner | None = None,
        max_tool_launches: int = 4,
        max_parallel_tools: int = 2,
        timeout_s: float = 5.0,
    ) -> None:
        if max_tool_launches < 0 or max_parallel_tools < 1 or timeout_s <= 0:
            raise ValueError("invalid speculation limits")
        self.tracker = tracker
        self.backend = backend
        self.policy = policy or RepeatedPrefixPolicy()
        self.tools = dict(tools or {})
        self.planner = planner or (lambda text, final: ())
        self.max_tool_launches = max_tool_launches
        self.max_parallel_tools = max_parallel_tools
        self.timeout_s = timeout_s
        self.owner: tuple[str, int] | None = None
        self.hypotheses: list[str] = []
        self.prefix = ""
        self.actions: list[Action] = []
        self.errors: list[str] = []
        self._tasks: dict[ToolCall, asyncio.Task[Any]] = {}
        self._retired: set[asyncio.Task[Any]] = set()
        self._launches = 0
        self._pending_prefix: str | None = None
        self._prefill_task: asyncio.Task[None] | None = None
        self._final = False
        self._epoch = 0
        self._lifecycle_lock = asyncio.Lock()

    def _relevant(self) -> bool:
        return (
            self.owner is not None
            and self.tracker.is_current_turn(self.owner[0])
            and self.tracker.is_latest(*self.owner)
            and not self.tracker.is_committed(*self.owner)
        )

    async def begin(self, turn_id: str, revision: int) -> None:
        """Bind work to an already allocated turn; never allocate/observe a turn."""
        async with self._lifecycle_lock:
            await self._cancel_locked()
            self.owner = (turn_id, revision)
            if not self._relevant():
                self.owner = None
                raise ValueError("turn must be current and uncommitted in the shared tracker")
            self.hypotheses = []
            self.prefix = ""
            self.actions = []
            self.errors = []
            self._launches = 0
            self._final = False

    async def update(self, text: str) -> Action:
        if not self._relevant() or self._final:
            await self.cancel()
            return Action.CANCEL
        self.hypotheses.append(text)
        # Retain bounded recent hypotheses, including unchanged updates.
        self.hypotheses = self.hypotheses[-32:]
        prefix = self.policy.stable_prefix(self.hypotheses)
        revised = bool(self.prefix and not prefix.startswith(self.prefix))
        if prefix != self.prefix:
            self.prefix = prefix
            if prefix and self.backend is not None:
                self._pending_prefix = prefix
                if self._prefill_task is None or self._prefill_task.done():
                    self._prefill_task = asyncio.create_task(self._prefill())
            elif not prefix:
                self._pending_prefix = None
        # Planner receives the full replacement hypothesis, so corrections and
        # late constraints invalidate a broad call even before they are stable.
        proposed = set(self.planner(text, False)) if prefix else set()
        revised = revised or bool(set(self._tasks) - proposed)
        self._invalidate(proposed)
        for call in proposed:
            tool = self.tools.get(call.name)
            if tool is None or not tool.allow_speculation or call in self._tasks:
                continue
            if (
                self._launches >= self.max_tool_launches
                or sum(not t.done() for t in self._tasks.values()) + sum(not t.done() for t in self._retired)
                >= self.max_parallel_tools
            ):
                continue
            self._tasks[call] = asyncio.create_task(self._run_tool(tool, call))
            self._launches += 1
        action = Action.REVISE if revised else Action.SPECULATE if prefix else Action.WAIT
        self.actions.append(action)
        return action

    def _invalidate(self, proposed: set[ToolCall]) -> None:
        for call in list(self._tasks):
            if call not in proposed:
                task = self._tasks.pop(call)
                task.cancel()
                self._retired.add(task)

    async def _run_tool(self, tool: Tool, call: ToolCall) -> Any:
        return await asyncio.wait_for(tool.execute(call.arguments), self.timeout_s)

    async def _prefill(self) -> None:
        # One worker; rapid partials replace pending work instead of queuing it.
        while self._pending_prefix is not None and self._relevant() and not self._final:
            prefix, self._pending_prefix = self._pending_prefix, None
            try:
                assert self.backend is not None
                await asyncio.wait_for(self.backend.prefill(prefix), self.timeout_s)
            except Exception as exc:
                self.errors.append(type(exc).__name__)

    async def finalize(self, text: str) -> Finalized:
        """Validate the final plan; do not launch missing calls or release audio.

        Missing/failed speculative calls use the normal tool path. The caller
        must render the complete final prompt before decoding, including all
        final constraints and this turn's fixed conversation/config snapshot.
        """
        if not self._relevant() or self._final or not text.strip():
            await self.cancel()
            return Finalized(Action.CANCEL)
        epoch = self._epoch
        self._final = True
        self._pending_prefix = None
        approved = set(self.planner(text, True))
        self._invalidate(approved)
        tasks = list(self._tasks.values()) + list(self._retired)
        if self._prefill_task is not None:
            tasks.append(self._prefill_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if epoch != self._epoch:
            return Finalized(Action.CANCEL)
        owner = self.owner
        if not self._relevant() or owner is None or self.tracker.try_is_latest_after_reopen_grace(*owner) is not True:
            await self.cancel()
            return Finalized(Action.CANCEL)
        results = {}
        for call, task in self._tasks.items():
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                self.errors.append(type(error).__name__)
            else:
                results[call] = task.result()
        self._tasks.clear()
        self._retired.clear()
        self.actions.append(Action.COMMIT)
        return Finalized(Action.COMMIT, results, list(self.errors))

    async def cancel(self) -> None:
        async with self._lifecycle_lock:
            await self._cancel_locked()

    async def _cancel_locked(self) -> None:
        self._epoch += 1
        self.owner = None
        self._pending_prefix = None
        tasks = list(self._tasks.values()) + list(self._retired)
        if self._prefill_task is not None:
            tasks.append(self._prefill_task)
        self._tasks.clear()
        self._retired.clear()
        self._prefill_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
