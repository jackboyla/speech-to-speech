"""Private prefill adapter for a dedicated llama.cpp server slot.

Use the same adapter, server slot and frozen prompt renderer for final decode.
Full rendered prompts are tokenized afresh: character-prefix stability alone
cannot establish token-prefix stability at a word or chat-template boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx


class LlamaPrefill:
    def __init__(
        self,
        client: httpx.AsyncClient,
        render_prompt: Callable[[str], str],
        *,
        slot_id: int = 0,
    ) -> None:
        if slot_id < 0:
            raise ValueError("a dedicated slot is required")
        self.client = client
        self.render_prompt = render_prompt
        self.slot_id = slot_id
        self._lock = asyncio.Lock()
        self.reports: list[dict[str, Any]] = []

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    async def complete(self, text: str, *, n_predict: int = 1, cache_prompt: bool = True) -> dict[str, Any]:
        if n_predict < 0:
            raise ValueError("use a bounded decode budget")
        async with self._lock:
            tokens = await self._post(
                "/tokenize", {"content": self.render_prompt(text), "add_special": True, "parse_special": True}
            )
            result = await self._post(
                "/completion",
                {
                    "prompt": tokens["tokens"],
                    "n_predict": n_predict,
                    "id_slot": self.slot_id,
                    "cache_prompt": cache_prompt,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                },
            )
            if n_predict == 0:
                # Some llama.cpp builds sample once even for n_predict=0.
                # That sample is never used, exposed, or forwarded to decode.
                sampled = result.get("tokens_predicted", 0)
                if sampled > 1:
                    raise RuntimeError("server exceeded the private prefill budget")
                result = {key: value for key, value in result.items() if key not in ("content", "tokens", "prompt")}
                result["discarded_samples"] = sampled
            # Retain metrics only; do not log conversation content.
            self.reports.append(
                {
                    key: result[key]
                    for key in ("timings", "tokens_cached", "tokens_evaluated", "discarded_samples")
                    if key in result
                }
            )
            return result

    async def prefill(self, text: str) -> dict[str, Any]:
        return await self.complete(text, n_predict=0)
