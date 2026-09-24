#!/usr/bin/env python3
"""A deliberately slow, deterministic language model.

Why this exists: on a fast local stack the pipeline answers a short question in
about 300 ms, while a speaker needs roughly 330 ms of audio before voice
activity detection even reports that they have started talking. The window in
which a reply is being generated but has not yet been heard — the window issue
#308 is about — is therefore shorter than the fastest possible interruption,
and the race cannot be driven by talking at it.

Slowing the model down widens that window to whatever is asked for. The bug
under test lives in turn bookkeeping, not in the model, so replacing the model
changes when the race can be hit and nothing about whether it exists.

The replies are fixed and keyed to the question, which is the other half of the
point: if the pipeline lets stale work through, the speaker hears "Paris" after
asking about planets, and no interpretation is needed.

Usage::

    python slow_llm.py --port 18820 --ttft-ms 3000

Then point the pipeline at it::

    speech-to-speech serve --llm_backend chat-completions \
        --responses_api_base_url http://127.0.0.1:18820/v1 --model_name slow-stub
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Keyed on a word in the question. Each answer names its own subject, so a reply
# arriving for a question the speaker has already replaced is obvious on sight.
ANSWERS: list[tuple[tuple[str, ...], str]] = [
    (("capital", "france", "paris"), "The capital of France is Paris."),
    (("planet", "solar"), "There are eight planets in the solar system."),
    (("weather",), "The weather in Paris is mild today."),
    (("boil", "temperature"), "Water boils at one hundred degrees Celsius."),
]
DEFAULT_ANSWER = "I am a fixed test reply."

app = FastAPI()
settings = {"ttft_ms": 3000, "rate_ms": 25}


def answer_for(messages: list[dict]) -> str:
    """Pick the canned answer for the last thing the user said."""
    text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                text = content.lower()
            elif isinstance(content, list):
                text = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).lower()
            break
    for keys, reply in ANSWERS:
        if any(key in text for key in keys):
            return reply
    return DEFAULT_ANSWER


def chunk(chunk_id: str, model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse({"object": "list", "data": [{"id": "slow-stub", "object": "model"}]})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "slow-stub")
    reply = answer_for(body.get("messages", []))
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    ttft = settings["ttft_ms"] / 1000

    if not body.get("stream"):
        # Warmup and any non-streaming caller. Same delay, so a warmup call is
        # not accidentally the fast path.
        await asyncio.sleep(ttft)
        return JSONResponse({
            "id": chunk_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def stream():
        yield chunk(chunk_id, model, {"role": "assistant", "content": ""})
        # The whole point: nothing is emitted for this long, so the response is
        # in flight and silent for a window wide enough to interrupt.
        await asyncio.sleep(ttft)
        for word in reply.split(" "):
            yield chunk(chunk_id, model, {"content": word + " "})
            await asyncio.sleep(settings["rate_ms"] / 1000)
        yield chunk(chunk_id, model, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18820)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ttft-ms", type=int, default=3000, help="Silence before the first word.")
    parser.add_argument("--rate-ms", type=int, default=25, help="Gap between words once talking.")
    args = parser.parse_args()
    settings["ttft_ms"] = args.ttft_ms
    settings["rate_ms"] = args.rate_ms

    import uvicorn

    print(f"slow-llm on http://{args.host}:{args.port}/v1  ttft={args.ttft_ms} ms")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
