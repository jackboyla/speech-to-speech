# Speculation before the turn ends

This is an opt-in prototype for two uses of partial STT: prepare the LLM's KV
cache and start read-only tools while the user speaks. The default voice server
still starts its response from final STT. This branch does not add a server flag,
train a decision model, or change the Realtime wire protocol.

## What works

`TurnSpeculation` accepts replacement hypotheses and uses the existing
`SpeculativeTurnTracker` for turn identity. A replaceable prefix policy decides
when to prepare the cache. The baseline requires words to agree across two
hypotheses and leaves out the last word. This tests word stability, not whether
an unfinished request has enough meaning to answer.

One worker prepares the cache. New partials replace queued work. The llama.cpp
adapter tokenizes each full rendered prompt and asks the server to reuse its
common token prefix. Changed words and token boundaries cause the server to
recompute the affected suffix. Do not append chat-template endings or assume
that text which extends a string also extends its token sequence.

Tools need explicit `allow_speculation=True`. Calls stay private, and launch and
parallel limits bound wasted work. The planner sees each full hypothesis, so
late corrections can cancel an old call. Final planning must produce the same
tool name and arguments before the controller returns its result. A year filter
can reuse broad retrieval only when the final plan still calls for that same
retrieval and applies the filter afterwards. Changed dates or destinations
require a new call unless a tool's contract explicitly supports broad results.

`COMMIT` means final validation of private work. It does not commit the shared
turn tracker or accept assistant output. Existing response ownership, reopen
gates, history, cancellation and TTS must still govern public output. Missing or
failed tools fall back to the normal tool path. Speculative failures do not fail
the final response.

## Measured results

Machine: radiance-ws, CPU with four threads. Model: SmolLM2-135M-Instruct Q4_K_M.
Both GPUs held other services and received no new work. Five handwritten ASR
sequences, three repeats each, alternating the order of the baseline and
candidate. Both arms first warm the same conversation history.

| Measurement | Baseline | Candidate | Median paired saving |
| --- | ---: | ---: | ---: |
| Cache component, final one-token request | 12.47 ms | 8.49 ms | 4.41 ms |
| Controller with partial events every 200 ms | 12.75 ms | 11.04 ms | 2.39 ms |

All 15 first tokens match in each comparison. That checks only the first token,
not full answer quality. The timed run includes pending prefill in final latency
and uses the event adapter and shared tracker. The component run finishes all
prefill work before timing the final request; it is an upper bound on overlap.
Neither includes actual audio, STT, network speech delivery, or TTS. This small
CPU model does not support a claim of 500 ms or more saved by prefill.

The component run prepares 16–39 ms of work across several updates per request
while saving only a few milliseconds at final input. Speculation can increase
total compute. Existing history caching already removes much of the prefill
cost; measuring a cold-history baseline would overstate the added benefit.

The separate tool benchmark uses a synthetic one-second search and a 600 ms
speech tail. It takes about 1001 ms after final input in the baseline and 401 ms
with a reused search. A late year filter retains that saving. A corrected
destination cancels one search, launches a fresh one, and takes about 1001 ms.
These are controlled scheduling results, not real arXiv or flight measurements.

Raw reports:

- [Cache component](prefill-cpu.json)
- [Timed controller](prefill-timed-cpu.json)
- [Synthetic tools](tools-synthetic.json)

## Run again

From the checkout root, set up project-local dependencies:

```bash
uv sync --group dev
mkdir -p progress/logs
hostname
uptime
df -h
free -h
nvidia-smi
docker ps
tmux ls || true
ps -eo user,pid,pgid,stat,etime,%cpu,%mem,cmd --sort=-%mem | head -30
```

Use an existing local model, or download this small fixture once into persistent
storage. The model file used here has SHA256
`2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.

```bash
mkdir -p "$HOME/workspace/models/SmolLM2-GGUF"
uv run hf download bartowski/SmolLM2-135M-Instruct-GGUF \
  SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --local-dir "$HOME/workspace/models/SmolLM2-GGUF"
sha256sum "$HOME/workspace/models/SmolLM2-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
```

Start a dedicated server. Do not use a busy voice server's slot for this test.
The pinned image used here is llama.cpp build 11096, commit `c550d2f60`.

```bash
tmux new -s turn-speculation-cpu
```

Inside tmux:

```bash
set -o pipefail
docker run --rm --name turn-speculation-cpu --cpus 4 --memory 2g \
  -p 127.0.0.1:18093:8080 \
  -v "$HOME/workspace/models/SmolLM2-GGUF:/models:ro" \
  ghcr.io/ggml-org/llama.cpp@sha256:0192ab2545efcbe79c240645e34abd8fffbe4813aedcef5a0e3a886ef6d6d82f \
  -m /models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 -ngl 0 -np 1 -c 4096 -t 4 -tb 4 --no-warmup \
  2>&1 | tee progress/logs/turn-speculation-server.log
```

In another shell, verify the server and run the benchmarks in tmux:

```bash
curl --fail http://127.0.0.1:18093/health
docker top turn-speculation-cpu -eo pid,cmd
tail -20 progress/logs/turn-speculation-server.log
tmux new -s turn-speculation-benchmark
```

Inside tmux:

```bash
export CUDA_VISIBLE_DEVICES=''
uv run python benchmarks/turn_speculation/benchmark.py --mode prefill \
  --repeats 3 --output benchmarks/turn_speculation/prefill-cpu.json
uv run python benchmarks/turn_speculation/benchmark.py --mode prefill \
  --partial-interval 0.2 --repeats 3 \
  --output benchmarks/turn_speculation/prefill-timed-cpu.json
uv run python benchmarks/turn_speculation/benchmark.py --mode tools \
  --tool-delay 1 --speaking-tail 0.6 \
  --output benchmarks/turn_speculation/tools-synthetic.json
```

## Connect the prototype

Create one controller per connection on the service's asyncio loop. Supply a
planner, read-only tools, and a backend whose prompt renderer holds a fixed
snapshot of history, instructions, tools and model settings for this turn.
Use a dedicated server slot per connection and use that same slot for final
decode. A normal completion sent elsewhere cannot reuse this adapter's cache.

The existing `TranscriptionNotifier` emits the input events; its implementation
needs no change. Subscribe alongside the normal service handler:

```python
from speech_to_speech.experimental.transcription_speculation import TranscriptionSpeculation
from speech_to_speech.experimental.turn_speculation import TurnSpeculation

controller = TurnSpeculation(shared_tracker, backend=backend, tools=tools, planner=planner)
observer = TranscriptionSpeculation(controller)

# The existing lifecycle owner supplies this reference. Do not infer it from
# an unrelated response.create or tool continuation.
await controller.begin(turn_id, revision)

# Each PartialTranscriptionEvent is a full replacement hypothesis.
await observer.observe(partial_event)

# After validating final input ownership, response policy, and reopen state:
private = await observer.finalize(final_event)
# Match private.tools against the final tool plan. Run missing tools normally.
# Final decode must render the full final transcript and actual tool results.
# Application hook: include actual tool results in the final prompt, keep the
# same backend slot, and recheck ownership before releasing generated output.
reply = await decode_final(final_event.transcript, private.tools, backend=backend)

# On clear, reset, disconnect, revision, or a new turn:
await controller.cancel()
```

`decode_final` is an application hook, not an API added by this prototype.
This sketch is not installed in `RealtimeService` by this branch. Production
wiring must respect client `create_response`, transcription-only sessions,
out-of-band requests, tool continuation ownership, changing history/config,
connection shutdown and turn reopen. Rebuild or cancel the controller when its
snapshot changes. Use nonblocking async tools that cooperate with cancellation.
Cancelling an HTTP client does not guarantee that a remote server stops work;
tools must be safe if they finish after cancellation. Final results still remain
private and get rejected when ownership changes during a wait.

The tested llama.cpp build samples one token even with `n_predict=0`, despite
its documented cache-only behavior. The adapter discards that sample, removes
all generated content from its prefill result, records `discarded_samples`, and
rejects a server that produces more than one. Prefix prefill never supplies a
reply. Final decode always uses the complete final prompt.

## Checks and next experiments

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run pytest tests/ -x -q
uv run ruff check src/ tests/ benchmarks/turn_speculation/benchmark.py
uv run ruff format --check src/ tests/ benchmarks/turn_speculation/benchmark.py
uv run mypy src/
```

The next experiment should use recorded conversational audio, real ASR
replacement hypotheses and a representative LLM. Compare ordinary history
caching with partial prefill, and measure final-STT-to-first-audio time, total
compute, wasted calls, incorrect reuse, revisions and answer quality. Keep the
repeated-prefix policy as the baseline. Label prefix survival and tool reuse
from final transcripts before fitting a small decision model. Prefix survival
alone is not a label for safe tool execution.

Established APIs used in the design:
[llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/c550d2f60/tools/server/README.md)
and [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
