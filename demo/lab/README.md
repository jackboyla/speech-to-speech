# Voice Lab — developer mode

An instrumented view of what the realtime voice demo is actually doing: a
timeline for every turn, the latency broken down from the moment the user stops
talking, the raw protocol stream, and a scenario runner that drives scripted
conversations from prerecorded audio.

Press **DEV** in the demo header to open or close the lab. The choice is saved
in this browser. The module loads when first opened, so an ordinary visit does
not fetch it. `?debug=1` still opens it for a direct link or scripted run.

```
http://localhost:7860/?debug=1
```

---

## Why

Realtime voice bugs are timing bugs. "It answered the wrong question" and "it
talked over me" are reports about the order of events a few hundred
milliseconds apart, and neither the transcript nor the console tells you which
event came first. The panel exists so those questions have answers you can read
off a screen instead of guess at.

The second reason is that some of these bugs cannot be reproduced by hand at
all. See [the turn-ordering race](#the-turn-ordering-race) below: the window it
lives in is narrower than the time it takes a person to start talking.

---

## What it measures

Everything after the user stops talking is quoted as an offset from that
moment, because that is when the user starts waiting:

```
user speech start       0 ms
user speech end      1256 ms
STT final             +12 ms
response created     +796 ms
assistant first text +798 ms
first audio packet   +899 ms
audible playback     +899 ms
response done       +1103 ms
```

| Measurement | Where it comes from |
|---|---|
| End of speech → final STT | `input_audio_buffer.speech_stopped` → `conversation.item.input_audio_transcription.completed` |
| End of speech → first assistant token | → first `response.output_audio_transcript.delta` |
| End of speech → first audio | → first `response.output_audio.delta` |
| Audio packet → audible playback | main thread hands PCM to the worklet → the worklet emits its first sample |
| Barge-in → silence | interrupting `speech_started` → the worklet stops emitting |
| Tool-call duration | `response.function_call_arguments.done` → the matching `function_call_output` item |
| Total turn latency | end of speech → `response.done` |

A measurement that did not happen shows as `—`. It is never shown as zero and
never inferred from a neighbouring mark.

### Three things the panel is careful about

**Audible playback is measured in the audio clock, not the message queue.** The
playback worklet stamps the render quantum that emits the first sample and the
page converts that onto its own clock. Timing it by when the worklet's message
*arrived* would add main-thread scheduling jitter to a number meant to measure
the audio path. The device's own output latency is not included, because the
page cannot observe it; the panel prints it separately.

**"Committed" is inferred, and the panel says so.** Issue #308 distinguishes
work the user has begun hearing — which must be allowed to finish — from work
still in flight and silent, which must not. The page cannot read the pipeline's
commit flag. It can only tell whether audio for a response had started, which
is where #308 draws the line too. It is a fair proxy, but it is a proxy.

**WebSocket sees more than WebRTC.** Over WebSocket the page decodes the PCM and
drives playback itself, so audio packets and the playback queue are both
visible. Over WebRTC the media never passes through the page: audio arrives on a
peer connection and plays out of an element. The `first audio packet` and
`audible playback` columns are therefore empty over WebRTC, and the fallback —
polling the output level every 50 ms — is reported with that resolution
attached. This is not an oversight to fix; it is a real difference in what the
two transports let a client observe, and pretending otherwise would put
confident numbers next to a coarse estimate.

---

## Scenarios

The Scenarios tab drives scripted conversations. Turn on **Scripted
microphone**, press Restart, and the session captures from fixture WAVs through
a `MediaStreamAudioDestinationNode` instead of a real microphone — the capture
worklet, the noise gate, the resampler and the 40 ms send cadence all still run.
No microphone is opened at all, so a scripted run cannot pick up the room.

Fixture audio plays in real time. Server-side turn detection reacts to how audio
is paced, so posting a whole utterance at once would measure a path no user ever
takes.

| Scenario | What it is for |
|---|---|
| Normal turn | The baseline every other scenario is read against. |
| Rapid stop → resume | A speaker who pauses mid-sentence. Should stay one turn. |
| Barge-in while speaking | Interrupt once the assistant is audible. Measures time to silence. |
| Turn 2 while turn 1 is still generating | The issue #308 race. See below. |
| Repeated short fragments | Several very short utterances in a row. |
| Tool call, then interruption | Interrupt while a tool call is in flight. |

Each scenario carries a **check** that reports whether the window it was aiming
at actually opened. Driving a race is not the same as hitting one, and a run
that missed says so rather than showing a pass.

### Generating the fixtures

```bash
# From an environment with kokoro installed — the repo venv has it:
.venv/bin/python demo/scripts/make_lab_fixtures.py
```

Kokoro is used rather than the pipeline's own TTS so the input to the
measurement is not also a component under test. The WAVs are not committed;
regenerate them from the pinned text in the script.

### Running a scenario without a browser window

```bash
cd demo
node scripts/run_scenario.mjs --scenario normal-turn
node scripts/run_scenario.mjs --scenario superseded-turn --repeat 10 --out ../progress/traces
node scripts/run_scenario.mjs --list
```

Same page, same client, same audio graph — Playwright just does the clicking.
This is how a race gets run thirty times in a row.

---

## Reaching it from another machine

The page needs a secure context: `getUserMedia` and `AudioWorklet` are refused
over plain HTTP anywhere but `localhost`, so opening the demo on a LAN address
gives a dead orb and a console error. Tailscale hands out a real certificate for
the machine's tailnet name, which solves it without a reverse proxy or a
self-signed certificate to click through.

Both the page *and* the backend need exposing. The page is fetched by the
browser, but so is the realtime WebSocket — the demo hands the URL to the
client, which dials it directly — and a page on `https:` may only open `wss:`.

```bash
# the page, and the backend it will dial
tailscale serve --bg --https=10000 http://127.0.0.1:7871
tailscale serve --bg --https=10001 http://127.0.0.1:18775

# tell the demo to advertise the backend's tailnet address, not localhost
SPEECH_TO_SPEECH_URL=wss://<machine>.<tailnet>.ts.net:10001/v1/realtime \
  uvicorn --app-dir demo server:app --port 7871
```

Then open `https://<machine>.<tailnet>.ts.net:10000/?debug=1`. Ports 443 and
8443 are the other two Tailscale serves on HTTPS; 10000 upwards are free.

Undo with `tailscale serve --https=10000 off`.

---

## The turn-ordering race

Issue [#308](https://github.com/huggingface/speech-to-speech/issues/308) is
about what happens to a turn that is replaced before it has been answered. The
rule it argues for has two halves:

- Work the user has already begun hearing is **committed** and must be allowed
  to finish.
- Work still in flight and silent when a newer turn arrives is **stale** and
  must be dropped. Otherwise the user hears an answer to a question they have
  already moved on from.

PR [#578](https://github.com/huggingface/speech-to-speech/pull/578) enforces the
second half globally rather than per turn id.

### The window is narrower than a human

On the local stack measured here, a short question is answered like this:

```
end of speech            0 ms
response.created      +796 ms      ← first output is ready
first audio           +899 ms
response.done        +1103 ms
```

The pipeline emits `response.created` when the first output exists, not when
generation starts, so from the browser the gap between "generating" and
"speaking" is about **100 ms**. Voice activity detection needs roughly **330 ms**
of audio before it reports that a new speaker has started. The interruption
cannot be delivered faster than the answer arrives. Reproducing this by talking
at it is not difficult — it is impossible.

### Opening the window on purpose

Two knobs make it reachable, and both are stated here because they are the
difference between a demonstration and a trick:

1. **A slow language model.** `demo/scripts/slow_llm.py` is an OpenAI-compatible
   stub with a settable time to first token and fixed, question-keyed answers.
   Ask about France and it says Paris; ask about planets and it says eight. If
   stale work escapes, the speaker hears "Paris" after asking about planets and
   no interpretation is needed. The bug under test is in turn bookkeeping, not
   in the model, so replacing the model changes when the race can be hit and
   nothing about whether it exists.

2. **Short reopen windows.** By default, speech resuming within
   `speculative_reopen_ms` (800 ms) — or `unanswered_reopen_ms` (7 s) while the
   turn is still unanswered — *reopens* the same turn instead of starting a new
   one. That is deliberate and correct: it is how a speaker who pauses
   mid-sentence is handled. But it means a second utterance is a continuation,
   not a replacement, so the supersede path is never taken. Setting both low
   makes the second utterance a genuinely new turn. Smart Turn must be off as
   well, since it clamps the grace period back up to 800 ms.

```bash
# 1. the stub model, silent for three seconds before it says anything
python demo/scripts/slow_llm.py --port 18820 --ttft-ms 3000

# 2. a pipeline that treats a second utterance as a new turn
speech-to-speech serve --host 127.0.0.1 --port 18775 \
  --stt parakeet-tdt --tts qwen3 \
  --llm_backend chat-completions --model_name slow-stub \
  --responses_api_base_url http://127.0.0.1:18820/v1 --responses_api_api_key '' \
  --num_pipelines 1 \
  --no_smart_turn --speculative_reopen_ms 200 --unanswered_reopen_ms 200

# 3. the demo, pointed at it
SPEECH_TO_SPEECH_URL=ws://127.0.0.1:18775/v1/realtime \
  uvicorn --app-dir demo server:app --port 7871

# 4. interrupt turn 1 at a chosen offset after it ends
cd demo
node scripts/run_scenario.mjs --url http://127.0.0.1:7871/ \
  --scenario superseded-turn --interrupt-ms 1200
```

With that harness the state is reached on demand: turn 1 finishes, its
generation is still running and silent, and turn 2 replaces it.

```
turns replaced while waiting      1   ← the #308 window
   answered anyway (leaked)       0
   dropped                        1   ✓
```

### What was actually observed

Measured results, so they are worth reading carefully:

| Interruption offset after turn 1 ends | Window opened | `main` | `fix/issue-308-turn-order` |
|---:|---|---|---|
| 1200 ms | yes | dropped ✓ | dropped ✓ |
| 2400 ms | yes | dropped ✓ | dropped ✓ |
| 2800 ms | no | — | — |
| 3000 ms | no | — | — |
| 3200 ms | no | — | — |

**The end-to-end bug was not reproduced, on either build.** At every offset that
opened the window, turn 1's work was dropped and only turn 2 was answered. The
two builds behave identically through the browser. That is a real result and it
is not the one this section was built to show.

The reason is that the tracker is not the only thing standing between a replaced
turn and the speaker. The language-model handler keeps its own staleness check
on a global generation counter and discards turn 1's generation before any
output reaches the service layer `SpeculativeTurnTracker` guards. On this
configuration that earlier gate hides the tracker defect end to end.

The defect itself is real, and it takes three lines to show at the level it
exists:

```python
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

tracker = SpeculativeTurnTracker()
tracker.observe("turn_1", 0)   # on main; the fix branch calls start_turn()
tracker.observe("turn_2", 0)   # a new user turn replaces it
tracker.is_latest("turn_1", 0)
```

| | `is_latest("turn_1", 0)` after turn 2 starts |
|---|---|
| `main` | `True` — stale output from the replaced turn is accepted |
| `fix/issue-308-turn-order` | `False` — it is dropped |

and committed work is still allowed to finish, which is the other half of #308:

```python
tracker = SpeculativeTurnTracker()
tracker.start_turn()
tracker.commit("turn_1", 0)    # the user has begun hearing this
tracker.start_turn()
tracker.is_latest("turn_1", 0)  # True — it finishes
```

`main` reaches the wrong answer because `is_latest` compares a revision against
a map keyed by turn id: `turn_1` revision 0 is still the newest revision *of
turn_1*, so the check passes even though `turn_2` exists. Ordering is per turn
where it needs to be global. That is what #578 changes.

So the honest claim is narrower than "here is the bug in a browser":

- The **window** is reproducible on demand, which it was not before. Opening it
  needed a slow model and closed reopen windows, and finding that out was most
  of the work.
- The **instrumentation** reports which side of the commit line a replaced turn
  fell on, per turn, with the evidence next to the verdict.
- The **defect** is demonstrated at the level it exists — in the tracker — not
  by this harness.
- The **fix causes no observable end-to-end change** on this configuration,
  which is worth knowing before shipping it.

A harness that reported a bug it did not see would be worse than useless, so the
scenario check reports what happened and nothing more.

---

## Comparing two builds

The page cannot tell which server build it is talking to, so it does not guess.
Run the scenario against one build, press **Save as baseline**, point the demo
at the other build and run it again. Both columns are measured; neither is
asserted.

`Copy trace` puts the whole run on the clipboard as JSON — turns, per-turn
metrics, per-response verdicts and the raw event list — which is what to paste
into an issue.

---

## Limits worth knowing

- **Response attribution is inferred.** Assistant output carries no
  `previous_item_id` back to the user item it answers, so the panel decides
  which turn a response belongs to by order, preferring the turn still
  legitimately owed an answer. The bias is deliberate: when two turns could own
  a response the live one wins, so the panel under-reports stale output rather
  than inventing it. A trace that *does* report stale output is worth believing;
  a trace that reports none is weaker evidence.
- **Fixtures are synthesized.** TTS audio is cleaner than a microphone in a
  room, so latencies measured from it are a lower bound.
- **The scripted microphone replaces the capture source**, so a scripted run and
  a spoken run are not byte-identical. Every stage downstream of the source is
  the same, and all the reported latencies are measured between two server
  events or between a server event and the local audio clock, so a constant
  input delay cancels out of them.
