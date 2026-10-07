# OpenAI-compatible STT endpoint

The `openai` STT backend keeps VAD, turns, sessions, conversation state, and
response handling inside speech-to-speech while delegating recognition to an
external server:

```text
VAD audio -> POST /v1/audio/transcriptions
```

Each request uploads an in-memory mono PCM16 WAV at 16 kHz and accepts either
JSON with a string `text` field or a plain-text response. Word timings use
`words: [{"word": "hello", "start": 0.1, "end": 0.5}]`, in seconds relative
to each request. `verbose_json` requests word timestamps from compatible servers. With live
transcription enabled, progressive updates upload the accumulated utterance
again unless bounded windows are enabled below.

## vLLM with Qwen3-ASR

Run a supported ASR model behind vLLM. Install vLLM in its own environment on
the machine serving STT; it does not need to be inside this repository.

```bash
pip install "vllm[audio]"
vllm serve Qwen/Qwen3-ASR-1.7B --port 8000
```

Check the server, then select the remote backend. The LLM and TTS flags can
point at any other supported backends.

```bash
curl http://localhost:8000/v1/models

speech-to-speech local \
  --stt openai \
  --openai_stt_base_url http://localhost:8000/v1 \
  --openai_stt_model Qwen/Qwen3-ASR-1.7B \
  --openai_stt_window_seconds 30 \
  --openai_stt_overlap_seconds 4 \
  --openai_stt_language en \
  --openai_stt_aligner_model Qwen/Qwen3-ForcedAligner-0.6B-hf
```

## Bounded audio windows

Set `--openai_stt_window_seconds 30` to cap each transcription upload at 30
seconds. `--openai_stt_overlap_seconds 4` keeps four seconds of shared audio.
The default window is `0` (disabled). The default boundary mode is `aligned`;
when windows are enabled, the handler requires at least half a second of overlap.

The handler retains completed window text for the current turn and transcribes
only the outstanding tail on later updates. A reopened turn reuses completed
windows when their audio and language still match. The last window stays
revisable. Internal boundaries do not end the turn or trigger a reply. Final
requests process all remaining windows even when progressive work was skipped.

Aligned mode cuts at a confirmed interval of digital silence when possible.
For continuous speech, it matches adjacent words by both text and audio time,
then keeps the newer text after that shared position. It preserves punctuation
and repeated words. If the match is unclear, it transcribes one bounded section
across the boundary and checks both joins. Missing, invalid or still ambiguous
word times produce a clear failure; the handler does not guess a join. Entirely
silent windows bypass recognition to avoid silent-tail hallucinations.

Use word times from your server, or set `--openai_stt_aligner_model
Qwen/Qwen3-ForcedAligner-0.6B-hf`. The optional local model loads when a boundary
needs it. It requires a Transformers version with native Qwen3 forced-alignment
support (validated with 5.19.0), a known language and extra memory. Install it in
a project environment with `uv pip install 'transformers>=5.19,<6'`. Set
`--openai_stt_aligner_device cuda:0` only when that GPU is available; the default
is CPU. Supported language hints are `en`, `zh`, `yue`, `fr`, `de`, `it`, `ja`,
`ko`, `pt`, `ru` and `es`. Japanese and Korean may need the processor's optional
tokenizer dependencies. Alignment locates recognized words; it cannot correct
recognition errors or guarantee that supplied timestamps describe the audio.

`--openai_stt_boundary_mode text` selects the earlier text-only join. It permits
zero overlap and can repeat or omit words, especially in repeated phrases.
Use aligned mode for new windowed deployments. Windowing remains opt-in until
speech quality, supported languages and alignment cost justify a wider default.

This caps recognition requests, not the VAD audio buffer. The VAD still supplies
cumulative turn audio. The handler caches text and audio hashes for up to eight
turn/language keys per pipeline, with no second retained audio copy. Session end
clears this cache. Verified joins between completed windows are cached with
their audio, text and timings, so later updates do not repeat earlier bridge
requests. Audio changes and turn eviction discard those joins. Final and progressive workers can independently decode a
window during a race, while the existing cancellation and stale-result checks
still govern dispatch and publication.

Known vLLM audio duration, upload size and decoded-audio limit errors produce a
sanitized hint to enable or reduce the window. Set the cap below any stricter
limit imposed by your provider. Setup still sends one second of silence to test
the endpoint, independently of the turn window setting.

## Window validation

The HTTP stress test covers 30, 120 and 600 seconds of synthetic speech, with
word timestamps and resumed finals. All bounded requests stay within 30 seconds
and produce exact test transcripts. At 600 seconds, progressive uploads total
2,798 seconds versus 36,900 without windows: 92.4% less audio. This measures
transport work, not recognition quality or real model latency. Reproduce it:

```bash
uv run python scripts/benchmark_asr_windows.py --durations 30 120 600 \
  --windows 0 30 --overlap-seconds 4 --boundary-mode aligned \
  --reopen-seconds 5 --max-request-seconds 60 --output /tmp/asr-window-work.json
```

A small English LibriSpeech check uses Qwen3-ASR-0.6B-hf and the Qwen forced
aligner with the production planner and joins. Clean 120-second speech has
4.46% word error rate with windows, versus 4.83% for whole-recording recognition.
A 62-second recording matches the baseline at 4.64%; shifted and noisy variants
also complete. All 20 repetitions of one sentence survive, versus 18 in the
whole-recording baseline. One 20-second bridge resolves a difficult boundary.
These figures include recognition mistakes and cannot isolate boundary WER
without reference word timestamps. They do not establish quality for other
languages, Qwen3-ASR-1.7B or vLLM serving.

CPU recognition plus alignment costs about 64 seconds on the clean 120-second
case, versus 47 seconds for the whole recording. Whole-recording recognition
therefore remains cheaper for this final-only CPU example; bounded windows
address repeated progressive work and provider limits. Check quality and cost
on your own audio before deployment.

A native Transformers GPU check on one L40S completes six cases, including a
600-second recording made from repeated labeled English clips. That long case
has 3.54% word error rate (41 substitutions, 10 deletions, zero insertions).
Nine progressive calls at 60-second intervals plus the final submit 840 seconds
of audio; no backend call exceeds 30 seconds. The final call takes 6.96 seconds
after earlier work has completed. Prefixes feed immediately rather than waiting
for live playback. This checks real model work with the production planner; it
does not measure HTTP or vLLM latency, unique ten-minute speech or boundary-only
error rates.

## OpenAI-hosted transcription

The same backend can call OpenAI's hosted
[Transcription API](https://developers.openai.com/api/docs/guides/speech-to-text).
Export an API key and select a transcription model:

```bash
export OPENAI_API_KEY=...

speech-to-speech local \
  --stt openai \
  --openai_stt_base_url https://api.openai.com/v1 \
  --openai_stt_model gpt-transcribe \
  --openai_stt_response_format json
```

The LLM and TTS backends remain independently configurable; using OpenAI for
STT does not require using it for the rest of the pipeline.

## Authentication and compatibility

Set `--openai_stt_api_key` when the endpoint requires bearer authentication.
When the base URL is `https://api.openai.com/v1` and this flag is omitted, the
handler uses `OPENAI_API_KEY` if present. Other endpoints never receive that
environment credential implicitly.

The request and response shapes follow OpenAI's Audio API. In particular,
`gpt-transcribe` sends language hints with the official plural `languages[]`
field and reads the first detected language code from the plural `languages`
response. Older models and compatible servers continue to use the singular
`language` field.

The client accepts JSON and text responses. Use
`--openai_stt_response_format text` for a plain-text server. Transport and HTTP
errors are sanitized before they are surfaced to realtime clients, and failed
final requests do not create LLM work.

During setup, the handler transcribes one second of synthetic silence through
the configured endpoint. Endpoint, authentication, model, or response-format
failures therefore prevent the realtime server from accepting sessions.

## Turn and session lifecycle

Each pipeline delivers STT results asynchronously so HTTP work does not block
session teardown. Final requests for distinct turns retain their order within
that pipeline. They run independently of progressive work. Each pipeline retains
at most eight pending finals in addition to its active final request. Once this
queue is full, additional finals receive a sanitized `TranscriptionFailure`
without uploading audio or retrying. Obsolete pending requests are removed before
checking the limit, and accepted finals keep their order. This bounds the number
of utterances retained behind a stalled request; it is not an estimate of server
capacity.

Progressive requests are best-effort: one is active per pipeline, and only the
latest waiting cumulative window is retained. A final request cancels matching
active progressive work and discards its pending window. Newer turn revisions
invalidate older queued work and cancel older active requests. Relevance is
checked before HTTP dispatch and while a request is running.

Session end cancels active work and clears pending work. Results and failures
carry a session generation that is checked atomically with queue publication,
preventing old completions from appearing after teardown or in a reused session.
Shutdown cancels the remaining requests and joins the pipeline's request workers.
If a request worker cannot start, its pending work is cleared, final requests
receive a sanitized failure, and shutdown still reaches the pipeline boundary.

Cancellation remains active through connection establishment, upload, and
response reads, even if a connection handoff consumes an initial cancellation
signal. Transport cleanup completes before the request worker is reused, so an
old blocked upload does not hold up a new session until its HTTP timeout.
Server-side inference cancellation is best-effort: closing the client connection
does not guarantee that the server stops GPU work. Stale-result filtering remains
required even when a request has been cancelled.

Client-side queue bounds apply separately to each pipeline. Pipelines do not
share an admission queue or endpoint concurrency budget. STT server capacity,
fleet routing, and provider quotas belong to the inference service or its shared
proxy.

## Stateful streaming STT

Use `--stt openai-realtime` or `--stt vllm-realtime` to send incremental PCM
over a persistent WebSocket. Local VAD controls which audio is sent and when
an utterance is committed. Streaming preserves the configured pre-speech
padding, trailing silence, and short-fragment merge gaps; idle microphone
audio outside those windows is not uploaded. Smart Turn and speculative
reopening still control when the assistant responds.

Both streaming backends forward provider partial transcripts as they arrive,
regardless of `--enable_live_transcription`. That flag enables repeated
whole-utterance requests for non-streaming STT backends; it is not needed for
native streaming.

### OpenAI Realtime transcription

```bash
export OPENAI_API_KEY=...

speech-to-speech local \
  --stt openai-realtime \
  --openai_realtime_stt_model gpt-live-transcribe
```

The hosted endpoint uses a transcription session with `intent=transcription`;
the model is configured in the session, not in the WebSocket URL. Hosted
OpenAI requires 24 kHz PCM, so the handler resamples the pipeline's 16 kHz
audio. Leave `--openai_realtime_stt_audio_sample_rate` at its default of
`24000`; other values fail during setup. See the
[OpenAI Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription).

`--openai_realtime_stt_api_key` accepts an explicit bearer token. The default
hosted endpoint uses `OPENAI_API_KEY` when that option is omitted; custom
endpoints do not receive the environment key implicitly.

### vLLM Realtime transcription (experimental)

The client uses vLLM's separate Realtime transcription protocol with 16 kHz
PCM. `--vllm_realtime_stt_model` must match a Realtime-capable model identifier
served by your endpoint; check `/v1/models`, including any custom served name.
The default `Qwen/Qwen3-ASR-1.7B` only works when that identifier is served.

For a server serving `mistralai/Voxtral-Mini-4B-Realtime-2602`:

```bash
curl http://localhost:8000/v1/models

speech-to-speech local \
  --stt vllm-realtime \
  --vllm_realtime_stt_base_url ws://localhost:8000/v1 \
  --vllm_realtime_stt_model mistralai/Voxtral-Mini-4B-Realtime-2602
```

Set `--vllm_realtime_stt_api_key` if the server requires authentication. See
[vLLM's Realtime API documentation](https://docs.vllm.ai/en/stable/serving/online_serving/speech_to_text/#realtime-api)
for supported models and server configuration.
