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
The default window is `0` (disabled). Windows require at least half a second
of overlap.

The handler retains completed window text for the current turn and transcribes
only the outstanding tail on later updates. A reopened turn reuses completed
windows when their audio and language still match. The last window stays
revisable. Internal boundaries do not end the turn or trigger a reply. Final
requests process all remaining windows even when progressive work was skipped.

The handler cuts at a confirmed interval of digital silence when possible.
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

The handler keeps one rolling state per turn: completed text, an audio position
and the last window's word times. Later updates resume there instead of joining
every earlier window again. A hash of the completed audio rejects changed
prefixes; shortened audio, language changes and session end reset the state.
At most eight turn/language states are retained, without keeping another audio
copy. The VAD still supplies cumulative audio, so this bounds recognition work
rather than the VAD buffer. Concurrent workers keep the existing cancellation
and stale-result checks.

Windowing stays opt-in because alignment needs a supported language and adds
model work. Alignment cannot correct recognition mistakes.

Known vLLM audio duration, upload size and decoded-audio limit errors produce a
sanitized hint to enable or reduce the window. Set the cap below any stricter
limit imposed by your provider. Setup still sends one second of silence to test
the endpoint, independently of the turn window setting.

## Window validation

Tests cover 30/120/600-second requests, repeated words, resumed finals, changed
audio, language changes, cancellation and invalid timings. Run the bounded-STT
checks listed in the README.

Small English Qwen3-ASR-0.6B checks matched or improved whole-recording word
error rates. The repeated-sentence case kept all 20 sentences, versus 18 in the
baseline; one bounded bridge repaired its difficult join. These checks do not
establish other-language quality or boundary-only error rates. CPU alignment
adds work; measure quality and latency with your model and audio before use.

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
