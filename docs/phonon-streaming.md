# Phonon native streaming

`--stt phonon` sends incremental audio to a separately managed Phonon speech server.
It uses the [native WebSocket protocol](https://github.com/fermionresearch/phonon/blob/main/docs/server.md#get-v1audiostream-websocket),
with binary mono PCM16 at 16 kHz. It does not upload or retranscribe each growing audio window.
The speech server chooses its model; the WebSocket has no model selection field.

## Run

Install the speech pipeline as described in the main README. Keep the Phonon runtime
in its own environment. For a CPU server on Linux:

```bash
uv venv .venv-phonon --python 3.11
uv pip install --python .venv-phonon/bin/python fermion-research==0.2.7
uv pip install --python .venv-phonon/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-phonon/bin/python safetensors soundfile scipy zstandard
mkdir -p progress/logs
# Run inside tmux for a persistent server:
tmux new -s phonon-server
CUDA_VISIBLE_DEVICES='' .venv-phonon/bin/fermion serve phonon-2 \
  --port 18090 --threads 4 2>&1 | tee progress/logs/phonon-server.log
```

In another shell, check readiness and start the pipeline:

```bash
curl --fail http://127.0.0.1:18090/health
speech-to-speech serve --stt phonon \
  --phonon_stt_base_url ws://127.0.0.1:18090/v1 \
  --num_pipelines 1
```

The normal LLM and TTS flags still apply. Native partials work without
`--enable_live_transcription`; that flag controls growing-window transcription
for offline backends. Pipeline VAD settings still decide when the LLM starts.
The first utterance can include server warmup time.

For Apple Silicon, use the [Fermion installation instructions](https://github.com/fermionresearch/phonon)
and the same `fermion serve phonon-2` command. For NVIDIA, use the
[Phonon-2 CUDA image](https://github.com/fermionresearch/phonon/blob/main/docs/cuda.md).
Choose GPU IDs after checking other jobs, and preserve the downloaded model cache.
If exposing the demo to another device, use the repository's Tailscale instructions.

## Settings and behavior

| Flag | Default | Purpose |
|---|---|---|
| `--phonon_stt_base_url` | `ws://localhost:8000/v1` | Base URL or full `/v1/audio/stream` endpoint; HTTP(S) converts to WS(S). |
| `--phonon_stt_api_key` | unset | Bearer header for authenticated servers. |
| `--phonon_stt_connect_timeout` | 10 seconds | WebSocket connection timeout. |
| `--phonon_stt_final_timeout` | 60 seconds | Bound from local VAD commit until `done`. |

Each `partial` replaces the phrase's prior hypothesis. A server `final` completes
one segment, which the adapter joins with the next phrase's partials. It remains
a pipeline partial: only `done` after the local VAD boundary emits a final
`Transcription`. The adapter closes that socket and opens a new one for the next
utterance. Reopened revisions retain the prior committed text through the shared
streaming runtime. Cancellation and discarded speech close the connection so
rejected audio cannot enter a later turn.

A busy worker, dropped connection, invalid transcript, or final timeout produces
a typed transcription failure. The adapter does not replay audio automatically.
Phonon permits one live stream per worker; concurrent pipelines pointed at one
worker can fail with its busy response. Run one pipeline per worker for this
prototype. Do not expect timestamps, prompt conditioning, or language selection:
Phonon-2 transcribes English, reported as `en`.

The Realtime service already turns whole hypotheses into stable append-only
transcript deltas. Standard clients may therefore see text later than the
handler's first partial. Clients that opt into
[`speech_to_speech.input_audio_transcription.snapshot`](../src/speech_to_speech/api/openai_realtime/README.md#speculative-input-transcription-snapshots)
see the latest whole hypothesis, including word corrections. The final
`completed.transcript` remains authoritative for both paths.

## Verify and measure

```bash
uv run pytest -q tests/test_phonon_stt_handler.py tests/test_streaming_stt_handler.py tests/test_backend_registry.py
```

Use the [benchmark guide](phonon-benchmark.md) for corpus format, paced input,
accuracy normalization, latency definitions, and measured results. Published
model speed and first-partial claims are not measurements of this pipeline.
