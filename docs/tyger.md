# Tyger speech input (experimental)

[Tyger](https://huggingface.co/nineninesix/tyger) combines English speech recognition with learned endpoint and interruption detection. Selecting `--stt tyger` replaces the usual VAD, Smart Turn, and STT stages. The LLM and TTS remain independently selectable.

## Run

Use the normal installation and your chosen LLM/TTS configuration, selecting Tyger for input. For example, with a hosted LLM and local Qwen3-TTS:

```bash
export OPENAI_API_KEY=...
speech-to-speech serve \
  --stt tyger \
  --tyger_device cuda \
  --llm_backend responses-api \
  --tts qwen3
```

Connect with the [browser demo](../demo/README.md#quick-start-local), using **WebSocket** transport. This backend requires the demo's aligned playback reference. The packaged `local`/`talk` audio client, stock clients without this extension, and WebRTC are unsupported.

The model downloads about 2.6 GB of weights into the normal Hugging Face cache. Budget GPU memory for Tyger in addition to the selected LLM and TTS. Each configured pipeline loads its own model. `--tyger_device cuda:N` selects a GPU; `cpu` is available for offline use, without a real-time performance guarantee. Use `--tyger_model_name` for a local bundle and `--tyger_revision` to pin a Hub revision.

`--tyger_endpoint_threshold` defaults to `0.91`; `--tyger_barge_in_threshold` defaults to `0.5`. Both require two consecutive 80 ms frames after a six-frame warmup. Lower thresholds accept more detections and may cause false interruptions. Tyger decides endpoints directly; Silero thresholds and Smart Turn delays do not apply. Silence alone does not force an endpoint.

## Behavior and limits

- The model receives aligned microphone and rendered assistant audio in 320 ms hops, including silence. It keeps eight seconds of context for decisions and transcribes long user turns in bounded blocks.
- Playback activity continues through sentence pauses and queued audio after generation ends. Muting sends zeroed microphone samples while keeping the playback reference flowing.
- Normal speech/transcription/response events remain unchanged. Session `create_response` and `interrupt_response` flags still control replies and interruptions. Manual audio commits and disabling automatic turn detection are unsupported.
- This version does not synthesize backchannels. It does not combine with diarization or VAD audio enhancement.
- Published model scores are offline diagnostics, not a benchmark of this integration. Evaluate premature endpoints, missed interruptions, and echo with your microphones and speakers before production use. The model weights use [OpenMDW-1.1](https://huggingface.co/nineninesix/tyger/blob/main/LICENSE.md).

## Client extension

A Tyger server advertises `session.speech_to_speech.input_audio_reference: true` in `session.created`. Each `input_audio_buffer.append` must then contain:

- `audio`: base64 mono PCM16 microphone samples at the configured input rate.
- `playback_audio`: base64 PCM16 of the same length and rate, captured from rendered playback on the microphone's sample clock; use silence when nothing plays.
- `playback_active`: a boolean indicating that an assistant reply has started rendering and has not finished playing, including pauses while more audio is generated.

Send both streams in the same event, continuously during playback. Keep the input sample rate fixed for the connection. Missing or malformed reference fields return `invalid_audio_reference` without consuming the audio. Audio sent to a client is not a substitute for the audio that it renders.
