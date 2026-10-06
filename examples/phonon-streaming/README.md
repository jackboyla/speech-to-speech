# Record a Phonon STT demo

This terminal demo uses the repo's `PhononSTTHandler`. It captures microphone
audio, sends PCM16 chunks, shows corrected partial text, then shows a final
transcript. Capture ends after 20 seconds by default; Ctrl+C cancels.
It commits at the end of capture rather than using pipeline VAD. No LLM or TTS
runs. A video of this demo shows the STT adapter, not full voice-agent latency.

## Install the pipeline

Use a checkout that includes the Phonon adapter. Follow the source install in
the [main README](../../README.md), then activate its environment:

```bash
uv sync --python 3.11
source .venv/bin/activate
```

## Start or reuse the Phonon server

Check the existing server first:

```bash
curl --fail http://127.0.0.1:18090/health
```

If it reports Phonon-2, reuse it. Otherwise, install and run the server in a
separate environment. These are the versions used in our Mac measurement;
the server's MLX dependencies stay separate from the pipeline's environment:

```bash
uv venv .venv-phonon --python 3.11
uv pip install --python .venv-phonon/bin/python \
  fermion-research==0.2.7 mlx==0.32.3 mlx-audio==0.5.7 \
  mlx-lm torch soundfile scipy zstandard
mkdir -p progress/logs
# Keep this server running in tmux:
tmux new -s phonon-video-server
.venv-phonon/bin/fermion serve phonon-2 --port 18090 --threads 4 \
  2>&1 | tee progress/logs/phonon-video-server.log
```

In another terminal, activate the pipeline environment again, then run:

```bash
curl --fail http://127.0.0.1:18090/health
python examples/phonon-streaming/demo.py \
  --seconds 20 --save-audio progress/phonon-demo.wav
```

Allow Terminal to use your microphone when macOS asks. If it does not use the
right input, list devices and select an index or name:

```bash
python examples/phonon-streaming/demo.py --list-devices
python examples/phonon-streaming/demo.py --seconds 20 --device 1
```

If the server uses authentication, set `PHONON_API_KEY` in your shell. Use
`--base-url` for another endpoint. Keep one live client per Phonon worker.

## Record yourself using it

On your Mac, press **Shift–Command–5**, select the terminal area, then select
**Options → Microphone → MacBook microphone** and start recording. Run the
20-second demo command. Speak a few sentences and pause briefly between them.
For example: “I am testing native Phonon streaming in speech-to-speech.
The words update as I speak. Now I will finish the recording.”

Let the final transcript appear, then stop the screen recording. A 20–30 second
clip is enough. The saved WAV is optional; the screen recorder creates the video.
Use microphone mode for a video of yourself using STT. File replay should be
labeled as replay, not microphone capture.

Apple documents the recorder and microphone option in its
[screen-recording guide](https://support.apple.com/en-is/guide/mac-help/-mh26782/mac).

## Compare against the default Parakeet baseline

Use **Parakeet TDT 0.6B v3**, the repo's default STT, as the baseline. Record
one clip and replay that exact WAV through both handlers. Separate microphone
recordings would change the speech, timing and background sound.

Before recording, write the words you will read. Do not copy either model's
output into the reference:

```bash
mkdir -p progress
cat > progress/reference.txt <<'TEXT'
Today I am testing speech recognition with the same recording. Both models should transcribe these words. I will pause briefly and then finish.
TEXT
python examples/phonon-streaming/demo.py \
  --seconds 20 --save-audio progress/phonon-demo.wav
```

Read those words during capture. Prepare one shared manifest:

```bash
python - <<'PYTHON'
import json
from pathlib import Path
root = Path("progress")
record = {
    "id": "microphone-demo",
    "audio": "phonon-demo.wav",
    "text": (root / "reference.txt").read_text().strip(),
}
(root / "demo-manifest.jsonl").write_text(json.dumps(record) + "\n")
PYTHON
```

On the Mac, run each case separately inside tmux. Both use 32 ms packets,
a 0.5-second progressive update setting, four PyTorch threads and one excluded
warmup. Phonon's native server controls its own partial cadence. Setup and model
downloads are outside the timed clips. MLX keeps its native thread policy.

```bash
python scripts/benchmark_phonon.py \
  --manifest progress/demo-manifest.jsonl \
  --backend phonon --device mps --threads 4 --warmup 1 \
  --chunk-ms 32 --partial-interval 0.5 \
  --base-url ws://127.0.0.1:18090/v1 \
  --output progress/demo-phonon.json
```

For an 8 GiB Mac, stop **your own idle** Phonon server after that case to free
its memory before loading Parakeet. Use Ctrl+C in its server terminal or tmux
pane. Keep user apps unchanged. Then run:

```bash
python scripts/benchmark_phonon.py \
  --manifest progress/demo-manifest.jsonl \
  --backend parakeet-tdt --device mps --threads 4 --warmup 1 \
  --chunk-ms 32 --partial-interval 0.5 \
  --output progress/demo-parakeet.json
python scripts/summarize_phonon_comparison.py \
  progress/demo-phonon.json progress/demo-parakeet.json
```

Restore your Phonon server afterward. On Linux, use `cpu` or `cuda` instead of
`mps`; choose the GPU after checking existing jobs. The baseline enables
Parakeet's live text path, matching the progressive STT use case rather than
comparing it only in final-only mode.

Record the comparison table and both final transcripts as the last part of your
video. Describe this part as **paced replay of the same microphone recording**.
The summary checks audio hashes, reference text, capture settings, warmup count
and failures before comparing. Keep both JSON files as evidence. First text may
change; final wait excludes microphone VAD, LLM and TTS. A single clip cannot
establish a general accuracy or speed advantage.

## Attach the video

Drag the recorded `.mov`, `.mp4` or `.webm` into a GitHub description or comment,
then wait for the upload to finish. Preview the clip before submitting. Label
it as a timed microphone STT demo, since it does not use pipeline VAD. See
[GitHub's attachment guide](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files).

## Replay a recording

For a repeatable check without microphone access:

```bash
python examples/phonon-streaming/demo.py --audio progress/phonon-demo.wav
```

Replay runs at capture speed, not as a fast batch transcription. The script
supports SoundFile inputs, averages stereo channels and resamples to 16 kHz.
The saved audio contains what the handler received; video and audio files are
local artifacts and should not be committed.
