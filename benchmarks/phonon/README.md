# Phonon streaming measurements

The [benchmark guide](../../docs/phonon-benchmark.md) defines the measurements,
commands, dataset selection and limits. The initial reports in `2026-10-01/`
cover ten labeled clips on CPU, RTX 5090 and an M2 MacBook Air. They include a
separate warmup and each final transcript. The corpus provenance pins the source revision and
hashes. Downloaded audio and models are not checked in.

To regenerate the audio:

```bash
uv run --no-project --with pyarrow --with huggingface-hub \
  python scripts/prepare_phonon_corpus.py \
  --revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --count 10 --output-dir progress/evaluations/librispeech-mini
```

Absolute audio and manifest paths in the reports identify the original local
run. Use the new manifest path when repeating a run on another machine.
The CPU reports predate added package/source/manifest metadata fields; their
Torch version, machine, cadence and thread settings remain in the files.

The CPU Parakeet progressive path failed inside nano-parakeet 0.2.1's timestamp
decoder, so those first-partial values are null. Its CUDA path works. Phonon's
client CPU and RSS fields exclude its external server. The Mac report has ten
successful finals after a Phonon keepalive fix; the failed first run and single-clip
retry remain separate reports. Mac final waits include a 12.5-second slow case.
The later sequential Mac comparison uses `parakeet-mini-mac.json` and
`phonon-mini-mac-paired.json`. Both completed ten clips. Phonon delivered earlier
partials and finals; Parakeet made five fewer word errors. Model revision/file
sizes are in `parakeet-mac-model.json`; restored Phonon server details are in
`phonon-mac-paired-health-end.json`. Memory pressure and background tasks limit
the timing comparison. The small clean fixture cannot establish general accuracy.

## Mac candidate sweep

The sweep tests Whisper tiny/base/small/turbo through the repo's MLX Audio
Whisper handler, Moonshine tiny/small/medium through a benchmark-only bridge,
and fresh Parakeet/Phonon runs. Moonshine is not a registered production backend.
Its own VAD determines segments; the bridge joins corrected lines and forces a
final at the clip end. Speaker recognition is off. The requested 0.5 s update
interval is a floor for Moonshine; its library may adapt to decoder load.

Use the Mac environment and pinned corpus in the [guide](../../docs/phonon-benchmark.md).
Install Moonshine and start the sweep inside tmux. Stop only your own idle Phonon
server first, so it does not share memory with other models. The last case starts
it again and leaves it running:

```bash
uv pip install --python .venv-phonon/bin/python moonshine-voice==0.1.5
mkdir -p progress/logs progress/evaluations
# Run this inside a tmux session, using your project environment:
.venv-phonon/bin/python scripts/benchmark_mac_candidates.py \
  --config benchmarks/phonon/mac-candidates.json \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --output-dir progress/evaluations/mac-candidate-sweep \
  2>&1 | tee progress/logs/mac-candidate-sweep.log
```

The driver stages each model in its persistent cache, records its files and
hashes, then runs one case per process. It writes per-case predictions, logs,
metadata, resource traces every five seconds, an append-only ledger and
`summary.json`. Each benchmark process has a 15-minute limit, excluding model
downloads; timeouts and setup failures
remain visible and make the sweep exit nonzero. HF cases run from cached
snapshots with `HF_HUB_OFFLINE=1`. Record the resolved revision before comparing
a new model release to an older run.

Individual configurations can also run through `scripts/benchmark_phonon.py`
with `--backend mlx-audio-whisper --model <HF-ID>` or
`--backend moonshine --model small-streaming`. `--model` also selects a Parakeet
checkpoint; it does not change the external Phonon server. The runner includes
synchronous last-chunk decode in final latency and treats an empty final on
labeled speech as failure. Older reports predate that timestamp refinement.
All measured sweep cases use the refined code.

The `--threads 4` setting applies to PyTorch; MLX and Moonshine keep their native
thread/device policies. Resource numbers include process memory, not total Metal
or unified memory. Phonon client resource fields still exclude its server. These
are speech-handler timings, not end-to-end voice-agent reply times.

Whisper also has a final-only configuration file, since live text is optional in
normal voice-agent use. Run the driver again with
`--config benchmarks/phonon/mac-whisper-final-only.json` and a new output directory.
That configuration sets the update interval above every clip's length, so the
runner feeds the whole final once and produces no partials. It uses the same
weights and decoder defaults; it does not change decoding settings. Stop your own
idle Phonon server before this follow-up and restore it afterward.

The driver also stages the original Whisper processor files before going offline.
The MLX handler now maps released `whisper-base-mlx` and `whisper-small-mlx` names
to their matching original processors. The initial missing-processor setup
failures remain separate reports; they do not contribute accuracy numbers.

Saved sweep results and transcripts are in the [Mac candidate report](2026-10-01/mac-candidates/README.md).

The current MLX Audio ASR checkpoint check uses
`--config benchmarks/phonon/mac-whisper-current.json`. Run it in a new output
directory after pausing your own idle Phonon server, then restore that server.
The [MLX Audio Whisper guide](https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/whisper.md)
uses the `whisper-large-v3-turbo-asr-fp16` checkpoint. The check also tests
`whisper-tiny-asr-fp16`; old checkpoint reports stay separate.
