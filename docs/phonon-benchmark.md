# Phonon / Parakeet streaming benchmark

`scripts/benchmark_phonon.py` feeds the same labeled audio at capture speed to
Phonon's native streaming handler and Parakeet's growing-window handler. It
records first nonempty partial, final wait after the last audio packet, word
errors, startup, failures and client CPU/RSS. It excludes microphone capture,
VAD endpoint detection, LLM and TTS. Phonon's resource fields exclude its server.

## Repeat

Install the pipeline and start Phonon as described in the [setup guide](phonon-streaming.md).
Export the ten-clip fixture into a local folder:

```bash
uv run --no-project --with pyarrow --with huggingface-hub \
  python scripts/prepare_phonon_corpus.py \
  --revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --count 10 --output-dir progress/evaluations/librispeech-mini
mkdir -p progress/logs
```

Run the benchmarks one at a time inside tmux. Choose the device after checking
other jobs. The following uses an Apple Silicon Mac; use `cpu` or `cuda` on Linux:

```bash
tmux new -s phonon-compare
python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend phonon --device mps --threads 4 \
  --base-url ws://127.0.0.1:18090/v1 \
  --output progress/evaluations/phonon.json \
  2>&1 | tee progress/logs/phonon-compare.log
python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend parakeet-tdt --device mps --threads 4 \
  --output progress/evaluations/parakeet.json \
  2>&1 | tee progress/logs/parakeet-compare.log
```

Keep the model caches. For a memory-constrained Mac, stop only your own idle
Phonon server before the Parakeet case, then restore it. Four threads sets
PyTorch's policy; it does not fix MLX's thread policy.

The manifest supports WAV/FLAC paths and reference transcripts:

```json
{"id":"clip-1","audio":"clip-1.flac","text":"The reference transcript."}
```

Relative paths resolve against the manifest folder. The runner converts audio
to mono 16 kHz, sends 32 ms packets and requests growing-window updates every
0.5 seconds. One warmup is excluded by default. References fold case, remove
punctuation and use word edit distance; no abbreviation expansion applies.
Failed warmups and measured clips make the command exit nonzero.

## Evidence and limits

The [saved evidence](https://github.com/jackboyla/speech-to-speech/tree/9b190c0040c561373bb91dfb098d47725b274f57/benchmarks/phonon/2026-10-01)
contains every transcript, dataset hashes, package versions and model details.
It stays outside this adapter change. The later candidate sweep used an extended
runner and included time lost during audio feeding; those reports remain separate.

The initial paired M2 Air comparison completed ten clips per backend:

| Backend | Median first partial | Median final wait | Word errors / 254 |
|---|---:|---:|---:|
| Phonon-2 | 0.654 s | 0.311 s | 12 |
| Parakeet TDT v3 | 2.210 s | 7.377 s | 7 |

These are handler timings on a shared M2 Air with 8 GiB and memory pressure.
The Mac benchmark used MLX Audio 0.5.7 and MLX 0.32.3, which differ from the
repo's pinned full installation. Background tasks and swap limit comparisons.
The clean fixture has ten clips from one speaker, 109.755 seconds of audio and
254 words; it cannot establish broad accuracy or a new default backend.
Earlier CPU/CUDA results and the failed first Mac attempt remain in the evidence.
Do not present published model throughput as end-to-end voice-agent latency.
