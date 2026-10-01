# Phonon / Parakeet streaming benchmark

`scripts/benchmark_phonon.py` compares Phonon's native streaming handler with the
current Parakeet TDT growing-window handler. It feeds the same labeled audio at
capture speed and reports hypotheses, final latency, normalized word error rate
(WER), failures, setup time, and client resource use. It does not run microphone
capture, VAD, the Realtime service, the LLM, or TTS.

## Prepare labeled audio

A JSONL manifest contains one file and reference transcript per line. Relative
paths resolve against the manifest directory:

```json
{"id":"clip-1","audio":"clip-1.flac","text":"The reference transcript."}
```

The runner reads WAV/FLAC and other SoundFile formats, averages stereo to mono,
and resamples to 16 kHz before timing. References must contain words. A small
public fixture can be exported in an isolated uv environment:

```bash
uv run --no-project --with pyarrow --with huggingface-hub \
  python scripts/prepare_phonon_corpus.py \
  --revision 5be91486e11a2d616f4ec5db8d3fd248585ac07a \
  --count 10 --output-dir progress/evaluations/librispeech-mini
```

The export writes `manifest.jsonl`, FLAC files and `provenance.json` with the
resolved dataset revision, source parquet hash and each audio hash. It uses the
first ten rows of `hf-internal-testing/librispeech_asr_dummy`, a clean fixture
with repeated speakers. Use a larger, varied held-out corpus for accuracy claims.
The benchmark accepts your own manifest without downloading any dataset.

## Run

Start the [Phonon server](phonon-streaming.md) and wait for its health check.
Check the workstation before a compute run:

```bash
hostname
uptime
df -h
free -h
nvidia-smi
docker ps
tmux ls || true
ps -eo user,pid,pgid,stat,etime,%cpu,%mem,cmd --sort=-%mem | head -30
```

For four-thread CPU measurements, run these commands inside separate tmux
sessions. For isolated latency, finish one before starting the other:

```bash
mkdir -p progress/logs progress/evaluations
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend phonon --base-url ws://127.0.0.1:18090/v1 --threads 4 \
  --output progress/evaluations/phonon-mini-cpu.json \
  2>&1 | tee progress/logs/phonon-mini-cpu.log

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend parakeet-tdt --device cpu --threads 4 \
  --output progress/evaluations/parakeet-mini-cpu.json \
  2>&1 | tee progress/logs/parakeet-mini-cpu.log
```

Use `--device mps` for the Mac Parakeet backend, or `CUDA_VISIBLE_DEVICES=<chosen
GPU ID> --device cuda` for NVIDIA. The Phonon client always sends PCM to its
server; `--device` does not select the external server's hardware. Run one
stream per worker, and never overlap clients on a single Phonon endpoint.
Use `--log-level debug` to expose backend errors hidden during progressive
transcription. Check the JSON error fields as well as logs; exit status is
nonzero for a failed warmup, failed final or failed run. Failed clips remain in the report;
aggregate WER includes successful finals only, with failure counts alongside it.

## Measurement definitions

- **First partial:** elapsed wall time from the start of paced capture to the
  first nonempty handler hypothesis. It includes the first PCM chunk's capture
  time and lazy WebSocket setup. Standard Realtime deltas can arrive later while
  the service waits for stable words; opt-in snapshots show whole hypotheses.
- **Final latency:** time from the last paced audio chunk to the final handler
  transcript. The native runner explicitly commits then sends `end`; it does
  not measure Phonon's natural silence endpointing or the pipeline's VAD delay.
- **Pacing:** 32 ms chunks by default. Phonon receives each chunk once. Parakeet
  receives accumulated windows at the default 0.5 s interval. A busy offline
  decoder retains at most the latest pending window; the final replaces pending
  progressive work. The report records actual feed time to show scheduling delay.
- **Warmup:** one paced pass over the first clip by default, recorded separately
  and excluded from the measured clips. `--warmup 0` measures the first stream.
- **Startup:** handler construction only. For Parakeet it includes model setup;
  Phonon's model lives in an external server and its client connects lazily.
  These startup values are not a model-startup comparison. Record server launch
  to readiness and `/health` load time separately.
- **WER:** casefold text, remove punctuation, split on whitespace, then compute
  word edit distance. Sum word errors and reference words across successful
  clips. No number expansion, vocabulary substitutions, or text-specific rules.
- **Resources:** per-clip client CPU seconds and process peak RSS. Linux reports
  RSS in KiB; macOS reports bytes, converted to bytes in the output. Phonon's
  external server is excluded. Record its CPU, RSS and GPU memory separately;
  a client-only number must not become a model resource-use claim.

Reports include each hypothesis's final text, per-clip WER, partial count,
latencies, failures, warmup results, machine, Python/Torch versions and settings.
The current script also records package versions and source/manifest hashes.
The initial measured reports predate those added metadata fields; their pinned
corpus provenance and experiment ledger supply that information.

## Initial CPU results (2026-10-01)

Machine: `radiance-ws`; four CPU threads per backend. Phonon server:
fermion-research 0.2.7, CPU AVX512-VNNI; server Torch 2.14.1+cu130. Client and
Parakeet: Torch 2.11.0+cu130, nano-parakeet 0.2.1. Both GPUs hosted existing
services and neither CPU run used them. The two CPU runs overlapped on the
workstation, so these are shared-machine measurements, not isolated latency.

Ten clips, 109.755 seconds of audio, 254 normalized reference words, one warmup:

| Metric | Phonon-2 native | Parakeet TDT current CPU handler |
|---|---:|---:|
| Successful finals | 10 / 10 | 10 / 10 |
| Median first partial | 596.6 ms | Unavailable: timestamp decoding errors |
| Median wait after audio end | 81.2 ms | 765.0 ms |
| Word errors | 12 / 254 | 8 / 254 |
| WER | 4.72% | 3.15% |
| Client CPU across measured clips | 1.35 s; server excluded | 323.01 s |
| Client peak RSS | 0.50 GiB; server excluded | 6.86 GiB |

Phonon delivered a first partial in 373–1,109 ms from file capture start; leading
silence varies between clips. The native server finalized sooner in this sample,
while Parakeet made four fewer word errors. This small fixture does not establish
general accuracy or hardware performance.

Parakeet's installed CPU timestamp decoder repeatedly raised `not enough values
to unpack (expected 3, got 2)` during progressive windows. The handler caught
those errors, returned no partials and still transcribed finals. We kept that
baseline unchanged. Its null first-partial result is a failed progressive path,
not a timing measurement. Resolve that dependency failure before claiming a
first-partial speed comparison.

Phonon's health endpoint reports a 163,515,201-byte download and 14.07-second
model load on this first setup. Idle server RSS was about 1.77 GiB. Sampling
part of the ten-clip run observed 1.85 GiB server RSS and 19.71 CPU seconds across
53.56 seconds; those samples cover only the latter part of the run and do not
establish whole-run peak RSS or CPU use. The packed model is about 170 MiB on
disk, and the CPU engine writes an additional roughly 290 MiB cache. Download
size, model storage, generated caches and process memory are separate metrics.

The JFK smoke clip (11 seconds, 22 words) produced no word errors on either
backend: Phonon first partial 372.8 ms and final wait 20.9 ms; Parakeet final wait
855.4 ms. A single familiar clip is functional evidence only.

## RTX 5090 results (2026-10-01)

Same ten clips and one warmup, with runs performed sequentially on GPU 0. Other
resident services stayed present and idle. Phonon used the official CUDA image
`ghcr.io/fermionresearch/phonon-cuda:1.0.5` at digest
`sha256:6ae947bf6c4a4ee3e3215433ee3c566df788eeb1dee2ffc107c1d8aef9c09265`,
dense bfloat16 and CUDA graphs. Parakeet used nano-parakeet 0.2.1 and Torch
2.11.0+cu130 on CUDA. Both produced ten successful finals.

| Metric | Phonon-2 native | Parakeet TDT CUDA |
|---|---:|---:|
| Median first partial | 629.0 ms | 1,038.8 ms |
| Median wait after audio end | 30.9 ms | 19.6 ms |
| Word errors | 13 / 254 | 8 / 254 |
| WER | 5.12% | 3.15% |
| Sampled worker GPU memory | 2,212 MiB | 2,074 MiB |

Phonon delivered the first partial about 410 ms earlier on this fixture. Parakeet
returned the final about 11 ms sooner and made five fewer word errors. Phonon's
smaller stored model does not yield smaller GPU memory here: its CUDA runtime
expands weights to a dense representation. These are process memory samples,
not peak measurements. Parakeet CUDA partials work; its CPU failure is separate.

Phonon CUDA model loading took 16.8 seconds in its startup log; bucket warmup
reported 0.4 seconds. Parakeet handler construction took 14.06 seconds with a
cached model and includes its warmup. Those boundaries differ, so do not compare
them as equivalent startup measurements. The cached Parakeet `.nemo` distribution
is 2,509,332,480 bytes at HF revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; Phonon's reported model download is
163,515,201 bytes. The CUDA runtime image itself occupies 13.1 GB unpacked in
Docker, separate from those model downloads.

## Mac status

The available M2 MacBook Air has 8 GiB RAM. The first setup lacked MLX speech
dependencies; the local environment setup was updated to install them. SSH then
timed out on repeated checks, so neither server readiness nor inference could
be verified. There are no Mac latency or WER results. The exact rerun script is
`progress/start-phonon-mac.sh`, and the attempted run is recorded in the ledger.
The vendor's M5 figures are not measurements of this M2 or this adapter.

## Artifacts

The [committed reports and corpus hashes](../benchmarks/phonon/README.md) preserve
the ten-clip CPU and CUDA results. Session logs, references and the append-only
run ledger also live in `progress/evaluations/` and `progress/experiment-log.md`. The audio can be
regenerated with the pinned export command; no downloaded models belong in Git.

## Checks

```bash
uv run pytest -q tests/test_benchmark_phonon.py tests/test_phonon_stt_handler.py
uv run ruff check scripts/benchmark_phonon.py scripts/prepare_phonon_corpus.py src/ tests/
uv run mypy src/
```
