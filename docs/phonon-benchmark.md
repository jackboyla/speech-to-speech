# Phonon / Parakeet streaming benchmark

`scripts/benchmark_phonon.py` compares Phonon's native streaming handler with the
current Parakeet TDT growing-window handler. It feeds the same labeled audio at
capture speed and reports hypotheses, final latency, normalized word error rate
(WER), failures, setup time, and client resource use. It does not run microphone
capture, VAD, the Realtime service, the LLM, or TTS.

For the wider Mac comparison, see the [candidate sweep](../benchmarks/phonon/README.md#mac-candidate-sweep) and its saved reports.

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

## M2 Mac results (2026-10-01)

The M2 MacBook Air has 8 GiB RAM and runs macOS 27.0. Client and server ran
on the Mac through localhost, with Fermion 0.2.7, MLX 0.32.3, mlx-audio 0.5.7,
and Torch 2.14.1. User apps stayed open; this was a shared-machine test.
The initial server reported 19.74 seconds to load and 10.5 seconds for its first stream
warmup. The reported run used an already warm server and a separate warmup clip.

| Metric | Phonon-2 native, M2 |
|---|---:|
| Successful finals | 10 / 10 |
| Median first partial | 658.0 ms |
| Median wait after audio end | 560.2 ms |
| Final wait range | 160.7–12,493.0 ms |
| Word errors | 12 / 254 |
| WER | 4.72% |

The first run completed nine clips. The tenth received `done` on the wire but
lost it during a connection reset. A single-clip retry succeeded with another
reset in its close log. Phonon stops reading frames after `end`, so a client
keepalive ping sent during final decode can remain unread when the server
closes TCP. The adapter now disables periodic pings for Phonon; the full repeat
completed ten clips with no failures. These reports preserve the failed run and
retry separately. A reset still appeared after one final had reached the handler;
disabling pings does not repair the server's close handshake. Timing varies
between runs; the improved median does not
establish that the transport fix sped up decoding. The 12.5-second final wait
also prevents a claim of steady low latency on this Mac.

### Sequential Parakeet comparison

We then ran the current Parakeet MPS handler and a fresh Phonon run, one at a
time on the same Mac. Each used the same ten clips, one excluded warmup, 32 ms
capture chunks, and four CPU threads. Parakeet requested growing-window updates
every 0.5 seconds; Phonon used its native stream. We paused our idle Phonon server
during Parakeet and restored it afterward. User apps stayed open.

| Metric | Phonon-2 native | Parakeet TDT, MLX |
|---|---:|---:|
| Successful finals | 10 / 10 | 10 / 10 |
| Clips with a partial | 10 / 10 | 9 / 10 |
| Median first partial | 654.2 ms | 2,209.7 ms |
| Median wait after audio end | 310.8 ms | 7,377.3 ms |
| Longest final wait | 2,069.0 ms | 26,958.6 ms |
| Word errors | 12 / 254 | 7 / 254 |
| WER | 4.72% | 2.76% |

Parakeet's partial median covers the nine clips that produced a nonempty partial;
final medians cover all ten. The Phonon run finished with no connection errors in
its debug log. Both warmups succeeded. Parakeet made five fewer word errors;
Phonon delivered earlier partials and finals in this comparison. All original
reports remain available, including Phonon's earlier 12.5-second slow case.

Both used MLX 0.32.3 and mlx-audio 0.5.7 in the small benchmark environment,
rather than the repo's full pinned Mac install. Parakeet used
`mlx-community/parakeet-tdt-0.6b-v3` at revision
`ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`; its safetensors file is
2,508,288,736 bytes. The cached Parakeet handler setup took 8.24 seconds;
the restored Phonon server reported 11.41 seconds to load. These setup boundaries
differ and exclude the initial Parakeet download.

The Mac had memory pressure and active background tasks. Sampled system memory
free percentages ranged from 17–30% during Parakeet and reached 63% after its
process exited. During Phonon we sampled 21% free and about 8.9 GiB of swap in
use. We did not collect continuous memory or swap traces for both runs, so we
cannot attribute the slow clips to one cause. These numbers describe the current
handlers on this shared 8 GiB Mac, not isolated model speed. There is still no
full voice-agent test or representative accuracy study.

To repeat on Apple Silicon, prepare the corpus above and install the server in
its own project environment. Run the server command inside tmux, then run the
client command in a second tmux session:

```bash
uv venv .venv-phonon --python 3.11
uv pip install --python .venv-phonon/bin/python fermion-research==0.2.7 \
  mlx==0.32.3 mlx-audio==0.5.7 mlx-lm soundfile scipy zstandard
mkdir -p progress/logs progress/evaluations
.venv-phonon/bin/fermion serve phonon-2 --port 18090 --threads 4 \
  2>&1 | tee progress/logs/phonon-server.log

# In the client shell, from the same checkout:
uv pip install --python .venv-phonon/bin/python soxr openai==3.22.1 websockets==17.1
curl --fail http://127.0.0.1:18090/health
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-phonon/bin/python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend phonon --base-url ws://127.0.0.1:18090/v1 --device mps --threads 4 \
  --output progress/evaluations/phonon-mini-mac.json \
  2>&1 | tee progress/logs/phonon-mini-mac.log
```

For Parakeet, stop only your own idle Phonon server, then run this in tmux.
Restore that server after Parakeet exits. Record the printed snapshot revision;
the offline run uses the cached model without resolving a new revision:

```bash
uv pip install --python .venv-phonon/bin/python lingua-language-detector==2.1.1
.venv-phonon/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("mlx-community/parakeet-tdt-0.6b-v3"))
PY
HF_HUB_OFFLINE=1 PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-phonon/bin/python scripts/benchmark_phonon.py \
  --manifest progress/evaluations/librispeech-mini/manifest.jsonl \
  --backend parakeet-tdt --device mps --threads 4 \
  --output progress/evaluations/parakeet-mini-mac.json \
  2>&1 | tee progress/logs/parakeet-mini-mac.log
```

This small client environment uses the checked-out adapter, rather than installing
the full pipeline. Client RSS excludes the external Phonon server and does not
measure total Metal/unified memory for Parakeet. It cannot compare total model
memory. The vendor's M5 figures do not describe this M2.

## Artifacts

The [committed reports and corpus hashes](../benchmarks/phonon/README.md) preserve
the ten-clip CPU, CUDA and Mac results. Session logs, references and the append-only
run ledger also live in `progress/evaluations/` and `progress/experiment-log.md`. The audio can be
regenerated with the pinned export command; no downloaded models belong in Git.

## Checks

```bash
uv run pytest -q tests/test_benchmark_phonon.py tests/test_phonon_stt_handler.py
uv run ruff check scripts/benchmark_phonon.py scripts/prepare_phonon_corpus.py src/ tests/
uv run mypy src/
```
