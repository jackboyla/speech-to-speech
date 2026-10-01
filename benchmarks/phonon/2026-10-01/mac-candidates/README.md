# Mac speech recognition candidate measurements

Run on 2026-10-01: M2 MacBook Air, 8 GiB, macOS 27.0. User apps stayed open. Models ran one at a time. Ten labeled LibriSpeech fixture clips contain 109.755 seconds of audio and 254 reference words. Each case has a separate warmup. This is a small comparison of these handlers and settings, not a general accuracy ranking.

| Case | First words, median | Final wait, median | Worst final wait | Word errors | WER |
|---|---:|---:|---:|---:|---:|
| [moonshine-medium](initial/moonshine-medium.json) | 1.150 s | 0.185 s | 0.485 s | 13/254 | 5.12% |
| [moonshine-small](initial/moonshine-small.json) | 1.119 s | 0.097 s | 0.171 s | 16/254 | 6.30% |
| [moonshine-tiny](initial/moonshine-tiny.json) | 1.089 s | 0.068 s | 0.125 s | 30/254 | 11.81% |
| [parakeet](initial/parakeet.json) | 2.458 s | 3.488 s | 77.249 s | 7/254 | 2.76% |
| [phonon](initial/phonon.json) | 0.657 s | 0.489 s | 3.085 s | 12/254 | 4.72% |
| [whisper-tiny](initial/whisper-tiny.json) | 3.731 s | 5.338 s | 7.921 s | 171/254 | 67.32% |
| [whisper-base](processor-retry/whisper-base.json) | 5.178 s | 4.908 s | 8.207 s | 257/254 | 101.18% |
| [whisper-small](processor-retry/whisper-small.json) | 11.290 s | 8.945 s | 18.113 s | 186/254 | 73.23% |
| [whisper-turbo](processor-retry/whisper-turbo.json) | 9.261 s | 3.395 s | 6.748 s | 15/254 | 5.91% |
| [whisper-base-final-only](final-only/whisper-base-final-only.json) | none | 2.346 s | 6.904 s | 197/254 | 77.56% |
| [whisper-small-final-only](final-only/whisper-small-final-only.json) | none | 3.830 s | 10.522 s | 116/254 | 45.67% |
| [whisper-tiny-final-only](final-only/whisper-tiny-final-only.json) | none | 3.050 s | 5.997 s | 368/254 | 144.88% |
| [whisper-turbo-final-only](final-only/whisper-turbo-final-only.json) | none | 1.467 s | 1.825 s | 15/254 | 5.91% |
| [whisper-tiny-current](current-checkpoints/whisper-tiny-current.json) | 3.972 s | 6.012 s | 9.774 s | 448/254 | 176.38% |
| [whisper-turbo-current](current-checkpoints/whisper-turbo-current.json) | 7.416 s | 2.992 s | 9.056 s | 15/254 | 5.91% |

## Process resource observations

| Case | Handler setup | Peak process RSS | Median CPU seconds / audio second |
|---|---:|---:|---:|
| moonshine-medium | 0.379 s | 1130 MiB | 1.42 |
| moonshine-small | 0.300 s | 885 MiB | 1.04 |
| moonshine-tiny | 0.139 s | 482 MiB | 0.62 |
| parakeet | 5.157 s | 1746 MiB | 0.24 |
| phonon | 0.007 s | 223 MiB | 0.03 |
| whisper-tiny | 6.400 s | 510 MiB | 1.53 |
| whisper-base | 4.742 s | 703 MiB | 1.46 |
| whisper-small | 13.508 s | 876 MiB | 1.50 |
| whisper-turbo | 6.146 s | 1173 MiB | 0.21 |
| whisper-base-final-only | 6.974 s | 653 MiB | 0.25 |
| whisper-small-final-only | 8.123 s | 797 MiB | 0.27 |
| whisper-tiny-final-only | 4.829 s | 600 MiB | 0.30 |
| whisper-turbo-final-only | 5.994 s | 888 MiB | 0.07 |
| whisper-tiny-current | 6.032 s | 522 MiB | 1.56 |
| whisper-turbo-current | 6.231 s | 755 MiB | 0.23 |

Setup starts after package imports and staged downloads and includes any handler warmup. Phonon setup, RSS and CPU describe its client only, with an already-loaded server. MLX RSS does not measure all device allocations; these values cannot rank total memory use. CPU seconds can exceed wall time when several threads work together.

Final wait includes time lost while feeding paced audio: for each clip, `final_latency_s + audio_feed_s - duration_s`, then the median or maximum. This keeps Moonshine’s synchronous audio-feed work in the comparison. Raw reports also retain final latency measured from the actual last packet. First words means the first nonempty hypothesis, which may change; it does not promise a correct word. Forced finalization occurs at the clip boundary, so these waits exclude microphone/VAD endpoint detection, LLM and TTS.

Phonon has the earliest first text in this sweep. Moonshine small/medium offer much shorter final waits with more word errors than Parakeet. Parakeet has the fewest errors on this fixture but long delays, including a 77-second case. Whisper Turbo improves when live text is off. Moonshine remains a benchmark-only bridge; it is not a registered speech-to-speech backend.

## What the Whisper results establish

The legacy tiny/base/small checkpoints produced many inserted and repeated words through the repo’s MLX Audio handler. These runs do not establish their usual model accuracy. Turning live text off did not fix their transcripts. WER can exceed 100% when insertions outnumber reference words. The current `whisper-tiny-asr-fp16` checkpoint also produced many insertions (448/254 errors). Current `whisper-large-v3-turbo-asr-fp16` made the same 15 errors as the older Turbo, with a 2.992-second median final wait in live mode. These checks do not explain the tiny/base/small failures; diagnosing the MLX handler/runtime remains separate work. Current ASR checkpoints are recorded separately rather than substituted for earlier results. Decoder defaults were unchanged; no accuracy tuning was applied.

The initial base/small/turbo setup attempts failed because the offline cache lacked processor files. All failed reports and their logs remain under `initial/`. The retry staged those files and used matching base/small processors. Failed setup attempts do not contribute WER or latency.

## Scope and reproducibility

- MLX Audio 0.5.7, MLX 0.32.3, Moonshine Voice 0.1.5, Fermion 0.2.7. The Mac benchmark environment differs from the repo’s pinned full installation; results apply to this measured stack.
- Moonshine uses its native library engine, VAD and default thread policy with speaker recognition off. Phonon uses its external streaming server. Parakeet and Whisper show live text by rerunning growing audio windows; final-only Whisper cases run the whole final once.
- Background load, swap and memory pressure vary across serial cases. Per-case metadata and five-second resource traces preserve observations; timings are not from an idle, isolated machine.
- PyTorch requests four threads. This does not fix MLX or Moonshine threads. Process RSS is not total Metal/unified memory; Phonon client CPU/RSS exclude its server.
- Word scoring folds case, removes punctuation and uses word edit distance. It does not expand “Mr.” into “Mister,” so some counted errors reflect spelling conventions.
- [Source provenance](source-provenance.json), per-case metadata, model file hashes and resolved HF revisions document the run. Initial metadata predates added source-hash fields; the provenance file records those sources explicitly.
- [Machine-readable comparison](comparison.json) contains all completed cases and setup failures. Original reports retain every transcript and capture-end timing.
- Repeat commands and all three configurations are in the [benchmark README](../../README.md#mac-candidate-sweep). Audio and weights stay in persistent caches and are not checked in.
