# Phonon streaming measurements

The [benchmark guide](../../docs/phonon-benchmark.md) defines the measurements,
commands, dataset selection and limits. The initial reports in `2026-10-01/`
compare ten labeled clips on CPU and RTX 5090. They include a separate warmup
and each final transcript. The corpus provenance pins the source revision and
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
client CPU and RSS fields exclude its external server. No Mac inference result
was obtained. The small clean fixture cannot establish general model accuracy.
