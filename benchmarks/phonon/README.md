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
