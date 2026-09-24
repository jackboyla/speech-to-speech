#!/usr/bin/env python3
"""Generate the utterances the lab's scenario runner speaks.

Run once. The WAVs land in ``demo/lab/fixtures`` and are not committed: audio
regenerated from the pinned text below is reproducible enough for turn-taking
work, and a repository is a bad place to keep a few megabytes of speech.

Synthesized speech is used rather than recordings of people, which is worth
stating plainly: TTS audio is cleaner than a microphone in a room, so latencies
measured from it are a lower bound on what a person would see. What it buys is
the thing the scenarios need — the same utterance, paced the same way, every
run.

Kokoro is used rather than the pipeline's own TTS so the input to the
measurement is not also a component under test.

Usage::

    # From the speech-to-speech virtualenv, which already has kokoro:
    .venv/bin/python demo/scripts/make_lab_fixtures.py

    # Or any environment with kokoro installed:
    python demo/scripts/make_lab_fixtures.py --out demo/lab/fixtures --voice af_heart
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

KOKORO_RATE_HZ = 24_000
BYTES_PER_SAMPLE = 2

# Each line is chosen for a job in demo/lab/scenarios.js.
#
# The two halves of `half_question` are split mid-clause on purpose: spoken back
# to back with a short gap they are one sentence, so a pipeline that ends the
# turn at the pause answers a question nobody asked. `yes` is deliberately tiny,
# short enough to sit near the VAD's lower bound. The rest are short questions
# with short answers, so a reply measures how fast the first word arrives rather
# than how long the model likes to talk.
FIXTURES: list[tuple[str, str]] = [
    ("capital_france", "What is the capital of France?"),
    ("planet_count", "How many planets are in the solar system?"),
    ("half_question", "What is the capital"),
    ("half_question_tail", "of France?"),
    ("interrupt_wait", "Actually, wait, forget that."),
    ("yes", "Yes."),
    ("weather_query", "What is the weather in Paris right now?"),
]

_PIPELINE_CACHE: dict[str, object] = {}


def _pipeline(voice: str):
    """Build the Kokoro pipeline once; loading it per line dominates runtime."""
    from kokoro import KPipeline

    # Kokoro picks its grapheme-to-phoneme frontend from the voice's first
    # letter: 'a' for American English, 'b' for British, and so on.
    code = voice[0] if voice else "a"
    if code not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[code] = KPipeline(lang_code=code)
    return _PIPELINE_CACHE[code]


def synthesize(text: str, voice: str) -> np.ndarray:
    """Render one line to mono float32 at Kokoro's native rate."""
    pipeline = _pipeline(voice)
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    if not chunks:
        raise RuntimeError(f"kokoro produced no audio for {text!r}")
    return np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])


def trim_silence(samples: np.ndarray, floor: float = 3e-3, pad_ms: int = 40) -> np.ndarray:
    """Trim leading and trailing near-silence, keeping a short pad.

    Scenario timing is quoted from speech start and speech end, so silence
    baked into a fixture would show up as pipeline latency that nothing in the
    pipeline caused.
    """
    loud = np.flatnonzero(np.abs(samples) > floor)
    if loud.size == 0:
        return samples
    pad = int(KOKORO_RATE_HZ * pad_ms / 1000)
    start = max(0, int(loud[0]) - pad)
    end = min(len(samples), int(loud[-1]) + pad)
    return samples[start:end]


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", type=Path, default=here.parent / "lab" / "fixtures")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--peak", type=float, default=0.7, help="Normalise each fixture to this peak.")
    args = parser.parse_args()

    try:
        import kokoro  # noqa: F401
    except ImportError:
        print(
            "error: kokoro is not installed in this interpreter.\n"
            "Run this from an environment that has it, for example:\n"
            "  .venv/bin/python demo/scripts/make_lab_fixtures.py",
            file=sys.stderr,
        )
        return 2

    entries = []
    for name, text in FIXTURES:
        samples = trim_silence(synthesize(text, args.voice))
        peak = float(np.max(np.abs(samples))) or 1.0
        # A fixture quieter than the noise gate would never open the gate, and
        # the scenario would look like a pipeline failure instead of a bad file.
        samples = samples * (args.peak / peak)
        filename = f"{name}.wav"
        write_wav(args.out / filename, samples, KOKORO_RATE_HZ)
        duration = len(samples) / KOKORO_RATE_HZ
        entries.append({"name": name, "file": filename, "text": text, "durationS": round(duration, 3)})
        print(f"{name:20s} {duration:5.2f}s  {text}")

    manifest = {
        "voice": args.voice,
        "sampleRate": KOKORO_RATE_HZ,
        "generator": "demo/scripts/make_lab_fixtures.py",
        "fixtures": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(entries)} fixtures + manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
