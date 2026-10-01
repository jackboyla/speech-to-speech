"""Export a small labeled LibriSpeech fixture without changing the project environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "hf-internal-testing/librispeech_asr_dummy"
FILE = "clean/validation-00000-of-00001.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    revision = HfApi().dataset_info(REPO, revision=args.revision).sha
    source = Path(hf_hub_download(REPO, FILE, repo_type="dataset", revision=revision))
    rows = pq.read_table(source).to_pylist()
    if args.count > len(rows):
        parser.error(f"Fixture has only {len(rows)} rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    hashes = {}
    for row in rows[: args.count]:
        name = f"{row['id']}.flac"
        raw = row["audio"]["bytes"]
        (args.output_dir / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        manifest.append({"id": row["id"], "audio": name, "text": row["text"]})
    (args.output_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": revision,
                "file": FILE,
                "selection": f"first {args.count} rows",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "audio_sha256": hashes,
                "limit": "Small clean fixture, not representative corpus accuracy; repeated speaker.",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output_dir / "manifest.jsonl")


if __name__ == "__main__":
    main()
