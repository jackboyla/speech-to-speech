"""Run a sequential Mac STT sweep; see benchmarks/phonon/README.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from threading import Event, Thread


def resources():
    commands = [["uptime"], ["df", "-h", "/"], ["memory_pressure"], ["sysctl", "vm.swapusage"]]
    return {
        "time": time.time(),
        **{" ".join(cmd): subprocess.run(cmd, capture_output=True, text=True, check=False).stdout for cmd in commands},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-timeout", type=float, default=900)
    parser.add_argument("--ledger", type=Path, default=Path("progress/experiment-log.md"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in json.loads(args.config.read_text()):
        label = case["id"]
        output = args.output_dir / f"{label}.json"
        metadata = {
            "case": case,
            "resources_before": resources(),
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path("scripts/benchmark_phonon.py"),
                    Path("scripts/benchmark_moonshine_handler.py"),
                    Path("src/speech_to_speech/STT/streaming_handler.py"),
                    Path("src/speech_to_speech/STT/parakeet_tdt_handler.py"),
                    Path("src/speech_to_speech/STT/mlx_audio_whisper_handler.py"),
                ]
            },
        }
        print(f"Starting {label}", flush=True)
        try:
            if case["backend"] in {"mlx-audio-whisper", "parakeet-tdt"}:
                from huggingface_hub import snapshot_download

                snapshot = Path(snapshot_download(case["model"]))
                metadata["model_revision"] = snapshot.name
                metadata["model_files"] = [
                    {
                        "name": str(p.relative_to(snapshot)),
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in sorted(snapshot.rglob("*"))
                    if p.is_file()
                ]
                if case["backend"] == "mlx-audio-whisper":
                    name = case["model"].rsplit("/", 1)[-1]
                    size = next(
                        (size for size in ["tiny", "base", "small"] if name.startswith(f"whisper-{size}")), None
                    )
                    processor_repo = f"openai/whisper-{size}" if size else "openai/whisper-large-v3"
                    processor = Path(
                        snapshot_download(
                            processor_repo,
                            allow_patterns=[
                                "config.json",
                                "preprocessor_config.json",
                                "tokenizer*",
                                "special_tokens_map.json",
                                "added_tokens.json",
                                "merges.txt",
                                "vocab.json",
                                "normalizer.json",
                            ],
                        )
                    )
                    metadata["processor"] = {"repo": processor_repo, "revision": processor.name}
            elif case["backend"] == "moonshine":
                from moonshine_voice import ModelArch, get_model_for_language

                arch = getattr(ModelArch, case["model"].replace("-", "_").upper())
                metadata["model_path"] = get_model_for_language("en", arch)[0]
            elif case["backend"] == "phonon":
                server = shlex.quote(str(Path(sys.executable).with_name("fermion")))
                subprocess.run(
                    [
                        "/opt/homebrew/bin/tmux",
                        "new-session",
                        "-d",
                        "-s",
                        "phonon-server",
                        f"{server} serve phonon-2 --port 18090 --threads 4 > progress/logs/phonon-server-sweep.log 2>&1",
                    ],
                    check=True,
                )
                deadline = time.monotonic() + 180
                while True:
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:18090/health", timeout=5) as response:
                            metadata["server_health_start"] = json.load(response)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Phonon server readiness timeout") from None
                        time.sleep(1)
            cmd = [
                sys.executable,
                "scripts/benchmark_phonon.py",
                "--manifest",
                str(args.manifest),
                "--backend",
                case["backend"],
                "--device",
                case["device"],
                "--threads",
                "4",
                "--output",
                str(output),
                "--log-level",
                "debug",
            ]
            if case.get("model"):
                cmd.extend(["--model", case["model"]])
            if case.get("partial_interval"):
                cmd.extend(["--partial-interval", str(case["partial_interval"])])
            metadata["command"] = cmd
            env = {**os.environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            if case["backend"] in {"mlx-audio-whisper", "parakeet-tdt"}:
                env["HF_HUB_OFFLINE"] = "1"
            stopped = Event()

            def monitor():
                with (args.output_dir / f"{label}-resources.jsonl").open("w") as log:
                    while not stopped.is_set():
                        log.write(json.dumps(resources()) + "\n")
                        log.flush()
                        stopped.wait(5)

            observer = Thread(target=monitor, daemon=True)
            observer.start()
            try:
                with (args.output_dir / f"{label}.log").open("w") as log:
                    run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.case_timeout)
                metadata["exit_code"] = run.returncode
            finally:
                stopped.set()
                observer.join(timeout=10)
            if output.exists():
                report = json.loads(output.read_text())
                metadata["result"] = {
                    k: report.get(k)
                    for k in [
                        "successful_clips",
                        "failed_clips",
                        "warmup_failures",
                        "median_first_partial_s",
                        "median_final_latency_s",
                        "word_errors_total",
                        "reference_words_total",
                        "wer",
                        "error",
                    ]
                }
            if case["backend"] == "phonon":
                with urllib.request.urlopen("http://127.0.0.1:18090/health", timeout=5) as response:
                    metadata["server_health_end"] = json.load(response)
        except Exception as exc:
            metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["resources_after"] = resources()
        (args.output_dir / f"{label}-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary.append(metadata)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        with args.ledger.open("a") as ledger:
            ledger.write(
                f"\n## {time.strftime('%Y-%m-%d')}—mac-sweep-{label}\n\n"
                "### Intent\n\nSame-corpus sequential candidate comparison against Phonon/Parakeet.\n\n"
                f"### Environment\n\n- Machine: {platform.node()}, M2 Air 8 GiB\n"
                f"- GPU: Apple M2 MLX for MPS; Moonshine default native engine\n"
                f"- CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}\n"
                "- Git commit/branch: copied checkout; see source hashes and the session ledger\n"
                f"- Python environment: {sys.executable}\n- Docker: none\n\n"
                "### Resource check before launch\n\n```bash\nuptime\ndf -h /\nmemory_pressure\nsysctl vm.swapusage\n```\n\n"
                f"Exact observations: {label}-metadata.json and {label}-resources.jsonl.\n\n"
                f"### Commands\n\n#### Eval\n\n```bash\n{shlex.join(metadata.get('command', [])) or '# Setup failed before inference'}\n```\n\n"
                "Setup: driver snapshots the HF model and hashes its files, or stages Moonshine in its persistent cache.\n"
                "PYTHONPATH=src, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4; HF_HUB_OFFLINE=1 for offline HF backends.\n\n"
                f"### Artifacts\n\n- Logs: {args.output_dir / (label + '.log')}\n"
                f"- Predictions/metrics: {output}\n- Model/source/resources: {label}-metadata.json\n"
                f"- Resource trace: {label}-resources.jsonl\n- Checkpoints: persistent cache; no trained checkpoint\n\n"
                f"### Result\n\n{json.dumps(metadata.get('result') or {'error': metadata.get('error')})}\n\n"
                f"Exit: {metadata.get('exit_code')}; errors: {metadata.get('error')}.\n\n"
                "### Decision\n\nPreserve results and failures; compare after all cases finish.\n\n"
                "### Notes\n\nShared Mac; user apps unchanged. No universal accuracy/speed claim.\n"
            )
        print(
            json.dumps(
                {
                    "id": label,
                    "exit_code": metadata.get("exit_code"),
                    "result": metadata.get("result"),
                    "error": metadata.get("error"),
                }
            ),
            flush=True,
        )
    return int(any(row.get("error") or row.get("exit_code") != 0 for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
