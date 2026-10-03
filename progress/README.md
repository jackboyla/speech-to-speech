# Turn speculation session

The source, public rerun commands, raw benchmark reports and limits are in
[benchmarks/turn_speculation/README.md](../benchmarks/turn_speculation/README.md).

The workstation's shared environment contains Transformers 5.17.0. Upstream
requires 5.18.0 for Nemotron diarization tests. Rather than changing a running
service's environment, this worktree has a local overlay. Exact setup:

```bash
uv venv .venv --python /home/jack/workspace/Desktop/speech-to-speech/.venv/bin/python
.venv/bin/python - <<'PY'
import site
from pathlib import Path
Path(site.getsitepackages()[0], 'workstation-dependencies.pth').write_text(
    '/home/jack/workspace/Desktop/speech-to-speech/.venv/lib/python3.11/site-packages\n'
)
PY
uv pip install --python .venv/bin/python --no-deps transformers==5.18.0
```

Public setup uses `uv sync --group dev`; the local overlay is workstation-only.
Run local checks with:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest tests/ -x -q
PYTHONPATH=src .venv/bin/python -m ruff check src/ tests/ benchmarks/turn_speculation/benchmark.py
PYTHONPATH=src .venv/bin/python -m ruff format --check src/ tests/ benchmarks/turn_speculation/benchmark.py
PYTHONPATH=src .venv/bin/python -m mypy src/
```

Tmux launch scripts and logs are in `progress/logs/`. Baseline checks used the
main checkout's environment with `PYTHONPATH=src CUDA_VISIBLE_DEVICES=''`.
