# Experiment ledger

Append-only; timings must distinguish simulated costs from real inference.

## 2026-10-03—controller-check-1

### Intent
Check private speculation against replacement hypotheses and existing notifier events.

### Environment
- Machine: radiance-ws
- GPUs used: none; CUDA_VISIBLE_DEVICES=''
- Git commit: 81b688b + uncommitted prototype
- Branch: feat/turn-speculation
- Python environment: main checkout .venv (PYTHONPATH=worktree/src)
- Docker image: not used

### Resource check before launch
```bash
hostname
uptime
df -h / /home
free -h
nvidia-smi
docker ps
tmux ls || true
ps -eo user,pid,pgid,stat,etime,%cpu,%mem,cmd --sort=-%mem | head -30
```
Both GPUs occupied; CPU checks only.

### Commands
#### Eval
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /home/jack/workspace/Desktop/speech-to-speech/.venv/bin/pytest -q tests/test_turn_speculation.py tests/test_transcription_notifier.py
```

### Artifacts
- Source: src/speech_to_speech/experimental/
- Tests: tests/test_turn_speculation.py

### Result
Base: unmodified notifier tests.
Candidate: first check 8 failed / 9 passed; corrected is_current_turn call signature.
Second check: 17 passed in 0.30 s.
- Failure modes: incorrect tracker method arity in initial implementation.
- GPU utilization: no GPU compute.

### Decision
Keep prototype, proceed to real prefill and timing comparison.

### Notes
No learned model or live microphone gain claimed.

## 2026-10-03—llama-prefill-compatibility

### Intent
Check whether the installed llama.cpp server honors cache-only requests.

### Environment
- Machine: radiance-ws
- GPUs used: none; no GPU devices passed to Docker
- CUDA_VISIBLE_DEVICES: '' for Python clients
- Git commit: 81b688b + prototype
- Branch: feat/turn-speculation
- Python environment: shared .venv via PYTHONPATH=src
- Docker image: ghcr.io/ggml-org/llama.cpp@sha256:0192ab2545efcbe79c240645e34abd8fffbe4813aedcef5a0e3a886ef6d6d82f
- llama.cpp: build 11096 / c550d2f60
- Model: cached SmolLM2-135M-Instruct-Q4_K_M.gguf

### Resource check before launch
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
free -h
uptime
docker ps
tmux ls
```
~18GiB available RAM. Other GPU services active; use isolated 4-CPU / 2GiB container.

### Commands
#### Setup
```bash
tmux new-session -d -s turn-speculation-cpu 'bash /home/jack/workspace/Desktop/speech-to-speech/.claude/worktrees/turn-speculation/progress/logs/turn-speculation-server.sh'
```
Exact Docker command in script and benchmark README.
#### Diagnostics
```bash
docker ps --filter name=turn-speculation-cpu
curl -sS http://127.0.0.1:18093/health
docker top turn-speculation-cpu -eo pid,cmd
tail -20 progress/logs/turn-speculation-server.log
curl -sS http://127.0.0.1:18093/completion -H 'Content-Type: application/json' -d '{"prompt":"User: Hello\nAssistant:","n_predict":0,"id_slot":0,"cache_prompt":true,"stream":false}'
```

### Artifacts
- Logs: progress/logs/turn-speculation-server.log
- Source inspection: progress/evaluations/llama-server-context.cpp (upstream c550d2f60)
- Regression: compatibility sample test in tests/test_turn_speculation.py

### Result
Base: documented n_predict=0 produces no token.
Candidate: real build returns tokens_predicted=1 and a sampled newline; prompt KV retained, no subsequent decode step.
- First benchmark stopped on the adapter's strict zero-token check.
- Changed adapter to discard at most one sample, strip generated content, record count, reject >1.
- Resource use: 234MiB container RAM at idle; no GPU allocation.
- Runtime: server ready in ~0.2 s; initial health probe was too early and failed, next probe healthy.

### Decision
Keep compatibility handling; never expose the sampled partial answer.

### Notes
The source confirms sampling occurs immediately after prompt processing, before checking the zero-token budget. This mismatch is recorded rather than treated as a true cache-only backend.

## 2026-10-03—prefill-and-tools-r2

### Intent
Measure partial prefill against ordinary history caching and tool reuse against final-only execution.

### Environment
- Machine: radiance-ws
- GPUs used: none
- CUDA_VISIBLE_DEVICES: ''
- Git commit: 81b688b + prototype
- Branch: feat/turn-speculation
- Python environment: shared .venv, then project-local overlay for timed run
- Docker image/model: same as preceding entry; one dedicated slot

### Resource check before launch
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
docker stats --no-stream turn-speculation-cpu
docker ps
tmux ls
```

### Commands
#### Compare
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /home/jack/workspace/Desktop/speech-to-speech/.venv/bin/python benchmarks/turn_speculation/benchmark.py --mode prefill --repeats 3 --output benchmarks/turn_speculation/prefill-cpu.json
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /home/jack/workspace/Desktop/speech-to-speech/.venv/bin/python benchmarks/turn_speculation/benchmark.py --mode tools --output benchmarks/turn_speculation/tools-synthetic.json
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/turn_speculation/benchmark.py --mode prefill --partial-interval 0.2 --repeats 3 --output benchmarks/turn_speculation/prefill-timed-cpu.json
```
All launched through tmux scripts (run-benchmarks.sh / run-timed.sh); verified logs and output JSON.

### Artifacts
- Predictions/metrics: benchmarks/turn_speculation/{prefill-cpu,prefill-timed-cpu,tools-synthetic}.json
- Eval summary/commands: benchmarks/turn_speculation/README.md
- Logs: progress/logs/{prefill-cpu,prefill-timed-cpu,tools-synthetic}.log
- Checkpoints: none; no training

### Result
Base: same history warmed in both arms; final one-token wall time 12.47 ms (component), 12.75 ms (timed controller).
Candidate: 8.49 ms (component), 11.04 ms (timed controller).
Delta: median paired saving 4.41 ms / 2.39 ms. All 15 first tokens match in each run.
- Prefill total extra work: 16–39 ms across several updates; gain is small.
- Synthetic search: ~1001 ms base vs ~401 ms reused after final; changed destination ~1001 ms, 1 old call cancelled + 1 fresh call.
- Failure modes: no incorrect final reuse in these fixtures; no live ASR/quality measurement.
- GPU utilization: none
- RAM: small isolated CPU server
- Runtime: each run well under a minute

### Decision
Keep experimental controller. Tool overlap warrants a real-service test; do not claim large prefill or end-to-end voice gains from this tiny model.

### Notes
Handwritten ASR fixtures; tools and remaining speech duration use controlled sleeps. Learned policy, full answer quality, real speech and production service integration remain future work.

## 2026-10-03—repository-checks

### Intent
Verify the new module and unchanged default pipeline on latest upstream.

### Environment
- Machine: radiance-ws
- GPUs used: none; CUDA_VISIBLE_DEVICES=''
- Git commit: 81b688b + final prototype
- Branch: feat/turn-speculation
- Python environment: worktree .venv overlay, Transformers 5.18.0; other dependencies read from shared .venv
- Docker image: not used for tests

### Resource check before launch
Previously checked; existing GPU services left alone. OMP_NUM_THREADS=4 / MKL_NUM_THREADS=4.

### Commands
#### Setup
Exact overlay setup in progress/README.md. Shared Transformers 5.17.0 lacks the new upstream Nemotron diarization classes; never change that environment while services use it.
#### Eval
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest tests/ -x -q
PYTHONPATH=src .venv/bin/python -m ruff check src/ tests/ benchmarks/turn_speculation/benchmark.py
PYTHONPATH=src .venv/bin/python -m ruff format --check src/ tests/ benchmarks/turn_speculation/benchmark.py
PYTHONPATH=src .venv/bin/python -m mypy src/
```

### Artifacts
- Logs: progress/logs/check-all.log (failed collection), check-all-r2.log (full success)
- Tests: 13 new behavioral cases including notifier boundary, correction, stale turn/revision, cancellation race, reopen, timeout, privacy, coalescing and tool limits

### Result
Base: initial full run fails collection because shared environment has Transformers 5.17.0.
Candidate with project-local 5.18.0: 2217 passed / 3 skipped in 93.23 s; Ruff lint/format pass; mypy 123 files pass.
- Failure modes: environment mismatch resolved locally; no source workaround or shared service changes.
- GPU compute: none

### Decision
Keep prototype; ready for review as experimental work.
