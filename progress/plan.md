# Turn speculation prototype — 2026-10-03

## Instructions
- Implement the user's #2 (incremental prefill) and #3 (early tool execution) proposal.
- Preserve existing work and services. Do not merge or send external messages.
- Keep commands in README and append meaningful runs to experiment-log.md.

## Plan
1. Audit partial/final STT, turn ownership, established prefix-cache APIs.
2. Build an opt-in experimental controller with replaceable decision policy,
   token-safe prefill, private tool results, revision/cancellation, and final validation.
3. Consume existing TranscriptionNotifier events through an opt-in adapter.
4. Add deterministic adversarial tests and reproducible timing benchmark.
5. Run a real backend cache comparison if a safe local CPU backend is available.
6. Review, run repository checks, record results and limits.

## Decisions
- Base: upstream/main 81b688b; branch feat/turn-speculation in this worktree.
- Existing partials are replacement hypotheses, not append-only strings.
- Default behavior stays unchanged; no early assistant audio/history/protocol output.
- Start with a repeat-prefix rule as a measured baseline, not a claimed learned model.
- Exact final tool-call identity controls reuse; late constraints must be revalidated.
- Both RTX5090s have ~30/31GiB occupied. No new GPU model or shared-server cache mutation.

## Status
Prototype complete; scope defaults to prototype + benchmarks, as stated at the start.

- 13 new behavior tests; full suite 2217 passed / 3 skipped.
- Ruff lint/format pass, mypy 123 source files pass.
- Real cache component: 12.47 → 8.49 ms; paired saving 4.41 ms.
- Actual controller on timed synthetic partial events: 12.75 → 11.04 ms;
  paired saving 2.39 ms; 15/15 first tokens match.
- Synthetic one-second tool / 600 ms speech tail: ~1001 → ~401 ms when reused;
  changed destination cancels and falls back at ~1001 ms.
- Dedicated CPU server remains in tmux turn-speculation-cpu, loopback port18093,
  Docker turn-speculation-cpu, ~234MiB RAM. Existing services unchanged.
- Local Transformers5.18 overlay used for new upstream tests; shared5.17 untouched.

## Outstanding research / production work
- No learned model trained. Need real ASR traces and labels before fitting one.
- No default service wiring or server flag. Existing protocol and output unchanged.
- No measured audio/STT/TTS latency or representative-model answer quality.
- The public README specifies the next experiments and integration requirements.
- Prototype can be reviewed independently; no upstream PR or merge requested.

## Commands
- git fetch upstream main
- git worktree add .claude/worktrees/turn-speculation -b feat/turn-speculation upstream/main
- hostname; uptime; df -h / /home; free -h; nvidia-smi; docker ps; tmux ls
- ps -eo user,pid,pgid,stat,etime,%cpu,%mem,cmd --sort=-%mem | head -30

## Artifacts
- Source: src/speech_to_speech/experimental/
- Tests: tests/test_turn_speculation.py
- Benchmarks, raw reports, provenance, commands: benchmarks/turn_speculation/
- Full chronological ledger: progress/experiment-log.md
- Workstation setup: progress/README.md
- Logs and tmux launch scripts: progress/logs/
