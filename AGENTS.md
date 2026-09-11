# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is TinyWatch Environment v1, Reward v1, observation v1
and tool schema v1. The project directory is TinyWatchAgent. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths, experiment journals or LLM judges on
the main eval path.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_watchlist` terminal result via `finalize_watchlist`
with `reward_valid=true`.

Do not start training, merge models, Flash collection, or the 100-task
evaluation unless the user explicitly requests execution.
