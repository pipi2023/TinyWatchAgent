# Package layout

```text
environment/   TinyWatch v1, BM25, tool schema v1, Reward v1
generation/    IMDb catalog import, task synthesis, Oracle policy
collection/    Flash rollout loop and Reward filter
training/sft/  action-only labels
training/grpo/ group-relative advantages
evaluation/    four deterministic panels, no LLM Judge
cli.py         smoke + offline evaluate
smoke.py       CPU contract checks
```

Launchers stay in repository `scripts/`.
