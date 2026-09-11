"""Lightweight GRPO utilities: group-relative advantages, no veRL required."""

from __future__ import annotations


def group_advantages(rewards: list[float], *, eps: float = 1e-8) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    std = variance ** 0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(item - mean) / (std + eps) for item in rewards]


def select_reward_varying_groups(group_ids: list, rewards: list[float]) -> tuple[list[int], dict]:
    buckets: dict = {}
    for index, (group_id, reward) in enumerate(zip(group_ids, rewards)):
        buckets.setdefault(group_id, []).append((index, float(reward)))
    kept = []
    skipped = 0
    for items in buckets.values():
        values = {reward for _, reward in items}
        if len(values) <= 1:
            skipped += 1
            continue
        kept.extend(index for index, _ in items)
    kept.sort()
    return kept, {"kept_group_count": len(buckets) - skipped, "skipped_constant_groups": skipped}


def grpo_policy_loss(logprobs: list[float], advantages: list[float]) -> float:
    if len(logprobs) != len(advantages) or not logprobs:
        raise ValueError("logprobs and advantages must be aligned and non-empty")
    return -sum(lp * adv for lp, adv in zip(logprobs, advantages)) / len(logprobs)
