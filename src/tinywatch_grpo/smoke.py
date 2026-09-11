"""Pure-CPU smoke checks for TinyWatch Environment v1 / Reward v1 contracts."""

from __future__ import annotations

from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS
from tinywatch_grpo.evaluation.metrics import compute_deterministic_metrics
from tinywatch_grpo.evaluation.trajectory import normalize_trajectory
from tinywatch_grpo.generation.oracle import run_oracle
from tinywatch_grpo.generation.task_gen import generate_task
from tinywatch_grpo.generation.vocab import mini_catalog
from tinywatch_grpo.training.grpo.loop import select_reward_varying_groups
from tinywatch_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example


class _CharacterTemplate:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                text += f"[tool={call['function']['name']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _next_int(n, state):
    state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
    return state[0] % n


def run_cpu_smoke() -> dict:
    checks = []
    names = {schema["function"]["name"] for schema in TINYWATCH_TOOL_SCHEMAS}
    required = {
        "search_movies",
        "open",
        "view_crew",
        "compare",
        "add_to_watchlist",
        "view_draft",
        "finalize_watchlist",
        "abort",
    }
    if not required <= names:
        raise AssertionError("movie tool schema is incomplete")
    checks.append("action_schema")

    catalog = mini_catalog()
    state = [7]

    def nxt(n):
        return _next_int(n, state)

    task = generate_task(1, catalog, difficulty="easy", nxt=nxt, solvable=True)
    env = TinyWatchEnv(catalog, task, max_steps=14)
    trajectory = run_oracle(env)
    detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
    if detail.get("reward_type") != "gold_watchlist" or detail.get("reward_valid") is not True:
        raise AssertionError(f"oracle smoke failed: {detail}")
    checks.append("oracle_gold")

    normalized = normalize_trajectory(trajectory)
    metrics = compute_deterministic_metrics(normalized, task=task)
    if not metrics["reward_and_outcome"]["strict_gold_success"]:
        raise AssertionError("strict gold success sample failed")
    checks.append("reward_sample")

    tokenizer = _CharacterTemplate()
    supervised = build_supervised_example(
        messages=[
            {"role": "user", "content": "选一部科幻片"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_movies",
                            "arguments": '{"query":"科幻"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "private observation"},
        ],
        tools=[],
        tokenizer=tokenizer,
        max_length=1000,
    )
    if supervised is None:
        raise AssertionError("SFT label-mask sample was rejected")
    labeled = [token for token in supervised["labels"] if token != IGNORE_INDEX]
    if not labeled or len(labeled) == len(supervised["labels"]):
        raise AssertionError("SFT labels are not assistant-only")
    checks.append("sft_label_mask")

    selected, diagnostics = select_reward_varying_groups(["a", "a", "b", "b"], [0.0, 1.0, 0.5, 0.5])
    if selected != [0, 1] or diagnostics["kept_group_count"] != 1:
        raise AssertionError("dynamic sampling grouping changed")
    checks.append("grpo_grouping")

    return {"schema_version": "tinywatch-cpu-smoke-v1", "checks": checks}
