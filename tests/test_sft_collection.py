from tinywatch_grpo.collection.reward_filter import acceptance_reasons, build_sft_row
from tinywatch_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example, split_rows_by_task


def test_split_rows_by_task_disjoint():
    rows = [{"task_id": i, "messages": []} for i in range(10)]
    train, val = split_rows_by_task(rows, validation_ratio=0.2, seed=42)
    train_ids = {row["task_id"] for row in train}
    val_ids = {row["task_id"] for row in val}
    assert train_ids.isdisjoint(val_ids)
    assert len(train) + len(val) == 10


class _Tok:
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
        return {"input_ids": [ord(ch) for ch in text]}


def test_action_only_labels():
    example = build_supervised_example(
        messages=[
            {"role": "user", "content": "plan"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "search_movies", "arguments": '{"query":"x"}'}}],
            },
            {"role": "tool", "content": "obs"},
        ],
        tools=[],
        tokenizer=_Tok(),
        max_length=500,
    )
    labeled = [token for token in example["labels"] if token != IGNORE_INDEX]
    assert labeled
    assert len(labeled) < len(example["labels"])


def test_acceptance_requires_valid_reward():
    trajectory = {
        "task_id": 1,
        "status": "done",
        "done": True,
        "error": None,
        "messages": [],
        "steps": [{"tool_name": "finalize_watchlist"}],
        "terminal_result": {
            "reward_detail": {
                "reward_version": "tinywatch-reward-v1",
                "reward_valid": False,
                "reward": 1.0,
            }
        },
    }
    ok, reasons = acceptance_reasons(trajectory)
    assert ok is False
    assert "reward_invalid" in reasons
