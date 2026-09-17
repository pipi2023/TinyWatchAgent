from tinywatch_grpo.training.grpo.loop import (
    group_advantages,
    grpo_policy_loss,
    is_collapsed_rollout,
    ppo_clip_loss,
    select_reward_varying_groups,
    select_trainable_varying_indices,
)
from tinywatch_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example


def test_group_advantages_zero_mean():
    adv = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(adv)) < 1e-8


def test_constant_group_dropped():
    kept, diagnostics = select_reward_varying_groups(["a", "a", "b", "b"], [0.2, 0.2, 0.1, 0.9])
    assert kept == [2, 3]
    assert diagnostics["kept_group_count"] == 1


def test_policy_loss_sign():
    loss = grpo_policy_loss([-0.2, -0.4], [1.0, -1.0])
    assert loss < 0


def test_shaped_reward_prefers_gold_overlap():
    from tinywatch_grpo.training.grpo.reward_shaping import shaped_grpo_reward

    task = {"gold_watchlist": {"movie_ids": ["g1", "g2"]}, "constraints": {}}
    closer = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 0.6, "reward_type": "valid_alternative"},
            "watchlist": {"movie_ids": ["g1", "x"]},
        },
        "steps": [],
    }
    farther = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 0.6, "reward_type": "valid_alternative"},
            "watchlist": {"movie_ids": ["x", "y"]},
        },
        "steps": [],
    }
    assert shaped_grpo_reward(closer, task) > shaped_grpo_reward(farther, task)


def test_collapsed_rollout_detects_no_tool_call_and_template_leak():
    assert is_collapsed_rollout({"error": "no_tool_call", "messages": []})
    assert is_collapsed_rollout(
        {
            "error": None,
            "messages": [
                {
                    "role": "assistant",
                    "content": "user\n<tool_response>\n{\"movie_id\": \"tt1\"}",
                    "tool_calls": [],
                }
            ],
        }
    )
    assert is_collapsed_rollout(
        {
            "error": "too_many_guard_rejections",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "open"}}],
                }
            ],
        }
    )


def test_collapsed_rollout_detects_illegal_movie_id():
    from tinywatch_grpo.training.grpo.loop import has_illegal_movie_id

    fake = {
        "error": None,
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "open"}}]}],
        "steps": [{"tool_name": "open", "parameters": {"movie_id": 1000000001}}],
    }
    assert has_illegal_movie_id(fake)
    assert is_collapsed_rollout(fake)
    legal = {
        "error": None,
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "open"}}]}],
        "steps": [{"tool_name": "open", "parameters": {"movie_id": "tt0477348"}}],
    }
    assert not has_illegal_movie_id(legal)
    assert not is_collapsed_rollout(legal)


def test_trainable_varying_drops_collapsed_then_requires_reward_spread():
    rewards = [1.05, 1.05, 0.05, 1.05]
    kept, diagnostics = select_trainable_varying_indices(rewards, [True, True, False, True])
    assert kept == []
    assert diagnostics["skipped_constant_groups"] == 1
    kept, diagnostics = select_trainable_varying_indices([1.05, 0.05, 0.05, 1.05], [True, True, False, True])
    assert kept == [0, 1, 3]
    assert diagnostics["kept_group_count"] == 1


def test_ppo_clip_limits_large_ratio():
    unclipped_scale = ppo_clip_loss(new_logprob=0.0, old_logprob=-2.0, advantage=1.0, clip_ratio=0.2)
    assert abs(unclipped_scale - (-1.2)) < 1e-6
    negative = ppo_clip_loss(new_logprob=0.0, old_logprob=-2.0, advantage=-1.0, clip_ratio=0.2)
    assert negative > 0


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


def test_overflow_truncate_keeps_tail_labels():
    messages = [
        {"role": "user", "content": "x" * 80},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "search_movies", "arguments": '{"query":"x"}'}}],
        },
        {"role": "tool", "content": "obs"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "finalize_watchlist", "arguments": "{}"}}],
        },
    ]
    dropped = build_supervised_example(messages, tools=[], tokenizer=_Tok(), max_length=40, overflow="drop")
    assert dropped is None
    truncated = build_supervised_example(
        messages, tools=[], tokenizer=_Tok(), max_length=40, overflow="truncate"
    )
    assert truncated is not None
    assert len(truncated["input_ids"]) == 40
    assert any(label != IGNORE_INDEX for label in truncated["labels"])


def test_scale_advantage_downweights_negatives():
    from tinywatch_grpo.training.grpo.loop import scale_advantage

    assert scale_advantage(1.0, neg_coef=0.5) == 1.0
    assert scale_advantage(-1.0, neg_coef=0.5) == -0.5


def test_shuffled_n2_schedule_oversamples_multi():
    from tinywatch_grpo.training.grpo.loop import build_task_schedule

    tasks = [
        {"task_id": 1, "constraints": {"n_movies": 1, "directors": ["A"]}},
        {"task_id": 2, "constraints": {"n_movies": 2, "directors": ["B"]}},
        {"task_id": 3, "constraints": {"n_movies": 1}},
        {"task_id": 4, "constraints": {"n_movies": 2}},
    ]
    schedule = build_task_schedule(tasks, 100, mode="shuffled_n2", n2_oversample=2.0, seed=0)
    assert len(schedule) == 100
    n2 = sum(1 for item in schedule if item["constraints"]["n_movies"] >= 2)
    assert n2 > 40


def test_shaped_reward_credits_n2_partial_hits():
    from tinywatch_grpo.training.grpo.reward_shaping import shaped_grpo_reward

    task = {
        "gold_watchlist": {"movie_ids": ["g1", "g2"]},
        "constraints": {"n_movies": 2},
    }
    one_hit = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 0.1, "reward_type": "partial"},
            "watchlist": {"movie_ids": ["g1"]},
        },
        "steps": [
            {"tool_name": "open", "parameters": {"movie_id": "g1"}},
            {"tool_name": "add_to_watchlist", "parameters": {"movie_id": "g1"}},
        ],
    }
    zero_hit = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 0.1, "reward_type": "partial"},
            "watchlist": {"movie_ids": ["x"]},
        },
        "steps": [{"tool_name": "open", "parameters": {"movie_id": "x"}}],
    }
    assert shaped_grpo_reward(one_hit, task) > shaped_grpo_reward(zero_hit, task)


def test_n2_verified_staircase_beats_unverified_wrong_list():
    from tinywatch_grpo.training.grpo.reward_shaping import (
        shaped_grpo_reward,
        verified_gold_hit_fraction,
    )

    task = {
        "gold_watchlist": {"movie_ids": ["g1", "g2"]},
        "constraints": {"n_movies": 2, "directors": ["A"]},
    }
    verified_one = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": -0.8, "reward_type": "wrong_watchlist"},
            "watchlist": {"movie_ids": ["g1", "x"]},
        },
        "steps": [
            {"tool_name": "open", "parameters": {"movie_id": "g1"}},
            {"tool_name": "view_crew", "parameters": {"movie_id": "g1"}},
            {"tool_name": "add_to_watchlist", "parameters": {"movie_id": "g1"}},
        ],
    }
    unverified_wrong = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": -0.8, "reward_type": "wrong_watchlist"},
            "watchlist": {"movie_ids": ["x", "y"]},
        },
        "steps": [{"tool_name": "add_to_watchlist", "parameters": {"movie_id": "x"}}],
    }
    assert verified_gold_hit_fraction(verified_one, task) == 0.5
    assert verified_gold_hit_fraction(unverified_wrong, task) == 0.0
    assert shaped_grpo_reward(verified_one, task) > shaped_grpo_reward(unverified_wrong, task)
    gold = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 1.0, "reward_type": "gold_watchlist"},
            "watchlist": {"movie_ids": ["g1", "g2"]},
        },
        "steps": [],
    }
    assert shaped_grpo_reward(gold, task) == 1.0
    alt = {
        "terminal_result": {
            "reward_detail": {"reward_valid": True, "reward": 0.6, "reward_type": "valid_alternative"},
            "watchlist": {"movie_ids": ["g1", "x"]},
        },
        "steps": [
            {"tool_name": "open", "parameters": {"movie_id": "g1"}},
            {"tool_name": "view_crew", "parameters": {"movie_id": "g1"}},
        ],
    }
    score = shaped_grpo_reward(alt, task)
    assert 0.60 <= score <= 0.80


def test_n2_policy_rollout_count_reserves_oracle_slot():
    from tinywatch_grpo.training.grpo.loop import n2_policy_rollout_count

    assert n2_policy_rollout_count(4, 2, oracle_mix=True) == 3
    assert n2_policy_rollout_count(4, 1, oracle_mix=True) == 4
    assert n2_policy_rollout_count(4, 2, oracle_mix=False) == 4


def test_window_rates_ignore_oracle_and_prefer_verified_hits():
    from tinywatch_grpo.training.grpo.trainer import _select_checkpoint, _window_rates

    history = [
        {
            "n_movies": 2,
            "optimizer_step": True,
            "mean_reward": 0.2,
            "collapsed_count": 0,
            "reward_types": ["wrong_watchlist", "wrong_watchlist", "partial", "gold_watchlist"],
            "oracle_flags": [False, False, False, True],
            "verified_hit_fracs": [0.5, 0.0, 0.5, 1.0],
        }
    ]
    stats = _window_rates(history, 0, 1)
    assert stats["n2_gold_rate"] == 0.0
    assert abs(stats["n2_verified_hit_rate"] - (1.0 / 3.0)) < 1e-8
    assert stats["select_score"] == stats["n2_verified_hit_rate"]
    low = {
        "step": 25,
        "select_score": 0.1,
        "n2_gold_rate": 0.0,
        "n1_gold_rate": 0.8,
        "n2_null_rate": 0.1,
        "null_rate": 0.1,
    }
    high = {
        "step": 50,
        "select_score": 0.3,
        "n2_gold_rate": 0.0,
        "n1_gold_rate": 0.6,
        "n2_null_rate": 0.2,
        "null_rate": 0.2,
    }
    assert _select_checkpoint([low, high])["step"] == 50