from tinywatch_grpo.smoke import run_cpu_smoke


def test_cpu_smoke():
    result = run_cpu_smoke()
    assert result["checks"] == [
        "action_schema",
        "oracle_gold",
        "reward_sample",
        "sft_label_mask",
        "grpo_grouping",
    ]
