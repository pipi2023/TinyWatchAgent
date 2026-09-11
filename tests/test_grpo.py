from tinywatch_grpo.training.grpo.loop import group_advantages, grpo_policy_loss, select_reward_varying_groups


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
