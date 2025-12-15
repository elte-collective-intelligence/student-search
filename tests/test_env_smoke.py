import numpy as np
import torch


def test_env_reset_and_step_smoke(make_env):
    env = make_env(
        num_missing=2,
        num_rescuers=2,
        num_trees=1,
        num_safe_zones=2,
        max_cycles=5,
        seed=0,
    )

    td = env.reset()

    # Basic keys and shapes
    assert "observation" in td
    assert td["observation"].dtype == torch.float32
    assert td["observation"].shape == env.observation_spec["observation"].shape

    # Observation should be finite numbers (1e6 sentinel is fine, it's finite)
    assert torch.isfinite(td["observation"]).all()

    done = False
    steps = 0

    while not done and steps < 20:
        # valid discrete action in [0..4]
        action = torch.tensor(np.random.randint(0, 5), dtype=torch.int64)
        td["action"] = action

        td_next = env.step(td)

        # Required outputs exist and are well-typed
        assert ("next", "reward") in td_next.keys(True)
        assert ("next", "done") in td_next.keys(True)
        assert ("next", "terminated") in td_next.keys(True)
        assert ("next", "truncated") in td_next.keys(True)

        assert td_next["next", "reward"].shape == torch.Size([1])
        assert td_next["next", "done"].shape == torch.Size([1])

        # Step returns finite observation too
        assert torch.isfinite(td_next["next", "observation"]).all().item()

        done = bool(td_next["next", "done"].item())
        td = td_next
        steps += 1

    # Must finish by truncation within max_cycles (or earlier if all victims saved)
    assert done is True
    assert steps <= env.max_cycles
