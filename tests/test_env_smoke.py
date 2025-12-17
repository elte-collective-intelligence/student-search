import numpy as np
import torch


def test_env_reset_and_step_smoke(make_env):
    """Test basic environment functionality with CTDE setup."""
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
    # Single-agent format for TorchRL PPO compatibility
    assert td["observation"].shape == env.observation_spec["observation"].shape

    # Global state for CTDE critic
    assert "state" in td
    assert td["state"].dtype == torch.float32
    assert td["state"].shape == env.observation_spec["state"].shape

    # All observations available for multi-agent access
    assert "all_observations" in td
    assert td["all_observations"].shape[0] == env.num_rescuers

    # Observation should be finite numbers
    assert torch.isfinite(td["observation"]).all()
    assert torch.isfinite(td["state"]).all()

    done = False
    steps = 0

    while not done and steps < 20:
        # Single action (policy is shared, environment handles multi-agent internally)
        action = torch.tensor(np.random.randint(0, 5), dtype=torch.int64)
        td["action"] = action

        td_next = env.step(td)

        # Required outputs exist and are well-typed
        assert ("next", "reward") in td_next.keys(True)
        assert ("next", "done") in td_next.keys(True)
        assert ("next", "terminated") in td_next.keys(True)
        assert ("next", "truncated") in td_next.keys(True)

        # Shared team reward (cooperative MARL)
        assert td_next["next", "reward"].shape == torch.Size([1])
        assert td_next["next", "done"].shape == torch.Size([1])

        # Step returns finite observation
        assert torch.isfinite(td_next["next", "observation"]).all().item()

        done = bool(td_next["next", "done"].item())
        td = td_next
        steps += 1

    # Must finish by truncation within max_cycles (or earlier if all victims saved)
    assert done is True
    assert steps <= env.max_cycles


def test_ctde_global_state(make_env):
    """Test that global state contains full information for CTDE."""
    env = make_env(
        num_missing=3,
        num_rescuers=3,
        num_trees=2,
        num_safe_zones=3,
        seed=42,
    )

    td = env.reset()

    # Global state should exist
    assert "state" in td

    # Global state size calculation:
    # - All rescuer positions: num_rescuers * 2
    # - All rescuer velocities: num_rescuers * 2
    # - All victim positions: num_missing * 2
    # - All victim states: num_missing * 1
    # - All tree positions: num_trees * 2
    # - All safe zone positions: num_safe_zones * 2
    expected_size = (
        env.num_rescuers * 2  # positions
        + env.num_rescuers * 2  # velocities
        + env.num_missing * 2  # victim positions
        + env.num_missing * 1  # victim states
        + env.num_trees * 2  # tree positions
        + env.num_safe_zones * 2  # safe zone positions
    )

    assert (
        td["state"].shape[-1] == expected_size
    ), f"Expected global state size {expected_size}, got {td['state'].shape[-1]}"


def test_ctde_architecture(make_env):
    """Test that CTDE architecture is correctly set up."""
    from src.rl.models import make_mappo_models

    env = make_env(
        num_missing=2,
        num_rescuers=2,
        num_trees=1,
        num_safe_zones=2,
        seed=0,
    )

    actor, critic = make_mappo_models(env, device="cpu")

    # Actor should use "observation" (local, decentralized)
    assert "observation" in actor.in_keys

    # Critic should use "state" (global, centralized)
    assert "state" in critic.in_keys

    env.close()
