import numpy as np


def _other_pos_slice(env):
    """Get slice for other rescuer positions in observation vector."""
    # Observation format: [vel(2), landmarks(6), other_rescuers, victims]
    # No absolute position anymore (removed for proper partial observability)
    start = 2 + 6  # vel(2) + landmarks(6)
    end = start + (env.num_agents - 1) * 2
    return slice(start, end)


def test_occlusion_blocks_other_agent_in_observation(make_env):
    """Test that trees block vision and visibility masks are set correctly."""
    env = make_env(
        num_missing=0,
        num_rescuers=2,
        num_trees=1,
        num_safe_zones=0,
        seed=123,
        vision_radius=1.0,  # Use larger vision to test occlusion
    )

    resc0, resc1 = env.rescuers[0], env.rescuers[1]
    tree = env.trees[0]

    # Put agents within vision, and the tree directly between them.
    resc0.p_pos = np.array([-0.5, 0.0])
    resc1.p_pos = np.array([0.5, 0.0])
    tree.p_pos = np.array([0.0, 0.0])
    tree.size = 0.2  # make sure intersection is guaranteed

    assert env._get_distance(resc0, resc1) <= env.vision
    assert env._is_blocked_by_obstacle(resc0, resc1) is True

    obs_blocked, vis_mask_rescuers, vis_mask_victims = env._get_observation(resc0)
    sl = _other_pos_slice(env)
    first_other = slice(sl.start, sl.start + 2)

    # When blocked, position should be zeros (not sentinel values) and mask should be 0
    assert np.allclose(obs_blocked[first_other], np.array([0.0, 0.0], dtype=np.float32))
    assert vis_mask_rescuers[0] == 0.0  # Not visible

    # Move tree away: should be visible now
    tree.p_pos = np.array([0.0, 0.8])
    assert env._is_blocked_by_obstacle(resc0, resc1) is False

    obs_visible, vis_mask_rescuers, vis_mask_victims = env._get_observation(resc0)
    # relative position = resc1 - resc0 = [1.0, 0.0]
    assert np.allclose(
        obs_visible[sl], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6
    )
    assert vis_mask_rescuers[0] == 1.0  # Now visible


def test_vision_radius_masks_distant_entities(make_env):
    """Test that entities beyond vision radius are masked."""
    env = make_env(
        num_missing=1,
        num_rescuers=2,
        num_trees=0,
        num_safe_zones=0,
        seed=42,
        vision_radius=0.5,  # Explicit vision radius for testing
    )

    resc0, resc1 = env.rescuers[0], env.rescuers[1]
    victim = env.victims[0]

    # Place resc1 and victim within vision (< 0.5)
    resc0.p_pos = np.array([0.0, 0.0])
    resc1.p_pos = np.array([0.3, 0.0])
    victim.p_pos = np.array([0.2, 0.0])

    obs, vis_rescuers, vis_victims = env._get_observation(resc0)
    assert vis_rescuers[0] == 1.0  # resc1 visible
    assert vis_victims[0] == 1.0  # victim visible

    # Move resc1 beyond vision radius (> 0.5)
    resc1.p_pos = np.array([0.7, 0.0])

    obs, vis_rescuers, vis_victims = env._get_observation(resc0)
    assert vis_rescuers[0] == 0.0  # resc1 not visible (too far)
    assert vis_victims[0] == 1.0  # victim still visible


def test_occlusion_no_block_when_tree_outside_segment(make_env):
    """Test that trees behind observer or target don't block vision."""
    env = make_env(
        num_missing=0, num_rescuers=2, num_trees=1, num_safe_zones=0, seed=42
    )

    resc0, resc1 = env.rescuers[0], env.rescuers[1]
    tree = env.trees[0]

    resc0.p_pos = np.array([-0.5, 0.0])
    resc1.p_pos = np.array([0.5, 0.0])
    tree.size = 0.1

    # Tree behind observer (should NOT block)
    tree.p_pos = np.array([-0.8, 0.0])
    assert env._is_blocked_by_obstacle(resc0, resc1) is False

    # Tree behind target (should NOT block)
    tree.p_pos = np.array([0.8, 0.0])
    assert env._is_blocked_by_obstacle(resc0, resc1) is False

    # Tree between (should block)
    tree.p_pos = np.array([0.0, 0.0])
    assert env._is_blocked_by_obstacle(resc0, resc1) is True
