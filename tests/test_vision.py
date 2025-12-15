import numpy as np


def _other_pos_slice(env):
    start = 2 + 2 + 6
    end = start + (env.num_agents - 1) * 2
    return slice(start, end)


def test_occlusion_blocks_other_agent_in_observation(make_env):
    env = make_env(
        num_missing=0, num_rescuers=2, num_trees=1, num_safe_zones=0, seed=123
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

    obs_blocked = env._get_observation(resc0)
    sl = _other_pos_slice(env)
    first_other = slice(sl.start, sl.start + 2)
    assert np.allclose(obs_blocked[first_other], np.array([1e6, 1e6], dtype=np.float32))

    # Move tree away: should be visible now
    tree.p_pos = np.array([0.0, 0.8])
    assert env._is_blocked_by_obstacle(resc0, resc1) is False

    obs_visible = env._get_observation(resc0)
    # relative position = resc1 - resc0 = [1.0, 0.0]
    assert np.allclose(
        obs_visible[sl], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6
    )
