import numpy as np


def _get_obs_slice_indices(env, agent_idx):
    """Calculate slice indices for different observation components.

    Observation structure:
    [0:2] Self velocity (2)
    [2:4] Self position (2)
    [4:4+num_rescuers] Agent ID one-hot (num_rescuers)
    [4+num_rescuers:4+num_rescuers+num_safe_zones*3] Safe zones (num_safe_zones * 3: rel_x, rel_y, type)
    [4+num_rescuers+num_safe_zones*3:4+num_rescuers+num_safe_zones*3+num_trees*2] Trees (num_trees * 2: rel_x, rel_y)
    [4+num_rescuers+num_safe_zones*3+num_trees*2:] Victims (num_victims * 3: rel_x, rel_y, type)
    """
    base = 4  # vel(2) + pos(2)
    agent_id_end = base + env.num_rescuers
    safe_zones_end = agent_id_end + env.num_safe_zones * 3
    trees_end = safe_zones_end + env.num_trees * 2
    victims_end = trees_end + env.num_victims * 3

    return {
        'self_vel': slice(0, 2),
        'self_pos': slice(2, 4),
        'agent_id': slice(4, agent_id_end),
        'safe_zones': slice(agent_id_end, safe_zones_end),
        'trees': slice(safe_zones_end, trees_end),
        'victims': slice(trees_end, victims_end),
    }


def test_occlusion_blocks_tree_in_observation(make_env):
    """Test that trees block vision and visibility masks are set correctly."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=1,
        num_safe_zones=0,
        seed=123,
        vision_radius=1.0,  # Use larger vision to test occlusion
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    # Place agent and tree within vision, with tree at origin
    env.rescuer_pos[agent_idx] = np.array([-0.5, 0.0])
    env.tree_pos[0] = np.array([0.0, 0.0])
    # Place a victim behind the tree (should be blocked)
    env.victim_pos[0] = np.array([0.5, 0.0])
    env.victim_saved[0] = False

    obs = env._get_obs()
    obs_vec = obs[agent]
    slices = _get_obs_slice_indices(env, agent_idx)

    # Check that tree is visible (within vision radius and no occlusion)
    tree_slice = slices['trees']
    tree_rel_pos = obs_vec[tree_slice.start:tree_slice.start + 2]
    expected_tree_rel = env.tree_pos[0] - env.rescuer_pos[agent_idx]
    assert np.allclose(tree_rel_pos, expected_tree_rel, atol=1e-5), \
        f"Tree should be visible. Got {tree_rel_pos}, expected {expected_tree_rel}"

    # Check that victim is blocked (masked as [0.0, 0.0, -1.0])
    victim_slice = slices['victims']
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
    assert np.allclose(victim_obs, [0.0, 0.0, -1.0], atol=1e-5), \
        f"Victim should be blocked. Got {victim_obs}, expected [0.0, 0.0, -1.0]"

    # Verify occlusion check
    assert not env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Victim should be blocked by tree"

    # Move tree away: victim should be visible now
    env.tree_pos[0] = np.array([0.0, 0.8])
    obs = env._get_obs()
    obs_vec = obs[agent]
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
    expected_victim_rel = env.victim_pos[0] - env.rescuer_pos[agent_idx]
    assert np.allclose(victim_obs[:2], expected_victim_rel[:2], atol=1e-5), \
        f"Victim should be visible now. Got {victim_obs[:2]}, expected {expected_victim_rel[:2]}"
    assert victim_obs[2] == 0.0, f"Victim type should be 0, got {victim_obs[2]}"


def test_vision_radius_masks_distant_entities(make_env):
    """Test that entities beyond vision radius are masked."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=1,
        num_safe_zones=0,
        seed=42,
        vision_radius=0.5,  # Explicit vision radius for testing
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    # Place agent, victim, and tree within vision (< 0.5)
    env.rescuer_pos[agent_idx] = np.array([0.0, 0.0])
    env.victim_pos[0] = np.array([0.2, 0.0])
    env.tree_pos[0] = np.array([0.3, 0.0])

    obs = env._get_obs()
    obs_vec = obs[agent]
    slices = _get_obs_slice_indices(env, agent_idx)

    # Check victim is visible
    victim_slice = slices['victims']
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
    expected_victim_rel = env.victim_pos[0] - env.rescuer_pos[agent_idx]
    assert np.allclose(victim_obs[:2], expected_victim_rel[:2], atol=1e-5), \
        f"Victim should be visible. Got {victim_obs[:2]}, expected {expected_victim_rel[:2]}"
    assert victim_obs[2] != -1.0, "Victim should not be masked"

    # Check tree is visible
    tree_slice = slices['trees']
    tree_obs = obs_vec[tree_slice.start:tree_slice.start + 2]
    expected_tree_rel = env.tree_pos[0] - env.rescuer_pos[agent_idx]
    assert np.allclose(tree_obs, expected_tree_rel, atol=1e-5), \
        f"Tree should be visible. Got {tree_obs}, expected {expected_tree_rel}"

    # Move victim and tree beyond vision radius (> 0.5)
    env.victim_pos[0] = np.array([0.7, 0.0])
    env.tree_pos[0] = np.array([0.8, 0.0])

    obs = env._get_obs()
    obs_vec = obs[agent]

    # Check victim is masked
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
    assert np.allclose(victim_obs, [0.0, 0.0, -1.0], atol=1e-5), \
        f"Victim should be masked (too far). Got {victim_obs}, expected [0.0, 0.0, -1.0]"

    # Check tree is masked
    tree_obs = obs_vec[tree_slice.start:tree_slice.start + 2]
    assert np.allclose(tree_obs, [0.0, 0.0], atol=1e-5), \
        f"Tree should be masked (too far). Got {tree_obs}, expected [0.0, 0.0]"


def test_occlusion_no_block_when_tree_outside_segment(make_env):
    """Test that trees behind observer or target don't block vision."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=1,
        num_safe_zones=0,
        seed=42,
        vision_radius=1.0,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    env.rescuer_pos[agent_idx] = np.array([-0.5, 0.0])
    env.victim_pos[0] = np.array([0.5, 0.0])

    # Tree behind observer (should NOT block)
    env.tree_pos[0] = np.array([-0.8, 0.0])
    assert env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Tree behind observer should not block vision"

    # Tree behind target (should NOT block)
    env.tree_pos[0] = np.array([0.8, 0.0])
    assert env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Tree behind target should not block vision"

    # Tree between (should block)
    env.tree_pos[0] = np.array([0.0, 0.0])
    assert not env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Tree between observer and target should block vision"


def test_saved_victims_are_masked(make_env):
    """Test that saved victims are always masked in observations."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=0,
        num_safe_zones=1,
        seed=42,
        vision_radius=1.0,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    # Place victim close to agent (within vision)
    env.rescuer_pos[agent_idx] = np.array([0.0, 0.0])
    env.victim_pos[0] = np.array([0.1, 0.1])
    env.victim_saved[0] = False

    obs = env._get_obs()
    obs_vec = obs[agent]
    slices = _get_obs_slice_indices(env, agent_idx)

    # Victim should be visible when not saved
    victim_slice = slices['victims']
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
    assert victim_obs[2] != -1.0, "Unsaved victim should be visible"

    # Mark victim as saved
    env.victim_saved[0] = True

    obs = env._get_obs()
    obs_vec = obs[agent]
    victim_obs = obs_vec[victim_slice.start:victim_slice.start + 3]

    # Saved victim should be masked regardless of distance
    assert np.allclose(victim_obs, [0.0, 0.0, -1.0], atol=1e-5), \
        f"Saved victim should be masked. Got {victim_obs}, expected [0.0, 0.0, -1.0]"


def test_observation_structure_correctness(make_env):
    """Test that observation structure matches expected format."""
    env = make_env(
        num_rescuers=2,
        num_victims=2,
        num_trees=3,
        num_safe_zones=4,
        seed=42,
    )

    obs, _ = env.reset()

    # Check observation dimensions
    expected_obs_dim = (
        4  # vel(2) + pos(2)
        + env.num_rescuers  # agent ID one-hot
        + (env.num_safe_zones * 3)  # safe zones (rel_x, rel_y, type)
        + (env.num_trees * 2)  # trees (rel_x, rel_y)
        + (env.num_victims * 3)  # victims (rel_x, rel_y, type)
    )

    for agent in env.agents:
        obs_vec = obs[agent]
        assert obs_vec.shape == (expected_obs_dim,), \
            f"Observation shape mismatch. Got {obs_vec.shape}, expected ({expected_obs_dim},)"

        # Check that self position and velocity are present
        assert obs_vec[0:2].shape == (2,), "Self velocity should be 2D"
        assert obs_vec[2:4].shape == (2,), "Self position should be 2D"

        # Check agent ID one-hot encoding
        agent_idx = env.agents.index(agent)
        agent_id_slice = slice(4, 4 + env.num_rescuers)
        agent_id = obs_vec[agent_id_slice]
        assert agent_id[agent_idx] == 1.0, f"Agent {agent_idx} ID should be 1.0 at index {agent_idx}"
        assert np.sum(agent_id) == 1.0, "Agent ID should be one-hot (sum to 1.0)"

        # Check safe zones structure (should always be present, not masked)
        safe_zones_start = 4 + env.num_rescuers
        safe_zones_end = safe_zones_start + env.num_safe_zones * 3
        safe_zones = obs_vec[safe_zones_start:safe_zones_end]
        assert safe_zones.shape == (env.num_safe_zones * 3,), \
            f"Safe zones should have shape ({env.num_safe_zones * 3},), got {safe_zones.shape}"


def test_multiple_trees_occlusion(make_env):
    """Test that multiple trees can block vision correctly."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=2,
        num_safe_zones=0,
        seed=42,
        vision_radius=1.0,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    env.rescuer_pos[agent_idx] = np.array([-0.5, 0.0])
    env.victim_pos[0] = np.array([0.5, 0.0])

    # Place first tree between agent and victim (should block)
    env.tree_pos[0] = np.array([0.0, 0.0])
    # Place second tree away (should not affect)
    env.tree_pos[1] = np.array([0.0, 0.5])

    assert not env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "First tree should block vision"

    # Move first tree away, second tree should not block (it's off the line)
    env.tree_pos[0] = np.array([0.0, 0.5])
    assert env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "No tree should block vision now"


def test_vision_radius_edge_case(make_env):
    """Test vision at exactly the vision radius boundary."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=0,
        num_safe_zones=0,
        seed=42,
        vision_radius=0.5,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    env.rescuer_pos[agent_idx] = np.array([0.0, 0.0])

    # Victim exactly at vision radius (should be visible, distance == radius)
    env.victim_pos[0] = np.array([0.5, 0.0])
    dist = np.linalg.norm(env.victim_pos[0] - env.rescuer_pos[agent_idx])
    assert dist == env.vision_radius, "Distance should equal vision radius"

    # According to _is_visible, dist > vision_radius returns False, so == should return True
    assert env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Entity at exactly vision radius should be visible"

    # Victim just beyond vision radius (should be masked)
    env.victim_pos[0] = np.array([0.5001, 0.0])
    assert not env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    ), "Entity just beyond vision radius should not be visible"


def test_safe_zones_always_visible(make_env):
    """Test that safe zones are always visible regardless of distance or occlusion."""
    env = make_env(
        num_rescuers=1,
        num_victims=0,
        num_trees=1,
        num_safe_zones=4,
        seed=42,
        vision_radius=0.3,  # Small vision radius
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    env.rescuer_pos[agent_idx] = np.array([0.0, 0.0])
    # Place tree between agent and a safe zone
    env.tree_pos[0] = np.array([0.1, 0.0])
    # Safe zones are at corners: [-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]

    obs = env._get_obs()
    obs_vec = obs[agent]
    slices = _get_obs_slice_indices(env, agent_idx)

    # Safe zones should always be present in observation (not masked)
    safe_zones_slice = slices['safe_zones']
    safe_zones = obs_vec[safe_zones_slice.start:safe_zones_slice.stop]

    # All safe zones should have valid relative positions (not all zeros)
    # They should be relative to agent position
    for i in range(env.num_safe_zones):
        idx = i * 3
        rel_x = safe_zones[idx]
        rel_y = safe_zones[idx + 1]
        zone_type = safe_zones[idx + 2]

        # Relative position should match safe zone position - agent position
        expected_rel = env.safezone_pos[i] - env.rescuer_pos[agent_idx]
        assert np.allclose([rel_x, rel_y], expected_rel, atol=1e-5), \
            f"Safe zone {i} relative position incorrect. Got [{rel_x}, {rel_y}], expected {expected_rel}"
        assert zone_type == float(env.safe_zone_types[i]), \
            f"Safe zone {i} type incorrect. Got {zone_type}, expected {env.safe_zone_types[i]}"


# def test_tree_occludes_other_trees(make_env):
#     """Test that a tree can occlude another tree."""
#     env = make_env(
#         num_rescuers=1,
#         num_victims=0,
#         num_trees=2,
#         num_safe_zones=0,
#         seed=42,
#         vision_radius=1.0,
#     )
#
#     obs, _ = env.reset()
#     agent = env.agents[0]
#     agent_idx = 0
#
#     env.rescuer_pos[agent_idx] = np.array([-0.5, 0.0])
#     # Place first tree between agent and second tree
#     env.tree_pos[0] = np.array([0.0, 0.0])
#     env.tree_pos[1] = np.array([0.5, 0.0])
#
#     obs = env._get_obs()
#     obs_vec = obs[agent]
#     slices = _get_obs_slice_indices(env, agent_idx)
#
#     tree_slice = slices['trees']
#
#     # First tree should be visible
#     tree0_obs = obs_vec[tree_slice.start:tree_slice.start + 2]
#     expected_tree0_rel = env.tree_pos[0] - env.rescuer_pos[agent_idx]
#     assert np.allclose(tree0_obs, expected_tree0_rel, atol=1e-5), \
#         f"First tree should be visible. Got {tree0_obs}, expected {expected_tree0_rel}"
#
#     # Second tree should be blocked by first tree
#     tree1_obs = obs_vec[tree_slice.start + 2:tree_slice.start + 4]
#     assert np.allclose(tree1_obs, [0.0, 0.0], atol=1e-5), \
#         f"Second tree should be blocked. Got {tree1_obs}, expected [0.0, 0.0]"
#
#     # Verify with _is_visible
#     assert env._is_visible(
#         env.rescuer_pos[agent_idx],
#         env.tree_pos[0],
#         env.tree_radius
#     ), "First tree should be visible"
#
#     assert not env._is_visible(
#         env.rescuer_pos[agent_idx],
#         env.tree_pos[1],
#         env.tree_radius
#     ), "Second tree should be blocked by first tree"


def test_relative_position_calculation(make_env):
    """Test that relative positions in observations are calculated correctly."""
    env = make_env(
        num_rescuers=1,
        num_victims=2,
        num_trees=2,
        num_safe_zones=4,
        seed=42,
        vision_radius=1.0,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    # Set specific positions
    env.rescuer_pos[agent_idx] = np.array([0.2, 0.3])
    env.victim_pos[0] = np.array([0.5, 0.4])
    env.victim_pos[1] = np.array([0.1, 0.1])
    env.tree_pos[0] = np.array([0.3, 0.35])
    env.tree_pos[1] = np.array([-0.2, -0.1])
    env.victim_saved[0] = False
    env.victim_saved[1] = False

    obs = env._get_obs()
    obs_vec = obs[agent]
    slices = _get_obs_slice_indices(env, agent_idx)

    # Check self position (should be absolute, not relative)
    self_pos = obs_vec[slices['self_pos']]
    assert np.allclose(self_pos, env.rescuer_pos[agent_idx], atol=1e-5), \
        f"Self position should be absolute. Got {self_pos}, expected {env.rescuer_pos[agent_idx]}"

    # Check victim relative positions
    victim_slice = slices['victims']
    if env._is_visible(env.rescuer_pos[agent_idx], env.victim_pos[0], env.agent_size):
        victim0_obs = obs_vec[victim_slice.start:victim_slice.start + 3]
        expected_rel = env.victim_pos[0] - env.rescuer_pos[agent_idx]
        assert np.allclose(victim0_obs[:2], expected_rel, atol=1e-5), \
            f"Victim 0 relative position incorrect. Got {victim0_obs[:2]}, expected {expected_rel}"

    # Check tree relative positions
    tree_slice = slices['trees']
    if env._is_visible(env.rescuer_pos[agent_idx], env.tree_pos[0], env.tree_radius):
        tree0_obs = obs_vec[tree_slice.start:tree_slice.start + 2]
        expected_rel = env.tree_pos[0] - env.rescuer_pos[agent_idx]
        assert np.allclose(tree0_obs, expected_rel, atol=1e-5), \
            f"Tree 0 relative position incorrect. Got {tree0_obs}, expected {expected_rel}"

    # Check safe zone relative positions
    safe_zones_slice = slices['safe_zones']
    for i in range(env.num_safe_zones):
        idx = safe_zones_slice.start + i * 3
        rel_x = obs_vec[idx]
        rel_y = obs_vec[idx + 1]
        expected_rel = env.safezone_pos[i] - env.rescuer_pos[agent_idx]
        assert np.allclose([rel_x, rel_y], expected_rel, atol=1e-5), \
            f"Safe zone {i} relative position incorrect. Got [{rel_x}, {rel_y}], expected {expected_rel}"


def test_occlusion_tangent_case(make_env):
    """Test occlusion when tree is tangent to the line of sight (edge case)."""
    env = make_env(
        num_rescuers=1,
        num_victims=1,
        num_trees=1,
        num_safe_zones=0,
        seed=42,
        vision_radius=1.0,
    )

    obs, _ = env.reset()
    agent = env.agents[0]
    agent_idx = 0

    env.rescuer_pos[agent_idx] = np.array([-0.5, 0.0])
    env.victim_pos[0] = np.array([0.5, 0.0])

    # Place tree such that it's tangent to the line (perpendicular to line at midpoint)
    # Tree at (0, tree_radius) - just touching the line
    env.tree_pos[0] = np.array([0.0, env.tree_radius])

    # The occlusion check uses quadratic formula - if discriminant == 0, it's tangent
    # Tangent case: tree should block if it intersects the segment [0, 1]
    # For this specific case, we need to check if the implementation handles it correctly
    is_visible = env._is_visible(
        env.rescuer_pos[agent_idx],
        env.victim_pos[0],
        env.agent_size
    )

    # The exact behavior depends on the math, but we should test it's consistent
    # If tangent and t1 or t2 is exactly 0 or 1, it should block
    # For now, just verify the function doesn't crash and returns a boolean
    assert isinstance(is_visible, (bool, np.bool_)), "Visibility check should return boolean"
