"""
Vision and occlusion tests for SearchAndRescueEnv.

Tests cover:
- Basic visibility and masking
- Tree occlusion mechanics
- Vision radius boundaries
- Observation structure and relative positions
- Edge cases and special scenarios
"""

import numpy as np

from conftest import (
    get_obs_slices,
    get_tree_obs,
    get_victim_obs,
    get_safe_zone_obs,
    is_masked_victim,
    is_masked_tree,
    place_agent,
    place_victim,
    place_tree,
    check_visibility,
    assert_visible,
    assert_not_visible,
    assert_obs_matches,
)


# =============================================================================
# Basic Occlusion Tests
# =============================================================================


class TestBasicOcclusion:
    """Tests for basic occlusion behavior."""

    def test_tree_blocks_victim(self, make_simple_env):
        """Trees block vision to entities behind them."""
        env = make_simple_env(seed=123)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (-0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        env.victim_saved[0] = False

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Tree should be visible
        tree_obs = get_tree_obs(obs_vec, slices, 0)
        expected_tree_rel = env.tree_pos[0] - env.rescuer_pos[0]
        assert_obs_matches(tree_obs, expected_tree_rel, msg="Tree should be visible")

        # Victim should be blocked (masked)
        victim_obs = get_victim_obs(obs_vec, slices, 0)
        assert is_masked_victim(victim_obs), "Victim should be blocked by tree"

        # Verify with _is_visible
        assert_not_visible(env, env.rescuer_pos[0], env.victim_pos[0], env.agent_size)

        # Move tree away: victim should be visible
        place_tree(env, 0, (0.0, 0.8))
        obs = env._get_obs()
        obs_vec = obs[agent]

        victim_obs = get_victim_obs(obs_vec, slices, 0)
        expected_rel = env.victim_pos[0] - env.rescuer_pos[0]
        assert_obs_matches(
            victim_obs[:2], expected_rel[:2], msg="Victim should be visible now"
        )
        assert victim_obs[2] == 0.0

    def test_tree_behind_observer_no_block(self, env_simple):
        """Trees behind observer don't block vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (-0.8, 0.0))

        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree behind observer should not block",
        )

    def test_tree_behind_target_no_block(self, env_simple):
        """Trees behind target don't block vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.8, 0.0))

        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree behind target should not block",
        )

    def test_tree_between_blocks(self, env_simple):
        """Trees between observer and target block vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree between should block",
        )


# =============================================================================
# Vision Radius Tests
# =============================================================================


class TestVisionRadius:
    """Tests for vision radius behavior."""

    def test_entities_beyond_radius_masked(self, make_simple_env):
        """Entities beyond vision radius are masked."""
        env = make_simple_env(vision_radius=0.5)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.2, 0.0))
        place_tree(env, 0, (0.3, 0.0))

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Within radius - should be visible
        victim_obs = get_victim_obs(obs_vec, slices, 0)
        assert not is_masked_victim(victim_obs), "Victim should be visible"

        tree_obs = get_tree_obs(obs_vec, slices, 0)
        assert not is_masked_tree(tree_obs), "Tree should be visible"

        # Move beyond radius
        place_victim(env, 0, (0.7, 0.0))
        place_tree(env, 0, (0.8, 0.0))

        obs = env._get_obs()
        obs_vec = obs[agent]

        victim_obs = get_victim_obs(obs_vec, slices, 0)
        assert is_masked_victim(victim_obs), "Victim should be masked (too far)"

        tree_obs = get_tree_obs(obs_vec, slices, 0)
        assert is_masked_tree(tree_obs), "Tree should be masked (too far)"

    def test_exactly_at_radius_visible(self, make_simple_env):
        """Entity exactly at vision radius is visible."""
        env = make_simple_env(vision_radius=0.5, num_trees=0)
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.5, 0.0))

        dist = np.linalg.norm(env.victim_pos[0] - env.rescuer_pos[0])
        assert dist == env.vision_radius

        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Entity at exactly vision radius should be visible",
        )

    def test_just_beyond_radius_masked(self, make_simple_env):
        """Entity just beyond vision radius is masked."""
        env = make_simple_env(vision_radius=0.5, num_trees=0)
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.5001, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Entity just beyond radius should not be visible",
        )

    def test_zero_radius(self, make_simple_env):
        """Zero vision radius makes everything invisible."""
        env = make_simple_env(vision_radius=0.0, num_trees=0)
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.0, 0.0))  # Same position

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Zero radius should make everything invisible",
        )

    def test_large_radius_still_blocks(self, make_simple_env):
        """Large vision radius doesn't bypass occlusion."""
        env = make_simple_env(vision_radius=10.0)
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree should block even with large radius",
        )


# =============================================================================
# Multiple Trees Tests
# =============================================================================


class TestMultipleTrees:
    """Tests for multiple tree occlusion scenarios."""

    def test_multiple_trees_occlusion(self, make_simple_env):
        """Multiple trees can block vision correctly."""
        env = make_simple_env(num_trees=2)
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))  # Blocks
        place_tree(env, 1, (0.0, 0.5))  # Doesn't affect

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="First tree should block",
        )

        # Move first tree away
        place_tree(env, 0, (0.0, 0.5))

        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="No tree should block now",
        )

    def test_tree_chain_blocks(self, make_simple_env):
        """Chain of trees along line of sight blocks vision."""
        env = make_simple_env(num_trees=3)
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (-0.2, 0.0))
        place_tree(env, 1, (0.0, 0.0))
        place_tree(env, 2, (0.2, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Chain of trees should block",
        )

    def test_tree_occludes_other_tree(self, make_simple_env):
        """A tree can occlude another tree."""
        env = make_simple_env(num_victims=0, num_trees=2)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (-0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))
        place_tree(env, 1, (0.5, 0.0))

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # First tree visible
        tree0_obs = get_tree_obs(obs_vec, slices, 0)
        expected = env.tree_pos[0] - env.rescuer_pos[0]
        assert_obs_matches(tree0_obs, expected, msg="First tree should be visible")

        # Second tree blocked
        tree1_obs = get_tree_obs(obs_vec, slices, 1)
        assert is_masked_tree(tree1_obs), "Second tree should be blocked"

        # Verify with _is_visible
        assert_visible(
            env,
            env.rescuer_pos[0],
            env.tree_pos[0],
            env.tree_radius,
            exclude_tree_idx=0,
            msg="First tree visible (excludes itself)",
        )
        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.tree_pos[1],
            env.tree_radius,
            msg="Second tree blocked by first",
        )


# =============================================================================
# Victim Masking Tests
# =============================================================================


class TestVictimMasking:
    """Tests for victim visibility and masking."""

    def test_saved_victims_masked(self, make_simple_env):
        """Saved victims are always masked in observations."""
        env = make_simple_env(num_trees=0, num_safe_zones=4)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.1, 0.1))
        env.victim_saved[0] = False

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        victim_obs = get_victim_obs(obs_vec, slices, 0)
        assert not is_masked_victim(victim_obs), "Unsaved victim should be visible"

        # Mark as saved
        env.victim_saved[0] = True

        obs = env._get_obs()
        obs_vec = obs[agent]
        victim_obs = get_victim_obs(obs_vec, slices, 0)

        assert is_masked_victim(victim_obs), "Saved victim should be masked"


# =============================================================================
# Safe Zone Tests
# =============================================================================


class TestSafeZones:
    """Tests for safe zone observation behavior."""

    def test_safe_zones_visibility_depends_on_distance(self, make_simple_env):
        """Safe zones are masked based on distance and occlusion."""
        env = make_simple_env(num_victims=0, num_trees=1, vision_radius=0.3)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.0, 0.0))
        place_tree(env, 0, (0.1, 0.0))

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Check each safe zone
        for i in range(env.num_safe_zones):
            zone_obs = get_safe_zone_obs(obs_vec, slices, i)
            dist = np.linalg.norm(env.safezone_pos[i] - env.rescuer_pos[0])

            if dist <= env.vision_radius and check_visibility(
                env, env.rescuer_pos[0], env.safezone_pos[i], env.safe_zone_radius
            ):
                expected_rel = env.safezone_pos[i] - env.rescuer_pos[0]
                assert_obs_matches(
                    zone_obs[:2],
                    expected_rel,
                    msg=f"Safe zone {i} should have correct relative position",
                )
                assert zone_obs[2] == float(env.safe_zone_types[i])


# =============================================================================
# Observation Structure Tests
# =============================================================================


class TestObservationStructure:
    """Tests for observation structure correctness."""

    def test_observation_dimensions(self, make_env):
        """Observation structure matches expected format."""
        env = make_env(
            num_rescuers=2,
            num_victims=2,
            num_trees=3,
            num_safe_zones=4,
            seed=42,
        )
        obs, _ = env.reset()

        expected_obs_dim = (
            4  # vel(2) + pos(2)
            + env.num_rescuers  # agent ID one-hot
            + (env.num_safe_zones * 3)  # safe zones
            + (env.num_trees * 2)  # trees
            + (env.num_victims * 3)  # victims
        )

        for agent in env.agents:
            obs_vec = obs[agent]
            assert obs_vec.shape == (expected_obs_dim,)

            # Check self position and velocity
            assert obs_vec[0:2].shape == (2,)
            assert obs_vec[2:4].shape == (2,)

            # Check agent ID one-hot
            agent_idx = env.agents.index(agent)
            slices = get_obs_slices(env)
            agent_id = obs_vec[slices["agent_id"]]
            assert agent_id[agent_idx] == 1.0
            assert np.sum(agent_id) == 1.0

    def test_relative_positions_correct(self, make_simple_env):
        """Relative positions in observations are calculated correctly."""
        env = make_simple_env(num_victims=2, num_trees=2, num_safe_zones=4)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.2, 0.3))
        place_victim(env, 0, (0.5, 0.4))
        place_victim(env, 1, (0.1, 0.1))
        place_tree(env, 0, (0.3, 0.35))
        place_tree(env, 1, (-0.2, -0.1))
        env.victim_saved[0] = False
        env.victim_saved[1] = False

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Self position should be absolute
        self_pos = obs_vec[slices["self_pos"]]
        assert_obs_matches(
            self_pos, env.rescuer_pos[0], msg="Self position should be absolute"
        )

        # Check victim relative positions if visible
        if check_visibility(env, env.rescuer_pos[0], env.victim_pos[0], env.agent_size):
            victim0_obs = get_victim_obs(obs_vec, slices, 0)
            expected = env.victim_pos[0] - env.rescuer_pos[0]
            assert_obs_matches(victim0_obs[:2], expected)

        # Check tree relative positions if visible
        if check_visibility(env, env.rescuer_pos[0], env.tree_pos[0], env.tree_radius):
            tree0_obs = get_tree_obs(obs_vec, slices, 0)
            expected = env.tree_pos[0] - env.rescuer_pos[0]
            assert_obs_matches(tree0_obs, expected)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_tangent_case(self, env_simple):
        """Occlusion when tree is tangent to line of sight."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, env.tree_radius))  # Tangent to line

        is_visible = check_visibility(
            env, env.rescuer_pos[0], env.victim_pos[0], env.agent_size
        )
        assert isinstance(is_visible, (bool, np.bool_))

    def test_partial_overlap(self, env_simple):
        """Tree partially overlapping line of sight blocks vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.02))  # Small offset

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree with partial overlap should block",
        )

    def test_tree_at_observer_position(self, env_simple):
        """Tree at observer position handled gracefully."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))

        is_visible = check_visibility(
            env, env.rescuer_pos[0], env.victim_pos[0], env.agent_size
        )
        assert isinstance(is_visible, (bool, np.bool_))

    def test_tree_at_target_position(self, env_simple):
        """Tree at target position handled gracefully."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.5, 0.0))

        is_visible = check_visibility(
            env, env.rescuer_pos[0], env.victim_pos[0], env.agent_size
        )
        assert isinstance(is_visible, (bool, np.bool_))

    def test_diagonal_line_of_sight(self, env_simple):
        """Occlusion with diagonal line of sight."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.3, -0.3))
        place_victim(env, 0, (0.3, 0.3))

        # Tree on diagonal line
        place_tree(env, 0, (0.0, 0.0))
        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree on diagonal should block",
        )

        # Tree off diagonal line
        place_tree(env, 0, (0.0, 0.2))
        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree off diagonal should not block",
        )

    def test_exclude_tree_parameter(self, make_simple_env):
        """exclude_tree_idx parameter works correctly."""
        env = make_simple_env(num_victims=0, num_trees=2)
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_tree(env, 0, (0.0, 0.0))
        place_tree(env, 1, (0.5, 0.0))

        # Without exclude: tree1 blocked by tree0
        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.tree_pos[1],
            env.tree_radius,
            msg="Tree1 should be blocked by tree0",
        )

        # With exclude_tree_idx=0: tree1 visible
        assert_visible(
            env,
            env.rescuer_pos[0],
            env.tree_pos[1],
            env.tree_radius,
            exclude_tree_idx=0,
            msg="Tree1 visible when tree0 excluded",
        )

    def test_tree_self_visibility(self, make_simple_env):
        """Trees can be visible using exclude_tree_idx for themselves."""
        env = make_simple_env(num_victims=0, num_trees=2)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.0, 0.0))
        place_tree(env, 0, (0.2, 0.0))
        place_tree(env, 1, (0.4, 0.0))

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Tree 0 should be visible (excludes itself)
        tree0_obs = get_tree_obs(obs_vec, slices, 0)
        expected = env.tree_pos[0] - env.rescuer_pos[0]
        assert_obs_matches(tree0_obs, expected, msg="Tree 0 should be visible")

        # Verify with _is_visible
        assert_visible(
            env,
            env.rescuer_pos[0],
            env.tree_pos[0],
            env.tree_radius,
            exclude_tree_idx=0,
            msg="Tree visible when excluding itself",
        )

    def test_very_close_tree_to_observer(self, env_simple):
        """Tree very close to observer blocks vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.01, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree very close to observer should block",
        )

    def test_very_close_tree_to_target(self, env_simple):
        """Tree very close to target blocks vision."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.49, 0.0))

        assert_not_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree very close to target should block",
        )

    def test_parallel_trees_no_block(self, env_simple):
        """Trees parallel to line of sight (offset) don't block."""
        env = env_simple
        env.reset()

        place_agent(env, 0, (-0.5, 0.0))
        place_victim(env, 0, (0.5, 0.0))
        place_tree(env, 0, (0.0, 0.1))  # Offset perpendicular

        # With tree_radius=0.05 and offset 0.1, should be visible
        assert_visible(
            env,
            env.rescuer_pos[0],
            env.victim_pos[0],
            env.agent_size,
            msg="Tree offset from line should not block",
        )

    def test_observation_masking_consistency(self, make_simple_env):
        """Observation masking is consistent with visibility check."""
        env = make_simple_env(num_victims=2, num_trees=2)
        env.reset()
        agent = env.agents[0]

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.3, 0.0))
        place_victim(env, 1, (0.6, 0.0))
        place_tree(env, 0, (0.15, 0.0))  # Blocks victim 0
        place_tree(env, 1, (0.45, 0.0))  # Blocks victim 1
        env.victim_saved[0] = False
        env.victim_saved[1] = False

        obs = env._get_obs()
        obs_vec = obs[agent]
        slices = get_obs_slices(env)

        # Check consistency for each victim
        for v_idx in range(2):
            victim_obs = get_victim_obs(obs_vec, slices, v_idx)
            is_vis = check_visibility(
                env, env.rescuer_pos[0], env.victim_pos[v_idx], env.agent_size
            )

            if is_vis:
                assert not is_masked_victim(
                    victim_obs
                ), f"Visible victim {v_idx} should not be masked"
            else:
                assert is_masked_victim(
                    victim_obs
                ), f"Blocked victim {v_idx} should be masked"
