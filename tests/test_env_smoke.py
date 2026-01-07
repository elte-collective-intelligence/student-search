"""
Smoke tests for SearchAndRescueEnv core functionality.

Tests cover:
- Reset behavior and state initialization
- Step mechanics and physics
- Collision handling
- Victim dynamics and saving
- Reward structure
"""

import numpy as np

from conftest import noop_action, noop_actions, place_agent, place_victim, place_tree
from src.seed_utils import set_seed


# =============================================================================
# Reset Tests
# =============================================================================


class TestReset:
    """Tests for environment reset functionality."""

    def test_initializes_state_correctly(self, make_env):
        """Reset properly initializes all environment state."""
        env = make_env(
            num_rescuers=2,
            num_victims=3,
            num_trees=4,
            num_safe_zones=4,
        )

        obs, info = env.reset()

        # Verify counters and agent list
        assert env.steps == 0
        assert len(env.agents) == 2
        assert env.agents == env.possible_agents

        # Verify position shapes
        assert env.rescuer_pos.shape == (2, 2)
        assert env.victim_pos.shape == (3, 2)
        assert env.tree_pos.shape == (4, 2)
        assert env.safezone_pos.shape == (4, 2)

        # Verify positions are within spawn bounds
        assert np.all(env.rescuer_pos >= -0.8) and np.all(env.rescuer_pos <= 0.8)
        assert np.all(env.victim_pos >= -0.8) and np.all(env.victim_pos <= 0.8)
        assert np.all(env.tree_pos >= -0.8) and np.all(env.tree_pos <= 0.8)

        # Verify velocities are zeroed
        assert np.allclose(env.rescuer_vel, 0)
        assert np.allclose(env.victim_vel, 0)

        # Verify victim states are reset
        assert np.all(~env.victim_saved)
        assert np.all(env.victim_assignments == -1)

        # Verify observations returned for all agents
        assert len(obs) == 2
        for agent in env.agents:
            assert agent in obs
            assert isinstance(obs[agent], np.ndarray)

        # Verify info structure
        assert len(info) == 2
        for agent in env.agents:
            assert agent in info

    def test_seed_reproducibility(self, make_env):
        """Same seed produces identical initial state."""
        seed = 123
        set_seed(seed)

        env1 = make_env(num_rescuers=2, num_victims=2, num_trees=3)
        obs1, _ = env1.reset(seed=seed)

        env2 = make_env(num_rescuers=2, num_victims=2, num_trees=3)
        obs2, _ = env2.reset(seed=seed)

        assert np.allclose(env1.rescuer_pos, env2.rescuer_pos)
        assert np.allclose(env1.victim_pos, env2.victim_pos)
        assert np.allclose(env1.tree_pos, env2.tree_pos)

        for agent in env1.agents:
            assert np.allclose(obs1[agent], obs2[agent])

    def test_multiple_resets(self, make_env):
        """Reset can be called multiple times correctly."""
        set_seed(42)

        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=1,
        )

        env.reset(seed=100)
        pos1 = env.rescuer_pos[0].copy()

        # Modify state
        env.rescuer_pos[0] = np.array([0.5, 0.5])
        env.victim_saved[0] = True
        env.steps = 50

        env.reset(seed=200)
        pos2 = env.rescuer_pos[0].copy()

        # State should be reset
        assert env.steps == 0
        assert np.all(~env.victim_saved)
        # Different seeds should produce different positions
        assert not np.allclose(pos1, pos2)

    def test_safe_zones_fixed_positions(self, make_env):
        """Safe zones are at fixed corner positions by default."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        expected = np.array([[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]])
        assert np.allclose(env.safezone_pos, expected)

    def test_prev_distances_initialized(self, make_env):
        """Previous distances are initialized after reset."""
        env = make_env(
            num_rescuers=2,
            num_victims=2,
            num_trees=0,
        )
        env.reset()

        assert hasattr(env, "prev_agent_victim_dists")
        assert len(env.prev_agent_victim_dists) == 2
        assert all(d >= 0 for d in env.prev_agent_victim_dists)


# =============================================================================
# Step Mechanics Tests
# =============================================================================


class TestStepMechanics:
    """Tests for basic step mechanics."""

    def test_increments_counter(self, make_env):
        """Step increments the step counter."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        assert env.steps == 0

        env.step(noop_actions(env))
        assert env.steps == 1

        env.step(noop_actions(env))
        assert env.steps == 2

    def test_applies_actions(self, make_env):
        """Step correctly applies actions and updates positions."""
        env = make_env(
            num_rescuers=1,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        initial_pos = env.rescuer_pos[0].copy()
        initial_vel = env.rescuer_vel[0].copy()

        action = np.array([0.1, 0.1])
        actions = {env.agents[0]: action}
        env.step(actions)

        # Velocity should be updated (with damping)
        expected_vel = initial_vel * 0.8 + action * 0.1
        assert np.allclose(env.rescuer_vel[0], expected_vel, atol=1e-5)

        # Position should be updated
        expected_pos = initial_pos + env.rescuer_vel[0]
        assert np.allclose(env.rescuer_pos[0], expected_pos, atol=1e-5)

    def test_velocity_clamping(self, make_env):
        """Velocity is clamped to max speed."""
        env = make_env(
            num_rescuers=1,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        # Apply very large action
        actions = {env.agents[0]: np.array([10.0, 10.0])}
        env.step(actions)

        speed = np.linalg.norm(env.rescuer_vel[0])
        assert speed <= 0.08

    def test_observations_after_step(self, make_env):
        """Observations are returned correctly after step."""
        env = make_env(
            num_rescuers=2,
            num_victims=1,
            num_trees=1,
        )
        env.reset()

        agent_names = env.agents.copy()
        obs, _, _, _, _ = env.step(noop_actions(env))

        assert len(obs) == len(agent_names)
        for agent in agent_names:
            assert agent in obs
            assert isinstance(obs[agent], np.ndarray)
            assert obs[agent].shape == (env.obs_dim,)

    def test_rewards_structure(self, make_env):
        """Rewards are returned correctly for all agents."""
        env = make_env(
            num_rescuers=2,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        agent_names = env.agents.copy()
        _, rewards, _, _, _ = env.step(noop_actions(env))

        assert len(rewards) == len(agent_names)
        for agent in agent_names:
            assert agent in rewards
            assert isinstance(rewards[agent], (float, np.floating))


# =============================================================================
# Collision Tests
# =============================================================================


class TestCollisions:
    """Tests for collision handling."""

    def test_wall_collision(self, make_env):
        """Wall collisions are handled correctly."""
        env = make_env(
            num_rescuers=1,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        # Place agent near boundary with velocity toward wall
        place_agent(env, 0, (0.95, 0.0))
        env.rescuer_vel[0] = np.array([0.1, 0.0])

        env.step(noop_actions(env))

        # Position should be clamped
        assert env.rescuer_pos[0][0] <= 1.0
        # Velocity should be reflected
        assert env.rescuer_vel[0][0] < 0

    def test_tree_collision_penalty(self, make_env):
        """Tree collisions apply penalty and handle physics."""
        env = make_env(
            num_rescuers=1,
            num_victims=0,
            num_trees=1,
        )
        env.reset()

        agent_name = env.agents[0]
        place_agent(env, 0, (0.0, 0.0))
        place_tree(env, 0, (0.05, 0.0))

        actions = {agent_name: noop_action()}
        _, rewards, _, _, _ = env.step(actions)

        # Should receive collision penalty
        assert rewards[agent_name] < 0
        # Position should be adjusted
        dist = np.linalg.norm(env.rescuer_pos[0] - env.tree_pos[0])
        min_dist = env.agent_size + env.tree_radius
        assert dist >= min_dist - 1e-5

    def test_agent_repulsion(self, make_env):
        """Agent-agent repulsion works correctly."""
        env = make_env(
            num_rescuers=2,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_agent(env, 1, (0.05, 0.0))
        initial_dist = np.linalg.norm(env.rescuer_pos[0] - env.rescuer_pos[1])

        env.step(noop_actions(env))

        final_dist = np.linalg.norm(env.rescuer_pos[0] - env.rescuer_pos[1])
        # Repulsion should increase distance or keep it stable
        assert final_dist >= initial_dist - 1e-5

    def test_agent_collision_penalty(self, make_env):
        """Agent collisions apply penalty."""
        env = make_env(
            num_rescuers=2,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        agent_names = env.agents.copy()
        place_agent(env, 0, (0.0, 0.0))
        place_agent(env, 1, (0.01, 0.0))

        _, rewards, _, _, _ = env.step(noop_actions(env))

        # Both agents should receive collision penalty
        assert rewards[agent_names[0]] < 0
        assert rewards[agent_names[1]] < 0

    def test_boundary_penalty(self, make_env):
        """Agents near boundaries receive penalty.

        The boundary penalty is applied when |pos| > 0.95 after physics update.
        The inward push mechanism (at margin=0.85) reduces velocity toward the wall,
        so we need to position the agent such that it ends up > 0.95 after the step.

        Physics order:
        1. vel = vel * 0.8 + action * 0.1
        2. speed clamp to max 0.08
        3. inward push if |pos| > 0.85: adjust vel by ±0.02
        4. pos += vel
        5. wall collision clamp to [-1, 1]
        """
        env = make_env(
            num_rescuers=1,
            num_victims=0,
            num_trees=0,
        )
        env.reset()

        agent_name = env.agents[0]
        # Start at 0.94 with velocity 0.08 towards wall.
        # After step: vel = 0.08*0.8 = 0.064, inward push: vel -= 0.02 = 0.044
        # pos = 0.94 + 0.044 = 0.984 > 0.95, so penalty applies!
        place_agent(env, 0, (0.94, 0.0))
        env.rescuer_vel[0] = np.array([0.08, 0.0])

        _, rewards, _, _, _ = env.step({agent_name: noop_action()})

        # Verify agent is past threshold and received penalty
        assert (
            env.rescuer_pos[0][0] > 0.95
        ), f"Agent should be past 0.95, got {env.rescuer_pos[0][0]}"
        assert (
            rewards[agent_name] < 0
        ), f"Expected negative reward, got {rewards[agent_name]}"


# =============================================================================
# Victim Dynamics Tests
# =============================================================================


class TestVictimDynamics:
    """Tests for victim behavior."""

    def test_victim_assignment(self, make_env):
        """Victims assign to nearby agents."""
        env = make_env(
            num_rescuers=2,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_agent(env, 1, (0.5, 0.5))
        place_victim(env, 0, (0.15, 0.0))  # Within follow_radius (0.2) of agent 0

        env.step(noop_actions(env))

        assert env.victim_assignments[0] == 0

    def test_victim_follows_assigned_agent(self, make_env):
        """Assigned victims follow their assigned agent."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        place_agent(env, 0, (0.0, 0.0))
        place_victim(env, 0, (0.15, 0.0))
        initial_dist = np.linalg.norm(env.rescuer_pos[0] - env.victim_pos[0])

        # Move agent away
        actions = {env.agents[0]: np.array([0.05, 0.0])}
        env.step(actions)

        final_dist = np.linalg.norm(env.rescuer_pos[0] - env.victim_pos[0])
        # Victim should follow, so distance shouldn't increase much
        assert final_dist <= initial_dist + 0.1

    def test_victim_brownian_motion(self, make_env):
        """Unassigned victims move with Brownian motion."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        # Place victim far from agent (not assigned)
        place_agent(env, 0, (0.5, 0.5))
        place_victim(env, 0, (-0.5, -0.5))
        initial_pos = env.victim_pos[0].copy()

        env.step(noop_actions(env))

        # Position should change (Brownian motion)
        assert not np.allclose(env.victim_pos[0], initial_pos, atol=1e-6)

    def test_saved_victim_stationary(self, make_env):
        """Saved victims don't move."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        env.victim_saved[0] = True
        env.victim_vel[0] = np.array([0.1, 0.1])
        initial_pos = env.victim_pos[0].copy()

        env.step(noop_actions(env))

        assert np.allclose(env.victim_pos[0], initial_pos)
        assert np.allclose(env.victim_vel[0], 0)


# =============================================================================
# Saving and Termination Tests
# =============================================================================


class TestSavingAndTermination:
    """Tests for victim saving and episode termination."""

    def test_victim_saved_at_safe_zone(self, make_env):
        """Victims are saved when reaching correct safe zone."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
        )
        env.reset()

        agent_name = env.agents[0]
        victim_type = env.victim_types[0]
        target_zone = env.safezone_pos[victim_type]

        # Place agent near the safe zone so victim gets assigned
        place_agent(env, 0, target_zone.copy())

        # Place victim at safe zone center (close to agent)
        env.victim_pos[0] = target_zone.copy()

        # Manually assign victim to agent (or ensure they're close enough)
        env.victim_assignments[0] = 0

        _, rewards, _, _, _ = env.step({agent_name: noop_action()})

        assert env.victim_saved[0]
        assert rewards[agent_name] > 0

    def test_termination_all_victims_saved(self, make_env):
        """Episode terminates when all victims are saved."""
        env = make_env(
            num_rescuers=1,
            num_victims=2,
            num_trees=0,
        )
        env.reset()

        # Save first victim manually
        env.victim_saved[0] = True
        env.victim_pos[0] = env.safezone_pos[env.victim_types[0]]

        # Place second victim at its safe zone
        env.victim_pos[1] = env.safezone_pos[env.victim_types[1]]

        _, _, terminations, _, _ = env.step(noop_actions(env))

        assert all(terminations.values())
        assert len(env.agents) == 0

    def test_truncation_max_steps(self, make_env):
        """Episode truncates after max_steps."""
        env = make_env(
            num_rescuers=1,
            num_victims=1,
            num_trees=0,
            max_cycles=5,
        )
        env.reset()

        # Step until max_steps
        truncations = {}
        for _ in range(5):
            _, _, _, truncations, _ = env.step(noop_actions(env))
            if len(env.agents) == 0:
                break

        assert all(truncations.values())
        assert env.steps >= env.max_steps


# =============================================================================
# Additional Functionality Tests
# =============================================================================


class TestDiscreteActions:
    """Discrete action mode tests (Task 3 requirement)."""

    def test_discrete_action_space_defined(self, make_env):
        """Discrete action space has 5 actions."""
        env = make_env(continuous_actions=False)
        env.reset()
        assert env.action_spaces[env.agents[0]].n == 5

    def test_discrete_action_mapping(self, make_env):
        """Discrete actions map correctly to movements."""
        env = make_env(
            continuous_actions=False, num_rescuers=1, num_victims=0, num_trees=0
        )
        env.reset()

        # Test up action (1)
        env.rescuer_pos[0] = np.array([0.0, 0.0])
        env.rescuer_vel[0] = np.array([0.0, 0.0])
        env.step({env.agents[0]: 1})  # up
        assert env.rescuer_vel[0][1] > 0  # positive y velocity


class TestCurriculumIntegration:
    """Curriculum learning integration (Task 4 requirement)."""

    def test_set_num_trees_updates_on_reset(self, make_env):
        """set_num_trees changes tree count after reset."""
        env = make_env(num_trees=2, max_trees=10)
        env.reset()
        assert env.num_trees == 2

        env.set_num_trees(5)
        env.reset()
        assert env.num_trees == 5


class TestMetricsTracking:
    """Metrics collection tests (Task 5 requirement)."""

    def test_pop_episode_metrics_returns_data(self, make_env):
        """Metrics are collected when episode ends."""
        env = make_env(num_rescuers=1, num_victims=1, max_cycles=3)
        env.reset()

        for _ in range(5):
            if not env.agents:
                break
            env.step(noop_actions(env))

        metrics = env.pop_episode_metrics()
        assert len(metrics) >= 1
        assert "rescues_pct" in metrics[0]
        assert "collisions" in metrics[0]
        assert "coverage_cells" in metrics[0]


class TestDomainRandomization:
    """Domain randomization tests (Task 4 requirement)."""

    def test_randomize_safe_zones(self, make_env):
        """Safe zones randomize when flag is set."""
        env = make_env(randomize_safe_zones=True)

        positions = []
        for seed in [1, 2, 3]:
            env.reset(seed=seed)
            positions.append(env.safezone_pos.copy())

        # Not all should be identical
        assert not all(np.allclose(positions[0], p) for p in positions[1:])
