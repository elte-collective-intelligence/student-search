import numpy as np


def test_reset_initializes_state_correctly(make_env):
    """Test that reset properly initializes all environment state."""
    env = make_env(
        num_rescuers=2,
        num_victims=3,
        num_trees=4,
        num_safe_zones=4,
        seed=42,
    )

    obs, info = env.reset()

    # Check that steps counter is reset
    assert env.steps == 0, "Steps should be reset to 0"

    # Check that agents list is properly initialized
    assert len(env.agents) == 2, "Should have 2 agents"
    assert env.agents == env.possible_agents, "Agents should match possible_agents"

    # Check positions are initialized
    assert env.rescuer_pos.shape == (2, 2), "Rescuer positions should be (2, 2)"
    assert env.victim_pos.shape == (3, 2), "Victim positions should be (3, 2)"
    assert env.tree_pos.shape == (4, 2), "Tree positions should be (4, 2)"
    assert env.safezone_pos.shape == (4, 2), "Safe zone positions should be (4, 2)"

    # Check positions are within bounds
    assert np.all(env.rescuer_pos >= -0.8) and np.all(
        env.rescuer_pos <= 0.8
    ), "Rescuer positions should be in [-0.8, 0.8]"
    assert np.all(env.victim_pos >= -0.8) and np.all(
        env.victim_pos <= 0.8
    ), "Victim positions should be in [-0.8, 0.8]"
    assert np.all(env.tree_pos >= -0.8) and np.all(
        env.tree_pos <= 0.8
    ), "Tree positions should be in [-0.8, 0.8]"

    # Check velocities are zeroed
    assert np.allclose(env.rescuer_vel, 0), "Rescuer velocities should be zero"
    assert np.allclose(env.victim_vel, 0), "Victim velocities should be zero"

    # Check victim states are reset
    assert np.all(~env.victim_saved), "All victims should be unsaved"
    assert np.all(env.victim_assignments == -1), "All victims should be unassigned"

    # Check observations are returned for all agents
    assert len(obs) == 2, "Should return observations for 2 agents"
    for agent in env.agents:
        assert agent in obs, f"Observation should contain {agent}"
        assert isinstance(obs[agent], np.ndarray), "Observation should be numpy array"

    # Check info structure
    assert len(info) == 2, "Should return info for 2 agents"
    for agent in env.agents:
        assert agent in info, f"Info should contain {agent}"


def test_reset_with_seed_reproducibility(make_env):
    """Test that reset with same seed produces same initial state."""
    seed = 123

    env1 = make_env(num_rescuers=2, num_victims=2, num_trees=3, seed=seed)
    obs1, _ = env1.reset(seed=seed)

    env2 = make_env(num_rescuers=2, num_victims=2, num_trees=3, seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Check positions are identical
    assert np.allclose(
        env1.rescuer_pos, env2.rescuer_pos
    ), "Rescuer positions should match"
    assert np.allclose(
        env1.victim_pos, env2.victim_pos
    ), "Victim positions should match"
    assert np.allclose(env1.tree_pos, env2.tree_pos), "Tree positions should match"

    # Check observations are identical
    for agent in env1.agents:
        assert np.allclose(
            obs1[agent], obs2[agent]
        ), f"Observations for {agent} should match"


def test_reset_multiple_times(make_env):
    """Test that reset can be called multiple times correctly."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=1, seed=42)

    # First reset
    obs1, _ = env.reset(seed=100)
    pos1 = env.rescuer_pos[0].copy()

    # Modify state
    env.rescuer_pos[0] = np.array([0.5, 0.5])
    env.victim_saved[0] = True
    env.steps = 50

    # Second reset
    obs2, _ = env.reset(seed=200)
    pos2 = env.rescuer_pos[0].copy()

    # State should be reset
    assert env.steps == 0, "Steps should be reset"
    assert np.all(~env.victim_saved), "Victims should be unsaved"
    assert pos1 is not None and pos2 is not None, "Positions should be set"
    # Positions should likely be different (different seed)
    assert not np.allclose(
        pos1, pos2
    ), "Different seeds should produce different positions"


def test_reset_safe_zones_fixed_positions(make_env):
    """Test that safe zones are always at fixed corner positions."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)

    obs, _ = env.reset()

    expected_safe_zones = np.array([[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]])

    assert np.allclose(
        env.safezone_pos, expected_safe_zones
    ), "Safe zones should be at fixed corner positions"


def test_reset_prev_distances_initialized(make_env):
    """Test that previous distances are initialized after reset."""
    env = make_env(num_rescuers=2, num_victims=2, num_trees=0, seed=42)

    obs, _ = env.reset()

    assert hasattr(
        env, "prev_agent_victim_dists"
    ), "Should have prev_agent_victim_dists"
    assert len(env.prev_agent_victim_dists) == 2, "Should have distance for each agent"
    assert all(
        d >= 0 for d in env.prev_agent_victim_dists
    ), "All distances should be non-negative"


def test_step_increments_counter(make_env):
    """Test that step increments the step counter."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    assert env.steps == 0, "Initial steps should be 0"

    actions = {env.agents[0]: np.array([0.0, 0.0])}
    env.step(actions)

    assert env.steps == 1, "Steps should increment to 1"

    env.step(actions)
    assert env.steps == 2, "Steps should increment to 2"


def test_step_applies_actions(make_env):
    """Test that step correctly applies actions and updates positions."""
    env = make_env(num_rescuers=1, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    initial_pos = env.rescuer_pos[0].copy()
    initial_vel = env.rescuer_vel[0].copy()

    # Apply action
    action = np.array([0.1, 0.1])
    actions = {env.agents[0]: action}
    env.step(actions)

    # Velocity should be updated (with damping)
    expected_vel = initial_vel * 0.8 + action * 0.1
    assert np.allclose(
        env.rescuer_vel[0], expected_vel, atol=1e-5
    ), "Velocity should be updated according to physics"

    # Position should be updated
    expected_pos = initial_pos + env.rescuer_vel[0]
    assert np.allclose(
        env.rescuer_pos[0], expected_pos, atol=1e-5
    ), "Position should be updated based on velocity"


def test_step_velocity_clamping(make_env):
    """Test that velocity is clamped to max speed."""
    env = make_env(num_rescuers=1, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Apply very large action
    action = np.array([10.0, 10.0])
    actions = {env.agents[0]: action}
    env.step(actions)

    speed = np.linalg.norm(env.rescuer_vel[0])
    assert speed <= 0.08, f"Speed should be clamped to 0.08, got {speed}"


def test_step_wall_collision(make_env):
    """Test that wall collisions are handled correctly."""
    env = make_env(num_rescuers=1, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Place agent near boundary with velocity toward wall
    env.rescuer_pos[0] = np.array([0.95, 0.0])
    env.rescuer_vel[0] = np.array([0.1, 0.0])

    actions = {env.agents[0]: np.array([0.0, 0.0])}
    env.step(actions)

    # Position should be clamped
    assert env.rescuer_pos[0][0] <= 1.0, "Position should be clamped to 1.0"
    # Velocity should be reflected and damped
    assert env.rescuer_vel[0][0] < 0, "Velocity should be reflected"


def test_step_tree_collision_penalty(make_env):
    """Test that tree collisions apply penalty and handle physics."""
    env = make_env(num_rescuers=1, num_victims=0, num_trees=1, seed=42)
    obs, _ = env.reset()

    # Store agent name before step (agents list may be emptied on termination)
    agent_name = env.agents[0]

    # Place agent very close to tree
    env.rescuer_pos[0] = np.array([0.0, 0.0])
    env.tree_pos[0] = np.array([0.05, 0.0])

    actions = {agent_name: np.array([0.0, 0.0])}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Should receive collision penalty
    assert rewards[agent_name] < 0, "Should receive negative reward for collision"
    # Position should be adjusted to avoid interpenetration
    dist = np.linalg.norm(env.rescuer_pos[0] - env.tree_pos[0])
    min_dist = env.agent_size + env.tree_radius
    assert dist >= min_dist - 1e-5, "Agent should not penetrate tree"


def test_step_agent_repulsion(make_env):
    """Test that agent-agent repulsion works correctly."""
    env = make_env(num_rescuers=2, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Place agents very close together
    env.rescuer_pos[0] = np.array([0.0, 0.0])
    env.rescuer_pos[1] = np.array([0.05, 0.0])
    initial_dist = np.linalg.norm(env.rescuer_pos[0] - env.rescuer_pos[1])

    actions = {agent: np.array([0.0, 0.0]) for agent in env.agents}
    env.step(actions)

    # Agents should be pushed apart (or at least not get closer)
    final_dist = np.linalg.norm(env.rescuer_pos[0] - env.rescuer_pos[1])
    # Repulsion should increase distance or keep it stable
    assert (
        final_dist >= initial_dist - 1e-5
    ), "Agents should not get closer due to repulsion"


def test_step_victim_assignment(make_env):
    """Test that victims assign to nearby agents."""
    env = make_env(num_rescuers=2, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Place agent close to victim
    env.rescuer_pos[0] = np.array([0.0, 0.0])
    env.rescuer_pos[1] = np.array([0.5, 0.5])
    env.victim_pos[0] = np.array([0.15, 0.0])  # Within follow_radius (0.2) of agent 0

    actions = {agent: np.array([0.0, 0.0]) for agent in env.agents}
    env.step(actions)

    # Victim should be assigned to closest agent
    assert env.victim_assignments[0] == 0, "Victim should be assigned to agent 0"


def test_step_victim_follows_assigned_agent(make_env):
    """Test that assigned victims follow their assigned agent."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Place agent and victim close together
    env.rescuer_pos[0] = np.array([0.0, 0.0])
    env.victim_pos[0] = np.array([0.15, 0.0])
    initial_dist = np.linalg.norm(env.rescuer_pos[0] - env.victim_pos[0])

    # Move agent away
    actions = {env.agents[0]: np.array([0.05, 0.0])}
    env.step(actions)

    # Victim should move toward agent (distance should decrease or stay similar)
    final_dist = np.linalg.norm(env.rescuer_pos[0] - env.victim_pos[0])
    # Victim should follow, so distance shouldn't increase much
    assert final_dist <= initial_dist + 0.1, "Victim should follow assigned agent"


def test_step_victim_saved_at_safe_zone(make_env):
    """Test that victims are saved when reaching correct safe zone."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Store agent name before step (agents list may be emptied on termination)
    agent_name = env.agents[0]

    # Place victim at matching safe zone
    victim_type = env.victim_types[0]
    target_zone = env.safezone_pos[victim_type]
    env.victim_pos[0] = target_zone + np.array([0.1, 0.0])  # Just outside

    # Move victim into safe zone
    env.victim_pos[0] = target_zone  # At safe zone center

    actions = {agent_name: np.array([0.0, 0.0])}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Victim should be saved
    assert env.victim_saved[0], "Victim should be saved when at safe zone"
    # Agent should receive reward
    assert rewards[agent_name] > 0, "Agent should receive reward for saving victim"


def test_step_termination_all_victims_saved(make_env):
    """Test that episode terminates when all victims are saved."""
    env = make_env(num_rescuers=1, num_victims=2, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Save first victim
    victim0_type = env.victim_types[0]
    env.victim_pos[0] = env.safezone_pos[victim0_type]
    env.victim_saved[0] = True

    # Save second victim
    victim1_type = env.victim_types[1]
    env.victim_pos[1] = env.safezone_pos[victim1_type]

    actions = {env.agents[0]: np.array([0.0, 0.0])}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Should terminate
    assert all(terminations.values()), "Should terminate when all victims saved"
    assert len(env.agents) == 0, "Agents list should be empty on termination"


def test_step_truncation_max_steps(make_env):
    """Test that episode truncates after max_steps."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, max_cycles=5, seed=42)
    obs, _ = env.reset()

    actions = {env.agents[0]: np.array([0.0, 0.0])}

    truncations = {a: False for a in env.agents}

    # Step until max_steps
    for _ in range(5):
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if len(env.agents) == 0:
            break

    # Should truncate
    assert all(truncations.values()), "Should truncate after max_steps"
    assert env.steps >= env.max_steps, "Steps should be at least max_steps"


def test_step_rewards_structure(make_env):
    """Test that rewards are returned correctly for all agents."""
    env = make_env(num_rescuers=2, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Store agent names before step (agents list may be emptied on termination)
    agent_names = env.agents.copy()

    actions = {agent: np.array([0.0, 0.0]) for agent in agent_names}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Check rewards structure
    assert len(rewards) == len(agent_names), "Should return reward for each agent"
    for agent in agent_names:
        assert agent in rewards, f"Rewards should contain {agent}"
        assert isinstance(
            rewards[agent], (float, np.floating)
        ), "Reward should be float"


def test_step_observations_after_step(make_env):
    """Test that observations are returned correctly after step."""
    env = make_env(num_rescuers=2, num_victims=1, num_trees=1, seed=42)
    obs, _ = env.reset()

    # Store agent names before step (agents list may be emptied on termination)
    agent_names = env.agents.copy()

    actions = {agent: np.array([0.0, 0.0]) for agent in agent_names}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Check observations structure
    assert len(obs) == len(agent_names), "Should return observation for each agent"
    for agent in agent_names:
        assert agent in obs, f"Observations should contain {agent}"
        assert isinstance(obs[agent], np.ndarray), "Observation should be numpy array"
        assert obs[agent].shape == (
            env.obs_dim,
        ), f"Observation shape should match obs_dim ({env.obs_dim})"


def test_step_boundary_penalty(make_env):
    """Test that agents near boundaries receive penalty."""
    env = make_env(num_rescuers=1, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Store agent name before step (agents list may be emptied on termination)
    agent_name = env.agents[0]

    # Place agent near boundary
    env.rescuer_pos[0] = np.array([0.96, 0.0])

    actions = {agent_name: np.array([0.0, 0.0])}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Should receive boundary penalty
    assert rewards[agent_name] < 0, "Should receive penalty for being near boundary"


def test_step_agent_collision_penalty(make_env):
    """Test that agent collisions apply penalty."""
    env = make_env(num_rescuers=2, num_victims=0, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Store agent names before step (agents list may be emptied on termination)
    agent_names = env.agents.copy()

    # Place agents overlapping
    env.rescuer_pos[0] = np.array([0.0, 0.0])
    env.rescuer_pos[1] = np.array([0.01, 0.0])  # Very close

    actions = {agent: np.array([0.0, 0.0]) for agent in agent_names}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Both agents should receive collision penalty
    assert rewards[agent_names[0]] < 0, "Agent 0 should receive collision penalty"
    assert rewards[agent_names[1]] < 0, "Agent 1 should receive collision penalty"


def test_step_victim_brownian_motion(make_env):
    """Test that unassigned victims move with Brownian motion."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Place victim far from agent (not assigned)
    env.rescuer_pos[0] = np.array([0.5, 0.5])
    env.victim_pos[0] = np.array([-0.5, -0.5])
    initial_pos = env.victim_pos[0].copy()

    actions = {env.agents[0]: np.array([0.0, 0.0])}
    env.step(actions)

    # Victim should move (Brownian motion)
    final_pos = env.victim_pos[0]
    # Position should change (with some randomness)
    assert not np.allclose(
        initial_pos, final_pos, atol=1e-6
    ), "Unassigned victim should move with Brownian motion"


def test_step_saved_victim_stationary(make_env):
    """Test that saved victims don't move."""
    env = make_env(num_rescuers=1, num_victims=1, num_trees=0, seed=42)
    obs, _ = env.reset()

    # Mark victim as saved
    env.victim_saved[0] = True
    env.victim_vel[0] = np.array([0.1, 0.1])
    initial_pos = env.victim_pos[0].copy()

    actions = {env.agents[0]: np.array([0.0, 0.0])}
    env.step(actions)

    # Victim should not move
    assert np.allclose(env.victim_pos[0], initial_pos), "Saved victim should not move"
    assert np.allclose(env.victim_vel[0], 0), "Saved victim velocity should be zero"
