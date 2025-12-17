# noqa: D212, D415
"""
Search and Rescue Environment using TorchRL.

This environment simulates a search and rescue scenario where rescuers
must guide victims to their matching safe zones while avoiding obstacles (trees).

Victims are environmental entities (not agents) with three states:
- IDLE: Waiting to be found by a rescuer
- FOLLOW: Following the nearest rescuer within range
- STOP: Reached a safe zone and stopped (saved)
"""
from __future__ import annotations

from src.domain.victim import VictimState, Victim
from src.domain.agent import Agent
from src.domain.landmark import Landmark
import numpy as np
import torch
import pygame
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Unbounded, Categorical, Composite
from typing import Optional, Union


class SearchAndRescueEnv(EnvBase):
    """
    TorchRL-based Search and Rescue Environment.

    Rescuers must guide victims to their matching safe zones (same type).
    Trees act as obstacles that block movement and vision.

    Victims are environmental entities with three states:
    - IDLE: Waiting to be found by a rescuer
    - FOLLOW: Following the nearest rescuer within detection range
    - STOP: Reached matching safe zone and saved
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
        "name": "search_and_rescue_v2",
    }

    # Type colors for victims and safe zones
    TYPE_COLORS = {
        "A": np.array([1.0, 0.0, 0.0]),  # Red
        "B": np.array([0.0, 1.0, 0.0]),  # Green
        "C": np.array([0.0, 0.0, 1.0]),  # Blue
        "D": np.array([1.0, 1.0, 0.0]),  # Yellow
    }

    VICTIM_TYPES = ["A", "B", "C", "D"]
    SAFE_ZONE_TYPES = ["A", "B", "C", "D"]

    def __init__(
        self,
        num_missing: int = 4,
        num_rescuers: int = 3,
        num_trees: int = 8,
        num_safe_zones: int = 4,
        max_cycles: int = 120,
        continuous_actions: bool = False,
        render_mode: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        # Vision and behavior parameters
        vision_radius: float = 0.4,  # Agent's vision range (map spans -1 to 1)
        follow_range: float = 0.3,  # Range at which victims start following
        rescue_range: float = 0.1,  # Range at which victims are saved at safe zones
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.num_missing = num_missing
        self.num_rescuers = num_rescuers
        self.num_trees = num_trees
        self.num_safe_zones = num_safe_zones
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.vision = vision_radius  # Vision range for agents

        # Victim behavior parameters
        self.follow_range = follow_range
        self.rescue_range = rescue_range

        # World dimensions
        self.dim_p = 2  # Physical dimension (x, y)
        self.dim_c = 2  # Communication dimension

        # Physics parameters
        self.dt = 0.1
        self.damping = 0.25
        self.contact_force = 1e2
        self.contact_margin = 1e-3

        # Initialize entities
        self._create_entities()

        # Rendering
        self.screen = None
        self.clock = None
        self.width = 700
        self.height = 700

        # Step counter
        self._step_count = 0

        # Initialize random state
        self._np_random = np.random.RandomState()

        # Set seed if provided
        if seed is not None:
            self._set_seed(seed)

        # Define specs
        self._make_specs()

    def _create_entities(self):
        """Create all agents, victims, and landmarks."""
        # Create rescuer agents
        self.agents = []
        for i in range(self.num_rescuers):
            agent = Agent()
            agent.name = f"rescuer_{i}"
            self.agents.append(agent)

        # Create victims (environmental entities, not agents)
        self.victims = []
        for i in range(self.num_missing):
            victim = Victim()
            victim.name = f"victim_{i}"
            victim.type = self.VICTIM_TYPES[i % len(self.VICTIM_TYPES)]
            victim.color = self.TYPE_COLORS[victim.type].copy()
            self.victims.append(victim)

        # Create landmarks (trees + safe zones)
        self.landmarks = []
        for i in range(self.num_trees):
            landmark = Landmark()
            landmark.name = f"tree_{i}"
            landmark.tree = True
            landmark.collide = True
            landmark.size = 0.03
            landmark.color = np.array([0.35, 0.85, 0.35])  # Tree color
            self.landmarks.append(landmark)

        for i in range(self.num_safe_zones):
            landmark = Landmark()
            landmark.name = f"safe_zone_{i}"
            landmark.tree = False
            landmark.collide = False
            landmark.size = 0.1
            landmark.type = self.SAFE_ZONE_TYPES[i % len(self.SAFE_ZONE_TYPES)]
            landmark.color = self.TYPE_COLORS[landmark.type].copy()
            self.landmarks.append(landmark)

        # Agent names for compatibility
        self.possible_agents = [a.name for a in self.agents]
        self.agent_name_mapping = {a.name: i for i, a in enumerate(self.agents)}

    def _make_specs(self):
        """Define observation and action specs."""
        # Observation components (per-agent, partial observability):
        # For decentralized execution, agents do NOT have absolute position.
        # They only observe:
        # - own velocity (2)
        # - closest N landmarks relative positions (N*2) - only if within vision
        # - other rescuers relative positions (num_rescuers-1)*2 - only if visible
        # - victims relative positions + state (num_missing * 3) [x, y, state] - only if visible
        self.n_closest_landmarks = 3
        obs_size = (
            2  # own velocity
            + self.n_closest_landmarks * 2  # closest landmark positions (relative)
            + (self.num_rescuers - 1) * 2  # other rescuer positions (relative)
            + self.num_missing * 3  # victim positions + state (relative)
        )

        # Global state components (for CTDE critic - full observability):
        # - all rescuer positions: num_rescuers * 2
        # - all rescuer velocities: num_rescuers * 2
        # - all victim positions: num_missing * 2
        # - all victim states: num_missing * 1
        # - all tree positions: num_trees * 2
        # - all safe zone positions: num_safe_zones * 2
        self.global_state_size = (
            self.num_rescuers * 2  # all rescuer positions
            + self.num_rescuers * 2  # all rescuer velocities
            + self.num_missing * 2  # all victim positions
            + self.num_missing * 1  # all victim states
            + self.num_trees * 2  # all tree positions
            + self.num_safe_zones * 2  # all safe zone positions
        )

        # Visibility mask sizes
        vis_mask_rescuers_size = max(self.num_rescuers - 1, 0)
        vis_mask_victims_size = self.num_missing

        # Per-agent observation size for individual observations
        self.per_agent_obs_size = obs_size

        # Observation bounds (positions are relative, bounded by ~2x world size)
        obs_low = -3.0
        obs_high = 3.0

        # Global state bounds (absolute positions bounded by world size)
        state_low = -2.0
        state_high = 2.0

        # Observation spec - per-agent observation + shared global state
        # For compatibility with TorchRL PPO, we flatten to single-agent format
        # but the environment internally handles all agents
        self.observation_spec = Composite(
            observation=Bounded(
                low=obs_low,
                high=obs_high,
                shape=(obs_size,),  # Single agent observation
                dtype=torch.float32,
                device=self.device,
            ),
            state=Bounded(
                low=state_low,
                high=state_high,
                shape=(self.global_state_size,),  # Shared global state for CTDE critic
                dtype=torch.float32,
                device=self.device,
            ),
            # Store all agent observations for multi-agent access
            all_observations=Bounded(
                low=obs_low,
                high=obs_high,
                shape=(self.num_rescuers, obs_size),
                dtype=torch.float32,
                device=self.device,
            ),
            vis_mask_rescuers=Bounded(
                low=0.0,
                high=1.0,
                shape=(vis_mask_rescuers_size,),
                dtype=torch.float32,
                device=self.device,
            ),
            vis_mask_victims=Bounded(
                low=0.0,
                high=1.0,
                shape=(vis_mask_victims_size,),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=(),
        )

        # Action spec - single agent action (policy is shared, applied to each agent)
        if self.continuous_actions:
            # Continuous: 2D force vector
            self.action_spec = Bounded(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Discrete: 5 actions (no-op, left, right, down, up)
            self.action_spec = Categorical(
                n=5,
                shape=(),
                dtype=torch.int64,
                device=self.device,
            )

        # Reward spec
        self.reward_spec = Unbounded(
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        # Done spec
        self.done_spec = Composite(
            done=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=Categorical(
                n=2,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )

    def _set_seed(self, seed: Optional[int]):
        """Set random seed."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """Reset the environment."""
        self._step_count = 0
        self._current_agent_idx = 0  # Track which agent we're controlling

        # Initialize reward tracking counters
        self._prev_saved_count = 0
        self._prev_following_count = 0

        # Safe zone corner positions (will be shuffled to randomize type mapping)
        safe_zone_positions = [
            np.array([-1.0, 1.0]),  # Top-left
            np.array([-1.0, -1.0]),  # Bottom-left
            np.array([1.0, 1.0]),  # Top-right
            np.array([1.0, -1.0]),  # Bottom-right
        ]
        # Shuffle positions so type-to-corner mapping is random each episode
        # This prevents agents from learning fixed "type B -> corner X" associations
        self._np_random.shuffle(safe_zone_positions)

        # Reset rescuer agents
        for agent in self.agents:
            agent.p_pos = self._np_random.uniform(-0.3, 0.3, self.dim_p)
            agent.p_vel = np.zeros(self.dim_p)
            agent.c = np.zeros(self.dim_c)

        # Reset victims (scattered around the map)
        for victim in self.victims:
            victim.p_pos = self._np_random.uniform(-0.7, 0.7, self.dim_p)
            victim.p_vel = np.zeros(self.dim_p)
            victim.state = VictimState.IDLE
            victim.following_agent = None
            victim.color = self.TYPE_COLORS[victim.type].copy()

        # Reset landmarks - safe zones get shuffled positions
        safe_zone_idx = 0
        for landmark in self.landmarks:
            if landmark.tree:
                landmark.p_pos = self._np_random.uniform(-0.8, 0.8, self.dim_p)
            else:
                landmark.p_pos = safe_zone_positions[
                    safe_zone_idx % len(safe_zone_positions)
                ].copy()
                safe_zone_idx += 1
            landmark.p_vel = np.zeros(self.dim_p)

        # Get observations for ALL agents
        all_obs = []
        all_vis_mask_rescuers = []
        all_vis_mask_victims = []
        for agent in self.agents:
            obs, vis_mask_rescuers, vis_mask_victims = self._get_observation(agent)
            all_obs.append(obs)
            all_vis_mask_rescuers.append(vis_mask_rescuers)
            all_vis_mask_victims.append(vis_mask_victims)

        # Get global state for CTDE critic
        global_state = self._get_global_state()

        # For TorchRL compatibility, return observation for "current" agent
        # but store all observations for multi-agent access
        main_obs = all_obs[0]
        main_vis_rescuers = all_vis_mask_rescuers[0]
        main_vis_victims = all_vis_mask_victims[0]

        return TensorDict(
            {
                # Main observation (for TorchRL PPO compatibility)
                "observation": torch.tensor(
                    main_obs, dtype=torch.float32, device=self.device
                ),
                # Global state for CTDE critic
                "state": torch.tensor(
                    global_state, dtype=torch.float32, device=self.device
                ),
                # All agent observations (for multi-agent execution)
                "all_observations": torch.tensor(
                    np.stack(all_obs), dtype=torch.float32, device=self.device
                ),
                "vis_mask_rescuers": torch.tensor(
                    main_vis_rescuers, dtype=torch.float32, device=self.device
                ),
                "vis_mask_victims": torch.tensor(
                    main_vis_victims, dtype=torch.float32, device=self.device
                ),
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor(
                    [False], dtype=torch.bool, device=self.device
                ),
                "truncated": torch.tensor(
                    [False], dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one step in the environment.

        MARL with parameter sharing: The same policy is applied to each agent.
        For TorchRL compatibility, we accept a single action and apply the shared
        policy to all agents using their respective observations.
        """
        action = tensordict["action"]

        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # MARL with parameter sharing: Apply the SAME action logic to all agents
        # The policy was trained on individual observations, so we apply it per-agent
        # For now, use the provided action for the "main" agent and sample for others
        # In full MARL training, we'd collect separate transitions for each agent

        # Get all observations for all agents
        all_obs = []
        for agent in self.agents:
            obs, _, _ = self._get_observation(agent)
            all_obs.append(obs)

        # Apply action to main agent (index 0)
        self._set_action(action, self.agents[0])

        # For other agents in training: apply the same policy using their observations
        # This requires the actor to be passed in, but for now we use heuristic actions
        for i, agent in enumerate(self.agents[1:], 1):
            # Heuristic: move towards nearest idle/following victim
            heuristic_action = self._get_heuristic_action(agent)
            self._set_action(heuristic_action, agent)

        # Update victim states and movement
        self._update_victims()

        # Physics step for agents
        self._world_step()

        # Move victims (after agent physics)
        self._move_victims()

        # Check for rescues
        self._check_rescues()

        # Calculate shared team reward (cooperative MARL)
        reward = self._get_team_reward()

        # Get observations for ALL agents
        all_obs = []
        all_vis_mask_rescuers = []
        all_vis_mask_victims = []
        for agent in self.agents:
            obs, vis_mask_rescuers, vis_mask_victims = self._get_observation(agent)
            all_obs.append(obs)
            all_vis_mask_rescuers.append(vis_mask_rescuers)
            all_vis_mask_victims.append(vis_mask_victims)

        # Get global state for CTDE critic
        global_state = self._get_global_state()

        # Increment step counter
        self._step_count += 1

        # Check termination
        all_saved = all(v.state == VictimState.STOP for v in self.victims)
        truncated = self._step_count >= self.max_cycles
        terminated = all_saved
        done = terminated or truncated

        # For TorchRL PPO compatibility, return single-agent format observation
        main_obs = all_obs[0]
        main_vis_rescuers = all_vis_mask_rescuers[0]
        main_vis_victims = all_vis_mask_victims[0]

        return TensorDict(
            {
                # Main observation for PPO (single agent format)
                "observation": torch.tensor(
                    main_obs, dtype=torch.float32, device=self.device
                ),
                # Global state for CTDE critic
                "state": torch.tensor(
                    global_state, dtype=torch.float32, device=self.device
                ),
                # All observations for multi-agent access
                "all_observations": torch.tensor(
                    np.stack(all_obs), dtype=torch.float32, device=self.device
                ),
                "vis_mask_rescuers": torch.tensor(
                    main_vis_rescuers, dtype=torch.float32, device=self.device
                ),
                "vis_mask_victims": torch.tensor(
                    main_vis_victims, dtype=torch.float32, device=self.device
                ),
                # Shared team reward
                "reward": torch.tensor(
                    [reward], dtype=torch.float32, device=self.device
                ),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor(
                    [terminated], dtype=torch.bool, device=self.device
                ),
                "truncated": torch.tensor(
                    [truncated], dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
        )

    def _get_heuristic_action(self, agent) -> int:
        """Get a heuristic action for an agent (used for non-main agents).

        Heuristic:
        1. If a victim is following this agent, lead it to its matching safe zone
        2. Otherwise, move towards the nearest idle victim
        """
        # Check if any victim is following this specific agent
        following_me = None
        for victim in self.victims:
            if victim.state == VictimState.FOLLOW and victim.following is agent:
                following_me = victim
                break

        if following_me is not None:
            # Lead the victim to its matching safe zone
            matching_safe_zone = next(
                (sz for sz in self.safe_zones if sz.type == following_me.type), None
            )
            if matching_safe_zone is not None:
                direction = matching_safe_zone.p_pos - agent.p_pos
            else:
                # No matching safe zone, just stay put
                return 0
        else:
            # Find nearest idle victim to rescue
            nearest_victim = None
            min_dist = float("inf")

            for victim in self.victims:
                if victim.state == VictimState.IDLE:
                    dist = self._get_distance_pos(agent.p_pos, victim.p_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_victim = victim

            if nearest_victim is None:
                # No idle victims, check for following victims not assigned
                for victim in self.victims:
                    if victim.state == VictimState.FOLLOW:
                        dist = self._get_distance_pos(agent.p_pos, victim.p_pos)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_victim = victim

            if nearest_victim is None:
                return 0  # No-op if no victims to rescue

            direction = nearest_victim.p_pos - agent.p_pos

        # Convert to discrete action
        if abs(direction[0]) > abs(direction[1]):
            # Move horizontally
            return 1 if direction[0] < 0 else 2  # left or right
        else:
            # Move vertically
            return 3 if direction[1] < 0 else 4  # down or up

    def _get_team_reward(self) -> float:
        """Calculate shared team reward for cooperative MARL.

        In cooperative settings, all agents share the same reward signal.
        This encourages coordination and avoids credit assignment issues.

        Reward structure:
        - Large one-time bonus for newly saved victims (+10.0)
        - Small bonus for victims starting to follow (+1.0)
        - Distance-based shaping to guide exploration
        - Small time penalty to encourage efficiency
        - Collision penalties
        """
        reward = 0.0

        # Count victims in each state
        idle_victims = [v for v in self.victims if v.state == VictimState.IDLE]
        following_victims = [v for v in self.victims if v.state == VictimState.FOLLOW]
        saved_victims = [v for v in self.victims if v.state == VictimState.STOP]

        # Track state changes (one-time rewards)
        newly_saved = len(saved_victims) - self._prev_saved_count
        newly_following = len(following_victims) - self._prev_following_count

        # Update tracking for next step
        self._prev_saved_count = len(saved_victims)
        self._prev_following_count = len(following_victims)

        # Large one-time reward for newly saved victims
        if newly_saved > 0:
            reward += newly_saved * 10.0

        # One-time reward for victims starting to follow
        if newly_following > 0:
            reward += newly_following * 1.0

        # Shaping: encourage getting close to idle victims (team-level)
        for victim in idle_victims:
            # Find closest rescuer to this victim
            min_dist = min(
                self._get_distance_pos(agent.p_pos, victim.p_pos)
                for agent in self.agents
            )
            # Reward inversely proportional to distance (closer = better)
            reward += 0.1 / (1.0 + min_dist)

        # Shaping: encourage leading following victims to safe zones
        for victim in following_victims:
            # Find matching safe zone
            matching_safe_zone = next(
                (sz for sz in self.safe_zones if sz.type == victim.type), None
            )
            if matching_safe_zone:
                dist_to_safe = self._get_distance_pos(
                    victim.p_pos, matching_safe_zone.p_pos
                )
                # Reward for victim being close to matching safe zone
                reward += 0.2 / (1.0 + dist_to_safe)

        # Small time penalty to encourage faster completion
        reward -= 0.01

        # Penalty for any agent hitting trees
        for agent in self.agents:
            for landmark in self.landmarks:
                if landmark.tree:
                    if self._is_collision_pos(
                        agent.p_pos, agent.size, landmark.p_pos, landmark.size
                    ):
                        reward -= 0.5

            # Boundary penalty per agent
            reward -= self._bound_penalty(agent.p_pos)

        return reward

    def _update_victims(self):
        """Update victim states based on proximity to rescuers."""
        for victim in self.victims:
            if victim.state == VictimState.STOP:
                continue  # Already saved, don't update

            # Find nearest rescuer within follow range
            nearest_rescuer = None
            min_dist = float("inf")

            for rescuer in self.agents:
                dist = self._get_distance_pos(victim.p_pos, rescuer.p_pos)
                # Check if in range and not blocked by obstacle
                if dist < self.follow_range and dist < min_dist:
                    if not self._is_blocked_by_obstacle_pos(
                        victim.p_pos, rescuer.p_pos
                    ):
                        nearest_rescuer = rescuer
                        min_dist = dist

            # Update state based on proximity
            if nearest_rescuer is not None:
                victim.state = VictimState.FOLLOW
                victim.following_agent = nearest_rescuer
            else:
                # No rescuer in range
                if victim.state == VictimState.FOLLOW:
                    # Keep following last rescuer if they went out of extended range
                    if victim.following_agent is not None:
                        dist_to_following = self._get_distance_pos(
                            victim.p_pos, victim.following_agent.p_pos
                        )
                        # Stop following if too far (1.5x follow range)
                        if dist_to_following > self.follow_range * 1.5:
                            victim.state = VictimState.IDLE
                            victim.following_agent = None
                else:
                    victim.state = VictimState.IDLE
                    victim.following_agent = None

    def _move_victims(self):
        """Move victims based on their state."""
        for victim in self.victims:
            if victim.state == VictimState.STOP:
                victim.p_vel = np.zeros(self.dim_p)
                continue

            if victim.state == VictimState.IDLE:
                # Idle: small random movement or stay still
                victim.p_vel = np.zeros(self.dim_p)
                continue

            if (
                victim.state == VictimState.FOLLOW
                and victim.following_agent is not None
            ):
                # Follow: move towards the rescuer
                direction = victim.following_agent.p_pos - victim.p_pos
                dist = np.linalg.norm(direction)

                if dist > 0.05:  # Don't get too close
                    direction = direction / dist
                    victim.p_vel = direction * victim.speed
                else:
                    victim.p_vel = np.zeros(self.dim_p)

                # Update position
                victim.p_pos += victim.p_vel * self.dt

                # Clamp to bounds
                victim.p_pos = np.clip(victim.p_pos, -0.95, 0.95)

    def _check_rescues(self):
        """Check if any victims have reached their matching safe zones."""
        for victim in self.victims:
            if victim.state == VictimState.STOP:
                continue

            for safe_zone in self.safe_zones:
                if victim.type != safe_zone.type:
                    continue

                dist = self._get_distance_pos(victim.p_pos, safe_zone.p_pos)

                if dist < self.rescue_range:
                    victim.state = VictimState.STOP
                    victim.following_agent = None
                    victim.p_vel = np.zeros(self.dim_p)
                    victim.p_pos = safe_zone.p_pos.copy()  # Snap to safe zone
                    # Dim the color to indicate saved
                    victim.color = victim.color * 0.5
                    break

    def _set_action(self, action, agent):
        """Set action for an agent."""
        u = np.zeros(self.dim_p)

        if self.continuous_actions:
            u[0] = action[0] if len(action) > 0 else 0.0
            u[1] = action[1] if len(action) > 1 else 0.0
        else:
            # Discrete actions: 0=no-op, 1=left, 2=right, 3=down, 4=up
            if action == 1:
                u[0] = -1.0
            elif action == 2:
                u[0] = 1.0
            elif action == 3:
                u[1] = -1.0
            elif action == 4:
                u[1] = 1.0

        sensitivity = agent.accel if agent.accel is not None else 5.0
        u *= sensitivity

        # Store action in agent
        agent.action_u = u

    def _world_step(self):
        """Execute physics step for agents."""
        for agent in self.agents:
            u = agent.action_u
            if u is not None:
                # Apply action force
                agent.p_vel += u * self.dt

                # Apply damping
                agent.p_vel *= 1 - self.damping

                # Clamp to max speed
                speed = np.linalg.norm(agent.p_vel)
                if speed > agent.max_speed:
                    agent.p_vel = agent.p_vel / speed * agent.max_speed

        # Handle collisions
        self._handle_collisions()

        # Update agent positions
        for agent in self.agents:
            agent.p_pos += agent.p_vel * self.dt
            # Clamp to bounds
            agent.p_pos = np.clip(agent.p_pos, -1, 1)

    def _handle_collisions(self):
        """Handle collisions between entities."""
        # Agent-agent collisions
        for i, agent_a in enumerate(self.agents):
            for j, agent_b in enumerate(self.agents):
                if i >= j:
                    continue
                self._apply_collision_force(agent_a, agent_b)

        # Agent-landmark collisions (trees only)
        for agent in self.agents:
            for landmark in self.landmarks:
                if landmark.collide:
                    self._apply_collision_force(agent, landmark)

        # Victim-landmark collisions (trees only)
        for victim in self.victims:
            for landmark in self.landmarks:
                if landmark.collide:
                    self._apply_collision_force_victim(victim, landmark)

    def _apply_collision_force(self, entity_a, entity_b):
        """Apply collision forces between two entities."""
        delta_pos = entity_a.p_pos - entity_b.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity_a.size + entity_b.size

        if dist < dist_min:
            if dist > 0:
                direction = delta_pos / dist
            else:
                direction = np.array([1.0, 0.0])

            penetration = dist_min - dist
            force = self.contact_force * penetration * direction

            entity_a.p_vel += force * self.dt
            if hasattr(entity_b, "movable") and entity_b.movable:
                entity_b.p_vel -= force * self.dt

    @staticmethod
    def _apply_collision_force_victim(victim, landmark):
        """Apply collision force to push victim away from landmark."""
        delta_pos = victim.p_pos - landmark.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = victim.size + landmark.size

        if dist < dist_min:
            if dist > 0:
                direction = delta_pos / dist
            else:
                direction = np.array([1.0, 0.0])

            # Push victim out of collision
            penetration = dist_min - dist
            victim.p_pos += direction * penetration

    def _get_observation(self, agent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get observation for an agent.

        For decentralized execution (CTDE), agents should NOT have access to
        global information like their absolute position. They only observe:
        - Their own velocity
        - Relative positions of nearby landmarks (within vision)
        - Relative positions of nearby rescuers (within vision, not blocked)
        - Relative positions of nearby victims (within vision, not blocked)

        Returns:
            obs: The observation vector (bounded, no sentinel values)
            vis_mask_rescuers: Visibility mask for other rescuers (1=visible, 0=hidden)
            vis_mask_victims: Visibility mask for victims (1=visible, 0=hidden)
        """
        # Calculate distances to each landmark (trees and safe zones within vision)
        landmark_distances = []
        for landmark in self.landmarks:
            if not landmark.boundary:
                dist = self._get_distance_pos(agent.p_pos, landmark.p_pos)
                if dist <= self.vision:
                    delta_pos = landmark.p_pos - agent.p_pos
                    landmark_distances.append((dist, delta_pos))

        # Sort by distance and take N closest
        n_closest = self.n_closest_landmarks
        landmark_distances.sort(key=lambda x: x[0])
        closest_landmarks = landmark_distances[:n_closest]

        # Pad if not enough landmarks visible (use zeros for masked/missing landmarks)
        entity_pos = [pos for _, pos in closest_landmarks]
        while len(entity_pos) < n_closest:
            entity_pos.append(np.zeros(2))

        # Other rescuer positions (relative) with visibility mask
        other_rescuer_pos = []
        vis_mask_rescuers = []
        for other in self.agents:
            if other is agent:
                continue

            dist = self._get_distance_pos(agent.p_pos, other.p_pos)
            is_blocked = self._is_blocked_by_obstacle_pos(agent.p_pos, other.p_pos)
            is_visible = (dist <= self.vision) and (not is_blocked)

            if is_visible:
                other_rescuer_pos.append(other.p_pos - agent.p_pos)
                vis_mask_rescuers.append(1.0)
            else:
                # Use zeros instead of sentinel values
                other_rescuer_pos.append(np.zeros(2))
                vis_mask_rescuers.append(0.0)

        # Victim info: relative position + state with visibility mask
        victim_info = []
        vis_mask_victims = []
        for victim in self.victims:
            dist = self._get_distance_pos(agent.p_pos, victim.p_pos)
            is_blocked = self._is_blocked_by_obstacle_pos(agent.p_pos, victim.p_pos)
            is_visible = (dist <= self.vision) and (not is_blocked)

            if is_visible:
                rel_pos = victim.p_pos - agent.p_pos
                state_val = float(victim.state.value)  # 0=idle, 1=follow, 2=stop
                victim_info.append(np.array([rel_pos[0], rel_pos[1], state_val]))
                vis_mask_victims.append(1.0)
            else:
                # Use zeros instead of sentinel values
                victim_info.append(np.zeros(3))
                vis_mask_victims.append(0.0)

        # Concatenate observation (NO absolute position - only relative/local info)
        # Format: [velocity, relative_landmarks, relative_rescuers, relative_victims]
        obs = np.concatenate(
            [agent.p_vel] + entity_pos + other_rescuer_pos + victim_info
        )

        return (
            obs.astype(np.float32),
            np.array(vis_mask_rescuers, dtype=np.float32),
            np.array(vis_mask_victims, dtype=np.float32),
        )

    def _get_global_state(self) -> np.ndarray:
        """
        Get global state for CTDE (Centralized Training with Decentralized Execution).

        This provides full observability for the centralized critic during training,
        while actors only see their local observations.

        Global State Shape (fixed):
            - All rescuer positions: num_rescuers * 2
            - All rescuer velocities: num_rescuers * 2
            - All victim positions: num_missing * 2
            - All victim states: num_missing * 1  (0=idle, 1=follow, 2=stop)
            - All tree positions: num_trees * 2
            - All safe zone positions: num_safe_zones * 2

        Total size: self.global_state_size

        Returns:
            np.ndarray: Global state vector (shape: global_state_size,)
        """
        state_components = []

        # All rescuer positions (absolute)
        for agent in self.agents:
            state_components.append(agent.p_pos)

        # All rescuer velocities
        for agent in self.agents:
            state_components.append(agent.p_vel)

        # All victim positions (absolute)
        for victim in self.victims:
            state_components.append(victim.p_pos)

        # All victim states
        victim_states = np.array([float(v.state.value) for v in self.victims])
        state_components.append(victim_states)

        # All tree positions
        for landmark in self.landmarks:
            if landmark.tree:
                state_components.append(landmark.p_pos)

        # All safe zone positions
        for landmark in self.landmarks:
            if not landmark.tree and not landmark.boundary:
                state_components.append(landmark.p_pos)

        # Concatenate and clip to bounds
        global_state = np.concatenate(state_components)
        global_state = np.clip(global_state, -2.0, 2.0)

        return global_state.astype(np.float32)

    def _get_reward(self, agent) -> float:
        """Calculate reward for rescuer agent."""
        reward = 0.0

        # Count victims in each state
        idle_victims = [v for v in self.victims if v.state == VictimState.IDLE]
        following_victims = [v for v in self.victims if v.state == VictimState.FOLLOW]
        saved_victims = [v for v in self.victims if v.state == VictimState.STOP]

        # Reward for victims following (small positive)
        reward += len(following_victims) * 0.1

        # Big reward for saved victims
        reward += len(saved_victims) * 1.0

        # Shaping: encourage getting close to idle victims
        for victim in idle_victims:
            dist = self._get_distance_pos(agent.p_pos, victim.p_pos)
            reward -= dist * 0.05  # Small penalty for distance to idle victims

        # Shaping: encourage leading following victims to safe zones
        for victim in following_victims:
            # Find matching safe zone
            matching_safe_zone = next(
                (sz for sz in self.safe_zones if sz.type == victim.type), None
            )
            if matching_safe_zone:
                dist_to_safe = self._get_distance_pos(
                    victim.p_pos, matching_safe_zone.p_pos
                )
                reward += (
                    1.0 / (1 + dist_to_safe) * 0.2
                )  # Bonus for victim being close to safe zone

        # Penalty for hitting trees
        for landmark in self.landmarks:
            if landmark.tree:
                if self._is_collision_pos(
                    agent.p_pos, agent.size, landmark.p_pos, landmark.size
                ):
                    reward -= 0.5

        # Boundary penalty
        reward -= self._bound_penalty(agent.p_pos)

        return reward

    @staticmethod
    def _bound_penalty(pos: np.ndarray) -> float:
        """Calculate penalty for being near bounds."""
        penalty = 0.0
        x_min, x_max = -0.9, 0.9
        y_min, y_max = -0.9, 0.9

        if pos[0] < x_min:
            penalty += (x_min - pos[0]) * 10
        elif pos[0] > x_max:
            penalty += (pos[0] - x_max) * 10

        if pos[1] < y_min:
            penalty += (y_min - pos[1]) * 10
        elif pos[1] > y_max:
            penalty += (pos[1] - y_max) * 10

        return penalty

    @staticmethod
    def _is_collision_pos(pos_a, size_a, pos_b, size_b) -> bool:
        """Check if two entities are colliding."""
        delta_pos = pos_a - pos_b
        dist = np.linalg.norm(delta_pos)
        dist_min = size_a + size_b
        return dist < dist_min

    def _is_blocked_by_obstacle_pos(self, pos_a, pos_b) -> bool:
        """Check if line of sight is blocked by a tree.

        Only returns True if a tree is actually BETWEEN pos_a and pos_b,
        not if it's behind either endpoint.
        """
        for landmark in self.landmarks:
            if landmark.tree:
                if self._line_segment_intersects_circle(
                    pos_a, pos_b, landmark.p_pos, landmark.size
                ):
                    return True
        return False

    @staticmethod
    def _line_segment_intersects_circle(p1, p2, center, radius) -> bool:
        """
        Check if line SEGMENT (not infinite line) intersects a circle.

        Uses parametric form: P(t) = p1 + t*(p2-p1) for t in [0,1]
        Checks if the closest point on the segment to the circle center
        is within the radius.
        """
        # Direction vector from p1 to p2
        d = p2 - p1
        f = p1 - center  # Vector from circle center to p1

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius

        # Handle degenerate case (p1 == p2)
        if a < 1e-10:
            return np.linalg.norm(f) <= radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            # No intersection with infinite line
            return False

        discriminant = np.sqrt(discriminant)

        # Two intersection points with infinite line at t1 and t2
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if either intersection point is within segment bounds [0, 1]
        # Or if the segment is entirely inside the circle
        if 0 <= t1 <= 1:
            return True
        if 0 <= t2 <= 1:
            return True

        # Check if segment is entirely inside circle (both t values outside [0,1] but circle contains segment)
        if t1 < 0 and t2 > 1:
            return True

        return False

    @staticmethod
    def _get_distance_pos(pos_a, pos_b) -> float:
        """Calculate distance between two positions."""
        return np.linalg.norm(pos_a - pos_b)

    @property
    def safe_zones(self):
        """Get list of safe zone landmarks."""
        return [lm for lm in self.landmarks if not lm.tree]

    @property
    def trees(self):
        """Get list of tree landmarks."""
        return [lm for lm in self.landmarks if lm.tree]

    @property
    def rescuers(self):
        """Get list of rescuer agents (alias for agents)."""
        return self.agents

    @property
    def num_agents(self):
        """Get number of agents (rescuers)."""
        return len(self.agents)

    def _get_distance(self, entity_a, entity_b) -> float:
        """Calculate distance between two entities."""
        return self._get_distance_pos(entity_a.p_pos, entity_b.p_pos)

    def _is_blocked_by_obstacle(self, entity_a, entity_b) -> bool:
        """Check if line of sight between two entities is blocked by a tree."""
        return self._is_blocked_by_obstacle_pos(entity_a.p_pos, entity_b.p_pos)

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Search and Rescue")
            else:
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((240, 240, 240))

        # Calculate camera range
        cam_range = 1.0

        def to_screen(pos):
            """Convert world position to screen coordinates."""
            _x, _y = pos
            _y = -_y  # Flip y for display
            _x = (_x / cam_range) * self.width // 2 * 0.9 + self.width // 2
            _y = (_y / cam_range) * self.height // 2 * 0.9 + self.height // 2
            return int(_x), int(_y)

        def draw_circle(obj: Union[Landmark | Agent]) -> tuple[int, int, int]:
            """Draw a circle for an object and return screen coords and radius."""
            _x, _y = to_screen(obj.p_pos)
            _radius = int(obj.size * 350)
            _color = tuple((obj.color * 200).astype(int))
            pygame.draw.circle(self.screen, _color, (_x, _y), _radius)
            return _x, _y, _radius

        # Draw safe zones (background)
        for safe_zone in self.safe_zones:
            x, y = to_screen(safe_zone.p_pos)
            radius = int(safe_zone.size * 350)
            color = tuple((safe_zone.color * 100 + 100).astype(int))
            pygame.draw.circle(self.screen, color, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 2)

            # Draw type label
            font = pygame.font.Font(None, 24)
            text = font.render(safe_zone.type, True, (0, 0, 0))
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

        # Draw trees
        for tree in self.trees:
            x, y, radius = draw_circle(tree)
            pygame.draw.circle(self.screen, (0, 100, 0), (x, y), radius, 2)

        # Draw victims
        for victim in self.victims:
            x, y = to_screen(victim.p_pos)
            radius = int(victim.size * 350)
            color = tuple((victim.color * 200).astype(int))

            # Draw state indicator
            if victim.state == VictimState.IDLE:
                # Circle with question mark
                pygame.draw.circle(self.screen, color, (x, y), radius)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)
            elif victim.state == VictimState.FOLLOW:
                # Circle with movement indicator (arrow)
                pygame.draw.circle(self.screen, color, (x, y), radius)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius, 2)
                # Draw line to rescuer being followed
                if victim.following_agent is not None:
                    rx, ry = to_screen(victim.following_agent.p_pos)
                    pygame.draw.line(self.screen, (200, 200, 200), (x, y), (rx, ry), 1)
            elif victim.state == VictimState.STOP:
                # Dimmed circle with checkmark
                pygame.draw.circle(self.screen, color, (x, y), radius)
                pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 2)

            # Draw type label
            font = pygame.font.Font(None, 16)
            text = font.render(victim.type, True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

        # Draw rescuers
        for agent in self.agents:
            x, y, radius = draw_circle(agent)
            pygame.draw.circle(self.screen, (100, 0, 0), (x, y), radius, 2)

            # Draw vision range (faint circle) - uses self.vision
            vision_px = int(self.vision * 350)  # Convert to pixels (map scale)
            pygame.draw.circle(self.screen, (200, 200, 255), (x, y), vision_px, 1)

        # Draw HUD
        font = pygame.font.Font(None, 24)
        idle_count = sum(1 for v in self.victims if v.state == VictimState.IDLE)
        follow_count = sum(1 for v in self.victims if v.state == VictimState.FOLLOW)
        saved_count = sum(1 for v in self.victims if v.state == VictimState.STOP)

        hud_text = (
            f"Step: {self._step_count}/{self.max_cycles} | "
            + f"Idle: {idle_count} | Following: {follow_count} | Saved: {saved_count}"
        )
        text_surface = font.render(hud_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return None

    def close(self, raise_if_closed: bool = True):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ---- Public helpers (avoid protected-member access from outside) ----
    def is_collision(self, entity_a, entity_b) -> bool:
        """Public wrapper for collision check."""
        return self._is_collision_pos(
            pos_a=entity_a.p_pos,
            size_a=entity_a.size,
            pos_b=entity_b.p_pos,
            size_b=entity_b.size,
        )

    def bound_penalty(self, pos: np.ndarray) -> float:
        """Public wrapper for boundary penalty."""
        return self._bound_penalty(pos)

    def sample_discrete_action(self) -> int:
        """Sample a discrete action using the environment RNG."""
        return int(self._np_random.randint(0, 5))


# Factory functions for compatibility
def make_env(**kwargs):
    """Create a SearchAndRescueEnv instance."""
    return SearchAndRescueEnv(**kwargs)
