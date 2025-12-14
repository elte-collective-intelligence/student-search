# noqa: D212, D415
"""
Search and Rescue Environment using TorchRL.

This environment simulates a search and rescue scenario where rescuers (adversaries)
must guide victims (agents) to their matching safe zones while avoiding obstacles (trees).
"""

import numpy as np
import torch
import pygame
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, Unbounded, Categorical, Composite
from typing import Optional


class Agent:
    """Agent entity in the environment."""

    def __init__(self):
        self.name = ""
        self.is_rescuer = False  # True for rescuers, False for victims
        self.saved = False
        self.collide = True
        self.silent = True
        self.size = 0.025
        self.accel = 3.0
        self.max_speed = 0.3
        self.type = None  # Victim type (A, B, C, D)
        self.color = np.array([0.0, 0.0, 0.0])
        # State
        self.p_pos = np.zeros(2)  # Position
        self.p_vel = np.zeros(2)  # Velocity
        self.c = np.zeros(2)  # Communication


class Landmark:
    """Landmark entity in the environment (trees or safe zones)."""

    def __init__(self):
        self.name = ""
        self.tree = False
        self.collide = False
        self.movable = False
        self.size = 0.03
        self.boundary = False
        self.type = None  # Safe zone type (A, B, C, D)
        self.color = np.array([0.0, 0.0, 0.0])
        # State
        self.p_pos = np.zeros(2)
        self.p_vel = np.zeros(2)


class SearchAndRescueEnv(EnvBase):
    """
    TorchRL-based Search and Rescue Environment.

    Rescuers must guide victims to their matching safe zones (same type).
    Trees act as obstacles that block movement and vision.
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
    ):
        super().__init__(device=device, batch_size=torch.Size([]))

        self.num_missing = num_missing
        self.num_rescuers = num_rescuers
        self.num_trees = num_trees
        self.num_safe_zones = num_safe_zones
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.vision = 1  # Vision range for agents

        self.num_agents = num_rescuers + num_missing
        self.num_landmarks = num_trees + num_safe_zones

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
        """Create all agents and landmarks."""
        # Create agents (rescuers + victims)
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent()
            agent.is_rescuer = i < self.num_rescuers  # First agents are rescuers
            base_name = "rescuer" if agent.is_rescuer else "victim"
            base_index = i if agent.is_rescuer else i - self.num_rescuers
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.025 if agent.is_rescuer else 0.01
            agent.accel = 3.0 if agent.is_rescuer else 4.0
            agent.max_speed = 0.3

            # Assign type to victims
            if not agent.is_rescuer:
                agent.type = self.VICTIM_TYPES[base_index % len(self.VICTIM_TYPES)]
                agent.color = self.TYPE_COLORS[agent.type].copy()
            else:
                agent.color = np.array([0.85, 0.35, 0.35])  # Rescuer color

            self.agents.append(agent)

        # Create landmarks (trees + safe zones)
        self.landmarks = []
        for i in range(self.num_landmarks):
            landmark = Landmark()
            landmark.tree = i < self.num_trees
            base_name = "tree" if landmark.tree else "safe_zone"
            base_index = i if landmark.tree else i - self.num_trees
            landmark.name = f"{base_name}_{base_index}"
            landmark.collide = landmark.tree  # Only trees block movement
            landmark.movable = False
            landmark.size = 0.03 if landmark.tree else 0.1
            landmark.boundary = False

            # Assign type to safe zones
            if not landmark.tree:
                landmark.type = self.SAFE_ZONE_TYPES[
                    base_index % len(self.SAFE_ZONE_TYPES)
                ]
                landmark.color = self.TYPE_COLORS[landmark.type].copy()
            else:
                landmark.color = np.array([0.35, 0.85, 0.35])  # Tree color

            self.landmarks.append(landmark)

        # Define agent names for multi-agent handling
        self.possible_agents = [a.name for a in self.agents]
        self.agent_name_mapping = {a.name: i for i, a in enumerate(self.agents)}

    def _make_specs(self):
        """Define observation and action specs."""
        # Calculate observation size
        # For each agent: own velocity (2) + own position (2) +
        # closest N landmarks (N*2) + other agents positions (num_agents-1)*2 +
        # victim velocities (num_missing*2)
        n_closest_landmarks = 3
        obs_size = (
            2  # own velocity
            + 2  # own position
            + n_closest_landmarks * 2  # closest landmark positions
            + (self.num_agents - 1) * 2  # other agent positions
            + self.num_missing * 2  # victim velocities
        )

        # Observation spec - single agent observation
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(obs_size,),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=(),
        )

        # Action spec
        if self.continuous_actions:
            # Continuous: 2D force vector
            self.action_spec = BoundedTensorSpec(
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

        # Done spec (must be a Composite with done, terminated, truncated)
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

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        """Reset the environment."""
        self._step_count = 0

        # Safe zone fixed positions (corners)
        safe_zone_positions = [
            np.array([-1.0, 1.0]),  # Top-left
            np.array([-1.0, -1.0]),  # Bottom-left
            np.array([1.0, 1.0]),  # Top-right
            np.array([1.0, -1.0]),  # Bottom-right
        ]

        # Reset agents
        for agent in self.agents:
            agent.p_pos = self._np_random.uniform(-0.5, 0.5, self.dim_p)
            agent.p_vel = np.zeros(self.dim_p)
            agent.c = np.zeros(self.dim_c)
            agent.saved = False

            # Reset colors
            if agent.is_rescuer:
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = self.TYPE_COLORS[agent.type].copy()

        # Reset landmarks
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

        # Get observation for first rescuer (main controlled agent)
        obs = self._get_observation(self.rescuers[0])

        return TensorDict(
            {
                "observation": torch.tensor(
                    obs, dtype=torch.float32, device=self.device
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
        """Execute one step in the environment."""
        action = tensordict["action"]

        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Apply action to main rescuer
        main_rescuer = self.rescuers[0]
        self._set_action(action, main_rescuer)

        # Apply random actions to other rescuers (or scripted behavior)
        for rescuer in self.rescuers[1:]:
            random_action = self._np_random.randint(0, 5)
            self._set_action(random_action, rescuer)

        # Victims move randomly (or can be scripted)
        for victim in self.victims:
            if not victim.saved:
                random_action = self._np_random.randint(0, 5)
                self._set_action(random_action, victim)

        # Physics step
        self._world_step()

        # Check for rescues and update saved status
        self._check_rescues()

        # Calculate reward for main rescuer
        reward = self._get_reward(main_rescuer)

        # Get new observation
        obs = self._get_observation(main_rescuer)

        # Increment step counter
        self._step_count += 1

        # Check termination
        all_saved = all(v.saved for v in self.victims)
        truncated = self._step_count >= self.max_cycles
        terminated = all_saved
        done = terminated or truncated

        return TensorDict(
            {
                "observation": torch.tensor(
                    obs, dtype=torch.float32, device=self.device
                ),
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

    def _set_action(self, action, agent):
        """Set action for an agent."""
        # Initialize action force
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

        # Apply sensitivity/acceleration
        sensitivity = agent.accel if agent.accel is not None else 5.0
        u *= sensitivity

        # Store action in agent
        agent._action_u = u

    def _world_step(self):
        """Execute physics step for all entities."""
        # Apply forces and update velocities
        for agent in self.agents:
            if hasattr(agent, "_action_u"):
                # Apply action force
                agent.p_vel += agent._action_u * self.dt

                # Apply damping
                agent.p_vel *= 1 - self.damping

                # Clamp to max speed
                speed = np.linalg.norm(agent.p_vel)
                if speed > agent.max_speed:
                    agent.p_vel = agent.p_vel / speed * agent.max_speed

        # Handle collisions
        self._handle_collisions()

        # Update positions
        for agent in self.agents:
            if not agent.saved:  # Saved victims don't move
                agent.p_pos += agent.p_vel * self.dt

        # Stop saved victims
        for victim in self.victims:
            if victim.saved:
                victim.p_vel = np.zeros(self.dim_p)

    def _handle_collisions(self):
        """Handle collisions between entities."""
        # Agent-agent collisions
        for i, agent_a in enumerate(self.agents):
            for j, agent_b in enumerate(self.agents):
                if i >= j:
                    continue
                if agent_a.collide and agent_b.collide:
                    self._apply_collision_force(agent_a, agent_b)

        # Agent-landmark collisions (trees only)
        for agent in self.agents:
            for landmark in self.landmarks:
                if landmark.collide and agent.collide:
                    self._apply_collision_force(agent, landmark)

    def _apply_collision_force(self, entity_a, entity_b):
        """Apply collision forces between two entities."""
        delta_pos = entity_a.p_pos - entity_b.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity_a.size + entity_b.size

        if dist < dist_min:
            # Collision detected
            if dist > 0:
                direction = delta_pos / dist
            else:
                direction = np.array([1.0, 0.0])

            penetration = dist_min - dist
            force = self.contact_force * penetration * direction

            # Apply forces (only to movable entities)
            if hasattr(entity_a, "p_vel") and not getattr(entity_a, "movable", True):
                entity_a.p_vel += force * self.dt
            if hasattr(entity_b, "p_vel") and getattr(entity_b, "movable", False):
                entity_b.p_vel -= force * self.dt

    def _check_rescues(self):
        """Check if any victims have reached their matching safe zones."""
        rescue_radius = 0.1

        for victim in self.victims:
            if victim.saved:
                continue

            for safe_zone in self.safe_zones:
                if victim.type != safe_zone.type:
                    continue

                delta_pos = victim.p_pos - safe_zone.p_pos
                dist = np.linalg.norm(delta_pos)

                if dist < rescue_radius:
                    victim.saved = True
                    victim.p_vel = np.zeros(self.dim_p)
                    print(f"Victim {victim.name} saved at {safe_zone.name}!")
                    break

    def _get_observation(self, agent) -> np.ndarray:
        """Get observation for an agent."""
        # Calculate distances to each landmark
        landmark_distances = []
        for landmark in self.landmarks:
            if not landmark.boundary:
                dist = self._get_distance(agent, landmark)
                if dist <= self.vision:
                    delta_pos = landmark.p_pos - agent.p_pos
                    landmark_distances.append((dist, delta_pos))

        # Sort by distance and take N closest
        n_closest = 3
        landmark_distances.sort(key=lambda x: x[0])
        closest_landmarks = landmark_distances[:n_closest]

        # Pad if not enough landmarks
        entity_pos = [pos for _, pos in closest_landmarks]
        while len(entity_pos) < n_closest:
            entity_pos.append(np.zeros(2))

        # Other agent positions (relative)
        other_pos = []
        other_vel = []
        for other in self.agents:
            if other is agent:
                continue

            # Check if blocked by obstacle or out of vision
            if (
                self._is_blocked_by_obstacle(agent, other)
                or self._get_distance(agent, other) > self.vision
            ):
                other_pos.append(np.array([1e6, 1e6]))
            else:
                other_pos.append(other.p_pos - agent.p_pos)

            # Victim velocities
            if not other.is_rescuer:
                other_vel.append(other.p_vel.copy())

        # Concatenate observation
        obs = np.concatenate(
            [agent.p_vel] + [agent.p_pos] + entity_pos + other_pos + other_vel
        )

        return obs.astype(np.float32)

    def _get_reward(self, agent) -> float:
        """Calculate reward for an agent."""
        if agent.is_rescuer:
            return self._rescuer_reward(agent)
        else:
            return self._victim_reward(agent)

    def _victim_reward(self, agent) -> float:
        """Reward function for victims."""
        if agent.saved:
            return 0.0

        reward = 0.0
        reward -= self._bound_penalty(agent.p_pos)
        return reward

    def _rescuer_reward(self, agent) -> float:
        """Reward function for rescuers."""
        reward = 0.0
        shape = True

        unsaved_victims = [v for v in self.victims if not v.saved]

        if shape:
            for victim in unsaved_victims:
                # Penalize distance to victim
                dist_to_victim = np.linalg.norm(agent.p_pos - victim.p_pos)
                reward -= dist_to_victim * 0.1

                # Find matching safe zone
                matching_safe_zone = next(
                    (sz for sz in self.safe_zones if sz.type == victim.type), None
                )

                if matching_safe_zone:
                    dist_to_safe_zone = np.linalg.norm(
                        victim.p_pos - matching_safe_zone.p_pos
                    )
                    reward += 1.0 / (1 + dist_to_safe_zone)

                    # Bonus for rescue
                    if victim.saved:
                        reward += 100.0

        # Penalty for hitting obstacles
        if shape:
            for landmark in self.landmarks:
                if landmark.collide and self._is_collision(agent, landmark):
                    reward -= 10.0

        # Boundary penalty
        reward -= self._bound_penalty(agent.p_pos)

        return reward

    @staticmethod
    def _bound_penalty(pos: np.ndarray) -> float:
        """Calculate penalty for being out of bounds."""
        penalty = 0.0
        x_min, x_max = -0.7, 0.7
        y_min, y_max = -0.7, 0.7

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
    def _is_collision(entity_a, entity_b) -> bool:
        """Check if two entities are colliding."""
        delta_pos = entity_a.p_pos - entity_b.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity_a.size + entity_b.size
        return dist < dist_min

    def _is_blocked_by_obstacle(self, agent_a, agent_b) -> bool:
        """Check if line of sight is blocked by an obstacle."""
        for landmark in self.landmarks:
            if landmark.collide:
                if self._line_intersects_circle(
                    agent_a.p_pos, agent_b.p_pos, landmark.p_pos, landmark.size
                ):
                    return True
        return False

    @staticmethod
    def _line_intersects_circle(p1, p2, center, radius) -> bool:
        """Check if line segment intersects a circle."""
        p1_rel = p1 - center
        p2_rel = p2 - center

        dx, dy = p2_rel - p1_rel
        dr_sq = dx**2 + dy**2
        d = p1_rel[0] * p2_rel[1] - p2_rel[0] * p1_rel[1]

        discriminant = radius**2 * dr_sq - d**2
        return discriminant >= 0

    @staticmethod
    def _get_distance(entity_a, entity_b) -> float:
        """Calculate distance between two entities."""
        delta_pos = entity_a.p_pos - entity_b.p_pos
        return np.linalg.norm(delta_pos)

    @property
    def rescuers(self):
        """Get list of rescuer agents."""
        return [a for a in self.agents if a.is_rescuer]

    @property
    def victims(self):
        """Get list of victim agents."""
        return [a for a in self.agents if not a.is_rescuer]

    @property
    def safe_zones(self):
        """Get list of safe zone landmarks."""
        return [lm for lm in self.landmarks if not lm.tree]

    @property
    def trees(self):
        """Get list of tree landmarks."""
        return [lm for lm in self.landmarks if lm.tree]

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
        self.screen.fill((255, 255, 255))

        # Calculate camera range
        all_poses = [agent.p_pos for agent in self.agents] + [
            lm.p_pos for lm in self.landmarks
        ]
        cam_range = max(np.max(np.abs(np.array(all_poses))), 1.0)

        # Draw entities
        for entity in self.landmarks + self.agents:
            x, y = entity.p_pos
            y *= -1  # Flip y for display

            # Scale to screen coordinates
            x = (x / cam_range) * self.width // 2 * 0.9 + self.width // 2
            y = (y / cam_range) * self.height // 2 * 0.9 + self.height // 2

            radius = entity.size * 350
            color = tuple((entity.color * 200).astype(int))

            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(radius), 1)

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


# Factory functions for compatibility
def make_env(**kwargs):
    """Create a SearchAndRescueEnv instance."""
    return SearchAndRescueEnv(**kwargs)


def env(**kwargs):
    """Alias for make_env."""
    return make_env(**kwargs)


def parallel_env(**kwargs):
    """Create environment (single-agent wrapper for compatibility)."""
    return make_env(**kwargs)
