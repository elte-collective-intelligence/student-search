from typing import Optional, Union

import numpy as np
import pygame
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from torchrl.envs import PettingZooWrapper, TransformedEnv, RewardSum


class SearchAndRescueEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "search_rescue_v2"}

    def __init__(
        self,
        num_rescuers: int = 2,
        num_victims: int = 2,
        num_trees: int = 5,
        num_safe_zones: int = 4,
        max_cycles: int = 200,
        continuous_actions: bool = False,
        vision_radius: float = 0.5,
        randomize_safe_zones: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        self.num_rescuers = num_rescuers
        self.num_victims = num_victims
        self.num_trees = num_trees
        self.num_safe_zones = num_safe_zones
        self.render_mode = render_mode
        self.max_steps = max_cycles
        self.is_continuous = continuous_actions
        self.randomize_safe_zones = randomize_safe_zones

        # Initialize random number generator
        self._seed = seed
        self.np_random = np.random.RandomState(seed)

        # Parameters
        self.world_size = 2.0  # [-1, 1] range
        self.vision_radius = vision_radius
        self.rescue_radius = 0.15
        self.agent_size = 0.03
        self.tree_radius = 0.05
        self.safe_zone_radius = 0.15
        self.follow_radius = 0.2

        self.agents = [f"rescuer_{i}" for i in range(num_rescuers)]
        self.possible_agents = self.agents[:]
        self.victim_names = [f"victim_{i}" for i in range(num_victims)]

        # Assign types cyclically
        self.victim_types = [i % self.num_safe_zones for i in range(num_victims)]
        # safe_zone_types will be set in reset() based on randomize_safe_zones

        # Colors for rendering
        self.type_colors = {
            0: (255, 50, 50),  # Red (A)
            1: (50, 255, 50),  # Green (B)
            2: (50, 50, 255),  # Blue (C)
            3: (255, 255, 50),  # Yellow (D)
        }

        # Action Space: Either Discrete or Continuous
        if self.is_continuous:
            # Continuous acceleration (dx, dy)
            self.action_spaces = {
                agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # Discrete actions: 0=noop, 1=up, 2=down, 3=left, 4=right
            self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}

        # Observation Space Calculation (Partial Observability with Vision Masking)
        # [Self_Vel(2), Self_Pos(2), Agent_ID(num_rescuers),
        #  SafeZones(4 * 3: rel_x, rel_y, type_idx) - masked if occluded/far,
        #  Trees(N * 2: rel_x, rel_y) - masked if occluded/far,
        #  Victims(N * 3: rel_x, rel_y, type_idx) - masked if occluded/far/saved]
        # Note: All spatial entities use visibility masking based on vision_radius and occlusion
        # Note: Other rescuers removed from observation to focus learning on task

        self.obs_dim = (
            4
            + num_rescuers  # Agent ID one-hot encoding
            + (self.num_safe_zones * 3)
            + (self.num_trees * 2)
            + (self.num_victims * 3)
        )

        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            for agent in self.agents
        }

        self.screen = None
        self.clock = None

        # Metrics state (reset each episode)
        self._cell_size = 0.05
        self._episode_counter = 0
        self._visited_cells = []
        self._collision_events = 0
        self._completed_episode_metrics = []

    def _reset_metrics(self) -> None:
        """Reset per-episode metric trackers."""
        self._visited_cells = [set() for _ in range(self.num_rescuers)]
        self._collision_events = 0

    def _hash_pos(self, pos: np.ndarray) -> tuple[int, int]:
        return tuple(np.floor(pos / self._cell_size).astype(int))

    def pop_episode_metrics(self) -> list[dict]:
        """Return and clear metrics for episodes that completed since last call."""
        metrics = self._completed_episode_metrics
        self._completed_episode_metrics = []
        return metrics

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.steps = 0

        # Handle seeding
        if seed is not None:
            self._seed = seed
            self.np_random = np.random.RandomState(seed)

        # Reset Agents list (required by PettingZoo API)
        self.agents = self.possible_agents[:]

        # Metrics state (reset each episode)
        self._episode_counter += 1
        self._reset_metrics()

        # Positions: Rescuers, Victims, Trees, SafeZones (using self.np_random)
        self.rescuer_pos = self.np_random.uniform(-0.8, 0.8, (self.num_rescuers, 2))
        self.rescuer_vel = np.zeros((self.num_rescuers, 2))

        self.victim_pos = self.np_random.uniform(-0.8, 0.8, (self.num_victims, 2))
        self.victim_vel = np.zeros((self.num_victims, 2))
        self.victim_saved = np.zeros(self.num_victims, dtype=bool)

        # Track which agent each victim is committed to (-1 = none)
        self.victim_assignments = np.full(self.num_victims, -1, dtype=int)

        self.tree_pos = self.np_random.uniform(-0.8, 0.8, (self.num_trees, 2))

        # Safe zones: randomized or at fixed corners
        if self.randomize_safe_zones:
            # Randomize positions within the world bounds
            self.safezone_pos = self.np_random.uniform(
                -0.95, 0.95, (self.num_safe_zones, 2)
            )

            # Shuffle types to randomize which zone accepts which victim type
            # Keep types as 0,1,2,3 but in random order
            self.safe_zone_types = list(range(self.num_safe_zones))
            self.np_random.shuffle(self.safe_zone_types)
        else:
            # Fixed positions at corners (default behavior)
            self.safezone_pos = np.array(
                [[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]]
            )
            # Types stay in original order
            self.safe_zone_types = [0, 1, 2, 3]

        # Track previous distances for delta-based shaping
        self.prev_agent_victim_dists = self._compute_agent_victim_dists()

        return self._get_obs(), {a: {} for a in self.agents}

    def _is_visible(
        self, observer_pos, target_pos, target_radius, exclude_tree_idx=None
    ):
        """Checks whether a target is within vision range and not occluded by trees.
        Args:
            observer_pos: Position of the observer.
            target_pos: Position of the target to check visibility of.
            target_radius: Radius of the target. Currently unused, kept for API compatibility
                and potential future use in more detailed visibility calculations.
            exclude_tree_idx: Optional tree index to exclude from the occlusion check
                (e.g., when checking if a tree itself is visible). Occlusion is currently
                determined using the environment's ``tree_radius`` for all trees.
        """
        # If vision radius is zero, nothing is visible
        if self.vision_radius == 0.0:
            return False

        dist = np.linalg.norm(target_pos - observer_pos)
        if dist > self.vision_radius:
            return False

        # Check line of sight against all trees (excluding the tree being checked if specified)
        for t_idx in range(self.num_trees):
            if exclude_tree_idx is not None and t_idx == exclude_tree_idx:
                continue  # Skip the tree being checked

            tree_c = self.tree_pos[t_idx]

            # Skip trees that are at the target position (target tree should not block itself)
            tree_to_target_dist = np.linalg.norm(tree_c - target_pos)
            if tree_to_target_dist < 1e-6:
                continue  # Tree is at target position, skip it

            # Vector from observer to target
            d_vec = target_pos - observer_pos
            # Vector from observer to tree center
            f_vec = observer_pos - tree_c

            a = np.dot(d_vec, d_vec)
            # Handle edge case: observer and target at same position
            if a < 1e-10:
                # If observer and target are at same position, check if tree is at that position
                tree_dist = np.linalg.norm(tree_c - observer_pos)
                if tree_dist < self.tree_radius:
                    return False  # Tree is blocking (at same position)
                continue  # No blocking if tree is not at same position

            b = 2 * np.dot(f_vec, d_vec)
            c = np.dot(f_vec, f_vec) - self.tree_radius**2

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)

                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    return False  # Blocked
        return True

    def _get_obs(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            # If agent was removed (e.g. done), skip
            if agent not in self.agents:
                continue

            obs_vec = []
            my_pos = self.rescuer_pos[i]

            # 1. Self State (Vel, Pos)
            obs_vec.extend(self.rescuer_vel[i])
            obs_vec.extend(my_pos)

            # 2. Agent ID (one-hot encoding to break symmetry)
            agent_id_onehot = np.zeros(self.num_rescuers)
            agent_id_onehot[i] = 1.0
            obs_vec.extend(agent_id_onehot)

            # 3. Safe Zones (with visibility masking)
            for sz_i in range(self.num_safe_zones):
                if self._is_visible(
                    my_pos, self.safezone_pos[sz_i], self.safe_zone_radius
                ):
                    rel_pos = self.safezone_pos[sz_i] - my_pos
                    # We append the numeric type (0-3) so the network knows which zone is which
                    obs_vec.extend(
                        [rel_pos[0], rel_pos[1], float(self.safe_zone_types[sz_i])]
                    )
                else:
                    # Masked: not visible due to distance or occlusion
                    obs_vec.extend([0.0, 0.0, -1.0])

            # 4. Trees
            for t_i in range(self.num_trees):
                if self._is_visible(
                    my_pos, self.tree_pos[t_i], self.tree_radius, exclude_tree_idx=t_i
                ):
                    obs_vec.extend(self.tree_pos[t_i] - my_pos)
                else:
                    obs_vec.extend([0.0, 0.0])  # Masked

            # 5. Victims
            for v_i in range(self.num_victims):
                # If visible and not saved
                if not self.victim_saved[v_i] and self._is_visible(
                    my_pos, self.victim_pos[v_i], self.agent_size
                ):
                    rel = self.victim_pos[v_i] - my_pos
                    # Use numeric type (0-3) for observation
                    obs_vec.extend([rel[0], rel[1], float(self.victim_types[v_i])])
                else:
                    obs_vec.extend(
                        [0.0, 0.0, -1.0]
                    )  # Masked (Type -1 indicates not visible)

            # Other rescuers removed from observation to focus learning on task

            observations[agent] = np.array(obs_vec, dtype=np.float32)
        return observations

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # 1. Apply Actions
        for i, agent in enumerate(self.agents):
            if agent not in actions:
                continue

            action = actions[agent]

            # Convert discrete action to continuous if needed
            if not self.is_continuous:
                # Discrete actions: 0=noop, 1=up, 2=down, 3=left, 4=right
                if isinstance(action, np.ndarray):
                    action = int(action)

                action_map = {
                    0: np.array([0.0, 0.0]),  # noop
                    1: np.array([0.0, 1.0]),  # up
                    2: np.array([0.0, -1.0]),  # down
                    3: np.array([-1.0, 0.0]),  # left
                    4: np.array([1.0, 0.0]),  # right
                }
                action = action_map.get(action, np.array([0.0, 0.0]))

            # Physics
            self.rescuer_vel[i] = self.rescuer_vel[i] * 0.8 + action * 0.1
            speed = np.linalg.norm(self.rescuer_vel[i])
            if speed > 0.08:
                self.rescuer_vel[i] = (self.rescuer_vel[i] / speed) * 0.08

            self.rescuer_pos[i] += self.rescuer_vel[i]

            # --- Wall collision handling (reflect and damp) ---
            # If we cross the world bounds, clamp position to boundary and
            # reflect the corresponding velocity component with damping.
            for axis in range(2):
                if self.rescuer_pos[i][axis] > 1.0:
                    self.rescuer_pos[i][axis] = 1.0
                    # invert normal component (axis) and damp
                    self.rescuer_vel[i][axis] *= -0.5
                elif self.rescuer_pos[i][axis] < -1.0:
                    self.rescuer_pos[i][axis] = -1.0
                    self.rescuer_vel[i][axis] *= -0.5

            # Tree Collision (reflect away from tree center and damp)
            for t_pos in self.tree_pos:
                to_tree = self.rescuer_pos[i] - t_pos
                dist = np.linalg.norm(to_tree)
                min_dist = self.agent_size + self.tree_radius
                if dist < min_dist:
                    # Track collision metric
                    self._collision_events += 1
                    # Tree collision penalty
                    rewards[agent] -= 1
                    # Compute penetration depth and normal
                    if dist > 1e-6:
                        n = to_tree / dist
                    else:
                        # Degenerate case: pick any normal (e.g., x-axis)
                        n = np.array([1.0, 0.0], dtype=float)
                    # Push the agent to the surface of the tree (no interpenetration)
                    self.rescuer_pos[i] = t_pos + n * min_dist
                    # Reflect velocity about the normal with damping
                    v = self.rescuer_vel[i]
                    vn = np.dot(v, n)
                    self.rescuer_vel[i] = (
                        v - (1.5 * vn) * n
                    )  # reflect and damp (~0.5 after reflection)

        # Agent-Agent Collision Physics (soft repulsion to prevent clustering)
        agent_repulsion_radius = 0.15
        repulsion_strength = 0.005
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                to_other = self.rescuer_pos[i] - self.rescuer_pos[j]
                dist = np.linalg.norm(to_other)

                if dist < agent_repulsion_radius and dist > 1e-6:
                    # Track collision between agents (count once per pair)
                    self._collision_events += 1
                    # Apply soft repulsion force inversely proportional to distance
                    repulsion_force = (
                        repulsion_strength * (agent_repulsion_radius - dist) / dist
                    )
                    direction = to_other / dist

                    # Apply equal and opposite forces
                    self.rescuer_vel[i] += direction * repulsion_force
                    self.rescuer_vel[j] -= direction * repulsion_force

        # 2. Victim Dynamics with Commitment System
        # Victims commit to an agent when approached, and maintain commitment
        # until the agent leaves or another agent is significantly closer
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                self.victim_vel[v_i] = 0
                continue

            # Find closest agent and their distance
            min_dist = float("inf")
            closest_agent_idx = -1
            for a_i, a_pos in enumerate(self.rescuer_pos):
                dist = np.linalg.norm(a_pos - self.victim_pos[v_i])
                if dist < min_dist:
                    min_dist = dist
                    closest_agent_idx = a_i

            current_assignment = self.victim_assignments[v_i]

            # Commitment logic with hysteresis
            if current_assignment == -1:
                # Not assigned - assign if agent is close enough
                if min_dist < self.follow_radius:
                    self.victim_assignments[v_i] = closest_agent_idx
                    current_assignment = closest_agent_idx
            else:
                # Already assigned - check if we should switch or release
                assigned_agent_pos = self.rescuer_pos[current_assignment]
                dist_to_assigned = np.linalg.norm(
                    assigned_agent_pos - self.victim_pos[v_i]
                )

                # Release if assigned agent is too far (1.5x follow radius = hysteresis)
                if dist_to_assigned > self.follow_radius * 1.5:
                    self.victim_assignments[v_i] = -1
                    current_assignment = -1
                    # Reassign if another agent is close
                    if min_dist < self.follow_radius:
                        self.victim_assignments[v_i] = closest_agent_idx
                        current_assignment = closest_agent_idx
                # Switch only if another agent is significantly closer (0.6x distance)
                elif (
                    closest_agent_idx != current_assignment
                    and min_dist < dist_to_assigned * 0.6
                ):
                    self.victim_assignments[v_i] = closest_agent_idx
                    current_assignment = closest_agent_idx

            # Movement based on assignment
            if current_assignment != -1:
                # Follow assigned agent
                assigned_pos = self.rescuer_pos[current_assignment]
                direction = (assigned_pos - self.victim_pos[v_i]) / (
                    np.linalg.norm(assigned_pos - self.victim_pos[v_i]) + 1e-6
                )
                follow_force = 0.02
                self.victim_vel[v_i] = (
                    self.victim_vel[v_i] * 0.8 + direction * follow_force
                )
            else:
                # Simple Brownian motion (using seeded RNG)
                noise = self.np_random.randn(2) * 0.0075
                self.victim_vel[v_i] = self.victim_vel[v_i] * 0.8 + noise

            self.victim_pos[v_i] += self.victim_vel[v_i]
            self.victim_pos[v_i] = np.clip(self.victim_pos[v_i], -1, 1)

        # 3. Logic: Rescues & Rewards
        saved_count = self._compute_rewards(rewards)

        # Track coverage each step (after physics updates)
        for i, agent in enumerate(self.agents):
            self._visited_cells[i].add(self._hash_pos(self.rescuer_pos[i].copy()))

        self.steps += 1

        # Termination conditions
        if saved_count == self.num_victims:
            terminations = {a: True for a in self.agents}
            self.agents = []  # PettingZoo requires emptying agents list on termination
        elif self.steps >= self.max_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

        if not self.agents:
            # Episode ended; aggregate metrics
            rescues_pct = (
                100.0
                * float(np.count_nonzero(self.victim_saved))
                / max(1, self.num_victims)
            )
            coverage_cells = len(set().union(*self._visited_cells))
            collisions = self._collision_events
            self._completed_episode_metrics.append(
                {
                    "episode": self._episode_counter,
                    "rescues_pct": rescues_pct,
                    "collisions": collisions,
                    "coverage_cells": coverage_cells,
                }
            )
            self._reset_metrics()

        return self._get_obs(), rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
            # Font for A/B/C/D labels
            self.font = pygame.font.SysFont("Arial", 18, bold=True)

        self.screen.fill((30, 30, 30))

        def to_screen(pos):
            x = (pos[0] + 1) / 2 * 600
            y = (1 - (pos[1] + 1) / 2) * 600
            return int(x), int(y)

        # Draw Safe Zones
        for i, pos in enumerate(self.safezone_pos):
            s_pos = to_screen(pos)
            r = int(self.safe_zone_radius * 300)

            type_idx = self.safe_zone_types[i]
            color = self.type_colors[type_idx]

            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, 50), (r, r), r)
            self.screen.blit(s, (s_pos[0] - r, s_pos[1] - r))
            pygame.draw.circle(self.screen, color, s_pos, r, 2)

        # Draw Trees
        for pos in self.tree_pos:
            pygame.draw.circle(
                self.screen,
                (100, 100, 100),
                to_screen(pos),
                int(self.tree_radius * 300),
            )

        # Draw Victims
        for i, pos in enumerate(self.victim_pos):
            if self.victim_saved[i]:
                continue

            type_idx = self.victim_types[i]
            color = self.type_colors[type_idx]

            screen_pos = to_screen(pos)
            pygame.draw.circle(
                self.screen, color, screen_pos, int(self.agent_size * 300)
            )

        # Draw Rescuers
        for i, pos in enumerate(self.rescuer_pos):
            pygame.draw.circle(
                self.screen, (200, 200, 200), to_screen(pos), int(self.agent_size * 300)
            )
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                to_screen(pos),
                int(self.agent_size * 300),
                2,
            )

        # Draw Vision Circles
        for i, pos in enumerate(self.rescuer_pos):
            screen_pos = to_screen(pos)
            vision_r = int(self.vision_radius * 300)
            pygame.draw.circle(
                self.screen, (255, 255, 255, 100), screen_pos, vision_r, 2
            )

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen:
            pygame.quit()

    def _get_matching_zone_idx(self, victim_type: int) -> Optional[int]:
        """Find the safe zone index that accepts the given victim type."""
        for zone_idx, zone_type in enumerate(self.safe_zone_types):
            if zone_type == victim_type:
                return zone_idx
        return None

    def _compute_agent_victim_dists(self) -> list[float]:
        """Compute min distance from each agent to nearest unsaved victim."""
        dists = []
        unsaved_indices = [k for k, saved in enumerate(self.victim_saved) if not saved]
        for i in range(self.num_rescuers):
            if unsaved_indices:
                min_dist = min(
                    np.linalg.norm(self.rescuer_pos[i] - self.victim_pos[k])
                    for k in unsaved_indices
                )
            else:
                min_dist = 0.0
            dists.append(min_dist)
        return dists

    def _compute_rewards(self, rewards) -> int:
        # Delta-based distance shaping (reward for getting closer, penalty for moving away)
        current_dists = self._compute_agent_victim_dists()
        for i, agent in enumerate(self.agents):
            if (
                hasattr(self, "prev_agent_victim_dists")
                and self.prev_agent_victim_dists[i] > 0
            ):
                delta = (
                    self.prev_agent_victim_dists[i] - current_dists[i]
                )  # positive = got closer
                rewards[agent] += delta * 0.1
        self.prev_agent_victim_dists = current_dists

        # Check safe zones
        saved_count = 0
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                saved_count += 1
                continue

            v_pos = self.victim_pos[v_i]
            v_type = self.victim_types[v_i]

            # Find safe zone with matching type (not index!)
            target_zone_idx = self._get_matching_zone_idx(v_type)
            if target_zone_idx is None:
                # Should not happen if num_safe_zones >= max victim types
                continue

            target_zone_pos = self.safezone_pos[target_zone_idx]
            dist_to_zone = np.linalg.norm(v_pos - target_zone_pos)

            # Victim got saved
            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1

                # Reward ONLY the agent that was assigned to (escorting) this victim
                assigned_agent_idx = self.victim_assignments[v_i]
                if assigned_agent_idx != -1 and assigned_agent_idx < len(self.agents):
                    # Reward only the assigned agent who did the work
                    rewards[self.agents[assigned_agent_idx]] += 100.0

        # Individual credit: reward assigned agent for escorting victim toward safe zone
        for v_i in range(self.num_victims):
            if not self.victim_saved[v_i]:
                assigned_agent_idx = self.victim_assignments[v_i]

                # Only reward the assigned agent (stronger signal for credit assignment)
                if assigned_agent_idx != -1 and assigned_agent_idx < len(self.agents):
                    agent = self.agents[assigned_agent_idx]

                    # Find the correct safe zone by matching type
                    v_type = self.victim_types[v_i]
                    target_zone_idx = self._get_matching_zone_idx(v_type)

                    if target_zone_idx is not None:
                        dist_to_zone = np.linalg.norm(
                            self.victim_pos[v_i] - self.safezone_pos[target_zone_idx]
                        )
                        # Reward proportional to inverse distance (closer to goal = higher reward)
                        rewards[agent] += 1.0 / (dist_to_zone + 1e-6)

        # Boundary penalties
        for i, agent in enumerate(self.agents):
            pos = self.rescuer_pos[i]
            if abs(pos[0]) > 0.95 or abs(pos[1]) > 0.95:
                rewards[agent] -= 1

        # Agent collision penalty (physical repulsion is now handled earlier)
        num_agents = len(self.agents)
        if num_agents > 1:
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(self.rescuer_pos[i] - self.rescuer_pos[j])
                    if dist < 0.15:  # Agents are overlapping/colliding
                        # Stronger penalty to discourage clustering (increased from -1.0)
                        rewards[self.agents[i]] -= 5.0
                        rewards[self.agents[j]] -= 5.0

        return saved_count


def make_env(device: Union[torch.device, str] = "cpu", **kwargs) -> TransformedEnv:
    env = SearchAndRescueEnv(**kwargs)
    group_map = {"agents": env.possible_agents}
    env = PettingZooWrapper(env, group_map=group_map, use_mask=True, device=device)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env.to(device)
