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
        continuous_actions: bool = True,
        vision_radius: float = 0.5,
        randomize_safe_zones: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_trees: Optional[
            int
        ] = None,  # For curriculum learning - fixes obs space size
    ):
        self.num_rescuers = num_rescuers
        self.num_victims = num_victims
        self.num_trees = num_trees
        self.num_safe_zones = num_safe_zones
        self.render_mode = render_mode
        self.max_steps = max_cycles
        self.is_continuous = continuous_actions
        self.randomize_safe_zones = randomize_safe_zones

        # For curriculum learning: use max_trees for obs space size, pad observations
        self.max_trees = max_trees if max_trees is not None else num_trees

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
        # Layout per agent:
        # [self_vel(2), self_pos(2), agent_id(num_rescuers),
        #  safe_zones(max_safe_zones * 3: rel_x, rel_y, type_idx or -1 if masked),
        #  trees(max_trees * 3: rel_x, rel_y, visible_bit),
        #  victims(num_victims * 3: rel_x, rel_y, type_idx or -1 if masked)]
        self.max_safe_zones = self.num_safe_zones  # kept for clarity

        self.obs_dim = (
            2  # self_vel
            + 2  # self_pos
            + self.num_rescuers  # agent_id one-hot
            + (self.max_safe_zones * 3)
            + (self.max_trees * 3)
            + (self.num_victims * 3)
        )

        # Build bounded observation space
        rel_low, rel_high = -3.0, 3.0  # generous bound for relative deltas
        obs_low = []
        obs_high = []

        # Self velocity (2)
        obs_low.extend([-0.1, -0.1])
        obs_high.extend([0.1, 0.1])

        # Self position (2)
        obs_low.extend([-1.0, -1.0])
        obs_high.extend([1.0, 1.0])

        # Agent ID one-hot
        obs_low.extend([0.0] * self.num_rescuers)
        obs_high.extend([1.0] * self.num_rescuers)

        # Safe zones: rel_x, rel_y, type (or -1 when masked)
        for _ in range(self.max_safe_zones):
            obs_low.extend([rel_low, rel_low, -1.0])
            obs_high.extend([rel_high, rel_high, float(self.num_safe_zones - 1)])

        # Trees: rel_x, rel_y (masked to 0), visible bit
        for _ in range(self.max_trees):
            obs_low.extend([rel_low, rel_low, 0.0])
            obs_high.extend([rel_high, rel_high, 1.0])

        # Victims: rel_x, rel_y, type (or -1 when masked)
        for _ in range(self.num_victims):
            obs_low.extend([rel_low, rel_low, -1.0])
            obs_high.extend([rel_high, rel_high, float(self.num_safe_zones - 1)])

        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array(obs_low, dtype=np.float32),
                high=np.array(obs_high, dtype=np.float32),
                dtype=np.float32,
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

            # 3. Safe Zones (always visible landmarks)
            for sz_i in range(self.num_safe_zones):
                rel_pos = self.safezone_pos[sz_i] - my_pos
                obs_vec.extend(
                    [rel_pos[0], rel_pos[1], float(self.safe_zone_types[sz_i])]
                )

            # 4. Trees (partial observability): mask rel_pos when not visible
            for t_i in range(self.max_trees):
                if t_i < self.num_trees:
                    visible = float(
                        self._is_visible(
                            my_pos,
                            self.tree_pos[t_i],
                            self.tree_radius,
                            exclude_tree_idx=t_i,
                        )
                    )
                    if visible > 0.5:
                        rel_pos = self.tree_pos[t_i] - my_pos
                        obs_vec.extend([rel_pos[0], rel_pos[1], 1.0])
                    else:
                        obs_vec.extend([0.0, 0.0, 0.0])
                else:
                    obs_vec.extend([0.0, 0.0, 0.0])  # unused slot

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

            # Inward push if we’re hugging the boundary to avoid sticking
            margin = 0.85
            inward_k = 0.02
            if self.rescuer_pos[i][0] > margin:
                self.rescuer_vel[i][0] -= inward_k
            elif self.rescuer_pos[i][0] < -margin:
                self.rescuer_vel[i][0] += inward_k
            if self.rescuer_pos[i][1] > margin:
                self.rescuer_vel[i][1] -= inward_k
            elif self.rescuer_pos[i][1] < -margin:
                self.rescuer_vel[i][1] += inward_k

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
                follow_force = 0.03
                self.victim_vel[v_i] = (
                    self.victim_vel[v_i] * 0.8 + direction * follow_force
                )
            else:
                # Simple Brownian motion (using seeded RNG)
                noise = self.np_random.randn(2) * 0.0075
                self.victim_vel[v_i] = self.victim_vel[v_i] * 0.8 + noise

            self.victim_pos[v_i] += self.victim_vel[v_i]
            self.victim_pos[v_i] = np.clip(self.victim_pos[v_i], -1, 1)

        # Gentle exploration if an agent sees no victims (breaks idling)
        for i, agent in enumerate(self.agents):
            sees_victim = any(
                not self.victim_saved[v_i]
                and self._is_visible(
                    self.rescuer_pos[i], self.victim_pos[v_i], self.agent_size
                )
                for v_i in range(self.num_victims)
            )
            if (not sees_victim) and np.linalg.norm(self.rescuer_vel[i]) < 0.01:
                self.rescuer_vel[i] += self.np_random.uniform(-0.02, 0.02, size=2)

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

    def set_num_trees(self, num_trees: int) -> None:
        """Update the active number of trees (used by curriculum)."""
        self.num_trees = int(np.clip(num_trees, 0, self.max_trees))
        # If reducing trees mid-run, truncate positions to keep arrays in sync
        if hasattr(self, "tree_pos"):
            self.tree_pos = self.tree_pos[: self.num_trees]

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

    def _bounded_zone_shaping(self, dist_to_zone: float) -> float:
        """
        Bounded shaping for escorting a victim to its safe zone.
        Returns a value in [0, 1], higher when closer.
        """
        # Smooth, bounded, stable (no explosion near 0)
        scale = 0.5  # adjust: smaller => steeper near zone
        return float(np.exp(-dist_to_zone / scale))

    def _nearest_unassigned_victim_dist(self, agent_idx: int) -> float:
        """
        Distance from an agent to the nearest *unassigned* and *unsaved* victim.
        If none exist, returns 0.0
        """
        best = float("inf")
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                continue
            if self.victim_assignments[v_i] != -1:
                continue  # already assigned to someone
            d = np.linalg.norm(self.rescuer_pos[agent_idx] - self.victim_pos[v_i])
            if d < best:
                best = d
        return 0.0 if best == float("inf") else float(best)

    def _compute_rewards(self, rewards) -> int:
        """
        Assignment-aware reward shaping:
        - If an agent is escorting (has an assigned victim), shape progress to the correct safe zone.
        - If not escorting, shape movement toward the nearest *unassigned* victim.
        Also keeps sparse success reward (+100) for saving, plus collision/boundary/idle penalties.
        """

        # -----------------------------
        # 1) PICKUP shaping (unassigned agents -> nearest unassigned victim)
        # -----------------------------
        # Track previous distances for delta shaping (per agent) for pickup mode
        if not hasattr(self, "prev_agent_pickup_dists"):
            self.prev_agent_pickup_dists = [0.0 for _ in range(self.num_rescuers)]

        # Determine which agents are currently escorting at least one victim
        agent_is_escorting = [False for _ in range(self.num_rescuers)]
        for v_i in range(self.num_victims):
            a = int(self.victim_assignments[v_i])
            if a != -1 and not self.victim_saved[v_i] and 0 <= a < self.num_rescuers:
                agent_is_escorting[a] = True

        # Pickup shaping for non-escorting agents only
        for i, agent in enumerate(self.agents):
            if agent_is_escorting[i]:
                continue  # escort shaping happens below

            d = self._nearest_unassigned_victim_dist(i)

            # Distance penalty (encourage getting close to a victim to start follow/commit)
            rewards[agent] -= 0.1 * d

            # Delta shaping (reward improvement)
            prev = self.prev_agent_pickup_dists[i]
            if prev > 0.0:
                rewards[agent] += 0.2 * (prev - d)  # positive if closer
            self.prev_agent_pickup_dists[i] = d

        # -----------------------------
        # 2) SAVE events (sparse success reward +100)
        # -----------------------------
        saved_count = 0
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                saved_count += 1
                continue

            v_pos = self.victim_pos[v_i]
            v_type = self.victim_types[v_i]

            target_zone_idx = self._get_matching_zone_idx(v_type)
            if target_zone_idx is None:
                continue

            target_zone_pos = self.safezone_pos[target_zone_idx]
            dist_to_zone = float(np.linalg.norm(v_pos - target_zone_pos))

            # Victim saved
            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1

                assigned_agent_idx = int(self.victim_assignments[v_i])
                if 0 <= assigned_agent_idx < len(self.agents):
                    rewards[self.agents[assigned_agent_idx]] += 100.0

        # -----------------------------
        # 3) ESCORT shaping (assigned agent -> bring victim closer to its zone)
        # -----------------------------
        # Track previous zone distances for delta shaping (per victim)
        if not hasattr(self, "prev_victim_zone_dists"):
            self.prev_victim_zone_dists = [None for _ in range(self.num_victims)]

        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                continue

            assigned_agent_idx = int(self.victim_assignments[v_i])
            if assigned_agent_idx == -1 or assigned_agent_idx >= len(self.agents):
                continue

            agent = self.agents[assigned_agent_idx]

            # Find correct safe zone by type
            v_type = self.victim_types[v_i]
            target_zone_idx = self._get_matching_zone_idx(v_type)
            if target_zone_idx is None:
                continue

            dist_to_zone = float(
                np.linalg.norm(
                    self.victim_pos[v_i] - self.safezone_pos[target_zone_idx]
                )
            )

            # Bounded dense shaping: in [0,1]
            # Encourages staying closer to the target zone (stable, no blow-ups)
            shaped = self._bounded_zone_shaping(dist_to_zone)
            rewards[agent] += 1.0 * shaped  # weight = 1.0, tune if needed

            # Delta shaping: reward reduction in distance (progress)
            prev = self.prev_victim_zone_dists[v_i]
            if prev is not None:
                rewards[agent] += 0.5 * (prev - dist_to_zone)  # positive if closer
            self.prev_victim_zone_dists[v_i] = dist_to_zone

        # -----------------------------
        # 4) Boundary penalties
        # -----------------------------
        for i, agent in enumerate(self.agents):
            pos = self.rescuer_pos[i]
            if abs(pos[0]) > 0.95 or abs(pos[1]) > 0.95:
                rewards[agent] -= 0.2  # softened from -1

        # -----------------------------
        # 5) Agent collision penalty (discourage clustering)
        # -----------------------------
        num_agents = len(self.agents)
        if num_agents > 1:
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(self.rescuer_pos[i] - self.rescuer_pos[j])
                    if dist < 0.15:
                        rewards[self.agents[i]] -= 1.0  # softened from -5
                        rewards[self.agents[j]] -= 1.0

        # -----------------------------
        # 6) Small penalty for idling
        # -----------------------------
        for i, agent in enumerate(self.agents):
            if np.linalg.norm(self.rescuer_vel[i]) < 1e-3:
                rewards[agent] -= 0.01

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
