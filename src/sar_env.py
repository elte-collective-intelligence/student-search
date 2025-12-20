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
        self.seed = seed

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
        self.victim_types = [i % 4 for i in range(num_victims)]
        self.safe_zone_types = [0, 1, 2, 3]  # TL, TR, BL, BR

        # Colors for rendering
        self.type_colors = {
            0: (255, 50, 50),  # Red (A)
            1: (50, 255, 50),  # Green (B)
            2: (50, 50, 255),  # Blue (C)
            3: (255, 255, 50),  # Yellow (D)
        }

        # Action Space: Continuous acceleration (dx, dy)
        self.action_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }

        # Observation Space Calculation
        # [Self_Vel(2), Self_Pos(2), Agent_ID(num_rescuers),
        #  SafeZones(4 * 3: rel_x, rel_y, type_idx),
        #  Trees(N * 2: rel_x, rel_y),
        #  Victims(N * 3: rel_x, rel_y, type_idx)]
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
        if seed is not None:
            np.random.seed(seed)

        # Reset Agents list (required by PettingZoo API)
        self.agents = self.possible_agents[:]

        # Metrics state (reset each episode)
        self._episode_counter += 1
        self._reset_metrics()

        # Positions: Rescuers, Victims, Trees, SafeZones
        self.rescuer_pos = np.random.uniform(-0.8, 0.8, (self.num_rescuers, 2))
        self.rescuer_vel = np.zeros((self.num_rescuers, 2))

        self.victim_pos = np.random.uniform(-0.8, 0.8, (self.num_victims, 2))
        self.victim_vel = np.zeros((self.num_victims, 2))
        self.victim_saved = np.zeros(self.num_victims, dtype=bool)

        # Track which agent each victim is committed to (-1 = none)
        self.victim_assignments = np.full(self.num_victims, -1, dtype=int)

        self.tree_pos = np.random.uniform(-0.8, 0.8, (self.num_trees, 2))

        # Safe zones at corners
        self.safezone_pos = np.array(
            [[-0.9, 0.9], [0.9, 0.9], [-0.9, -0.9], [0.9, -0.9]]
        )

        # Track previous distances for delta-based shaping
        self.prev_agent_victim_dists = self._compute_agent_victim_dists()

        return self._get_obs(), {a: {} for a in self.agents}

    def _is_visible(self, observer_pos, target_pos, target_radius):
        """Checks distance and occlusion by trees."""
        dist = np.linalg.norm(target_pos - observer_pos)
        if dist > self.vision_radius:
            return False

        # Check line of sight against all trees
        for t_idx in range(self.num_trees):
            tree_c = self.tree_pos[t_idx]

            if (tree_c == target_pos).all():
                # Tree is most likely itself, and we don't want it to block
                return True

            # Vector from observer to target
            d_vec = target_pos - observer_pos
            # Vector from observer to tree center
            f_vec = observer_pos - tree_c

            a = np.dot(d_vec, d_vec)
            b = 2 * np.dot(f_vec, d_vec)
            c = np.dot(f_vec, f_vec) - self.tree_radius**2

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a + 1e-6)
                t2 = (-b + discriminant) / (2 * a + 1e-6)

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

            # 3. Safe Zones (Global knowledge assumed for static zones)
            for sz_i in range(self.num_safe_zones):
                rel_pos = self.safezone_pos[sz_i] - my_pos
                # We append the numeric type (0-3) so the network knows which zone is which
                obs_vec.extend(
                    [rel_pos[0], rel_pos[1], float(self.safe_zone_types[sz_i])]
                )

            # 4. Trees
            for t_i in range(self.num_trees):
                if self._is_visible(my_pos, self.tree_pos[t_i], self.tree_radius):
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
                # Simple Brownian motion
                noise = np.random.randn(2) * 0.0075
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
            s.fill((*color, 50))
            self.screen.blit(s, (s_pos[0] - r, s_pos[1] - r))
            pygame.draw.rect(
                self.screen, color, (s_pos[0] - r, s_pos[1] - r, r * 2, r * 2), 2
            )

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
            v_type_idx = self.victim_types[v_i]

            # Find pos of safe zone with matching type
            target_zone_pos = self.safezone_pos[v_type_idx]

            dist_to_zone = np.linalg.norm(v_pos - target_zone_pos)

            # Victim got saved
            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1

                # Reward the assigned agent (who escorted the victim)
                # If no assignment, fall back to closest agent
                assigned_agent_idx = self.victim_assignments[v_i]
                if assigned_agent_idx != -1 and assigned_agent_idx < len(self.agents):
                    # Primary reward to assigned agent who did the work
                    rewards[self.agents[assigned_agent_idx]] += 100.0
                else:
                    # Fallback: reward closest agent
                    min_dist = float("inf")
                    closest_agent = None
                    for j, a in enumerate(self.agents):
                        dist = np.linalg.norm(self.rescuer_pos[j] - v_pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_agent = a
                    if closest_agent is not None:
                        rewards[closest_agent] += 100.0

                # Small bonus to nearby assisting agents (within follow radius)
                for j, a in enumerate(self.agents):
                    if j != assigned_agent_idx:
                        dist = np.linalg.norm(self.rescuer_pos[j] - v_pos)
                        if dist < self.follow_radius:
                            rewards[a] += 10.0  # 10% bonus for assisting

        # Individual credit: reward assigned agent for escorting victim toward safe zone
        for v_i in range(self.num_victims):
            if not self.victim_saved[v_i]:
                assigned_agent_idx = self.victim_assignments[v_i]

                # Only reward the assigned agent (stronger signal for credit assignment)
                if assigned_agent_idx != -1 and assigned_agent_idx < len(self.agents):
                    agent = self.agents[assigned_agent_idx]
                    dist_to_zone = np.linalg.norm(
                        self.victim_pos[v_i] - self.safezone_pos[self.victim_types[v_i]]
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
    env = PettingZooWrapper(env, group_map=group_map, use_mask=True)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env.to(device)
