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

        # --- Type Mapping (A..D) ---
        # Internally we use ints (0-3) for the Network, but strings for Humans
        self.type_map = {0: "A", 1: "B", 2: "C", 3: "D"}
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
        # [Self_Vel(2), Self_Pos(2),
        #  SafeZones(4 * 3: rel_x, rel_y, type_idx),
        #  Trees(N * 2: rel_x, rel_y),
        #  Victims(N * 3: rel_x, rel_y, type_idx)]
        # Note: Other rescuers removed from observation to focus learning on task

        self.obs_dim = (
            4
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.steps = 0
        if seed is not None:
            np.random.seed(seed)

        # Reset Agents list (required by PettingZoo API)
        self.agents = self.possible_agents[:]

        # Positions: Rescuers, Victims, Trees, SafeZones
        self.rescuer_pos = np.random.uniform(-0.8, 0.8, (self.num_rescuers, 2))
        self.rescuer_vel = np.zeros((self.num_rescuers, 2))

        self.victim_pos = np.random.uniform(-0.8, 0.8, (self.num_victims, 2))
        self.victim_vel = np.zeros((self.num_victims, 2))
        self.victim_saved = np.zeros(self.num_victims, dtype=bool)

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

            # 2. Safe Zones (Global knowledge assumed for static zones)
            for sz_i in range(self.num_safe_zones):
                rel_pos = self.safezone_pos[sz_i] - my_pos
                # We append the numeric type (0-3) so the network knows which zone is which
                obs_vec.extend(
                    [rel_pos[0], rel_pos[1], float(self.safe_zone_types[sz_i])]
                )

            # 3. Trees
            for t_i in range(self.num_trees):
                if self._is_visible(my_pos, self.tree_pos[t_i], self.tree_radius):
                    obs_vec.extend(self.tree_pos[t_i] - my_pos)
                else:
                    obs_vec.extend([0.0, 0.0])  # Masked

            # 4. Victims
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

            # prev_pos = self.rescuer_pos[i].copy()
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

        # 2. Victim Dynamics
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                self.victim_vel[v_i] = 0
                continue

            # Check for closest agent within follow radius
            min_dist = float("inf")
            closest_agent_pos = None
            for a_pos in self.rescuer_pos:
                dist = np.linalg.norm(a_pos - self.victim_pos[v_i])
                if dist < min_dist:
                    min_dist = dist
                    closest_agent_pos = a_pos

            if min_dist < self.follow_radius:
                # Follow the closest agent
                direction = (closest_agent_pos - self.victim_pos[v_i]) / (
                    min_dist + 1e-6
                )
                follow_force = 0.02  # Adjust speed towards agent
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

        self.steps += 1

        # Termination conditions
        if saved_count == self.num_victims:
            terminations = {a: True for a in self.agents}
            self.agents = []  # PettingZoo requires emptying agents list on termination
        elif self.steps >= self.max_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

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
            label = self.type_map[type_idx]  # "A", "B", etc.

            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            s.fill((*color, 50))
            self.screen.blit(s, (s_pos[0] - r, s_pos[1] - r))
            pygame.draw.rect(
                self.screen, color, (s_pos[0] - r, s_pos[1] - r, r * 2, r * 2), 2
            )

            # Draw Label
            text = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(text, (s_pos[0] - 10, s_pos[1] - 10))

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

    def _compute_agent_victim_dists(self):
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
                rewards[agent] += delta * 0.5  # 5x magnitude increase (was 0.1)
        self.prev_agent_victim_dists = current_dists

        # Check safe zones
        saved_count = 0
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                saved_count += 1
                continue

            v_pos = self.victim_pos[v_i]
            v_type_idx = self.victim_types[v_i]  # Int 0-3

            # Find pos of safe zone with matching type
            # Assuming safe_zone_types is [0, 1, 2, 3] aligned with safezone_pos indices
            target_zone_pos = self.safezone_pos[v_type_idx]

            dist_to_zone = np.linalg.norm(v_pos - target_zone_pos)

            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1
                # Big Reward to the closest agent
                min_dist = float("inf")
                closest_agent = None
                for j, a in enumerate(self.agents):
                    dist = np.linalg.norm(self.rescuer_pos[j] - v_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_agent = a
                if closest_agent is not None:
                    rewards[closest_agent] += 100.0

        # Individual credit: reward agent for leading victim toward safe zone
        for v_i in range(self.num_victims):
            if not self.victim_saved[v_i]:
                # Find which agent (if any) is following this victim
                for i, agent in enumerate(self.agents):
                    dist_to_victim = np.linalg.norm(
                        self.rescuer_pos[i] - self.victim_pos[v_i]
                    )
                    if dist_to_victim < self.follow_radius:
                        dist_to_zone = np.linalg.norm(
                            self.victim_pos[v_i]
                            - self.safezone_pos[self.victim_types[v_i]]
                        )
                        # 5x magnitude: 0.5 instead of 0.1
                        rewards[agent] += 0.5 / (dist_to_zone + 1e-6)

        # Boundary penalties
        for i, agent in enumerate(self.agents):
            pos = self.rescuer_pos[i]
            if abs(pos[0]) > 0.8 or abs(pos[1]) > 0.8:
                rewards[agent] -= 1

        # Agent collision penalty - penalize agents that are too close to each other
        num_agents = len(self.agents)
        if num_agents > 1:
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(self.rescuer_pos[i] - self.rescuer_pos[j])
                    if dist < 0.15:  # Agents are overlapping/colliding
                        # Strong penalty to both agents
                        rewards[self.agents[i]] -= 1.0
                        rewards[self.agents[j]] -= 1.0

        # Victim assignment bonus - reward agents for targeting different victims
        unsaved_indices = [k for k, saved in enumerate(self.victim_saved) if not saved]
        if num_agents > 1 and len(unsaved_indices) > 1:
            # Find which victim each agent is closest to
            agent_targets = []
            for i in range(num_agents):
                min_dist = float("inf")
                closest_victim = None
                for v_i in unsaved_indices:
                    dist = np.linalg.norm(self.rescuer_pos[i] - self.victim_pos[v_i])
                    if dist < min_dist:
                        min_dist = dist
                        closest_victim = v_i
                agent_targets.append(closest_victim)

            # Count unique victims being targeted
            unique_targets = len(set(agent_targets))
            # Bonus scales with task division (max when all agents target different victims)
            # 5x magnitude: 0.25 instead of 0.05
            if unique_targets > 1:
                division_bonus = (
                    0.25
                    * (unique_targets - 1)
                    / (min(num_agents, len(unsaved_indices)) - 1)
                )
                for a in self.agents:
                    rewards[a] += division_bonus

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
