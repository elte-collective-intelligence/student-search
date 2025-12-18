# noqa: D212, D415
from typing import Union, Optional

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
        render_mode=None,
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

        self.agents = [f"rescuer_{i}" for i in range(num_rescuers)]
        self.possible_agents = self.agents[:]
        self.victim_names = [f"victim_{i}" for i in range(num_victims)]

        # Types: 0:A, 1:B, 2:C, 3:D
        self.victim_types = [i % 4 for i in range(num_victims)]
        self.safe_zone_types = [0, 1, 2, 3]  # TL, TR, BL, BR

        # Colors for rendering
        self.type_colors = {
            0: (255, 50, 50),  # Red
            1: (50, 255, 50),  # Green
            2: (50, 50, 255),  # Blue
            3: (255, 255, 50),  # Yellow
        }

        # Action Space: Continuous acceleration (dx, dy)
        self.action_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            for agent in self.agents
        }

        # Observation Space:
        # [Self_Vel(2), Self_Pos(2),
        #  SafeZones(4*3 relative_pos+type),
        #  Trees(num_trees*2 relative_pos),
        #  Victims(num_victims*3 relative_pos+type),
        #  Other_Rescuers((N-1)*2 relative_pos)]
        # Entities are masked (set to 0) if occluded/out of range.

        self.obs_dim = (
            4
            + (self.num_safe_zones * 3)
            + (self.num_trees * 2)
            + (self.num_victims * 3)
            + ((self.num_rescuers - 1) * 2)
        )
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            for agent in self.agents
        }

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.steps = 0
        np.random.seed(seed)

        # --- Init State ---
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

        self.agents = self.possible_agents[:]
        return self._get_obs(), {a: {} for a in self.agents}

    def _is_visible(self, observer_pos, target_pos, target_radius):
        """Checks distance and occlusion by trees."""
        dist = np.linalg.norm(target_pos - observer_pos)
        if dist > self.vision_radius:
            return False

        # Line Segment Intersection with Tree Circles
        # Segment from A to B. Circle at C with radius R.
        for t_idx in range(self.num_trees):
            tree_c = self.tree_pos[t_idx]

            # Vector math to find closest point on segment to circle center
            d = target_pos - observer_pos
            f = observer_pos - tree_c

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.tree_radius**2  # Use tree radius for occlusion

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                # Potential intersection, check if it's within the segment
                discriminant = np.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a + 1e-6)
                t2 = (-b + discriminant) / (2 * a + 1e-6)

                if (0 <= t1 <= 1) or (0 <= t2 <= 1):
                    # One of the intersection points is on the segment
                    return False
        return True

    def _get_obs(self):
        observations = {}
        for i, agent in enumerate(self.agents):
            obs_vec = []
            my_pos = self.rescuer_pos[i]

            # 1. Self State (Vel, Pos) - Absolute pos usually kept for CTDE, relative for policy
            obs_vec.extend(self.rescuer_vel[i])
            obs_vec.extend(my_pos)

            # 2. Safe Zones (Always known/visible in this simplified task, or check vision)
            # Letting agents know where safe zones are globally helps navigation
            for sz_i in range(self.num_safe_zones):
                rel_pos = self.safezone_pos[sz_i] - my_pos
                # One-hot type or integer type? Float type for simplicity in single vector
                obs_vec.extend([rel_pos[0], rel_pos[1], self.safe_zone_types[sz_i]])

            # 3. Trees (Obstacles) - Subject to Vision
            for t_i in range(self.num_trees):
                if self._is_visible(my_pos, self.tree_pos[t_i], self.tree_radius):
                    obs_vec.extend(self.tree_pos[t_i] - my_pos)
                else:
                    obs_vec.extend([0.0, 0.0])  # Masked

            # 4. Victims - Subject to Vision
            for v_i in range(self.num_victims):
                if not self.victim_saved[v_i] and self._is_visible(
                    my_pos, self.victim_pos[v_i], self.agent_size
                ):
                    rel = self.victim_pos[v_i] - my_pos
                    obs_vec.extend([rel[0], rel[1], self.victim_types[v_i]])
                else:
                    obs_vec.extend([0.0, 0.0, -1.0])  # Masked

            # 5. Other Rescuers
            for j in range(self.num_rescuers):
                if i == j:
                    continue
                if self._is_visible(my_pos, self.rescuer_pos[j], self.agent_size):
                    obs_vec.extend(self.rescuer_pos[j] - my_pos)
                else:
                    obs_vec.extend([0.0, 0.0])

            observations[agent] = np.array(obs_vec, dtype=np.float32)
        return observations

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        # 1. Apply Actions (Rescuers)
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            # Simple physics: Apply force -> Velocity -> Position
            self.rescuer_vel[i] = self.rescuer_vel[i] * 0.9 + action * 0.1  # Damping

            # Clip velocity
            speed = np.linalg.norm(self.rescuer_vel[i])
            if speed > 0.1:
                self.rescuer_vel[i] = (self.rescuer_vel[i] / speed) * 0.1

            # Move
            prev_pos = self.rescuer_pos[i].copy()
            self.rescuer_pos[i] += self.rescuer_vel[i]

            # Bounds
            self.rescuer_pos[i] = np.clip(self.rescuer_pos[i], -1, 1)

            # Collision with Trees
            for t_pos in self.tree_pos:
                if np.linalg.norm(self.rescuer_pos[i] - t_pos) < (
                    self.agent_size + self.tree_radius
                ):
                    rewards[agent] -= 1.0  # Collision penalty
                    self.rescuer_pos[i] = prev_pos  # Bounce back
                    self.rescuer_vel[i] *= -0.5

        # 2. Victim Dynamics (Panic/Wander unless saved)
        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                self.victim_vel[v_i] = 0
                continue

            # Random brownian motion
            noise = np.random.randn(2) * 0.02
            self.victim_vel[v_i] = self.victim_vel[v_i] * 0.8 + noise
            self.victim_pos[v_i] += self.victim_vel[v_i]
            self.victim_pos[v_i] = np.clip(self.victim_pos[v_i], -1, 1)

        # 3. Check Rescues & Shaping
        saved_count = 0
        for i, agent in enumerate(self.agents):
            # Shaping: Distance to nearest unsaved victim
            dists = [
                np.linalg.norm(self.rescuer_pos[i] - self.victim_pos[v_i])
                for v_i in range(self.num_victims)
                if not self.victim_saved[v_i]
            ]
            if dists:
                rewards[agent] -= min(dists) * 0.1

        for v_i in range(self.num_victims):
            if self.victim_saved[v_i]:
                saved_count += 1
                continue

            # Check if in correct safe zone
            v_pos = self.victim_pos[v_i]
            v_type = self.victim_types[v_i]

            # Find matching zone
            target_zone_pos = self.safezone_pos[self.safe_zone_types.index(v_type)]
            dist_to_zone = np.linalg.norm(v_pos - target_zone_pos)

            if dist_to_zone < self.safe_zone_radius:
                self.victim_saved[v_i] = True
                saved_count += 1
                # Global Reward for team
                for a in self.agents:
                    rewards[a] += 100.0

        self.steps += 1
        if saved_count == self.num_victims:
            terminations = {a: True for a in self.agents}
        elif self.steps >= self.max_steps:
            truncations = {a: True for a in self.agents}

        return self._get_obs(), rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()

        self.screen.fill((30, 30, 30))  # Dark background

        def to_screen(pos):
            # map [-1, 1] to [0, 600]
            x = (pos[0] + 1) / 2 * 600
            y = (1 - (pos[1] + 1) / 2) * 600  # Flip Y
            return int(x), int(y)

        # Draw Safe Zones
        for i, pos in enumerate(self.safezone_pos):
            s_pos = to_screen(pos)
            r = int(self.safe_zone_radius * 300)
            color = self.type_colors[self.safe_zone_types[i]]
            # Draw semi-transparent square
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
                continue  # Don't draw if saved? Or draw ghost
            color = self.type_colors[self.victim_types[i]]
            pygame.draw.circle(
                self.screen, color, to_screen(pos), int(self.agent_size * 300)
            )
            # Draw 'V'
            # (Text rendering omitted for brevity, simple color circle is enough)

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

            # Draw Vision Radius (Debug)
            # pygame.draw.circle(self.screen, (50, 50, 50), to_screen(pos), int(self.vision_radius * 300), 1)

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen:
            pygame.quit()


def make_env(device: Union[torch.device, str] = "cpu", **kwargs) -> TransformedEnv:
    """Create the Search and Rescue environment wrapped in TorchRL format."""
    env = SearchAndRescueEnv(**kwargs)
    # Wrap standard PettingZoo env into TorchRL format
    env = PettingZooWrapper(env, use_mask=False)
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    return env.to(device)
