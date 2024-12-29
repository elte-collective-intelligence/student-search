# noqa: D212, D415


import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn



class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_missing=1,
        num_rescuers=3,
        num_trees=5,
        num_safezones = 4,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_missing=num_missing,
            num_rescuers=num_rescuers,
            num_trees=num_trees,
            num_safezones=num_safezones,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_missing, num_rescuers, num_trees, num_safezones)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "search_and_rescue_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


        


class Scenario(BaseScenario):
    def make_world(self, num_missing=1, num_rescuers=3, num_trees=5, num_safezones=4):
        victim_types = ["A", "B", "C", "D"]  # Example types for victims
        safe_zone_types = ["A", "B", "C", "D"]  # Matching types for safe zones
        type_colors = {
        "A": np.array([1.0, 0.0, 0.0]),  # Red
        "B": np.array([0.0, 1.0, 0.0]),  # Green
        "C": np.array([0.0, 0.0, 1.0]),  # Blue
        "D": np.array([1.0, 1.0, 0.0])   # Yellow
        }
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_missing_agents = num_missing
        num_rescuers = num_rescuers
        num_trees = num_trees
        num_safezones = num_safezones
        num_agents = num_rescuers + num_missing_agents
        num_landmarks = num_trees + num_safezones

        # Add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_rescuers else False
            agent.saved = False  # New attribute to mark saved agents
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_rescuers else i - num_rescuers
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.025 if agent.adversary else 0.01
            agent.accel = 3 if agent.adversary else 4
            agent.max_speed = 0.3 if agent.adversary else 0.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.tree = True if i < num_trees else False
            base_name = "tree" if landmark.tree else "safezone"
            base_index = i if i < num_trees else i - num_trees
            landmark.name = f"{base_name}_{base_index}"
            landmark.collide = True if landmark.tree else False
            landmark.movable = False
            landmark.size = 0.03 if landmark.tree else 0.1
            landmark.boundary = False

        # Assign victim types
        for i, agent in enumerate(world.agents):
            if not agent.adversary:  # Only for victims
                agent.type = victim_types[i % len(victim_types)]
                agent.color = type_colors[agent.type]  # Assign the corresponding color
        # Assign safe zone types
        for i, landmark in enumerate(world.landmarks):
            if not landmark.tree:  # Only for safe zones
                landmark.type = safe_zone_types[i % len(safe_zone_types)]
                landmark.color = type_colors[landmark.type]  # Assign the corresponding color

        return world

    def reset_world(self, world, np_random, reset_landmarks=True):
        type_colors = {
        "A": np.array([1.0, 0.0, 0.0]),  # Red
        "B": np.array([0.0, 1.0, 0.0]),  # Green
        "C": np.array([0.0, 0.0, 1.0]),  # Blue
        "D": np.array([1.0, 1.0, 0.0])   # Yellow
        }
        
        safe_zone_positions = [
                            [-1, 1],  # Top-left corner
                            [-1, -1],  # Bottom-left corner
                            [1, 1],   # Top-right corner
                            [1, -1]   # Bottom-right corner
                            ]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                type_colors[agent.type]
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color =(
                type_colors[landmark.type]
                if not landmark.tree
                else np.array([0.35, 0.85, 0.35])
            )
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-0.5, +0.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        if reset_landmarks:
            j = 0
            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    if landmark.tree:
                        landmark.state.p_pos = np_random.uniform(-0.8, +0.8, world.dim_p)
                        landmark.state.p_vel = np.zeros(world.dim_p)
                    else:
                        landmark.state.p_pos = safe_zone_positions[j]
                        landmark.state.p_vel = np.zeros(world.dim_p)
                        j+=1

        self.log_rescue_status(world)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_victim_rescued(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def step(self, world):
        for rescuer in self.adversaries(world):
            for victim in self.good_agents(world):
                if victim.saved:  # Skip saved victims
                    continue

            # Update victims' states
        for victim in self.good_agents(world):
            if victim.saved:  # Stop movement for saved agents
                victim.state.p_vel = np.zeros_like(victim.state.p_vel)

        self.log_rescue_status(world)


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]


    def is_victim_rescued(self, agent1, safezone, rescue_radius = 0.1):
        delta_pos = agent1.state.p_pos - safezone.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + safezone.size
        return True if dist < dist_min else False

        # delta_pos = rescuer.state.p_pos - victim.state.p_pos
        # distance = np.sqrt(np.sum(np.square(delta_pos)))
        # return distance <= rescue_radius

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def safezones(self, world):
        return [landmark for landmark in world.landmarks if not landmark.tree]

    def reward(self, agent, world):
        if agent.adversary:
            return self.adversary_reward(agent, world)
        else:
            return self.agent_reward(agent, world)

    def is_correctly_delegated(self, victim, safe_zone, rescue_radius=0.1):
        """
        Check if the victim is within the rescue radius of the matching safe zone.
        """
        if victim.saved:  # Already saved, skip
            return True

        delta_pos = victim.state.p_pos - safe_zone.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # Check if victim and safe zone types match
        if victim.type != safe_zone.type:
            return False

        # Check if the victim is within the rescue radius of the safe zone
        if dist < rescue_radius:
            victim.saved = True  # Mark the victim as saved
            victim.state.p_vel = np.zeros_like(victim.state.p_vel)  # Stop the victim
            print(f"Victim {victim.name} is saved at Safe Zone {safe_zone.name}!")
            return True

        return False



    def bound(self, pos):
        penalty = 0
        # Define boundaries for each axis
        x_min, x_max = -0.7, 0.7
        y_min, y_max = -0.7, 0.7

        # Check x-coordinate
        if pos[0] < x_min:
            penalty += (x_min - pos[0]) * 10  # Greater penalty for being further out
        elif pos[0] > x_max:
            penalty += (pos[0] - x_max) * 10

        # Check y-coordinate
        if pos[1] < y_min:
            penalty += (y_min - pos[1]) * 10
        elif pos[1] > y_max:
            penalty += (pos[1] - y_max) * 10

        return penalty

    def agent_reward(self, agent, world):
        """
        Reward function for victims (non-adversaries).
        Victims don't independently earn rewards, but their state impacts rescuers' rewards.
        """
        if agent.saved:
            return 0
    
        reward = 0
        shape = True
    
        # Penalize for being out of bounds
        reward -= self.bound(agent.state.p_pos)
    
        return reward

    def adversary_reward(self, agent, world):
        """
        Reward function for rescuers (adversaries).
        Rescuers are rewarded for successfully guiding victims to safe zones.
        """
        reward = 0
        shape = True
        victims = [v for v in self.good_agents(world) if not v.saved]  # Only unsaved victims
        safezones = self.safezones(world)

        # Reward shaping: Encourage rescuers to get closer to victims
        if shape:
            for victim in victims:
                # Penalize rescuers for being far from the victim
                distance_to_victim = np.linalg.norm(agent.state.p_pos - victim.state.p_pos)
                reward -= distance_to_victim * 0.1  # Penalize distance from victim

                # Reward for guiding the victim closer to a matching safe zone
                matching_safezone = next((sz for sz in safezones if sz.type == victim.type), None)
                if matching_safezone:
                    distance_to_safezone = np.linalg.norm(victim.state.p_pos - matching_safezone.state.p_pos)
                    reward += 1.0 / (1 + distance_to_safezone)  # Reward for reducing distance

                    # Bonus for successfully delegating the victim
                    if self.is_correctly_delegated(victim, matching_safezone):
                        reward += 100  # Large reward for successful rescue

        # Penalty for collisions with obstacles
        if shape:
            for obstacle in world.landmarks:
                if obstacle.collide and self.is_collision(agent, obstacle):
                    reward -= 10  # Penalty for hitting an obstacle

        # Penalize for going out of bounds
        reward -= self.bound(agent.state.p_pos)

        return reward


    def log_rescue_status(self, world):
        saved_agents = sum(1 for agent in self.good_agents(world) if agent.saved)
        print(f"Saved agents: {saved_agents}/{len(self.good_agents(world))}")



    
    def is_blocked_by_obstacle(self, agent1, agent2, world):
        """
        Check if the line of sight between two agents is blocked by any obstacle.
        """
        for obstacle in world.landmarks:
            if obstacle.collide:  # Only consider obstacles that can block line of sight
                if self.line_intersects_circle(agent1.state.p_pos, agent2.state.p_pos, obstacle.state.p_pos, obstacle.size):
                    return True
        return False

    def line_intersects_circle(self, point1, point2, circle_center, circle_radius):
        """
        Check if the line segment between two points intersects a circle.
        Args:
            point1 (array): The first point of the line segment (e.g., agent's position).
            point2 (array): The second point of the line segment (e.g., other agent's position).
            circle_center (array): The center of the circle (e.g., tree's position).
            circle_radius (float): The radius of the circle.

        Returns:
            bool: True if the line segment intersects the circle, False otherwise.
        """
        # Adjust points relative to circle center
        point1_rel = point1 - circle_center
        point2_rel = point2 - circle_center

        # Calculate coefficients of the quadratic equation
        dx, dy = point2_rel - point1_rel
        dr_squared = dx**2 + dy**2
        D = point1_rel[0] * point2_rel[1] - point2_rel[0] * point1_rel[1]

        # Calculate discriminant
        discriminant = circle_radius**2 * dr_squared - D**2

        return discriminant >= 0  # Intersects if discriminant is non-negative

    def get_distance(self, obj_a, obj_b):
        delta_pos = obj_a.state.p_pos - obj_b.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))

        return distance

    def observation(self, agent, world):
        # Calculate distances to each landmark
        landmark_distances = []
        for entity in world.landmarks:
            if not entity.boundary:
                distance = self.get_distance(agent, entity)
                if distance <= self.vision:
                    delta_pos = entity.state.p_pos - agent.state.p_pos
                    landmark_distances.append((distance, delta_pos))

        # Sort landmarks by distance and take the N closest
        N = 3  # Number of closest landmarks to consider
        landmark_distances.sort(key=lambda x: x[0])
        closest_landmarks = landmark_distances[:N]

        # Get positions of N closest landmarks
        entity_pos = [pos for _, pos in closest_landmarks]

        # Check if other agents are visible (not blocked by obstacles)
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent or self.is_blocked_by_obstacle(agent, other, world) or self.get_distance(agent, other) <= self.vision:
                other_pos.append(np.array([1e6, 1e6]))
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)

        observation_data = np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel
        )
        
        random_data_size = 5  # for example, appending 5 random values
        random_data = np.random.rand(random_data_size)
        # Append random data to observation
        # observation_data = np.concatenate([observation_data, random_data])
        # print(f"Shape : {observation_data.shape}")

        return observation_data