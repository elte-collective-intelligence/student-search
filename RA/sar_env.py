# noqa: D212, D415
"""
# Search and Rescue

```{figure} mpe_search_and_rescue.gif
:width: 140px
:name: search_and_rescue
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import search_and_rescue_v3`                 |
|--------------------|------------------------------------------------------------|
| Actions            | Discrete/Continuous                                        |
| Parallel API       | Yes                                                        |
| Manual Control     | No                                                         |
| Agents             | `agents= [adversary_0, adversary_1, adversary_2, agent_0]` |
| Agents             | 4                                                          |
| Action Shape       | (5)                                                        |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (50))                            |
| Observation Shape  | (14),(16)                                                  |
| Observation Values | (-inf,inf)                                                 |
| State Shape        | (62,)                                                      |
| State Values       | (-inf,inf)                                                 |


This is a predator-prey environment. Good agents (green) are faster and receive a negative reward for being hit by adversaries (red) (-10 for each collision). Adversaries are slower and are rewarded for hitting good agents (+10 for each collision). Obstacles (large black circles) block the way. By
default, there is 1 good agent, 3 adversaries and 2 obstacles.

So that good agents don't run to infinity, they are also penalized for exiting the area by the following function:

``` python
def bound(x):
      if x < 0.9:
          return 0
      if x < 1.0:
          return (x - 0.9) * 10
      return min(np.exp(2 * x - 2), 10)
```

Agent and adversary observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]`

Agent and adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
search_and_rescue_v3.env(num_missing=1, num_rescuers=3, num_trees=2, max_cycles=25, continuous_actions=False)
```



`num_missing`:  number of good agents

`num_rescuers`:  number of adversaries

`num_trees`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

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
        num_trees=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_missing=num_missing,
            num_rescuers=num_rescuers,
            num_trees=num_trees,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_missing, num_rescuers, num_trees)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "search_and_rescue_v3"




env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_missing=1, num_rescuers=3, num_trees=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_missing_agents = num_missing
        num_rescuers = num_rescuers
        num_agents = num_rescuers + num_missing_agents
        num_landmarks = num_trees
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_rescuers else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_rescuers else i - num_rescuers
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random, reset_landmarks=True):
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.25, 0.25, 0.98])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.35, 0.85, 0.35])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        if reset_landmarks:

            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)

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


    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]


    def is_victim_rescued(self, agent1, agent2, rescue_radius = 0.1):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

        # delta_pos = rescuer.state.p_pos - victim.state.p_pos
        # distance = np.sqrt(np.sum(np.square(delta_pos)))
        # return distance <= rescue_radius

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        print(main_reward)
        # Penalty for colliding with obstacles
        # obstacle_penalty = 0
        # for obstacle in world.landmarks:
        #     if obstacle.collide and self.is_collision(agent, obstacle):
        #         obstacle_penalty -= 5  # Penalty value, adjust as needed

        
        
        return main_reward# + obstacle_penalty

    def agent_reward(self, agent, world):
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - adv.state.p_pos))
                )
        if agent.collide:
            # print("Collide happened")
            for a in adversaries:
                if self.is_victim_rescued(a, agent):
                    # print("Rescue happened")
                    rew += 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (decreased reward for increased distance from agents)
                closest_distance = min(
                    np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                    for a in agents
                )
                # Reward increases as the adversary gets closer to any good agent
                rew += closest_distance
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_victim_rescued(ag, adv):
                        rew += 10
        return rew
    
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


    def observation(self, agent, world):
        # Calculate distances to each landmark
        landmark_distances = []
        for entity in world.landmarks:
            if not entity.boundary:
                delta_pos = entity.state.p_pos - agent.state.p_pos
                distance = np.sqrt(np.sum(np.square(delta_pos)))
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
            if other is agent or self.is_blocked_by_obstacle(agent, other, world):
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