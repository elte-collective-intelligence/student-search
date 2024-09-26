# noqa: D212, D415
"""
# Simple Tag

```{figure} mpe_simple_tag.gif
:width: 140px
:name: simple_tag
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_tag_v3`                 |
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
simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)
```



`num_good`:  number of good agents

`num_adversaries`:  number of adversaries

`num_obstacles`:  number of obstacles

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from SAR.sar_v0.simple_env_mod import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gymnasium.utils import seeding

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_agents=3,
        num_hostages=1,
        num_goals=1,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            num_hostages=num_hostages,
            num_goals=num_goals,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_agents, num_hostages, num_goals)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_sar_v0"




env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_agents=3, num_hostages=1, num_goals=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_landmarks = num_hostages + num_goals
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            #agent.adversary = True if i < num_adversaries else False
            #base_name = "adversary" if agent.adversary else "agent"
            #base_index = i if i < num_adversaries else i - num_adversaries
            #agent.name = f"{base_name}_{base_index}"
            agent.active = False
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.02
            agent.accel = 4.0
            agent.max_speed = 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.hostage = True if i < num_hostages else False
            base_name = "hostage" if landmark.hostage else "goal"
            base_index = i if i < num_hostages else i - num_hostages
            landmark.name = f"{base_name}_{base_index}"

            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.02 if landmark.hostage else 0.05
            landmark.boundary = False

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (np.array([1, 1, 1]))
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = (
                np.array([0, 1, 0])
                if not landmark.hostage
                else np.array([1, 1, 0])
            )

        self._seed()
        return world

    def reset_world(self, world, np_random):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-0.3, +0.3, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.active = False
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        #TODO
        #if agent.adversary:
        #    collisions = 0
        #    for a in self.good_agents(world):
        #        if self.is_collision(a, agent):
        #            collisions += 1
        #    return collisions
        #else:
        #    return 0
        return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def agents(self, world):
        return world.agents

    # return all agents that are not adversaries
    def hostages(self, world):
        return [landmark for landmark in world.landmarks if landmark.hostage]

    def goals(self, world):
        return [landmark for landmark in world.landmarks if not landmark.hostage]


    def reward(self, agent, world):
        # Returns the rewards for the agents
        return self.agent_reward(agent, world)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def agent_reward(self, agent, world):
        # Agents are rewarded for reaching a hostage and for delivering it to a goal zone
        rew = 0
        shape = False
        agents = self.agents(world)
        hostages = self.hostages(world)
        goals = self.goals(world)

        entities_in_range = agent.entities_in_range

        if agent.active:
            agent.color = (np.array([1, 0, 1]))
        else:
            agent.color = (np.array([1, 1, 1]))

        if agent.collide:
            for hsg in hostages:
                if self.is_collision(agent, hsg) and not agent.active:
                    # if agent collides with any hostage
                    rew += 100
                    agent.active = True
                    #self.reset_world(world,self.np_random)
            for gl in goals:
                if self.is_collision(agent, gl) and agent.active:
                    # if agent collides with any goal
                    rew += 300

        #print(agent.last_pos - agent.state.p_pos)
        #print("current :", agent.state.p_pos)
        #print("found stored :", agent.last_pos)


        #pygame.draw.circle(screen, (255, 0, 0), (-50, 300), 5)


        for ent in entities_in_range:
            if ent.hostage:
                x = self.heuristic(ent.state.p_pos, agent.state.p_pos)

                #print("for")
                #print("x :", x)
                #y = 1/3 * np.exp(-(1/3) * x + 2) (4)
                #y = 1/4 * np.exp(-(1/4) * x + 2) (5,6)
                #y = 1/4 * np.exp(-(1/2) * x + 3) (7)
                # y = 1/5 * np.exp(-(1/5) * x + 2) (8)
                #y = np.exp(-1 * x + 2) - 3
                #y = 1/4 * np.exp(-1 * x + 3) (10)
                #y = 1/4 * np.exp(-1 * x + 3)
                #y = np.exp(-1 * x + 3)
                y = (1/4) * np.exp(-1 * x + 3)

                #y = - np.sqrt(x) + 1.5

                d = 3*(0.5-x)
                #agent.color = (np.array([c, c, c]))
                #print("for :",x)
                #print("color :", c)

                #print("y :", y)
                rew += d

        #rew -= 0.5

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def heuristic(self, pos1, pos2):
        delta_pos = pos1 - pos2
        return np.sqrt(np.sum(np.square(delta_pos)))


    def observation(self, agent, world):


        hostages = self.hostages(world)
        goals = self.goals(world)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_list = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_list.append(entity)
                entity_pos.append(entity.state.p_pos)

        agent.entities_in_range = entity_list

        # communication of all other agents
        #comm = []
        #other_pos = []
        #other_vel = []
        #for other in world.agents:
        #    if other is agent:
        #        continue
        #    comm.append(other.state.c)
        #    other_pos.append(other.state.p_pos - agent.state.p_pos)
            #if not other.adversary:
            #    other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
        )