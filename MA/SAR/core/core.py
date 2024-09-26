import glob
import os
import numpy as np
import time

import supersuit as ss
from stable_baselines3 import PPO, DQN
from stable_baselines3.ppo import MlpPolicy
from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3 import DQN
# from stable_baselines3.dqn import CnnPolicy, MlpPolicy


def random_policy(env_type):
    """
    Uses
    :param env_type:
    """
    env = env_type.env(num_agents=10, num_hostages=1, num_goals=1, max_cycles=10000, continuous_actions=False, render_mode="human")

    env.reset()

    # print(env.num_agents)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            time.sleep(0.05)

        env.step(action)
    env.close()


def train(env_type, model_type, train_steps, num_agents, num_goals, num_hostages):
    env_kwargs = dict(
        num_agents=num_agents,
        num_hostages=num_hostages,
        num_goals=num_goals,
        max_cycles=50,
        render_mode=None,
        continuous_actions=False)

    env = env_type.parallel_env(**env_kwargs)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")


    model = select_model(model_type, env)

    # Train
    model.learn(total_timesteps=train_steps, progress_bar=True)

    model.save(f"{model_type}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def train_curr(env_type, model_type, train_steps, num_agents, num_goals, num_hostages):
    env_kwargs = dict(
        num_agents=num_agents,
        num_hostages=num_hostages,
        num_goals=num_goals,
        max_cycles=100,
        render_mode=None,
        continuous_actions=False)

    env = env_type.parallel_env(**env_kwargs, mode="goal")

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")


    model = select_model(model_type, env)


    # Train
    model.learn(total_timesteps=train_steps * 1 / 4, progress_bar=True)

    env.close

    env = env_type.parallel_env(**env_kwargs, mode="hostage")
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

    print(f"Resuming training.")
    model.env = env

    model.learn(total_timesteps=train_steps * 3 / 4, progress_bar=True)

    model.save(f"{model_type}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_type, model_type, eval_games, num_agents, num_goals, num_hostages):
    env_kwargs = dict(
        num_agents=num_agents,
        num_hostages=num_hostages,
        num_goals=num_goals,
        max_cycles=50,
        render_mode='human',
        continuous_actions=False)

    env = env_type.parallel_env(**env_kwargs)

    # Evaluate a trained agent vs a random agent
    env = env_type.env(**env_kwargs)
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={eval_games}, render_mode=human)"
    )

    try:
        latest_policy = max(
            glob.glob(f"{model_type}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = load_model(model_type, latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(eval_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if agent == 'agent_0':
                obs = np.append(obs, [0, 0])
            # print(obs)
            time.sleep(0.01)
            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / eval_games for agent in env.possible_agents
    }

    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward

def select_model(model_type, env):

    if model_type == "LSTM":
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            batch_size=256,
        )

    if model_type == "PPO":
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            batch_size=256,
            learning_rate=0.001,
            ent_coef=0.01,
            tensorboard_log="logs_300/",

        )

    if model_type == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            batch_size=256,
            learning_rate=0.001,
            #exploration_fraction=0.01,
        )


    return model

def load_model(model_type, latest_policy):

    if model_type == "LSTM":
        model = RecurrentPPO.load(latest_policy)

    if model_type == "PPO":
        model = PPO.load(latest_policy)

    if model_type == "DQN":
        model = DQN.load(latest_policy)

    return model
