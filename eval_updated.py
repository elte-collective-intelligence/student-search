from stable_baselines3 import PPO
import glob
import os
import time
import numpy as np
import time
from sar_env_updated import env as env_f
from sb3_contrib import RecurrentPPO


def eval(env_fn, num_games: int = 100, save_folder : str = "search_rescue_logs/", render_mode = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent


    env = env_f(render_mode=render_mode, **env_kwargs)
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    # Add option to manually specify policy name
    manual_policy_name = input(
        "Enter the policy name (e.g., policy_name.zip) or press Enter to load the latest: "
    ).strip()

    try:
        if manual_policy_name:
            # Load the manually specified policy
            if os.path.exists(manual_policy_name):
                model = PPO.load(manual_policy_name)
                print(f"Loaded policy: {manual_policy_name}")
            else:
                print(f"Policy '{manual_policy_name}' not found.")
                exit(0)
        else:
            # Fallback to the latest policy if no name is provided
            latest_policy = max(
                glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
            )
            model = PPO.load(latest_policy)
            print(f"Loaded the latest policy: {latest_policy}")
    except ValueError:
        print("Policy not found.")
        exit(0)

        # model = RecurrentPPO.load(latest_policy)
    #     model = DQN.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            # if agent == 'agent_0':
            #     obs=np.append(obs, [0,0])
            #print(obs)
            if render_mode== 'human':
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
        env.render()
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward
