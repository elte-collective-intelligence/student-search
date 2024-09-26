from pettingzoo.mpe._mpe_utils.simple_env import make_env

from sar_env import make_env, raw_env, parallel_wrapper_fn, parallel_env, env
from pettingzoo.test import parallel_api_test
import supersuit as ss


import glob
import os
import time
import numpy as np
import time
from eval import eval
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from train import train
# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3 import DQN
# from stable_baselines3.dqn import CnnPolicy, MlpPolicy



def main():
    # Set render_mode to None to reduce training time
    env_kwargs = dict(num_missing=1, num_rescuers=3, num_trees=8, max_cycles=120, continuous_actions=False)
    env_fn = "search_and_rescue"
    # train(env_fn, steps=1e5, seed=0, render_mode=None, **env_kwargs)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)
    
    
# if __name__ == "__main__":
#     main()

# def main():
#     seed = 22
#     # env = parallel_wrapper_fn(env)
#     env = parallel_env()

#     # env.reset(seed=seed)
#     # env_v =env()
#     print(f"Starting training on {str(env.metadata['name'])}.")
#     env = ss.multiagent_wrappers.pad_observations_v0(env)
#     env = ss.pettingzoo_env_to_vec_env_v1(env)
#     env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

#     parallel_api_test(env, num_cycles=1_000_000)







if __name__ == "__main__":
    main()