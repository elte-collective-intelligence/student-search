import argparse
from pettingzoo.mpe._mpe_utils.simple_env import make_env

from sar_env_updated import make_env, raw_env, parallel_wrapper_fn, parallel_env, env
from pettingzoo.test import parallel_api_test
import supersuit as ss


import glob
import os
import time
import numpy as np
import time
from eval_updated import eval
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from train_updated import train
# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3 import DQN
# from stable_baselines3.dqn import CnnPolicy, MlpPolicy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image captioning training pipeline")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training for the search and rescue environment.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a trained model in the search and rescue environment with graphical visualization.",
    )
    # parse_known_args avoids crashing when extra arguments are forwarded
    args, _ = parser.parse_known_args()
    return args

def main():
    # Set render_mode to None to reduce training time
    env_kwargs = dict(num_missing=1, num_rescuers=3, num_trees=8, num_safezones=4, max_cycles=120, continuous_actions=False)
    env_fn = "search_and_rescue"
    if args.train:
        train(env_fn, steps=int(1e5), seed=0, render_mode=None, **env_kwargs)
    elif args.eval:
        eval(env_fn, num_games=10, render_mode='human', **env_kwargs)
    else:
        print("Please specify either --train or --eval to run the respective process.")


if __name__ == "__main__":
    main()
