from pettingzoo.mpe._mpe_utils.simple_env import make_env

from sar_env_updated import make_env, raw_env, parallel_wrapper_fn, parallel_env, env
from pettingzoo.test import parallel_api_test
import supersuit as ss

import hydra
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    assert (
        cfg.train.active ^ cfg.eval.active
    ), "Please specify one of train.active=true or eval.active=true in the arguments."
    env_kwargs = {
        "num_missing": cfg.env.missing,
        "num_rescuers": cfg.env.rescuers,
        "num_trees": cfg.env.trees,
        "num_safezones": cfg.env.safezones,
        "max_cycles": cfg.env.max_cycles,
        "continuous_actions": cfg.env.continuous_actions,
    }
    env_fn = "search_and_rescue"
    if cfg.train.active:
        train(
            env_fn,
            steps=cfg.train.total_timesteps,
            seed=cfg.train.seed,
            render_mode=cfg.train.render_mode,
            **env_kwargs
        )
    elif cfg.eval.active:
        eval(
            env_fn,
            num_games=cfg.eval.games,
            render_mode=cfg.eval.render_mode,
            **env_kwargs
        )


if __name__ == "__main__":
    main()
