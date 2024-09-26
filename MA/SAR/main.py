from SAR.sar_v0 import env_sar_mod as sar_v0
from SAR.sar_v1 import env_sar_mod as sar_v1

import core.core as core

def main():

    # train
    # train_curr
    # eval
    # random
    mode = "random"

    # LSTM
    # PPO
    # DQN
    model_type = "PPO"

    # v0
    # v1
    version ="v0"

    num_agents = 4
    num_hostages = 1
    num_goals = 1
    train_steps = 300000
    eval_games = 10

    env_type = None

    if version == "v0":
        env_type = sar_v0
    if version == "v1":
        env_type = sar_v1

    assert env_type is not None

    # env = env_sar.env(num_good=1, num_adversaries=7, num_obstacles=5, max_cycles=1000, continuous_actions=True, render_mode="human")
    # env = sar_v0.env(num_agents=1, num_hostages=1, num_goals=0, max_cycles=10000, continuous_actions=True, render_mode="human")

    if mode == "train":
        core.train(env_type, model_type, train_steps, num_agents, num_goals, num_hostages)
    if mode == "train_curr":
        core.train_curr(env_type, model_type, train_steps, num_agents, num_goals, num_hostages)
    if mode == "eval":
        core.eval(env_type, model_type, eval_games, num_agents, num_goals, num_hostages)
    if mode == "random":
        core.random_policy(env_type)


if __name__ == "__main__":
    main()
