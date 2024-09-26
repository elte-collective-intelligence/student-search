import supersuit as ss
from sar_env import parallel_env
from stable_baselines3 import PPO
import time
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

def train(env_fn, steps: int = 100, seed = 0, **env_kwargs):
    env = parallel_env(**env_kwargs)

    env.reset(seed=seed)
    tmp_path = "search_rescue/sb3_log/"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])



    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.multiagent_wrappers.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 20, num_cpus=4, base_class="stable_baselines3")
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    # Model
    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, batch_size=256, n_epochs=40)

    model = PPO("MlpPolicy", env, verbose=1, batch_size=256, tensorboard_log="./search_rescue/")
    model.set_logger(new_logger)

    # Train
    model.learn(total_timesteps=steps, tb_log_name="MLP_Policy", callback=eval_callback)
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()
    
    