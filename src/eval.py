from time import sleep

from src.sar_env import make_env
from src.models import make_policy
from src.logger import RunContext, TensorboardLogger
import glob
import os
import torch
from torchrl.envs.utils import step_mdp


def find_latest_model(save_folder: str, env_name: str) -> str:
    """
    Finds the latest .pt file in the save_folder based on modification time.
    Searches recursively in dated subdirectories (e.g., save_folder/20251220-000020/*.pt)
    """
    # First try recursive search in dated subdirectories
    search_pattern = os.path.join(save_folder, "**", "*.pt")
    files = glob.glob(search_pattern, recursive=True)

    # Fallback to flat files in save_folder
    if not files:
        search_pattern = os.path.join(save_folder, "*.pt")
        files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(
            f"No model files (*.pt) found in {save_folder} or its subdirectories"
        )

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Auto-detected latest model: {latest_file}")
    return latest_file


def _get_metrics_env(env):
    base = getattr(env, "base_env", None)
    # Walk down wrappers until we find the environment exposing metrics
    while base is not None and not hasattr(base, "pop_episode_metrics"):
        base = getattr(base, "base_env", getattr(base, "_env", None))
    return base if base is not None else env


def evaluate(
    model_path: str = None,
    save_folder: str = "search_rescue_logs",
    num_games: int = 3,
    enable_logging: bool = True,
    **env_kwargs,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup logging
    run_ctx = RunContext(base_dir=save_folder, run_name="eval", create_subdirs=True)
    logger = TensorboardLogger.create(ctx=run_ctx, enabled=enable_logging)
    if enable_logging:
        print(f"TensorBoard logging enabled. Log directory: {run_ctx.tb_run_dir}")

    # 1. Resolve Model Path first (we need to load config from checkpoint)
    if not model_path:
        print(f"No model path provided. Searching in '{save_folder}'...")
        # Create temporary env to get metadata name
        temp_env = make_env(device=device, **env_kwargs)
        try:
            env_name = temp_env.base_env.metadata["name"]
            model_path = find_latest_model(save_folder, env_name)
        except Exception as e:
            print(f"Error finding model: {e}")
            return
        finally:
            temp_env.close()

    # 2. Load Model and Config
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 3. Use environment config from checkpoint if available
    if "env_config" in checkpoint:
        print("Using environment configuration from checkpoint:")
        saved_config = checkpoint["env_config"]
        for key, value in saved_config.items():
            print(f"  {key}: {value}")

        # Override env_kwargs with saved config (but keep render_mode from args)
        render_mode = env_kwargs.get("render_mode", "human")
        env_kwargs = saved_config.copy()
        env_kwargs["render_mode"] = render_mode
    else:
        print("Warning: No env_config in checkpoint, using provided arguments")

    # 4. Create Environment with correct configuration
    print("Initializing environment...")
    env = make_env(device=device, **env_kwargs)

    # 5. Create and Load Policy
    num_agents = env.action_spec["agents", "action"].shape[0]

    # Determine if we're using discrete or continuous actions
    is_discrete = not env.base_env.is_continuous
    print(f"Action type: {'Discrete' if is_discrete else 'Continuous'}")

    policy = make_policy(
        env, num_rescuers=num_agents, device=device, discrete=is_discrete
    )

    # Handle different saving formats
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        # Format from the robust training script
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(
            f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'N/A')} "
            f"(total frames: {checkpoint.get('total_frames', 'N/A')})"
        )
    elif isinstance(checkpoint, dict) and "actor" in checkpoint:
        # Format from older simple script
        policy.load_state_dict(checkpoint["actor"])
    else:
        # Raw state dict
        policy.load_state_dict(checkpoint)

    policy.eval()  # Set to evaluation mode

    # 6. Evaluation Loop
    print(f"Starting evaluation for {num_games} episodes...")

    # Track evaluation metrics
    episode_rewards = []
    episode_steps = []
    rescues_pct_log = []
    collisions_log = []
    coverage_log = []
    metrics_env = _get_metrics_env(env)

    for i in range(num_games):
        td = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0

        print(f"--- Episode {i + 1} ---")

        while not done:
            with torch.no_grad():
                td = policy(td)

            td = env.step(td)
            env.render()

            if "next" in td.keys():
                # Standard TorchRL behavior
                if td["next", "done"].any():
                    done = True
                td = step_mdp(td)
            else:
                # Flat behavior (PettingZooWrapper sometimes does this)
                # The 'td' returned IS the next state
                # Check for "done", "terminated", or "agents/done"
                if "done" in td.keys() and td["done"].any():
                    done = True
                elif "terminated" in td.keys() and td["terminated"].any():
                    done = True
                elif ("agents", "done") in td.keys(include_nested=True) and td[
                    "agents", "done"
                ].any():
                    done = True

            step_count += 1

            # Collect rewards for logging
            if ("agents", "reward") in td.keys(include_nested=True):
                rewards = td["agents", "reward"].detach().cpu().numpy()
                logger.log_scalar(
                    f"eval/episode_{i+1}", rewards.mean(), step=step_count
                )
                episode_reward += float(rewards.sum())

            sleep(0.1)

        # Log episode metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

        logger.log_scalar("eval/episode_reward", episode_reward, step=i + 1)
        logger.log_scalar("eval/episode_steps", step_count, step=i + 1)
        logger.log_scalar(
            "eval/mean_reward_per_step", episode_reward / max(step_count, 1), step=i + 1
        )

        # Log environment metrics (rescues %, collisions, coverage)
        metrics = metrics_env.pop_episode_metrics()
        if metrics:
            m = metrics[-1]
            rescues_pct_log.append(m["rescues_pct"])
            collisions_log.append(m["collisions"])
            coverage_log.append(m["coverage_cells"])
            logger.log_scalar("eval/rescues_pct", m["rescues_pct"], step=i + 1)
            logger.log_scalar("eval/collisions", m["collisions"], step=i + 1)
            logger.log_scalar("eval/coverage_cells", m["coverage_cells"], step=i + 1)

        print(
            f"Episode {i + 1} finished in {step_count} steps. Total reward: {episode_reward:.2f}"
        )

    # Log summary statistics
    if episode_rewards:
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        mean_steps = sum(episode_steps) / len(episode_steps)
        logger.log_scalar("eval/mean_episode_reward", mean_reward, step=num_games)
        logger.log_scalar("eval/mean_episode_steps", mean_steps, step=num_games)
        logger.log_scalar("eval/total_episodes", num_games, step=num_games)

    if rescues_pct_log:
        mean_rescues_pct = sum(rescues_pct_log) / len(rescues_pct_log)
        mean_collisions = sum(collisions_log) / len(collisions_log)
        mean_coverage = sum(coverage_log) / len(coverage_log)
        logger.log_dict(
            "eval/summary",
            {
                "rescues_pct": mean_rescues_pct,
                "collisions": mean_collisions,
                "coverage_cells": mean_coverage,
            },
            step=num_games,
        )

    logger.close()
    print("Evaluation finished.")
    if episode_rewards:
        print(f"Mean episode reward: {mean_reward:.2f}")
        print(f"Mean episode steps: {mean_steps:.1f}")
    if enable_logging:
        print(f"TensorBoard logs available at: {run_ctx.tb_run_dir}")
    env.close()
