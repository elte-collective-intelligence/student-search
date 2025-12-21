import time

import numpy as np
import torch
from torchrl.envs import check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss, ValueEstimators
import tqdm

from src.models import make_policy, make_critic
from src.sar_env import make_env
from src.logger import RunContext, TensorboardLogger
from src.curriculum import CurriculumScheduler


def _get_metrics_env(env):
    base = getattr(env, "base_env", None)
    while base is not None and not hasattr(base, "pop_episode_metrics"):
        base = getattr(base, "base_env", getattr(base, "_env", None))
    return base if base is not None else env


def train(
    steps: int = 100000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    frames_per_batch: int = 2048,
    seed: int = 0,
    save_folder: str = "search_rescue_logs/",
    enable_logging: bool = True,
    curriculum_enabled: bool = False,
    curriculum_min_trees: int = 0,
    curriculum_max_trees: int = 8,
    curriculum_num_stages: int = 5,
    **env_kwargs,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup logging
    run_ctx = RunContext(base_dir=save_folder, run_name=None, create_subdirs=True)
    logger = TensorboardLogger.create(ctx=run_ctx, enabled=enable_logging)
    if enable_logging:
        print(f"TensorBoard logging enabled. Log directory: {run_ctx.tb_run_dir}")

    # Initialize curriculum scheduler if enabled
    curriculum = None
    if curriculum_enabled:
        # Calculate iterations based on frames_per_batch
        total_iterations = steps // frames_per_batch
        curriculum = CurriculumScheduler(
            min_trees=curriculum_min_trees,
            max_trees=curriculum_max_trees,
            num_stages=curriculum_num_stages,
            total_iterations=total_iterations,
        )
        # Set initial trees and max_trees for fixed observation space
        env_kwargs["num_trees"] = curriculum.get_num_trees()
        env_kwargs["max_trees"] = curriculum_max_trees
        print(f"Curriculum learning enabled: {curriculum}")
        print(f"  Stages: {curriculum.num_stages}")
        print(f"  Trees progression: {curriculum.stage_trees}")
        print(f"  Iterations per stage: {curriculum.iterations_per_stage}")

    # Create environment
    env_kwargs["seed"] = seed
    env = make_env(device=device, **env_kwargs)

    # Check environment specs
    check_env_specs(env)

    num_agents = env.base_env.num_rescuers
    print(f"Starting MARL training on {env.base_env.metadata['name']}.")
    print(f"Number of agents: {num_agents}")
    print(f"Observation shape: {env.observation_spec['agents', 'observation'].shape}")
    # print(f"Global state shape: {env.observation_spec['state'].shape}")
    print(f"Action spec: {env.action_spec}")

    # 1. Policy Network (Actor) - Decentralized
    # Input: ("agents", "observation") -> [Batch, n_agents, obs_dim]
    # Output: ("agents", "loc"), ("agents", "scale") for continuous OR ("agents", "logits") for discrete
    is_discrete = not env.base_env.is_continuous
    policy = make_policy(
        env, num_rescuers=num_agents, device=device, discrete=is_discrete
    )

    # 2. Value Network (Critic) - Centralized
    # Input: All observations concatenated
    # Output: Value per agent
    critic = make_critic(env, num_rescuers=num_agents, device=device)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=steps,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.1,
        normalize_advantage=True,
        normalize_advantage_exclude_dims=(-1,),
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),  # TorchRL requires terminated
        value=("agents", "state_value"),  # Output of critic
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=0.99, lmbda=0.95)

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)

    pbar = tqdm.tqdm(total=steps, unit="frames")
    rewards_log = []
    iteration = 0
    total_frames = 0

    # Log hyperparameters
    hparams = {
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "frames_per_batch": frames_per_batch,
        "batch_size": batch_size,
        "total_steps": steps,
        "seed": seed,
        "num_agents": num_agents,
    }
    if curriculum is not None:
        hparams.update(
            {
                "curriculum_enabled": True,
                "curriculum_min_trees": curriculum.min_trees,
                "curriculum_max_trees": curriculum.max_trees,
                "curriculum_num_stages": curriculum.num_stages,
            }
        )
    logger.log_dict("hyperparameters", hparams, step=0)

    for batch in collector:
        metrics_env = _get_metrics_env(env)

        # 1. Prepare Batch
        # Add 'terminated' if missing (older versions of wrappers might miss it)
        if ("agents", "terminated") not in batch.keys(include_nested=True):
            batch["agents", "terminated"] = batch["agents", "done"]

        # Compute GAE
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        # Flatten batch time dimensions
        batch = batch.reshape(-1)
        replay_buffer.extend(batch)

        # 2. PPO Update
        avg_loss_objective = 0.0
        avg_loss_critic = 0.0
        avg_loss_entropy = 0.0
        avg_loss_total = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            for _ in range(frames_per_batch // batch_size):
                subdata = replay_buffer.sample(batch_size)
                # Ensure sampled minibatch tensors are on the training device
                try:
                    subdata = subdata.to(device)
                except AttributeError:
                    # Fallback: if `.to(device)` is unavailable, assume `subdata` is already on a compatible
                    # device or that `loss_module` handles device placement internally, so we skip adjustment.
                    pass
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optimizer.step()

                # Accumulate losses for logging
                avg_loss_objective += loss_vals["loss_objective"].item()
                avg_loss_critic += loss_vals["loss_critic"].item()
                avg_loss_entropy += loss_vals["loss_entropy"].item()
                avg_loss_total += loss_value.item()
                num_updates += 1

        # 3. Curriculum Progression
        if curriculum is not None:
            stage_changed = curriculum.step()
            if stage_changed:
                new_trees = curriculum.get_num_trees()
                print(
                    f"\n[Curriculum] Advanced to stage {curriculum.get_stage()}: {new_trees} trees"
                )
                # Update environment's tree count (takes effect on next episode reset)
                metrics_env.set_num_trees(new_trees)
                # Log curriculum change
                logger.log_dict("curriculum", curriculum.get_info(), step=iteration)

        # 4. Logging
        pbar.update(batch.numel())
        total_frames += batch.numel()
        iteration += 1

        # Log curriculum state each iteration
        if curriculum is not None:
            logger.log_dict("curriculum", curriculum.get_info(), step=iteration)

        # Log training losses
        if num_updates > 0:
            logger.log_dict(
                "train/loss",
                {
                    "objective": avg_loss_objective / num_updates,
                    "critic": avg_loss_critic / num_updates,
                    "entropy": avg_loss_entropy / num_updates,
                    "total": avg_loss_total / num_updates,
                },
                step=iteration,
            )

        # Check for completed episodes in the batch to log reward
        # Note: Using next/done to filter
        done_mask = batch["next", "agents", "done"].any(dim=-1)  # If any agent done
        if done_mask.any():
            mean_reward = (
                batch["next", "agents", "episode_reward"][done_mask].mean().item()
            )
            rewards_log.append(mean_reward)
            pbar.set_description(f"Mean Reward: {mean_reward:.2f}")

            # Log episode metrics
            logger.log_scalar("train/episode_reward", mean_reward, step=iteration)
            logger.log_scalar("train/total_frames", total_frames, step=iteration)

            # Pull environment metrics for the just-finished episodes
            metrics = metrics_env.pop_episode_metrics()
            if metrics:
                # Aggregate across episodes that finished in this batch
                rescues_pct = np.mean([m["rescues_pct"] for m in metrics])
                collisions = np.mean([m["collisions"] for m in metrics])
                coverage = np.mean([m["coverage_cells"] for m in metrics])
                logger.log_dict(
                    "train/episode_metrics",
                    {
                        "rescues_pct": rescues_pct,
                        "collisions": collisions,
                        "coverage_cells": coverage,
                    },
                    step=iteration,
                )

        collector.update_policy_weights_()

    pbar.close()

    # Save model in the run directory
    model_path = (
        run_ctx.run_dir
        / f"{env.base_env.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save environment configuration for evaluation
    env_config = {
        "num_victims": env.base_env.num_victims,
        "num_rescuers": env.base_env.num_rescuers,
        "num_trees": env.base_env.num_trees,
        "num_safe_zones": env.base_env.num_safe_zones,
        "max_cycles": env.base_env.max_steps,
        "continuous_actions": env.base_env.is_continuous,
        "vision_radius": env.base_env.vision_radius,
        "randomize_safe_zones": env.base_env.randomize_safe_zones,
    }

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "total_frames": total_frames,
            "iteration": iteration,
            "env_config": env_config,
        },
        str(model_path),
    )

    # Log final summary
    if rewards_log:
        final_mean_reward = sum(rewards_log) / len(rewards_log)
        logger.log_scalar("train/final_mean_reward", final_mean_reward, step=iteration)
        logger.log_scalar("train/total_episodes", len(rewards_log), step=iteration)

    logger.close()
    collector.shutdown()
    env.close()

    print(f"Finished MARL training on {env.base_env.metadata['name']}.")
    print(f"Model saved to: {model_path}")
    if enable_logging:
        print(f"TensorBoard logs available at: {run_ctx.tb_run_dir}")
