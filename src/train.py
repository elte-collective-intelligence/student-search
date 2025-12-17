"""
Training script using TorchRL with MAPPO for Multi-Agent Reinforcement Learning.

Implements CTDE (Centralized Training with Decentralized Execution):
- Actors: Use local observations for each agent (decentralized)
- Critic: Uses global state for value estimation (centralized)
- Parameter sharing: All agents share the same policy network

Based on: https://arxiv.org/abs/2103.01955 (MAPPO paper)
"""

import time
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter
from torchrl.envs.utils import check_env_specs
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.sar_env import SearchAndRescueEnv
from src.models import make_ppo_models, make_mappo_models


def make_env(env_kwargs, device="cpu"):
    """Create and wrap the environment."""
    env = SearchAndRescueEnv(**env_kwargs, device=device)
    env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    return env


def train(
    steps: int = 100000,
    batch_size: int = 256,
    seed: int = 0,
    save_folder: str = "search_rescue_logs/",
    algorithm: str = "mappo",
    **env_kwargs,
):
    """Train a PPO/MAPPO agent on the search and rescue environment.

    Implements MARL with CTDE:
    - All agents share the same policy (parameter sharing)
    - Critic uses global state for centralized training
    - Actors execute using only local observations

    Args:
        steps: Total training steps.
        batch_size: Batch size for training.
        seed: Random seed.
        save_folder: Folder to save logs and models.
        algorithm: "ppo" (local critic) or "mappo" (CTDE with global state critic).
        **env_kwargs: Environment configuration.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env_kwargs["seed"] = seed
    env = make_env(env_kwargs, device=device)

    # Check environment specs
    check_env_specs(env)

    num_agents = env.base_env.num_rescuers
    print(f"Starting MARL training on {env.base_env.metadata['name']}.")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Number of agents: {num_agents}")
    print(f"Observation shape (per-agent): {env.observation_spec['observation'].shape}")
    print(f"Global state shape: {env.observation_spec['state'].shape}")
    print(f"Action spec: {env.action_spec}")

    # Create models based on algorithm choice
    if algorithm.lower() == "mappo":
        # MAPPO: actor uses local obs, critic uses global state (CTDE)
        actor, critic = make_mappo_models(env, device=device)
        print("Critic using: global state (CTDE - Centralized Training)")
        print("Actor using: local observations (Decentralized Execution)")
    else:
        # PPO: both actor and critic use local observation
        actor, critic = make_ppo_models(env, device=device)
        print("Critic using: local observation (Independent PPO)")

    # Create GAE module for advantage estimation
    adv_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
    )

    # Create PPO loss module
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=0.5,
        loss_critic_type="smooth_l1",
    )

    # Create optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

    # Create data collector
    frames_per_batch = min(batch_size, steps // 4)
    print(f"Batch size (frames per batch): {frames_per_batch}")
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=steps,
        split_trajs=False,
        device=device,
    )

    # Setup tensorboard
    writer = SummaryWriter(log_dir=save_folder)

    # Training loop
    ppo_epochs = 4
    total_frames = 0
    start_time = time.time()

    pbar = tqdm(total=steps, desc="Training", unit="frames")

    for i, batch in enumerate(collector):
        total_frames += batch.numel()

        # Compute advantage using GAE
        with torch.no_grad():
            adv_module(batch)

        # Flatten batch for training
        batch_flat = batch.reshape(-1)

        # PPO update - iterate through batch with mini-batches
        minibatch_size = min(64, len(batch_flat))
        loss_sum = torch.tensor(0.0)
        loss_objective_sum = 0.0
        loss_critic_sum = 0.0
        loss_entropy_sum = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(len(batch_flat))

            for start_idx in range(0, len(batch_flat), minibatch_size):
                end_idx = min(start_idx + minibatch_size, len(batch_flat))
                mb_indices = indices[start_idx:end_idx]
                mb = batch_flat[mb_indices].to(device)

                # Compute PPO loss
                loss = loss_module(mb)

                # Aggregate losses
                loss_sum = (
                    loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                )

                # Track individual losses for logging
                loss_objective_sum += loss["loss_objective"].item()
                loss_critic_sum += loss["loss_critic"].item()
                loss_entropy_sum += loss["loss_entropy"].item()
                num_updates += 1

                optim.zero_grad()
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()

        # Logging
        reward = batch["next", "reward"].mean().item()
        done_rate = batch["next", "done"].float().mean().item()

        writer.add_scalar("reward/mean", reward, total_frames)
        writer.add_scalar("done_rate", done_rate, total_frames)
        writer.add_scalar("loss/total", loss_sum.item(), total_frames)

        if num_updates > 0:
            writer.add_scalar(
                "loss/objective", loss_objective_sum / num_updates, total_frames
            )
            writer.add_scalar(
                "loss/critic", loss_critic_sum / num_updates, total_frames
            )
            writer.add_scalar(
                "loss/entropy", loss_entropy_sum / num_updates, total_frames
            )

        elapsed = time.time() - start_time
        fps = total_frames / elapsed

        pbar.update(batch.numel())
        pbar.set_postfix(reward=f"{reward:.2f}", fps=f"{fps:.0f}")

    pbar.close()

    # Save model
    model_path = f"{save_folder}/{env.base_env.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "algorithm": algorithm,
            "num_agents": num_agents,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    writer.close()
    collector.shutdown()
    env.close()

    print(f"Finished MARL training on {env.base_env.metadata['name']}.")
