"""
Training script using TorchRL with PPO.
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

from sar_env import SearchAndRescueEnv
from models import make_ppo_models


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
    seed: int = 0,
    save_folder: str = "search_rescue_logs/",
    **env_kwargs,
):
    """Train a PPO agent on the search and rescue environment."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    env_kwargs["seed"] = seed
    env = make_env(env_kwargs, device=device)

    # Check environment specs
    check_env_specs(env)

    print(f"Starting training on {env.base_env.metadata['name']}.")
    print(f"Observation shape: {env.observation_spec['observation'].shape}")
    print(f"Action spec: {env.action_spec}")

    # Create models
    actor, critic = make_ppo_models(env, device=device)

    # Create GAE module
    adv_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
    )

    # Create PPO loss
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
    frames_per_batch = min(256, steps // 4)
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

        # Compute advantage
        with torch.no_grad():
            adv_module(batch)

        # Flatten batch for training
        batch_flat = batch.reshape(-1)

        # PPO update - iterate through batch directly
        batch_size = min(64, len(batch_flat))
        loss_sum = torch.tensor(0.0)

        for _ in range(ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(len(batch_flat))

            for start_idx in range(0, len(batch_flat), batch_size):
                end_idx = min(start_idx + batch_size, len(batch_flat))
                mb_indices = indices[start_idx:end_idx]
                mb = batch_flat[mb_indices].to(device)

                loss = loss_module(mb)

                loss_sum = (
                    loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                )

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
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    writer.close()
    collector.shutdown()
    env.close()

    print(f"Finished training on {env.base_env.metadata['name']}.")
