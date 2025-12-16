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
    env_name = getattr(env, "env_id", type(env).__name__)

    # Check environment specs
    check_env_specs(env)

    print(f"Starting training on {env_name}.")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Observation shape: {env.observation_spec['observation'].shape}")
    print(f"Global state shape: {env.observation_spec['state'].shape}")
    print(f"Action spec: {env.action_spec}")

    # Create models based on algorithm choice
    if algorithm.lower() == "mappo":
        # MAPPO: actor uses local obs, critic uses global state (CTDE)
        actor, critic = make_mappo_models(env, device=device)
        print("Critic using: global state (CTDE)")
    else:
        # PPO: both actor and critic use local observation
        actor, critic = make_ppo_models(env, device=device)
        print("Critic using: local observation")

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

        batch_frames = batch.numel()
        remaining = steps - total_frames
        effective_frames = min(batch_frames, remaining)
        if effective_frames <= 0:
            print("Reached the requested step budget; stopping data collection.")
            break

        # Truncate the batch to only process effective_frames so we don't compute adv/loss
        # on frames that shouldn't count toward the budget. Works for TensorDicts/tensors
        if effective_frames < batch_frames:
            try:
                batch = batch[:effective_frames]
            except (TypeError, AttributeError, RuntimeError, IndexError):
                # Best-effort fallback: flatten then slice
                batch = batch.reshape(-1)[:effective_frames]

        # If the final collected batch is larger than needed, only count the needed frames.
        total_frames += effective_frames

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
        fps = total_frames / elapsed if elapsed > 0 else 0.0

        pbar.update(effective_frames)
        pbar.set_postfix(reward=f"{reward:.2f}", fps=f"{fps:.0f}")

    pbar.close()

    # Save model
    model_path = f"{save_folder}/{env_name}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
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

    print(f"Finished training on {env_name}.")
