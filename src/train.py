import os
import time

import torch
from torchrl.envs import check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss, ValueEstimators
import tqdm

from src.models import make_policy, make_critic
from src.sar_env import make_env


def train(
    steps: int = 100000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    frames_per_batch: int = 2048,
    seed: int = 0,
    save_folder: str = "search_rescue_logs/",
    **env_kwargs,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
    # Output: ("agents", "loc"), ("agents", "scale")
    policy = make_policy(env, num_rescuers=num_agents, device=device)

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

    for batch in collector:
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

        minibatch_size = 128

        # 2. PPO Update
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample(minibatch_size)
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

        # 3. Logging
        pbar.update(batch.numel())

        # Check for completed episodes in the batch to log reward
        # Note: Using next/done to filter
        done_mask = batch["next", "agents", "done"].any(dim=-1)  # If any agent done
        if done_mask.any():
            mean_reward = (
                batch["next", "agents", "episode_reward"][done_mask].mean().item()
            )
            rewards_log.append(mean_reward)
            pbar.set_description(f"Mean Reward: {mean_reward:.2f}")

        collector.update_policy_weights_()

    pbar.close()

    os.makedirs(save_folder, exist_ok=True)

    # Save model
    model_path = f"{save_folder}/{env.base_env.metadata['name']}_{time.strftime('%Y%m%d-%H%M%S')}.pt"

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        model_path,
    )

    collector.shutdown()
    env.close()

    print(f"Finished MARL training on {env.base_env.metadata['name']}.")
