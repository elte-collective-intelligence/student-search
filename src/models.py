"""
Neural network models for MAPPO training and evaluation.

Implements CTDE (Centralized Training with Decentralized Execution):
- Actor: Uses local observations (decentralized execution)
- Critic: Uses global state (centralized training)

The environment handles multi-agent coordination internally.
Training is done on single-agent transitions with a shared policy.
"""

import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal


def make_actor(env, device="cpu", return_log_prob=True):
    """Create actor network for PPO/MAPPO.

    Uses local observation for decentralized execution.
    The same policy is shared across all agents (parameter sharing).

    Args:
        env: The environment to create the actor for.
        device: Device to place the model on.
        return_log_prob: Whether to return log probabilities (True for training, False for eval).

    Returns:
        ProbabilisticActor module.
    """
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_spec = env.action_spec

    is_continuous = hasattr(action_spec, "low")

    if is_continuous:
        action_dim = action_spec.shape[-1] if len(action_spec.shape) > 0 else 2

        actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * action_dim),  # mean and std
        )

        # Split network output into loc and scale
        class SplitLocScale(nn.Module):
            def __init__(self, net, action_dim):
                super().__init__()
                self.net = net
                self.action_dim = action_dim

            def forward(self, x):
                out = self.net(x)
                loc = out[..., : self.action_dim]
                scale = (
                    torch.nn.functional.softplus(
                        out[..., self.action_dim :]  # noqa: E203
                    )
                    + 1e-4
                )
                return loc, scale

        split_net = SplitLocScale(actor_net, action_dim)

        actor_module = TensorDictModule(
            split_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=return_log_prob,
        )
    else:
        n_actions = action_spec.space.n

        actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, n_actions),
        )

        actor_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["logits"],
        )

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=return_log_prob,
        )

    return actor.to(device)


def make_critic(env, device="cpu"):
    """Create critic (value) network for PPO.

    Uses local observation - suitable for independent PPO (IPPO).

    Args:
        env: The environment to create the critic for.
        device: Device to place the model on.

    Returns:
        ValueOperator module.
    """
    obs_dim = env.observation_spec["observation"].shape[-1]

    critic_net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    )

    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    return critic.to(device)


def make_mappo_critic(env, device="cpu"):
    """Create centralized critic for MAPPO (CTDE).

    Uses global state for centralized training with decentralized execution.
    The global state contains full observability:
        - All rescuer positions: num_rescuers * 2
        - All rescuer velocities: num_rescuers * 2
        - All victim positions: num_missing * 2
        - All victim states: num_missing * 1
        - All tree positions: num_trees * 2
        - All safe zone positions: num_safe_zones * 2

    Args:
        env: The environment to create the critic for.
        device: Device to place the model on.

    Returns:
        ValueOperator module that uses "state" key.
    """
    state_dim = env.observation_spec["state"].shape[-1]

    critic_net = nn.Sequential(
        nn.Linear(state_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    )

    critic = ValueOperator(
        module=critic_net,
        in_keys=["state"],  # Uses global state, not local observation
        out_keys=["state_value"],
    )

    return critic.to(device)


def make_ppo_models(env, device="cpu"):
    """Create actor and critic networks for PPO (IPPO).

    Uses local observations for both actor and critic.
    This is Independent PPO - each agent learns independently.

    Args:
        env: The environment to create models for.
        device: Device to place models on.

    Returns:
        Tuple of (actor, critic) modules.
    """
    actor = make_actor(env, device=device, return_log_prob=True)
    critic = make_critic(env, device=device)
    return actor, critic


def make_mappo_models(env, device="cpu"):
    """Create actor and critic networks for MAPPO (CTDE).

    Actor uses local observation (decentralized execution).
    Critic uses global state (centralized training).

    This implements the CTDE paradigm:
    - During training: Critic has access to global state for better value estimation
    - During execution: Actors only use local observations

    Args:
        env: The environment to create models for.
        device: Device to place models on.

    Returns:
        Tuple of (actor, critic) modules.
    """
    actor = make_actor(env, device=device, return_log_prob=True)
    critic = make_mappo_critic(env, device=device)
    return actor, critic
