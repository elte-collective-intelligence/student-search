from typing import Union

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueOperator
from torch.distributions import Categorical


class SplitLayer(nn.Module):
    def forward(self, x):
        # x shape: [..., 2, action_dim]
        return x[..., 0, :], x[..., 1, :].exp()  # loc, scale (positive)


def make_policy(
    env,
    num_rescuers: int,
    device: Union[torch.device, str] = "cpu",
    discrete: bool = False,
) -> ProbabilisticActor:
    if discrete:
        # Discrete action policy
        policy_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=5,  # 5 discrete actions
            n_agents=num_rescuers,
            centralised=False,  # strictly local
            share_params=True,  # Homogenous agents share weights
            device=device,
            depth=2,
            num_cells=64,
            activation_class=torch.nn.Tanh,
        ).to(device)

        policy_module = TensorDictModule(
            module=policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
        )

        return ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=[("agents", "logits")],
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
        )
    else:
        # Continuous action policy
        policy_net = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                n_agent_outputs=env.action_spec["agents", "action"].shape[-1]
                * 2,  # Mean + Std
                n_agents=num_rescuers,
                centralised=False,  # strictly local
                share_params=True,  # Homogenous agents share weights
                device=device,
                depth=2,
                num_cells=64,
                activation_class=torch.nn.Tanh,
            ),
            # Helper to split output into mean and log_std for sampling
            nn.Unflatten(-1, (2, env.action_spec["agents", "action"].shape[-1])),
        ).to(device)

        policy_module = TensorDictModule(
            module=nn.Sequential(policy_net, SplitLayer()),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        return ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )


def make_critic(env, num_rescuers: int, device: Union[torch.device, str] = "cpu"):

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=num_rescuers,
        centralised=True,  # MAPPO: Critic sees all
        share_params=True,
        device=device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    )

    return ValueOperator(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
