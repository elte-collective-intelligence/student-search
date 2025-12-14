"""
Evaluation script for the Search and Rescue environment.
Supports visualization and testing trained models.
"""

import glob
import os
import time
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import ProbabilisticActor

from sar_env import SearchAndRescueEnv


def make_actor(env, device="cpu"):
    """Create actor network matching training architecture."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_spec = env.action_spec

    is_continuous = hasattr(action_spec, "low")

    if is_continuous:
        action_dim = action_spec.shape[-1]
        actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * action_dim),
        )
        actor_module = TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=torch.distributions.Normal,
            return_log_prob=False,
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
            return_log_prob=False,
        )

    return actor.to(device)


def eval(
    env_fn,
    num_games: int = 100,
    save_folder: str = "search_rescue_logs/",
    render_mode=None,
    **env_kwargs,
):
    """Evaluate a trained agent with visualization support."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment with rendering
    env = SearchAndRescueEnv(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {env.metadata['name']} "
        f"(num_games={num_games}, render_mode={render_mode})"
    )

    # Create actor for loading weights
    actor = make_actor(env, device=device)

    # Option to manually specify policy name
    manual_policy_name = input(
        "Enter the policy name (e.g., policy_name.pt) "
        "or press Enter to load the latest: "
    ).strip()

    try:
        if manual_policy_name:
            if os.path.exists(manual_policy_name):
                checkpoint = torch.load(manual_policy_name, map_location=device)
                actor.load_state_dict(checkpoint["actor"])
                print(f"Loaded policy: {manual_policy_name}")
            else:
                print(f"Policy '{manual_policy_name}' not found.")
                env.close()
                return
        else:
            # Find latest policy
            pattern = f"{save_folder}/{env.metadata['name']}*.pt"
            policies = glob.glob(pattern)
            if not policies:
                print(f"No policies found matching {pattern}")
                print("Running with random actions instead.")
                actor = None
            else:
                latest_policy = max(policies, key=os.path.getctime)
                checkpoint = torch.load(latest_policy, map_location=device)
                actor.load_state_dict(checkpoint["actor"])
                print(f"Loaded the latest policy: {latest_policy}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("Running with random actions instead.")
        actor = None

    # Evaluation loop
    total_rewards = []
    total_rescues = []

    for game in range(num_games):
        td = env.reset()
        game_reward = 0.0

        while True:
            # Render
            if render_mode == "human":
                env.render()
                time.sleep(0.05)  # Slow down for visualization

            # Get action
            if actor is not None:
                with torch.no_grad():
                    td = actor(td)
                action = td["action"]
            else:
                # Random action
                action = torch.tensor(
                    env._np_random.randint(0, 5),
                    dtype=torch.int64,
                    device=device,
                )
                td["action"] = action

            # Step environment
            td = env.step(td)

            # Extract results
            reward = td["next", "reward"].item()
            done = td["next", "done"].item()

            game_reward += reward

            if done:
                break

            # Update td for next iteration
            td = td["next"].clone()

        # Count rescued victims
        rescued = sum(1 for v in env.victims if v.saved)
        total_rescues.append(rescued)
        total_rewards.append(game_reward)

        print(
            f"Game {game + 1}/{num_games}: "
            f"Reward={game_reward:.2f}, "
            f"Rescued={rescued}/{len(env.victims)}"
        )

    env.close()

    # Statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_rescues = sum(total_rescues) / len(total_rescues)

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average rescues: {avg_rescues:.2f}/{len(env.victims)}")
    print(f"  Total games: {num_games}")
    print("=" * 50)

    return avg_reward


def visualize_random(env_kwargs, num_steps=500):
    """Run visualization with random actions for testing."""
    env = SearchAndRescueEnv(render_mode="human", **env_kwargs)

    print(f"Running random visualization for {num_steps} steps...")

    td = env.reset()

    for step in range(num_steps):
        env.render()
        time.sleep(0.05)

        # Random action
        action = torch.tensor(
            env._np_random.randint(0, 5),
            dtype=torch.int64,
        )
        td["action"] = action

        td = env.step(td)

        if td["next", "done"].item():
            print(f"Episode finished at step {step}")
            td = env.reset()

    env.close()


if __name__ == "__main__":
    # Quick test
    env_kwargs = {
        "num_missing": 4,
        "num_rescuers": 3,
        "num_trees": 8,
        "num_safezones": 4,
        "max_cycles": 120,
    }
    visualize_random(env_kwargs, num_steps=200)
