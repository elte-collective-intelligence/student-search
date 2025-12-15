"""
Evaluation script for the Search and Rescue environment.
Supports visualization and testing trained models.
"""

import glob
import os
import time

import torch

from src.sar_env import SearchAndRescueEnv
from src.models import make_actor
from src.metrics import (
    EpisodeTracker,
    aggregate_logs,
    compute_summary,
    plot_core_metrics,
)


def evaluate(
    num_games: int = 100,
    save_folder: str = "search_rescue_logs/",
    render_mode=None,
    **env_kwargs,
):
    """Evaluate a trained agent with visualization support."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment with rendering
    env = SearchAndRescueEnv(render_mode=render_mode, **env_kwargs)
    env_name = getattr(env, "env_id", type(env).__name__)

    print(
        f"\nStarting evaluation on {env_name} "
        f"(num_games={num_games}, render_mode={render_mode})"
    )

    # Create actor for loading weights (no log prob needed for eval)
    actor = make_actor(env, device=device, return_log_prob=False)

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
                return None
        else:
            # Find latest policy
            pattern = f"{save_folder}/{env_name}*.pt"
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
    episode_logs = []

    for game in range(num_games):
        tracker = EpisodeTracker(game + 1)
        td = env.reset()
        game_reward = 0.0

        while True:
            # Render
            if render_mode == "human":
                env.render()
                time.sleep(0.05)  # Slow down for visualization

            current_actor = actor
            if current_actor is not None:
                actor_core = current_actor.get_submodule("actor")
                with torch.no_grad():
                    out = actor_core(td)

                # ensure action ends up in td
                if isinstance(out, dict) or hasattr(out, "get"):
                    td = out
                else:
                    td["action"] = out
            else:
                # Random action
                action = torch.tensor(
                    env.sample_discrete_action(),
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
            tracker.record(env, reward)

            if done:
                break

            # Update td for next iteration
            td = td["next"].clone()

        # Count rescued victims
        log = tracker.finalize(env)
        rescued = log.rescues
        total_rescues.append(rescued)
        total_rewards.append(game_reward)
        episode_logs.append(log)

        print(
            f"Game {game + 1}/{num_games}: "
            f"Reward={game_reward:.2f}, "
            f"Rescued={rescued}/{len(env.victims)}"
        )

    env.close()

    # Statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_rescues = sum(total_rescues) / len(total_rescues)

    df = aggregate_logs(episode_logs)
    summary = compute_summary(df, len(env.victims))
    plots = plot_core_metrics(df, os.path.join(save_folder, "plots"))

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average rescues: {avg_rescues:.2f}/{len(env.victims)}")
    print(f"  Rescues completed: {summary['rescues_pct']:.1f}%")
    print(f"  Average collisions: {summary['avg_collisions']:.2f}")
    print(f"  Average coverage cells: {summary['avg_coverage_cells']:.2f}")
    print(
        f"  Avg time to first rescue: {summary['avg_time_to_first_rescue']:.2f} steps"
    )
    print("  Plots saved:")
    for label, path in plots.items():
        print(f"    {label}: {path}")
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
            env.sample_discrete_action(),
            dtype=torch.int64,
        )
        td["action"] = action

        td = env.step(td)

        if td["next", "done"].item():
            print(f"Episode finished at step {step}")
            td = env.reset()

    env.close()
