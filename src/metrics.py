"""Utility helpers for computing evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol

import os
import numpy as np
import pandas as pd

# Matplotlib is only needed for offline plotting, so use a non-interactive backend.
import matplotlib
import matplotlib.pyplot as plt

import torch

matplotlib.use("Agg")


@dataclass
class EpisodeLog:
    """Container for per-episode statistics."""

    episode: int
    rewards: float = 0.0
    steps: int = 0
    rescues: int = 0
    collisions: int = 0
    boundary_violations: int = 0
    unique_cells: int = 0
    time_to_first_rescue: int = 0


def _hash_pos(pos: np.ndarray, cell_size: float) -> tuple[int, int]:
    return tuple(np.floor(pos / cell_size).astype(int))


def _count_collisions(env, main_rescuer) -> int:
    collisions = 0
    for agent in env.agents:
        if agent is main_rescuer:
            continue
        if env.is_collision(main_rescuer, agent):  # noqa: SLF001
            collisions += 1
    for tree in env.trees:
        if env.is_collision(main_rescuer, tree):  # noqa: SLF001
            collisions += 1
    return collisions


class PolicyFn(Protocol):
    def __call__(self, tensordict) -> torch.Tensor: ...


class EpisodeTracker:
    """Accumulates metrics during an episode without replaying it."""

    @property
    def unique_cells(self) -> int:
        return len(self._visited_cells)

    def __init__(self, episode_idx: int, cell_size: float = 0.05):
        self.log = EpisodeLog(episode=episode_idx)
        self._cell_size = cell_size
        self._visited_cells: set[tuple[int, int]] = set()

    def record(self, env, reward: float):
        self.log.steps += 1
        self.log.rewards += reward

        main_rescuer = env.rescuers[0]
        self._visited_cells.add(_hash_pos(main_rescuer.p_pos.copy(), self._cell_size))

        self.log.collisions += _count_collisions(env, main_rescuer)

        if env.bound_penalty(main_rescuer.p_pos) > 0:  # noqa: SLF001
            self.log.boundary_violations += 1

        current_rescues = sum(1 for v in env.victims if v.saved)
        if self.log.time_to_first_rescue == 0 and current_rescues > 0:
            self.log.time_to_first_rescue = self.log.steps

    def finalize(self, env) -> EpisodeLog:
        self.log.rescues = sum(1 for v in env.victims if v.saved)
        self.log.unique_cells = len(self._visited_cells)
        if self.log.time_to_first_rescue == 0:
            self.log.time_to_first_rescue = self.log.steps
        return self.log


def track_episode(env, tracker: EpisodeTracker, policy_fn: PolicyFn):
    td = env.reset()
    tracker.log.rescues = 0

    while True:
        action = policy_fn(td)
        td["action"] = action
        td = env.step(td)

        reward = td["next", "reward"].item()
        tracker.record(env, reward)

        if td["next", "done"].item():
            tracker.log.rescues = sum(1 for v in env.victims if v.saved)
            tracker.log.unique_cells = tracker.unique_cells
            if tracker.log.time_to_first_rescue == 0:
                tracker.log.time_to_first_rescue = tracker.log.steps
            break

        td = td["next"].clone()

    return tracker.log


def run_episode(
    env, policy_fn, episode_idx: int, cell_size: float = 0.05
) -> EpisodeLog:
    """Roll out one episode and capture metrics."""

    td = env.reset()
    total_reward = 0.0
    steps = 0
    collision_count = 0
    boundary_hits = 0
    visited_cells = set()
    time_to_first_rescue = None

    while True:
        steps += 1

        action = policy_fn(td)
        td["action"] = action
        td = env.step(td)

        reward = td["next", "reward"].item()
        done = td["next", "done"].item()

        total_reward += reward

        main_rescuer = env.rescuers[0]
        visited_cells.add(_hash_pos(main_rescuer.p_pos.copy(), cell_size))

        collision_count += _count_collisions(env, main_rescuer)

        if env.bound_penalty(main_rescuer.p_pos) > 0:  # noqa: SLF001
            boundary_hits += 1

        current_rescues = sum(1 for v in env.victims if v.saved)
        if time_to_first_rescue is None and current_rescues > 0:
            time_to_first_rescue = steps

        if done:
            break

        td = td["next"].clone()

    rescues = sum(1 for v in env.victims if v.saved)

    return EpisodeLog(
        episode=episode_idx,
        rewards=total_reward,
        steps=steps,
        rescues=rescues,
        collisions=collision_count,
        boundary_violations=boundary_hits,
        unique_cells=len(visited_cells),
        time_to_first_rescue=time_to_first_rescue or steps,
    )


def aggregate_logs(logs: List[EpisodeLog]) -> pd.DataFrame:
    return pd.DataFrame([log.__dict__ for log in logs])


def compute_summary(df: pd.DataFrame, victims_total: int) -> Dict[str, float]:
    return {
        "rescues_pct": 100.0 * df["rescues"].mean() / max(victims_total, 1),
        "avg_collisions": df["collisions"].mean(),
        "avg_coverage_cells": df["unique_cells"].mean(),
        "avg_time_to_first_rescue": df["time_to_first_rescue"].mean(),
    }


def plot_core_metrics(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    plot_plan = {
        "rescues": {
            "ylabel": "Victims Rescued",
            "title": "Rescues Completed per Episode",
        },
        "collisions": {
            "ylabel": "Collision Count",
            "title": "Collisions per Episode",
        },
        "unique_cells": {
            "ylabel": "Unique Cells Visited",
            "title": "Coverage per Episode",
        },
    }

    for key, meta in plot_plan.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["episode"], df[key], marker="o")
        ax.set_xlabel("Episode")
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(meta["title"])
        ax.grid(True, alpha=0.3)
        plot_path = Path(output_dir) / f"{key}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        plots[key] = str(plot_path)

    return plots
