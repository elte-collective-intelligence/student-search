"""
Shared test fixtures and helpers for SearchAndRescueEnv tests.

This module centralizes common testing utilities following DRY, KISS, and SOLID principles.
"""

import os
from typing import Optional

import numpy as np
import pytest

from src.sar_env import SearchAndRescueEnv
from src.seed_utils import set_seed


# =============================================================================
# Environment Setup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _headless_pygame():
    """Avoid pygame trying to open a real window in CI / headless envs."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    yield


@pytest.fixture
def make_env():
    """Factory fixture to create SearchAndRescueEnv with custom parameters.

    IMPORTANT: Explicitly sets continuous_actions=True to ensure tests work
    regardless of the environment's default value. This makes tests resilient
    to configuration changes.
    """

    def _make(seed=None, **kwargs) -> SearchAndRescueEnv:
        # Explicitly set continuous_actions=True unless overridden by caller
        if seed is not None:
            set_seed(seed)
        kwargs.setdefault("continuous_actions", True)
        return SearchAndRescueEnv(render_mode=None, **kwargs)

    return _make


@pytest.fixture
def make_simple_env(make_env):
    """Factory fixture for small environments commonly used in vision tests.

    Provides sensible defaults for simple test scenarios.
    """

    def _make(
        *,
        num_rescuers: int = 1,
        num_victims: int = 1,
        num_trees: int = 1,
        num_safe_zones: int = 4,
        seed: int = 42,
        vision_radius: float = 1.0,
        **kwargs,
    ) -> SearchAndRescueEnv:
        return make_env(
            num_rescuers=num_rescuers,
            num_victims=num_victims,
            num_trees=num_trees,
            num_safe_zones=num_safe_zones,
            seed=seed,
            vision_radius=vision_radius,
            **kwargs,
        )

    return _make


@pytest.fixture
def env_simple(make_simple_env) -> SearchAndRescueEnv:
    """Pre-configured simple environment, already reset."""
    env = make_simple_env()
    env.reset()
    return env


# =============================================================================
# Action Helpers
# =============================================================================


def noop_action() -> np.ndarray:
    """Return a no-op continuous action."""
    return np.array([0.0, 0.0], dtype=np.float32)


def noop_actions(env: SearchAndRescueEnv) -> dict[str, np.ndarray]:
    """Return no-op actions for all agents in the environment."""
    return {agent: noop_action() for agent in env.agents}


def move_action(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Return a continuous movement action."""
    return np.array([dx, dy], dtype=np.float32)


# =============================================================================
# Observation Helpers
# =============================================================================


def get_obs_slices(env: SearchAndRescueEnv) -> dict[str, slice]:
    """
    Updated observation structure:
    - [0:2] Self velocity (2)
    - [2:4] Self position (2)
    - [4:4+num_rescuers] Agent ID one-hot (num_rescuers)
    - [..:+1] Energy (optional, normalized)
    - landmarks (n_closest_landmarks * 5: rel_x, rel_y, visible_bit, is_safezone_bit, safezone_type)
    - victims (num_victims * 4: rel_x, rel_y, type, visible_bit)
    - other rescuers ((num_rescuers - 1) * 3: rel_x, rel_y, visible_bit)
    """
    base = 4  # vel(2) + pos(2)
    agent_id_end = base + env.num_rescuers

    energy_end = agent_id_end + (1 if getattr(env, "energy_enabled", False) else 0)

    landmarks_end = energy_end + env.n_closest_landmarks * 5
    victims_end = landmarks_end + env.num_victims * 4
    others_end = victims_end + (env.num_rescuers - 1) * 3

    slices = {
        "self_vel": slice(0, 2),
        "self_pos": slice(2, 4),
        "agent_id": slice(4, agent_id_end),
        "landmarks": slice(energy_end, landmarks_end),
        "victims": slice(landmarks_end, victims_end),
        "other_agents": slice(victims_end, others_end),
    }
    if getattr(env, "energy_enabled", False):
        slices["energy"] = slice(agent_id_end, energy_end)

    return slices


def get_landmark_obs(
    obs_vec: np.ndarray, slices: dict[str, slice], landmark_idx: int
) -> np.ndarray:
    """Extract observation for a specific landmark (rel_x, rel_y)."""
    lm_slice = slices["landmarks"]
    n_landmarks = (lm_slice.stop - lm_slice.start) // 5
    if not (0 <= landmark_idx < n_landmarks):
        raise IndexError(
            f"landmark_idx {landmark_idx} out of bounds for {n_landmarks} landmarks"
        )
    start = lm_slice.start + landmark_idx * 5
    return obs_vec[start : start + 2]  # noqa: E203


def get_victim_obs(
    obs_vec: np.ndarray, slices: dict[str, slice], victim_idx: int
) -> np.ndarray:
    """Extract observation for a specific victim."""
    victim_slice = slices["victims"]
    victim_span = victim_slice.stop - victim_slice.start
    num_victims = victim_span // 4 if victim_span >= 0 else 0
    if not (0 <= victim_idx < num_victims):
        raise IndexError(
            f"victim_idx {victim_idx} is out of range for {num_victims} victims"
        )
    start = victim_slice.start + victim_idx * 4
    return obs_vec[start : start + 4]  # noqa: E203


def is_masked_victim(victim_obs: np.ndarray, atol: float = 1e-5) -> bool:
    """Check if a victim observation is masked (invisible/saved)."""
    return np.allclose(victim_obs, [0.0, 0.0, 0.0, 0.0], atol=atol)


# =============================================================================
# Position Helpers
# =============================================================================


def place_agent(env: SearchAndRescueEnv, agent_idx: int, pos: tuple[float, float]):
    """Place an agent at a specific position."""
    if not (0 <= agent_idx < env.num_rescuers):
        raise ValueError(
            f"Invalid agent_idx {agent_idx}; expected 0 <= agent_idx < {env.num_rescuers}."
        )
    env.rescuer_pos[agent_idx] = np.array(pos, dtype=np.float64)


def place_victim(env: SearchAndRescueEnv, victim_idx: int, pos: tuple[float, float]):
    """Place a victim at a specific position."""
    num_victims = len(env.victim_pos)
    if victim_idx < 0 or victim_idx >= num_victims:
        raise ValueError(
            f"victim_idx {victim_idx} is out of range for {num_victims} victims"
        )
    env.victim_pos[victim_idx] = np.array(pos, dtype=np.float64)


def place_tree(env: SearchAndRescueEnv, tree_idx: int, pos: tuple[float, float]):
    """Place a tree at a specific position."""
    # Validate tree index to provide clearer failures in tests.
    num_trees = getattr(env, "num_trees", len(env.tree_pos))
    if not isinstance(tree_idx, int) or not (0 <= tree_idx < num_trees):
        raise ValueError(
            f"tree_idx {tree_idx} is out of bounds for environment with {num_trees} trees."
        )
    env.tree_pos[tree_idx] = np.array(pos, dtype=np.float64)


# =============================================================================
# Visibility Helpers (wrapper for testing private _is_visible method)
# =============================================================================


def check_visibility(
    env: SearchAndRescueEnv,
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    target_radius: float,
    exclude_tree_idx: Optional[int] = None,
) -> bool:
    """Check if target is visible from observer.

    This is a test helper that wraps the private _is_visible method.
    Used for testing visibility/occlusion logic.
    """
    # Use getattr to access protected method - intentional for testing
    is_visible_fn = getattr(env, "_is_visible")
    return is_visible_fn(observer_pos, target_pos, target_radius, exclude_tree_idx)


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_visible(
    env: SearchAndRescueEnv,
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    target_radius: float,
    exclude_tree_idx: Optional[int] = None,
    msg: str = "",
):
    """Assert that target is visible from observer."""
    result = check_visibility(
        env, observer_pos, target_pos, target_radius, exclude_tree_idx
    )
    assert result, (
        msg or f"Expected target to be visible from {observer_pos} to {target_pos}"
    )


def assert_not_visible(
    env: SearchAndRescueEnv,
    observer_pos: np.ndarray,
    target_pos: np.ndarray,
    target_radius: float,
    exclude_tree_idx: Optional[int] = None,
    msg: str = "",
):
    """Assert that target is NOT visible from observer."""
    result = check_visibility(
        env, observer_pos, target_pos, target_radius, exclude_tree_idx
    )
    assert not result, (
        msg or f"Expected target to NOT be visible from {observer_pos} to {target_pos}"
    )


def assert_obs_matches(
    actual: np.ndarray,
    expected: np.ndarray,
    scale: float = 1.0,
    atol: float = 1e-5,
    msg: str = "",
):
    """Assert that observation arrays match within tolerance, with optional scaling."""
    if scale != 1.0:
        expected = expected / scale
    assert np.allclose(actual, expected, atol=atol), (
        msg or f"Observation mismatch. Got {actual}, expected {expected}"
    )
