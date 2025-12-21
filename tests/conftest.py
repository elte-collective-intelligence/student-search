"""
Shared test fixtures and helpers for SearchAndRescueEnv tests.

This module centralizes common testing utilities following DRY, KISS, and SOLID principles.
"""

import os
from typing import Optional

import numpy as np
import pytest

from src.sar_env import SearchAndRescueEnv


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

    def _make(**kwargs) -> SearchAndRescueEnv:
        # Explicitly set continuous_actions=True unless overridden by caller
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
    """Calculate slice indices for different observation components.

    Observation structure:
    - [0:2] Self velocity (2)
    - [2:4] Self position (2)
    - [4:4+num_rescuers] Agent ID one-hot (num_rescuers)
    - Safe zones (num_safe_zones * 3: rel_x, rel_y, type)
    - Trees (num_trees * 2: rel_x, rel_y)
    - Victims (num_victims * 3: rel_x, rel_y, type)

    Returns:
        Dictionary mapping component names to their slice indices.
    """
    base = 4  # vel(2) + pos(2)
    agent_id_end = base + env.num_rescuers
    safe_zones_end = agent_id_end + env.num_safe_zones * 3
    trees_end = safe_zones_end + env.num_trees * 2
    victims_end = trees_end + env.num_victims * 3

    return {
        "self_vel": slice(0, 2),
        "self_pos": slice(2, 4),
        "agent_id": slice(4, agent_id_end),
        "safe_zones": slice(agent_id_end, safe_zones_end),
        "trees": slice(safe_zones_end, trees_end),
        "victims": slice(trees_end, victims_end),
    }


def get_tree_obs(obs_vec: np.ndarray, slices: dict, tree_idx: int) -> np.ndarray:
    """Extract observation for a specific tree."""
    start = slices["trees"].start + tree_idx * 2
    return obs_vec[start : start + 2]  # noqa: E203


def get_victim_obs(obs_vec: np.ndarray, slices: dict, victim_idx: int) -> np.ndarray:
    """Extract observation for a specific victim."""
    victim_slice = slices["victims"]
    victim_span = victim_slice.stop - victim_slice.start
    num_victims = victim_span // 3 if victim_span >= 0 else 0
    if not (0 <= victim_idx < num_victims):
        raise IndexError(
            f"victim_idx {victim_idx} is out of range for {num_victims} victims"
        )
    start = victim_slice.start + victim_idx * 3
    return obs_vec[start : start + 3]  # noqa: E203


def get_safe_zone_obs(obs_vec: np.ndarray, slices: dict, zone_idx: int) -> np.ndarray:
    """Extract observation for a specific safe zone."""
    start = slices["safe_zones"].start + zone_idx * 3
    return obs_vec[start : start + 3]  # noqa: E203


def is_masked_victim(victim_obs: np.ndarray, atol: float = 1e-5) -> bool:
    """Check if a victim observation is masked (invisible/saved)."""
    return np.allclose(victim_obs, [0.0, 0.0, -1.0], atol=atol)


def is_masked_tree(tree_obs: np.ndarray, atol: float = 1e-5) -> bool:
    """Check if a tree observation is masked (invisible)."""
    return np.allclose(tree_obs, [0.0, 0.0], atol=atol)


# =============================================================================
# Position Helpers
# =============================================================================


def place_agent(env: SearchAndRescueEnv, agent_idx: int, pos: tuple[float, float]):
    """Place an agent at a specific position."""
    env.rescuer_pos[agent_idx] = np.array(pos, dtype=np.float64)


def place_victim(env: SearchAndRescueEnv, victim_idx: int, pos: tuple[float, float]):
    """Place a victim at a specific position."""
    env.victim_pos[victim_idx] = np.array(pos, dtype=np.float64)


def place_tree(env: SearchAndRescueEnv, tree_idx: int, pos: tuple[float, float]):
    """Place a tree at a specific position."""
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
    atol: float = 1e-5,
    msg: str = "",
):
    """Assert that observation arrays match within tolerance."""
    assert np.allclose(actual, expected, atol=atol), (
        msg or f"Observation mismatch. Got {actual}, expected {expected}"
    )
