import os
import pytest

from src.domain.sar_env import SearchAndRescueEnv


@pytest.fixture(autouse=True)
def _headless_pygame():
    # Avoid pygame trying to open a real window in CI / headless envs
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    yield


@pytest.fixture
def make_env():
    def _make(**kwargs) -> SearchAndRescueEnv:
        return SearchAndRescueEnv(render_mode=None, device="cpu", **kwargs)

    return _make
