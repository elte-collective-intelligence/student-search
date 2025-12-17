from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import hydra


def to_abs_path(path: str) -> Path:
    """
    Resolve paths robustly even when Hydra changes the working directory.
    """
    if hydra is not None:
        return Path(hydra.utils.to_absolute_path(path)).expanduser().resolve()
    return Path(path).expanduser().resolve()


def make_run_id(
    env_name: str,
    algorithm: str,
    seed: int,
    timestamp: Optional[str] = None,
) -> str:
    """
    Produce a unique, readable run id.

    Example:
      SAR_MAPPO_seed0_20251216-021455
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{env_name}_{algorithm.upper()}_seed{seed}_{ts}"


@dataclass(frozen=True)
class RunContext:
    """
    Canonical, centralized run paths.

    save_root:
      Base folder for all run artifacts.
    tb_root:
      Root folder for TensorBoard runs (contains subfolders per run_id).
    ckpt_root:
      Folder for checkpoints.
    plots_root:
      Folder for plots / figures.
    run_id:
      Unique id for the run.
    """

    save_root: Path
    tb_root: Path
    ckpt_root: Path
    plots_root: Path
    run_id: str

    @property
    def tb_run_dir(self) -> Path:
        return self.tb_root / self.run_id

    def ensure_dirs(self) -> "RunContext":
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tb_root.mkdir(parents=True, exist_ok=True)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        self.plots_root.mkdir(parents=True, exist_ok=True)
        self.tb_run_dir.mkdir(parents=True, exist_ok=True)
        return self


def make_run_context(
    save_folder: str,
    env_name: str,
    algorithm: str,
    seed: int,
    run_name: Optional[str] = None,
) -> RunContext:
    """
    Create a RunContext and ensure all directories exist.

    Convention:
      <save_folder>/
        tensorboard/<run_id>/
        checkpoints/
        plots/
    """
    save_root = to_abs_path(save_folder)

    ctx = RunContext(
        save_root=save_root,
        tb_root=save_root / "tensorboard",
        ckpt_root=save_root / "checkpoints",
        plots_root=save_root / "plots",
        run_id=run_name or make_run_id(env_name, algorithm, seed),
    )
    return ctx.ensure_dirs()


def env_kwargs_from_cfg(cfg: Any) -> Dict[str, Any]:
    """
    Optional helper: map Hydra cfg -> SearchAndRescueEnv kwargs.
    Keeps main.py clean and makes config mapping explicit and testable.
    """
    return {
        "num_missing": int(cfg.env.missing),
        "num_rescuers": int(cfg.env.rescuers),
        "num_trees": int(cfg.env.trees),
        "num_safe_zones": int(cfg.env.safe_zones),
        "max_cycles": int(cfg.env.max_cycles),
        "continuous_actions": bool(cfg.env.continuous_actions),
    }
