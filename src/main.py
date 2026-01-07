"""
Main entry point for Search and Rescue training and evaluation.
"""

import subprocess

import hydra
from omegaconf import DictConfig

from src.eval import evaluate
from src.train import train


def launch_tensorboard(log_dir: str, port: int = 6006):
    """Launch TensorBoard in a subprocess."""
    print(f"Launching TensorBoard at http://localhost:{port}")
    print(f"Log directory: {log_dir}")
    try:
        subprocess.run(
            ["tensorboard", "--logdir", log_dir, "--port", str(port)], check=True
        )
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"TensorBoard error: {e}")
    except FileNotFoundError:
        print("TensorBoard not found. Install it with: pip install tensorboard")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Check for tensorboard mode
    tb_active = cfg.get("tensorboard", {}).get("active", False)

    if tb_active:
        launch_tensorboard(
            log_dir=cfg.save_folder, port=cfg.tensorboard.get("port", 6006)
        )
        return

    assert (
        cfg.train.active ^ cfg.eval.active
    ), "Please specify one of train.active=true or eval.active=true in the arguments."

    # Determine if logging is enabled (default to True if tensorboard config exists)
    enable_logging = cfg.get("tensorboard", {}).get("enabled", True)
    if "tensorboard" in cfg and "enabled" not in cfg.tensorboard:
        # If tensorboard config exists but enabled is not set, default to True
        enable_logging = True

    env_kwargs = {
        "num_victims": cfg.env.victims,
        "num_rescuers": cfg.env.rescuers,
        "num_trees": cfg.env.trees,
        "num_safe_zones": cfg.env.safe_zones,
        "max_cycles": cfg.env.max_cycles,
        "continuous_actions": cfg.env.continuous_actions,
        "vision_radius": cfg.env.vision_radius,
        "randomize_safe_zones": cfg.env.get("randomize_safe_zones", False),
        "n_closest_landmarks": cfg.env.get("n_closest_landmarks", 3),
        "energy_enabled": cfg.env.energy.enabled,
        "max_energy": cfg.env.energy.max_energy,
        "movement_cost_coeff": cfg.env.energy.movement_cost_coeff,
        "idle_cost": cfg.env.energy.idle_cost,
        "energy_depleted_action_scale": cfg.env.energy.depleted_action_scale,
        "num_chargers": cfg.env.energy.num_chargers,
        "recharge_radius": cfg.env.energy.recharge_radius,
        "recharge_rate": cfg.env.energy.recharge_rate,
        "randomize_chargers": cfg.env.energy.randomize_chargers,
    }

    if cfg.train.active:
        # Extract curriculum parameters if enabled
        curriculum_kwargs = {}
        if cfg.get("curriculum", {}).get("enabled", False):
            curriculum_kwargs = {
                "curriculum_enabled": True,
                "curriculum_min_trees": cfg.curriculum.min_trees,
                "curriculum_max_trees": cfg.curriculum.max_trees,
                "curriculum_num_stages": cfg.curriculum.num_stages,
            }

        train(
            steps=cfg.train.total_timesteps,
            batch_size=cfg.train.batch_size,
            frames_per_batch=cfg.train.frames_per_batch,
            seed=cfg.seed,
            save_folder=cfg.save_folder,
            enable_logging=enable_logging,
            render_mode=cfg.train.render_mode,
            num_epochs=cfg.train.n_epochs,
            **env_kwargs,
            **curriculum_kwargs,
        )
    elif cfg.eval.active:
        evaluate(
            num_games=cfg.eval.games,
            save_folder=cfg.save_folder,
            enable_logging=enable_logging,
            render_mode=cfg.eval.render_mode,
            seed=cfg.seed,
            model_path=cfg.eval.model_path,
            **env_kwargs,
        )


if __name__ == "__main__":
    main()
