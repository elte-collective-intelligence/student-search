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

    env_kwargs = {
        "num_victims": cfg.env.victims,
        "num_rescuers": cfg.env.rescuers,
        "num_trees": cfg.env.trees,
        "num_safe_zones": cfg.env.safe_zones,
        "max_cycles": cfg.env.max_cycles,
        "continuous_actions": cfg.env.continuous_actions,
        "vision_radius": cfg.env.vision_radius,
    }

    if cfg.train.active:
        train(
            steps=cfg.train.total_timesteps,
            batch_size=cfg.train.batch_size,
            seed=cfg.train.seed,
            render_mode=cfg.train.render_mode,
            save_folder=cfg.save_folder,
            **env_kwargs,
        )
    elif cfg.eval.active:
        evaluate(
            num_games=cfg.eval.games,
            render_mode=cfg.eval.render_mode,
            save_folder=cfg.save_folder,
            plot_cfg=cfg.eval.get("plot", {}),
            **env_kwargs,
        )


if __name__ == "__main__":
    main()
