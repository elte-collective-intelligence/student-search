from time import sleep, perf_counter

from src.sar_env import make_env
from src.models import make_policy
import glob
import os
import torch
from torchrl.envs.utils import step_mdp
from threading import Thread, Event
from queue import Queue, Empty


def find_latest_model(save_folder: str, env_name: str) -> str:
    """
    Finds the latest .pt file in the save_folder based on modification time.
    """
    # Check for flat files (based on user prompt naming convention)
    # Pattern: save_folder/search_rescue_v2_YYYYMMDD-HHMMSS.pt
    search_pattern = os.path.join(save_folder, "*.pt")

    files = glob.glob(search_pattern)

    if not files:
        raise FileNotFoundError(f"No model files (*.pt) found in {save_folder}")

    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Auto-detected latest model: {latest_file}")
    return latest_file


class LivePlotter:
    """
    Background matplotlib updater that keeps UI responsive without blocking the
    main evaluation loop. The main thread pushes (step, rewards[]) to a Queue.
    The plotter thread updates Line2D objects at a capped FPS.
    """

    def __init__(self, title_prefix: str = "Per-step rewards per agent", fps: int = 30):
        self.title_prefix = title_prefix
        self.fps = max(1, int(fps))
        self.queue: Queue = Queue()
        self.stop_evt = Event()
        self.thread: Thread | None = None
        self.fig = None
        self.ax = None
        self.lines = []
        self.num_agents = 0
        self.last_draw = 0.0

    def _worker(self):
        import matplotlib.pyplot as plt  # ensure backend is already chosen

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Reward")
        # Fix y-scale for better visibility (ignore +100 spikes)
        self.ax.set_ylim(-5, 10)
        colors = plt.cm.get_cmap("tab10")

        xs = [[] for _ in range(self.num_agents)]
        ys = [[] for _ in range(self.num_agents)]
        self.lines = []
        for a_idx in range(self.num_agents):
            (line,) = self.ax.plot(
                [], [], label=f"agent_{a_idx}", color=colors(a_idx % 10)
            )
            self.lines.append(line)
        self.ax.legend(loc="upper right")
        plt.show(block=False)

        min_dt = 1.0 / self.fps
        self.last_draw = perf_counter()

        while not self.stop_evt.is_set():
            updated = False
            # Drain queue quickly to batch updates
            try:
                while True:
                    msg = self.queue.get_nowait()
                    if msg[0] == "reset":
                        # msg: ("reset", episode_idx, num_agents)
                        _, epi, n_agents = msg
                        self.num_agents = n_agents
                        xs = [[] for _ in range(self.num_agents)]
                        ys = [[] for _ in range(self.num_agents)]
                        self.ax.cla()
                        self.ax.set_title(f"{self.title_prefix} (Episode {epi})")
                        self.ax.set_xlabel("Step")
                        self.ax.set_ylabel("Reward")
                        # Keep fixed y-scale on reset
                        self.ax.set_ylim(-5, 10)
                        colors = plt.cm.get_cmap("tab10")
                        self.lines = []
                        for a_idx in range(self.num_agents):
                            (line,) = self.ax.plot(
                                [], [], label=f"agent_{a_idx}", color=colors(a_idx % 10)
                            )
                            self.lines.append(line)
                        self.ax.legend(loc="upper right")
                        updated = True
                    elif msg[0] == "data":
                        # msg: ("data", step, rewards_array)
                        _, step_i, rewards = msg
                        for a_idx, r in enumerate(rewards):
                            if a_idx >= len(xs):
                                continue
                            xs[a_idx].append(step_i)
                            ys[a_idx].append(float(r))
                            self.lines[a_idx].set_data(xs[a_idx], ys[a_idx])
                        updated = True
                    elif msg[0] == "title":
                        _, title = msg
                        self.ax.set_title(title)
                        updated = True
                    else:
                        # Unknown message type
                        pass
            except Empty:
                pass

            now = perf_counter()
            if updated and (now - self.last_draw) >= min_dt:
                # Rescale x-axis based on data while keeping fixed y-limits
                self.ax.relim()
                try:
                    self.ax.autoscale(enable=True, axis="x", tight=False)
                except Exception:
                    # Fallback if autoscale signature differs
                    self.ax.autoscale_view(scalex=True, scaley=False)
                self.fig.canvas.draw_idle()
                # flush_events keeps UI responsive without blocking the main loop
                self.fig.canvas.flush_events()
                self.last_draw = now

            # Small sleep to avoid busy loop
            # Use a tiny pause to keep UI event loop responsive without blocking main loop
            import matplotlib.pyplot as plt

            plt.pause(0.001)

        # Final draw on stop
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def start(self, num_agents: int, episode_idx: int):
        self.num_agents = num_agents
        if self.thread is None or not self.thread.is_alive():
            self.stop_evt.clear()
            self.thread = Thread(target=self._worker, daemon=True)
            self.thread.start()
        self.queue.put(("reset", episode_idx, num_agents))

    def push(self, step_i: int, rewards):
        self.queue.put(("data", step_i, rewards))

    def set_title(self, title: str):
        self.queue.put(("title", title))

    def stop(self):
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        import matplotlib.pyplot as plt

        plt.ioff()


def evaluate(
    model_path: str = None,
    save_folder: str = "search_rescue_logs",
    num_games: int = 3,
    plot_cfg: dict | None = None,
    **env_kwargs,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Create Environment (Human Render Mode)
    # We need to create the env first to get the metadata name for auto-discovery
    print("Initializing environment...")
    env = make_env(device=device, **env_kwargs)

    # 2. Resolve Model Path
    if not model_path:
        print(f"No model path provided. Searching in '{save_folder}'...")
        try:
            env_name = env.base_env.metadata["name"]
            model_path = find_latest_model(save_folder, env_name)
        except Exception as e:
            print(f"Error finding model: {e}")
            return

    # 3. Load Model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_agents = env.action_spec["agents", "action"].shape[0]

    policy = make_policy(env, num_rescuers=num_agents, device=device)

    # Handle different saving formats
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        # Format from the robust training script
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print(
            f"Loaded checkpoint from step {checkpoint.get('steps', 'N/A')} "
            f"with reward {checkpoint.get('mean_reward', 'N/A')}"
        )
    elif isinstance(checkpoint, dict) and "actor" in checkpoint:
        # Format from older simple script
        policy.load_state_dict(checkpoint["actor"])
    else:
        # Raw state dict
        policy.load_state_dict(checkpoint)

    policy.eval()  # Set to evaluation mode

    # 4. Evaluation Loop
    print(f"Starting evaluation for {num_games} episodes...")

    # Live plotting (non-blocking) via background thread
    live = False
    plot_fps = 30
    if plot_cfg:
        live = bool(plot_cfg.get("live", False))
        plot_fps = int(plot_cfg.get("fps", 30))

    plotter = LivePlotter(fps=plot_fps) if live else None

    for i in range(num_games):
        td = env.reset()
        done = False
        step_count = 0

        print(f"--- Episode {i + 1} ---")

        # Reset plotting for this episode
        # Determine number of agents dynamically from env/action spec
        num_agents = env.action_spec["agents", "action"].shape[0]
        if plotter:
            plotter.start(num_agents=num_agents, episode_idx=i + 1)

        while not done:
            with torch.no_grad():
                td = policy(td)

            td = env.step(td)
            env.render()

            if "next" in td.keys():
                # Standard TorchRL behavior
                if td["next", "done"].any():
                    done = True
                td = step_mdp(td)
            else:
                # Flat behavior (PettingZooWrapper sometimes does this)
                # The 'td' returned IS the next state
                # Check for "done", "terminated", or "agents/done"
                if "done" in td.keys() and td["done"].any():
                    done = True
                elif "terminated" in td.keys() and td["terminated"].any():
                    done = True
                elif ("agents", "done") in td.keys(include_nested=True) and td[
                    "agents", "done"
                ].any():
                    done = True

            step_count += 1

            # Plot the rewards of each agent at each step (no console printing)
            if ("agents", "reward") in td.keys(include_nested=True):
                rewards = td["agents", "reward"].detach().cpu().numpy()

                # Send data to plotter thread (non-blocking)
                if plotter:
                    plotter.push(step_count, rewards)

            sleep(0.1)

        print(f"Episode {i + 1} finished in {step_count} steps.")
        # Brief spacing; plotter thread continues rendering
        if plotter:
            sleep(0.2)

    print("Evaluation finished.")
    env.close()
    if plotter:
        plotter.stop()
