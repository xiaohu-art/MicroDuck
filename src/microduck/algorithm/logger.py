"""Training metrics logger with optional TensorBoard output."""

import statistics
import time
from collections import deque

import torch

__all__ = ["Logger"]


class Logger:
    def __init__(self, log_dir: str | None, num_envs: int, num_steps_per_env: int, device) -> None:
        self.log_dir = log_dir
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.device = device

        self.writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            except ImportError:
                print("[Logger] tensorboard is not installed, logging to console only")

        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(num_envs, device=device)
        self.cur_episode_length = torch.zeros(num_envs, device=device)
        self.ep_extras: list[dict] = []

        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.start_time = time.time()

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Update episode statistics for one environment step."""
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])

        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self.rewbuffer.extend(self.cur_reward_sum[done_ids].tolist())
            self.lenbuffer.extend(self.cur_episode_length[done_ids].tolist())
            self.cur_reward_sum[done_ids] = 0.0
            self.cur_episode_length[done_ids] = 0.0

    def log(
        self,
        it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict[str, float],
        learning_rate: float,
        action_std: torch.Tensor,
    ) -> None:
        """Record and print metrics for one training iteration."""
        collection_size = self.num_steps_per_env * self.num_envs
        iteration_time = collect_time + learn_time
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        fps = int(collection_size / iteration_time)

        scalars: dict[str, float] = {}

        # Episode extras reported by the env (e.g. velocity tracking errors).
        for key in dict.fromkeys(k for ep in self.ep_extras for k in ep):
            values = [float(ep[key]) for ep in self.ep_extras if key in ep]
            scalars[key if "/" in key else f"Episode/{key}"] = statistics.mean(values)
        self.ep_extras.clear()

        for key, value in loss_dict.items():
            scalars[f"Loss/{key}"] = value
        scalars["Loss/learning_rate"] = learning_rate

        # Per-dim std, averaged over the batch: mean and min over action dims.
        std_per_dim = action_std.reshape(-1, action_std.shape[-1]).mean(dim=0)
        scalars["Policy/mean_std"] = std_per_dim.mean().item()
        scalars["Policy/min_std"] = std_per_dim.min().item()

        scalars["Perf/total_fps"] = fps
        scalars["Perf/collection_time"] = collect_time
        scalars["Perf/learning_time"] = learn_time

        if len(self.rewbuffer) > 0:
            scalars["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)

        if self.writer is not None:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, it)

        # Console summary.
        pad = 32
        lines = [f"{'#' * 20} Learning iteration {it}/{total_it} {'#' * 20}"]
        for key in ("Train/mean_reward", "Train/mean_episode_length"):
            if key in scalars:
                lines.append(f"{key + ':':>{pad}} {scalars[key]:.3f}")
        for key, value in scalars.items():
            if key.startswith(("Loss/", "Policy/", "Episode/")):
                lines.append(f"{key + ':':>{pad}} {value:.4g}")
        lines.append(f"{'Perf/fps:':>{pad}} {fps} (collect {collect_time:.2f}s, learn {learn_time:.2f}s)")
        lines.append(f"{'Total timesteps:':>{pad}} {self.tot_timesteps}")
        eta = self.tot_time / (it + 1) * (total_it - it - 1) if it + 1 < total_it else 0.0
        lines.append(f"{'Elapsed / ETA:':>{pad}} {self.tot_time:.1f}s / {eta:.1f}s")
        print("\n".join(lines))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()