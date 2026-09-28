"""Play the policy from the newest checkpoint."""

import argparse
from datetime import datetime
import glob
import os

import genesis as gs
import torch
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable

from microduck.algorithm.models import MLPModel
from microduck.env import Env

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENT_S = 5.0  # duration of each demo-schedule segment

# (name, [vx, vy, wz]) cycled while playing.
DEMO_SCHEDULE = [
    ("stand", [0.0, 0.0, 0.0]),
    ("forward", [0.3, 0.0, 0.0]),
    ("backward", [-0.3, 0.0, 0.0]),
    ("left", [0.0, 0.2, 0.0]),
    ("right", [0.0, -0.2, 0.0]),
    ("turn left", [0.0, 0.0, 0.8]),
    ("turn right", [0.0, 0.0, -0.8]),
]


def find_latest_checkpoint(root: str) -> str:
    candidates = glob.glob(os.path.join(root, "**", "model_*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt found under '{root}'")
    return max(candidates, key=os.path.getmtime)


def load_run_cfg(checkpoint: str) -> DictConfig:
    """Load the training config stored alongside a checkpoint."""
    path = os.path.join(os.path.dirname(checkpoint), ".hydra", "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Training config '{path}' not found; the checkpoint must stay in its run directory"
        )
    return OmegaConf.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the newest MicroDuck checkpoint.")
    parser.add_argument("--headless", action="store_true", help="run without the viewer")
    args = parser.parse_args()

    checkpoint = find_latest_checkpoint(os.path.join(REPO_ROOT, "outputs"))
    print(f"[play] checkpoint: {checkpoint}")
    cfg = load_run_cfg(checkpoint)
    cfg.env.sim.num_envs = 1
    cfg.env.sim.episode_length_s = 1e6  # no time-outs while playing

    backends = {"cpu": gs.cpu, "gpu": gs.gpu}
    gs.init(backend=backends[cfg.backend], seed=cfg.seed, logging_level="warning")
    env = Env(
        env_cfg=cfg.env,
        robot_cfg=cfg.robot,
        show_viewer=not args.headless,
        record_video=True,
    )
    obs = env.reset()

    # Only the actor is needed; its normalizer stats are part of the state dict.
    actor = MLPModel(obs, cfg.algo.obs_groups, "actor", env.action_dim, **cfg.algo.actor).to(env.device)
    loaded = torch.load(checkpoint, map_location=env.device, weights_only=False)
    actor.load_state_dict(loaded["actor_state_dict"])
    actor.eval()
    print(f"[play] loaded iteration {loaded.get('iter')}")

    command = env.command_term
    steps_per_segment = round(SEGMENT_S / env.step_dt)
    total_steps = steps_per_segment * len(DEMO_SCHEDULE)

    run_dir = os.path.dirname(checkpoint)                            # outputs/<date>/<time>
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(checkpoint))[0]   # model_999
    video_path = os.path.join(video_dir, f"{model_name}.mp4")

    env.record_camera.start_recording(video_path, fps=50)
    print(f"[play] recording video: {video_path}")

    summary_table = PrettyTable()
    summary_table.title = "Demo Schedule Summary"
    summary_table.field_names = [
        "Segment",
        "Cmd vx",
        "Cmd vy",
        "Cmd wz",
        "Avg vx",
        "Avg vy",
        "Avg wz",
        "Avg z (m)",
    ]
    summary_table.align["Segment"] = "l"
    vel_sum = torch.zeros(3, device=env.device)
    height_sum = 0.0
    segment_steps = 0
    step = 0
    segment, cmd = DEMO_SCHEDULE[0]
    try:
        with torch.inference_mode():
            while step < total_steps:
                if step % steps_per_segment == 0:
                    segment, cmd = DEMO_SCHEDULE[(step // steps_per_segment) % len(DEMO_SCHEDULE)]
                    command.set_fixed_command(cmd)
                    obs = env.get_observations()  # observation must see the new command

                actions = actor(obs)  # deterministic: mean action
                obs, _, dones, _ = env.step(actions)
                step += 1

                vel_sum += torch.stack([env.base_lin_vel[0, 0], env.base_lin_vel[0, 1], env.base_ang_vel[0, 2]])
                height_sum += env.base_pos[0, 2].item()
                segment_steps += 1
                if dones[0]:
                    print(f"{step * env.step_dt:6.1f}  ** fell, reset **")
                if segment_steps == steps_per_segment:
                    m = (vel_sum / segment_steps).tolist()
                    summary_table.add_row(
                        [
                            segment,
                            *(f"{value:.2f}" for value in cmd),
                            *(f"{value:.2f}" for value in m),
                            f"{height_sum / segment_steps:.3f}",
                        ]
                    )
                    vel_sum.zero_()
                    height_sum = 0.0
                    segment_steps = 0
    except KeyboardInterrupt:
        print("[play] interrupted")
    finally:
        if segment_steps:
            summary_table.add_row(
                [
                    f"{segment} (partial)",
                    *(f"{value:.2f}" for value in cmd),
                    *(f"{value:.2f}" for value in (vel_sum / segment_steps).tolist()),
                    f"{height_sum / segment_steps:.3f}",
                ]
            )
        print(summary_table)
        env.record_camera.stop_recording()
        print(f"[play] video saved: {video_path}")


if __name__ == "__main__":
    main()