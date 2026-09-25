"""Train MicroDuck with PPO.

    python scripts/train.py
    python scripts/train.py env.sim.num_envs=4096 algo.max_iterations=1500
    python scripts/train.py algo.algorithm.learning_rate=5e-4 seed=1

Checkpoints and TensorBoard logs go to the hydra run directory
(outputs/<date>/<time>/), next to the resolved config (.hydra/config.yaml).
"""

import genesis as gs
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from microduck.algorithm import PPO
from microduck.env import Env


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    backends = {"cpu": gs.cpu, "gpu": gs.gpu}
    gs.init(backend=backends[cfg.backend], seed=cfg.seed, logging_level="warning")

    env = Env(env_cfg=cfg.env, robot_cfg=cfg.robot, show_viewer=cfg.show_viewer)
    env.reset()

    log_dir = HydraConfig.get().runtime.output_dir
    print(f"Logging to {log_dir}")
    runner = PPO(env, cfg.algo, log_dir=log_dir)
    runner.learn(init_at_random_ep_len=True)


if __name__ == "__main__":
    main()