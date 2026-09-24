import torch
import hydra
import genesis as gs
from omegaconf import DictConfig, OmegaConf

from microduck.env import Env

@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):
    backends = {"cpu": gs.cpu, "gpu": gs.gpu}
    gs.init(backend=backends[cfg.backend], seed=cfg.seed)
    
    env = Env(
        env_cfg=cfg.env,
        robot_cfg=cfg.robot,
        show_viewer=cfg.show_viewer,
    )

    env.robot.set_dofs_position(env.default_joint_pos, env.motors_dof_idx)   # 初始化姿态
    zeros = torch.zeros(env.num_envs, env.action_dim, device=env.device)
    for i in range(250):
        env.action_term.process(zeros)
        env.action_term.apply()
        env.scene.step()

if __name__ == "__main__":
    main()