import math
import torch
import genesis as gs

from hydra.utils import instantiate

from .mdp import *
from ..assets import resolve_model_path

class Env:
    def __init__(
        self,
        env_cfg,
        robot_cfg,
        show_viewer=False,
    ):
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg

        self._init_scene(show_viewer)
        self._init_robot()
        self._init_mdp()
        self._init_buffers()


    """Create the scene, add ground and robot, then build it."""
    def _init_scene(self, show_viewer: bool = False):
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(resolve_model_path(self.robot_cfg.model_path)),
                pos=self.robot_cfg.pos,
                quat=self.robot_cfg.quat,
            )
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(0.5, 0.5))


    """Motor dof indices, PD gains and the robot's default state."""
    def _init_robot(self):
        cfg = self.robot_cfg
 
        self.motor_names = [j.name for j in cfg.joints]
        self.motors_dof_idx = [
            self.robot.get_joint(name).dofs_idx_local[0] for name in self.motor_names
        ]
        num_motors = len(self.motors_dof_idx)
 
        self.robot.set_dofs_kp([cfg.kp] * num_motors, self.motors_dof_idx)
        self.robot.set_dofs_kv([cfg.kv] * num_motors, self.motors_dof_idx)
        self.robot.set_dofs_force_range(
            [-cfg.effort_limit] * num_motors,
            [cfg.effort_limit] * num_motors,
            self.motors_dof_idx,
        )
 
        # Default state, used at reset.
        self.default_joint_pos = self._tensor([j.default_pos for j in cfg.joints])
        self.init_base_pos = self._tensor(cfg.pos)
        self.init_base_quat = self._tensor(cfg.quat)  # (w, x, y, z)


    """Instantiate MDP terms from env_cfg."""
    def _init_mdp(self):
        self.action_term = instantiate(self.env_cfg.action, env=self)


    """Per-env state buffers updated every step."""
    def _init_buffers(self):
        n, device = self.num_envs, self.device
 
        self.episode_length_buf = torch.zeros(n, dtype=gs.tc_int, device=device)
        self.reset_buf = torch.ones(n, dtype=torch.bool, device=device)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.rew_buf = torch.zeros(n, dtype=gs.tc_float, device=device)
 
        self.gravity_vec = self._tensor([0.0, 0.0, -1.0]).repeat(n, 1)


    def _tensor(self, data) -> torch.Tensor:
        return torch.tensor(list(data), dtype=gs.tc_float, device=self.device)


    @property
    def num_envs(self):
        return self.env_cfg.sim.num_envs


    @property
    def observation_dim(self):
        raise NotImplementedError


    @property
    def action_dim(self):
        return self.action_term.action_dim


    @property
    def physics_dt(self):
        return self.env_cfg.sim.physics_dt


    @property
    def step_dt(self):
        return self.env_cfg.sim.physics_dt * self.env_cfg.sim.decimation


    @property
    def device(self):
        return gs.device


    @property
    def max_episode_length(self):
        return math.ceil(self.env_cfg.sim.episode_length_s / self.step_dt)


    def resample_command(self):
        pass


    def reset_idx(self):
        pass


    def reset(self):
        pass


    def step(self):
        pass


    def apply_action(self):
        pass


    def get_observations(self):
        pass


    def get_states(self):
        pass


    def get_rewards(self):
        pass


    def get_dones(self):
        pass