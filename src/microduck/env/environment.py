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

        # init scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(resolve_model_path(robot_cfg.model_path)),
                pos=robot_cfg.pos,
                quat=robot_cfg.quat,
            )
        )

        self.scene.build(
            n_envs=self.num_envs,
            env_spacing=(0.5, 0.5),
        )

        self.action_term = instantiate(env_cfg.action, env = self)

        # init robot
        self.motors_dof_idx = [
            self.robot.get_joint(joint.name).dofs_idx_local[0]
            for joint in robot_cfg.joints
        ]

        self.robot.set_dofs_kp(
            [robot_cfg.kp] * len(self.motors_dof_idx),
            self.motors_dof_idx,
        )
        self.robot.set_dofs_kv(
            [robot_cfg.kv] * len(self.motors_dof_idx),
            self.motors_dof_idx,
        )
        self.robot.set_dofs_force_range(
            [-robot_cfg.effort_limit] * len(self.motors_dof_idx),
            [robot_cfg.effort_limit] * len(self.motors_dof_idx),
            self.motors_dof_idx,
        )

        self.default_joint_pos = torch.tensor(
            [joint.default_pos for joint in robot_cfg.joints],
            dtype=gs.tc_float,
            device=self.device,
        )

        self.init_base_pos = torch.tensor(
            robot_cfg.pos,
            dtype=gs.tc_float,
            device=self.device,
        )

        self.init_base_quat = torch.tensor(
            robot_cfg.quat,
            dtype=gs.tc_float,
            device=self.device,
        )

        self.gravity_direction = torch.tensor(
            [0.0, 0.0, -1.0],
            dtype=gs.tc_float,
            device=self.device,
        )

    @property
    def num_envs(self):
        return self.env_cfg.sim.num_envs

    @property
    def observation_dim(self):
        pass

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

    def init_scene(self):
        pass

    def init_robot(self):
        pass

    def init_buffer(self):
        pass
