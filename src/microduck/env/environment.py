import math
import torch
import genesis as gs

from hydra.utils import instantiate
from genesis.utils.geom import inv_quat, transform_by_quat


from .mdp import *
from ..assets import resolve_model_path

class Env:
    def __init__(
        self,
        env_cfg,
        robot_cfg,
        show_viewer=False,
    ) -> None:
        self.env_cfg = env_cfg
        self.robot_cfg = robot_cfg

        self._init_scene(show_viewer)
        self._init_robot()
        self._init_mdp()
        self._init_buffers()

        self._update_robot_state()


    def _init_scene(self, show_viewer: bool = False) -> None:
        """Create the scene, add ground and robot, then build it."""
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt),
            # The robot is ~15 cm tall: put the camera close to it.
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.6, -0.6, 0.35),
                camera_lookat=(0.0, 0.0, 0.1),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(resolve_model_path(self.robot_cfg.model_path))
            )
        )
        self.scene.build(n_envs=self.num_envs, env_spacing=(0.5, 0.5))
        if show_viewer:
            self.scene.viewer.follow_entity(self.robot)


    def _init_robot(self) -> None:
        """Motor dof indices, PD gains and the robot's default state."""
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
 
        # Feet: local link indices, used for contact sensing.
        self.feet_link_idx = [self.robot.get_link(name).idx_local for name in cfg.feet_link_names]
        self.feet_site_offset = torch.tensor(
            [list(o) for o in cfg.feet_site_offsets], dtype=gs.tc_float, device=self.device
        )  # (num_feet, 3), in the foot link frame

        # Default state, used at reset.
        self.default_joint_pos = self._tensor([j.default_pos for j in cfg.joints])
        self.init_base_pos = self._tensor(cfg.pos)
        self.init_base_quat = self._tensor(cfg.quat)  # (w, x, y, z)


    def _init_mdp(self) -> None:
        """Instantiate MDP terms from env_cfg."""
        self.command_term = instantiate(self.env_cfg.command, env=self)
        self.action_term = instantiate(self.env_cfg.action, env=self)

        self.observation_manager = instantiate(self.env_cfg.observation, env=self)
        self.reward_manager = instantiate(self.env_cfg.reward, env=self)
        self.termination_manager = instantiate(self.env_cfg.termination, env=self)


    def _init_buffers(self) -> None:
        """Per-env state buffers updated every step."""
        n, device = self.num_envs, self.device
 
        self.episode_length_buf = torch.zeros(n, dtype=gs.tc_int, device=device)
        self.reset_buf = torch.ones(n, dtype=torch.bool, device=device)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.rew_buf = torch.zeros(n, dtype=gs.tc_float, device=device)
 
        self.gravity_vec = self._tensor([0.0, 0.0, -1.0]).repeat(n, 1)

        # Per-foot time since the foot left / touched the ground [s], (num_envs, num_feet).
        num_feet = len(self.feet_link_idx)
        self.feet_air_time = torch.zeros(n, num_feet, dtype=gs.tc_float, device=device)
        self.feet_contact_time = torch.zeros(n, num_feet, dtype=gs.tc_float, device=device)
        # Highest sole height of the current swing, and its value at touchdown.
        self.feet_peak_height = torch.zeros(n, num_feet, dtype=gs.tc_float, device=device)
        self.feet_landing_peak = torch.zeros(n, num_feet, dtype=gs.tc_float, device=device)
        self.feet_first_contact = torch.zeros(n, num_feet, dtype=torch.bool, device=device)

        self.extras = {}

    def _tensor(self, data) -> torch.Tensor:
        return torch.tensor(list(data), dtype=gs.tc_float, device=self.device)


    @property
    def num_envs(self) -> int:
        return self.env_cfg.sim.num_envs


    @property
    def observation_dim(self) -> dict[str, int]:
        return self.observation_manager.group_dims


    @property
    def action_dim(self) -> int:
        return self.action_term.action_dim


    @property
    def physics_dt(self) -> float:
        return self.env_cfg.sim.physics_dt


    @property
    def step_dt(self) -> float:
        return self.env_cfg.sim.physics_dt * self.env_cfg.sim.decimation


    @property
    def device(self) -> torch.device:
        return gs.device


    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.env_cfg.sim.episode_length_s / self.step_dt)


    def _update_robot_state(self) -> None:
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_q)   # body frame
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_q)   # body frame
        self.projected_gravity = transform_by_quat(self.gravity_vec, inv_q)
        self.base_ang_vel_w = self.robot.get_ang()                            # world frame
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        # Net contact force on each foot link (world frame), from the last physics step.
        self.feet_contact_force = self.robot.get_links_net_contact_force()[:, self.feet_link_idx, :]
        self.feet_contact = self.feet_contact_force.norm(dim=-1) > self.robot_cfg.contact_force_threshold

        # Sole points (world frame): p = p_link + R_link r,  v = v_link + w_link x (R_link r).
        link_pos = self.robot.get_links_pos(self.feet_link_idx)          # (N, F, 3)
        link_quat = self.robot.get_links_quat(self.feet_link_idx)        # (N, F, 4)
        link_vel = self.robot.get_links_vel(self.feet_link_idx)          # (N, F, 3)
        link_ang = self.robot.get_links_ang(self.feet_link_idx)          # (N, F, 3)
        offset_w = transform_by_quat(self.feet_site_offset.expand_as(link_pos), link_quat)
        self.feet_pos = link_pos + offset_w
        self.feet_vel = link_vel + torch.cross(link_ang, offset_w, dim=-1)
        self.feet_height = self.feet_pos[..., 2]                         # flat ground at z = 0


    def _update_feet_timers(self, dt: float) -> None:
        """Advance per-foot air / contact timers by one env step."""
        # Touchdown: in contact now, but was in the air at the previous step.
        self.feet_first_contact = self.feet_contact & (self.feet_air_time > 0.0)
        in_air = ~self.feet_contact
        self.feet_peak_height = torch.where(
            in_air, torch.maximum(self.feet_peak_height, self.feet_height), self.feet_peak_height
        )
        self.feet_landing_peak = torch.where(self.feet_first_contact, self.feet_peak_height, 0.0)
        self.feet_peak_height = torch.where(self.feet_first_contact, 0.0, self.feet_peak_height)
        self.feet_air_time = torch.where(self.feet_contact, 0.0, self.feet_air_time + dt)
        self.feet_contact_time = torch.where(self.feet_contact, self.feet_contact_time + dt, 0.0)


    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset the given envs to the default standing state."""
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        self.robot.set_pos(self.init_base_pos.repeat(n, 1), envs_idx=envs_idx, zero_velocity=True)
        self.robot.set_quat(self.init_base_quat.repeat(n, 1), envs_idx=envs_idx, zero_velocity=True)

        self.robot.set_dofs_position(
            self.default_joint_pos.repeat(n, 1),
            self.motors_dof_idx,
            envs_idx=envs_idx,
            zero_velocity=True
        )

        # Episode statistics for logging. Must run before episode_length_buf is cleared.
        episode = {}
        episode.update(self.reward_manager.reset(envs_idx))
        episode.update(self.termination_manager.reset(envs_idx))
        episode.update({f"Metrics/{k}": v for k, v in self.command_term.reset(envs_idx).items()})
        self.extras["episode"] = episode

        self.action_term.reset(envs_idx)

        self.episode_length_buf[envs_idx] = 0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_contact_time[envs_idx] = 0.0
        self.feet_peak_height[envs_idx] = 0.0
        self.feet_landing_peak[envs_idx] = 0.0
        self.feet_first_contact[envs_idx] = False
        self.reset_buf[envs_idx] = False
        self.time_out_buf[envs_idx] = False

        self._update_robot_state()


    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all envs."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()


    def step(
        self, actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        self.extras = {}
        self.action_term.process(actions)
        for _ in range(self.env_cfg.sim.decimation):
            self.action_term.apply()
            self.scene.step()
        self.episode_length_buf += 1
        self._update_robot_state()
        self._update_feet_timers(self.step_dt)

        # Termination, then reward.
        self.reset_buf = self.termination_manager.compute()
        self.time_out_buf = self.termination_manager.time_outs
        self.rew_buf = self.reward_manager.compute(self.step_dt)

        dones = self.reset_buf.clone()
        if not self.env_cfg.is_finite_horizon:
            self.extras["time_outs"] = self.time_out_buf.clone()

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
        self.command_term.compute(self.step_dt)   # timer-based resampling

        obs = self.get_observations()
        return obs, self.rew_buf, dones, self.extras


    def get_observations(self) -> dict[str, torch.Tensor]:
        return self.observation_manager.compute()