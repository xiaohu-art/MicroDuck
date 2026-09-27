import torch
import genesis as gs

from ...utils import resolve_matching_names

__all__ = ["JointPositionAction"]

class JointPositionAction:
    def __init__(
        self,
        env,
        joint_names,
        scale: float = 1.0,
        clip: float | None = 1.0
    ) -> None:
        self.env = env
        self.scale = scale
        self.clip = clip

        # Indices into the motor list (= env.motors_dof_idx / env.default_joint_pos order).
        robot_joint_names = [j.name for j in env.robot_cfg.joints]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, robot_joint_names)

        self.default_pos = env.default_joint_pos[self.joint_ids]
        self.joint_targets = env.default_joint_pos.repeat(self.num_envs, 1)   # (N, num_motors)

        self.raw_actions = torch.zeros(
            self.num_envs, self.action_dim, dtype=gs.tc_float, device=self.device
        )
        # Previous step's raw_actions, for the action-rate reward.
        self.prev_actions = torch.zeros_like(self.raw_actions)


    @property
    def action_dim(self) -> int:
        return len(self.joint_ids)


    @property
    def device(self) -> torch.device:
        return self.env.device


    @property
    def num_envs(self) -> int:
        return self.env.num_envs


    def process(self, action: torch.Tensor) -> None:
        """Convert policy output to joint targets. Call once per env step."""
        if self.clip is not None:
            action = torch.clamp(action, -self.clip, self.clip)
        self.prev_actions[:] = self.raw_actions
        self.raw_actions[:] = action
        self.joint_targets[:, self.joint_ids] = self.default_pos + self.raw_actions * self.scale


    def apply(self) -> None:
        """Send joint targets of all motors to the PD controllers. Call once per physics step."""
        self.env.robot.control_dofs_position(self.joint_targets, self.env.motors_dof_idx)


    def reset(self, envs_idx: torch.Tensor) -> None:
        self.raw_actions[envs_idx] = 0.0
        self.prev_actions[envs_idx] = 0.0
        self.joint_targets[envs_idx] = self.env.default_joint_pos