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

        robot_joint_names = [j.name for j in env.robot_cfg.joints]
        default_map = {j.name: j.default_pos for j in env.robot_cfg.joints}
        _, self.joint_names = resolve_matching_names(joint_names, robot_joint_names)

        self.dofs_idx = [
            env.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names
        ]
        self.default_pos = torch.tensor(
            [default_map[name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=self.device,
        )

        self.raw_actions = torch.zeros(
            self.num_envs, self.action_dim, dtype=gs.tc_float, device=self.device
        )
        self.processed_actions = self.default_pos.repeat(self.num_envs, 1)
        # Previous step's raw_actions, for the action-rate reward.
        self.prev_actions = torch.zeros_like(self.raw_actions)


    @property
    def action_dim(self) -> int:
        return len(self.dofs_idx)


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
        self.processed_actions[:] = self.default_pos + self.raw_actions * self.scale


    def apply(self) -> None:
        """Send joint targets to the PD controllers. Call once per physics step."""
        self.env.robot.control_dofs_position(self.processed_actions, self.dofs_idx)


    def reset(self, envs_idx: torch.Tensor) -> None:
        self.raw_actions[envs_idx] = 0.0
        self.prev_actions[envs_idx] = 0.0
        self.processed_actions[envs_idx] = self.default_pos