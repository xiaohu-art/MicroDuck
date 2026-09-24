import torch
import genesis as gs

from ...utils import resolve_matching_names

class JointPositionAction:
    def __init__(
        self, 
        env, 
        joint_names, 
        scale: float = 1.0, 
        clip: float | None = 1.0
    ):
        self.env = env
        self.scale = scale
        self.clip = clip

        robot_joint_names = [j.name for j in env.robot_cfg.joints]
        default_map = {j.name: j.default_pos for j in env.robot_cfg.joints}
        _, self.joint_names = resolve_matching_names(joint_names, robot_joint_names)

        # (0, 6) is root translation and orientation
        self.dofs_idx = [
            env.robot.get_joint(name).dofs_idx_local[0] for name in self.joint_names
        ]
        self.default_pos = torch.tensor(
            [default_map[name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=env.device,
        )

        self.raw_actions = torch.zeros(
            env.num_envs, self.action_dim, dtype=gs.tc_float, device=env.device
        )
        self.processed_actions = self.default_pos.repeat(env.num_envs, 1)
 
    @property
    def action_dim(self) -> int:
        return len(self.dofs_idx)

    def process(self, action: torch.Tensor):
        """Convert policy output to joint targets. Call once per env step."""
        if self.clip is not None:
            action = torch.clamp(action, -self.clip, self.clip)
        self.raw_actions[:] = action
        self.processed_actions[:] = self.default_pos + self.raw_actions * self.scale

    def apply(self):
        self.env.robot.set_joint_positions(self.processed_actions, dofs_idx=self.dofs_idx)

    def reset(self, env_idx: torch.Tensor):
        self.raw_actions[env_idx] = 0.0
        self.processed_actions[env_idx] = self.default_pos