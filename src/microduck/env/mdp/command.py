import torch
import genesis as gs

__all__ = ["VelocityCommand"]

class VelocityCommand:
    def __init__(
        self,
        env,
        ranges,
        resample,
        standing_prob: float = 0.1,
        turn_in_place_prob: float = 0.15
    ):
        self.env = env

        self.lin_vel_x = tuple(float(v) for v in ranges["lin_vel_x"])
        self.lin_vel_y = tuple(float(v) for v in ranges["lin_vel_y"])
        self.ang_vel_z = tuple(float(v) for v in ranges["ang_vel_z"])
        self.resampling_time_range = tuple(float(v) for v in resample)
        self.standing_prob = float(standing_prob)
        self.turn_in_place_prob = float(turn_in_place_prob)

        n, device = self.num_envs, self.device
        self.vel_command_b = torch.zeros(n, 3, dtype=gs.tc_float, device=device)
        self.time_left = torch.zeros(n, dtype=gs.tc_float, device=device)
        self.is_standing_env = torch.zeros(n, dtype=torch.bool, device=device)
        self.is_turn_env = torch.zeros(n, dtype=torch.bool, device=device)

        self.metrics = {
            "error_vel_xy": torch.zeros(n, dtype=gs.tc_float, device=device),
            "error_vel_yaw": torch.zeros(n, dtype=gs.tc_float, device=device),
        }


    @property
    def command(self):
        """(num_envs, 3): vx, vy [m/s] and wz [rad/s] in the base frame."""
        return self.vel_command_b


    @property
    def device(self):
        return self.env.device
    

    @property
    def num_envs(self):
        return self.env.num_envs


    def reset(self, envs_idx):
        steps = self.env.episode_length_buf[envs_idx].clamp(min=1)
        extras = {}
        for name, value in self.metrics.items():
            extras[name] = (value[envs_idx] / steps).mean().item()   # mean error per step
            value[envs_idx] = 0.0
        self._resample(envs_idx)
        return extras


    def compute(self, dt: float):
        """Advance timers by `dt` and resample expired envs. Call once per env step."""
        self._update_metrics()
        self.time_left -= dt
        expired = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        self._resample(expired)


    def _resample(self, envs_idx: torch.Tensor):
        """Draw a new timer and a new command for `envs_idx`."""
        if len(envs_idx) == 0:
            return
        self.time_left[envs_idx] = self._uniform(len(envs_idx), self.resampling_time_range)
        self._resample_command(envs_idx)


    def _resample_command(self, envs_idx: torch.Tensor):
        n = len(envs_idx)
 
        # Default bucket: independent uniform sampling.
        self.vel_command_b[envs_idx, 0] = self._uniform(n, self.lin_vel_x)
        self.vel_command_b[envs_idx, 1] = self._uniform(n, self.lin_vel_y)
        self.vel_command_b[envs_idx, 2] = self._uniform(n, self.ang_vel_z)
 
        u = torch.rand(n, device=self.device)
        is_standing = u < self.standing_prob
        is_turn = (~is_standing) & (u < self.standing_prob + self.turn_in_place_prob)
        self.is_standing_env[envs_idx] = is_standing
        self.is_turn_env[envs_idx] = is_turn
 
        # Standing bucket: zero command.
        stand_idx = envs_idx[is_standing]
        self.vel_command_b[stand_idx] = 0.0
 
        # Turn-in-place bucket: no linear velocity, a clearly non-zero yaw rate.
        turn_idx = envs_idx[is_turn]
        if len(turn_idx) > 0:
            wz_max = max(abs(self.ang_vel_z[0]), abs(self.ang_vel_z[1]))
            magnitude = self._uniform(len(turn_idx), (0.4 * wz_max, wz_max))
            sign = torch.where(torch.rand(len(turn_idx), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[turn_idx, 0:2] = 0.0
            self.vel_command_b[turn_idx, 2] = sign * magnitude


    def _update_metrics(self):
        lin_err = torch.norm(self.vel_command_b[:, :2] - self.env.base_lin_vel[:, :2], dim=-1)
        yaw_err = torch.abs(self.vel_command_b[:, 2] - self.env.base_ang_vel[:, 2])
        self.metrics["error_vel_xy"] += lin_err       # plain sum
        self.metrics["error_vel_yaw"] += yaw_err

 
    def _uniform(self, n: int, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return torch.empty(n, dtype=gs.tc_float, device=self.device).uniform_(lo, hi)