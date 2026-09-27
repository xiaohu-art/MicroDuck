"""Reward terms and weighted reward manager."""

import re

import torch

from ...utils import resolve_matching_names

__all__ = [
    "RewardManager",
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "upright",
    "action_rate_l2",
    "feet_air_time",
    "pose",
    "body_ang_vel",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
]


# ----------------------------------------------------------------------
# Reward terms: fn(env, **params) -> (num_envs,)
# ----------------------------------------------------------------------
def track_lin_vel_xy(env, std: float) -> torch.Tensor:
    """Track planar velocity and penalize vertical velocity."""
    cmd = env.command_term.command
    v = env.base_lin_vel
    err = (cmd[:, :2] - v[:, :2]).square().sum(dim=1) + v[:, 2].square()
    return torch.exp(-err / std**2)


def track_ang_vel_z(env, std: float) -> torch.Tensor:
    """Track yaw velocity and penalize roll and pitch velocity."""
    cmd = env.command_term.command
    w = env.base_ang_vel
    err = (cmd[:, 2] - w[:, 2]).square() + w[:, :2].square().sum(dim=1)
    return torch.exp(-err / std**2)


def upright(env, std: float) -> torch.Tensor:
    """Reward an upright base orientation."""
    return torch.exp(-env.projected_gravity[:, :2].square().sum(dim=1) / std**2)


def action_rate_l2(env) -> torch.Tensor:
    """Return squared action change; use a negative weight."""
    return (env.action_term.raw_actions - env.action_term.prev_actions).square().sum(dim=1)


def _command_speed(env) -> torch.Tensor:
    """Return the combined linear and angular command magnitude."""
    cmd = env.command_term.command
    return cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()


def feet_air_time(
    env, threshold_min: float, threshold_max: float, command_threshold: float
) -> torch.Tensor:
    """Reward feet with swing duration in range when the command is active."""
    air = env.feet_air_time
    in_range = ((air > threshold_min) & (air < threshold_max)).float().sum(dim=1)
    moving = (_command_speed(env) > command_threshold).float()
    return in_range * moving


# Per-joint std tensors of `pose`, resolved once per (joint pattern, std map).
_POSE_CACHE: dict = {}


def _resolve_pose_std(env, joint_names, std_map) -> tuple[list[int], torch.Tensor]:
    """Return selected joint indices and their standard deviations."""
    key = (tuple(joint_names), tuple(std_map.items()))
    if key not in _POSE_CACHE:
        joint_idx, names = resolve_matching_names(list(joint_names), env.motor_names)
        # Every selected joint must match exactly one std pattern (raises otherwise).
        _, covered = resolve_matching_names(list(std_map.keys()), names)
        if len(covered) != len(names):
            missing = sorted(set(names) - set(covered))
            raise ValueError(f"pose std patterns {list(std_map)} do not cover joints {missing}")
        stds = [next(float(v) for p, v in std_map.items() if re.fullmatch(p, n)) for n in names]
        _POSE_CACHE[key] = (joint_idx, torch.tensor(stds, device=env.device))
    return _POSE_CACHE[key]


def pose(
    env, joint_names, std_standing: dict, std_walking: dict, walking_threshold: float
) -> torch.Tensor:
    """Reward poses near default, using separate tolerances for standing and walking."""
    idx, std_stand = _resolve_pose_std(env, joint_names, std_standing)
    _, std_walk = _resolve_pose_std(env, joint_names, std_walking)
    walking = (_command_speed(env) >= walking_threshold).unsqueeze(1)
    std = torch.where(walking, std_walk, std_stand)
    err = (env.dof_pos[:, idx] - env.default_joint_pos[idx]).square()
    return torch.exp(-(err / std.square()).mean(dim=1))


def _moving(env, command_threshold: float) -> torch.Tensor:
    """Return a mask for commands above the motion threshold."""
    return (_command_speed(env) > command_threshold).float()


def body_ang_vel(env) -> torch.Tensor:
    """Penalize base roll and pitch angular velocity."""
    return env.base_ang_vel_w[:, :2].square().sum(dim=1)


def foot_clearance(env, target_height: float, command_threshold: float) -> torch.Tensor:
    """Penalize moving feet that deviate from the target height."""
    speed_xy = env.feet_vel[..., :2].norm(dim=-1)
    cost = ((env.feet_height - target_height).abs() * speed_xy).sum(dim=1)
    return cost * _moving(env, command_threshold)


def foot_swing_height(env, target_height: float, command_threshold: float) -> torch.Tensor:
    """Penalize swing-height peaks that differ from the target height."""
    error = (env.feet_landing_peak / target_height - 1.0).square()
    cost = (error * env.feet_first_contact.float()).sum(dim=1)
    return cost * _moving(env, command_threshold)


def foot_slip(env, command_threshold: float) -> torch.Tensor:
    """Penalize horizontal foot velocity during ground contact."""
    speed_sq = env.feet_vel[..., :2].square().sum(dim=-1)
    cost = (speed_sq * env.feet_contact.float()).sum(dim=1)
    return cost * _moving(env, command_threshold)


_REWARD_FUNCTIONS = {
    "track_lin_vel_xy": track_lin_vel_xy,
    "track_ang_vel_z": track_ang_vel_z,
    "upright": upright,
    "action_rate_l2": action_rate_l2,
    "feet_air_time": feet_air_time,
    "pose": pose,
    "body_ang_vel": body_ang_vel,
    "foot_clearance": foot_clearance,
    "foot_swing_height": foot_swing_height,
    "foot_slip": foot_slip,
}


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------
class RewardManager:
    """Compute weighted rewards and track per-term episode sums."""

    def __init__(self, env, terms: dict) -> None:
        self.env = env
        self.term_names: list[str] = []
        self.term_funcs = []
        self.term_params: list[dict] = []
        self.term_weights: list[float] = []

        # Parse {name: {func, weight, params}}; `func` defaults to the term name.
        for name, term_cfg in terms.items():
            term_cfg = dict(term_cfg or {})
            func_name = term_cfg.pop("func", name)
            if func_name not in _REWARD_FUNCTIONS:
                raise ValueError(
                    f"Unknown reward function '{func_name}' (term '{name}'), "
                    f"expected one of {sorted(_REWARD_FUNCTIONS)}"
                )
            if "weight" not in term_cfg:
                raise ValueError(f"Reward term '{name}' is missing 'weight'")
            weight = float(term_cfg.pop("weight"))
            params = dict(term_cfg.pop("params", None) or {})
            if term_cfg:
                raise ValueError(f"Unknown keys in reward term '{name}': {list(term_cfg)}")
            self.term_names.append(name)
            self.term_funcs.append(_REWARD_FUNCTIONS[func_name])
            self.term_params.append(params)
            self.term_weights.append(weight)

        n, dev = env.num_envs, env.device
        self._reward_buf = torch.zeros(n, device=dev)
        self._episode_sums = {name: torch.zeros(n, device=dev) for name in self.term_names}
        # Unweighted, unscaled value of each term at the last step (for debugging).
        self.step_reward = torch.zeros(n, len(self.term_names), device=dev)

    def compute(self, dt: float) -> torch.Tensor:
        """Compute the weighted reward for the current step."""
        self._reward_buf[:] = 0.0
        for i, (name, func, params, weight) in enumerate(
            zip(self.term_names, self.term_funcs, self.term_params, self.term_weights)
        ):
            if weight == 0.0:
                self.step_reward[:, i] = 0.0
                continue
            value = func(self.env, **params)
            if value.shape != self._reward_buf.shape:
                raise ValueError(f"Reward '{name}' returned shape {tuple(value.shape)}, expected (num_envs,)")
            self.step_reward[:, i] = value
            value = value * weight * dt
            # A single env with corrupted physics must not poison the whole batch.
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._reward_buf += value
            self._episode_sums[name] += value
        return self._reward_buf

    def reset(self, envs_idx: torch.Tensor) -> dict[str, float]:
        """Return mean per-term episode rewards for selected environments."""
        episode_length_s = self.env.env_cfg.sim.episode_length_s
        extras = {}
        for name, sums in self._episode_sums.items():
            extras[f"Episode_Reward/{name}"] = sums[envs_idx].mean().item() / episode_length_s
            sums[envs_idx] = 0.0
        return extras

    def __repr__(self) -> str:
        rows = [
            f"  {n:<20} weight={w:<8g} params={p}"
            for n, w, p in zip(self.term_names, self.term_weights, self.term_params)
        ]
        return "RewardManager(\n" + "\n".join(rows) + "\n)"