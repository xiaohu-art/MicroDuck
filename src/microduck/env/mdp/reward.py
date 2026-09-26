"""Reward terms and weighted reward manager."""

import torch

__all__ = [
    "RewardManager",
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "upright",
    "action_rate_l2",
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


_REWARD_FUNCTIONS = {
    "track_lin_vel_xy": track_lin_vel_xy,
    "track_ang_vel_z": track_ang_vel_z,
    "upright": upright,
    "action_rate_l2": action_rate_l2,
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