"""Termination terms and manager."""

import torch

from .term import resolve_terms

__all__ = [
    "TerminationManager",
    "time_out",
    "bad_orientation",
    "root_height_below",
    "nan_state",
]


# ----------------------------------------------------------------------
# Termination terms: fn(env, **params) -> (num_envs,) bool
# ----------------------------------------------------------------------
def time_out(env) -> torch.Tensor:
    """Return environments that reached the episode limit."""
    return env.episode_length_buf >= env.max_episode_length


def bad_orientation(env, limit_angle: float) -> torch.Tensor:
    """Return environments whose tilt exceeds ``limit_angle`` radians."""
    cos_tilt = (-env.projected_gravity[:, 2]).clamp(-1.0, 1.0)
    return torch.acos(cos_tilt) > limit_angle


def root_height_below(env, min_height: float) -> torch.Tensor:
    """Return environments below ``min_height`` meters."""
    return env.base_pos[:, 2] < min_height


def nan_state(env) -> torch.Tensor:
    """Return environments with NaN base or joint state values."""
    return torch.isnan(env.base_pos).any(dim=-1) | torch.isnan(env.dof_pos).any(dim=-1)


_TERMINATION_FUNCTIONS = {
    "time_out": time_out,
    "bad_orientation": bad_orientation,
    "root_height_below": root_height_below,
    "nan_state": nan_state,
}


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------
class TerminationManager:
    """Evaluate termination terms and split failures from timeouts."""

    def __init__(self, env, terms: dict) -> None:
        self.env = env
        self.terms = resolve_terms(terms, _TERMINATION_FUNCTIONS, {"time_out": False})
        self.term_names = [t.name for t in self.terms]

        n, dev = env.num_envs, env.device
        self._terminated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._time_outs = torch.zeros(n, dtype=torch.bool, device=dev)
        self._term_dones = torch.zeros(n, len(self.terms), dtype=torch.bool, device=dev)
        # Which terms ended each env's last episode (kept until logged on reset).
        self._last_episode_dones = torch.zeros_like(self._term_dones)

    @property
    def terminated(self) -> torch.Tensor:
        """Failure terminations, shape ``(num_envs,)``."""
        return self._terminated

    @property
    def time_outs(self) -> torch.Tensor:
        """Timeout terminations, shape ``(num_envs,)``."""
        return self._time_outs

    @property
    def dones(self) -> torch.Tensor:
        return self._terminated | self._time_outs

    def compute(self) -> torch.Tensor:
        """Evaluate terms and return combined done flags."""
        self._terminated[:] = False
        self._time_outs[:] = False
        for i, term in enumerate(self.terms):
            value = term.func(self.env, **term.params)
            if term.extra["time_out"]:
                self._time_outs |= value
            else:
                self._terminated |= value
            self._term_dones[:, i] = value

        done_rows = self._term_dones.any(dim=1)
        self._last_episode_dones[done_rows] = self._term_dones[done_rows]
        return self.dones

    def reset(self, envs_idx: torch.Tensor) -> dict[str, float]:
        """Return termination rates for the selected environments."""
        extras = {
            f"Episode_Termination/{name}": self._last_episode_dones[envs_idx, i].float().mean().item()
            for i, name in enumerate(self.term_names)
        }
        self._last_episode_dones[envs_idx] = False
        return extras

    def __repr__(self) -> str:
        rows = [f"  {t.name:<20} {t.func.__name__:<20} time_out={t.extra['time_out']}" for t in self.terms]
        return "TerminationManager(\n" + "\n".join(rows) + "\n)"