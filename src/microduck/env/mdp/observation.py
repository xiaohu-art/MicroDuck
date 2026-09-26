"""Observation terms and grouped observation manager."""

import torch

__all__ = [
    "ObservationManager",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_command",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
]


# ----------------------------------------------------------------------
# Observation terms: fn(env, **params) -> (num_envs, dim)
# ----------------------------------------------------------------------
def base_lin_vel(env) -> torch.Tensor:
    """Return base linear velocity in the base frame."""
    return env.base_lin_vel


def base_ang_vel(env) -> torch.Tensor:
    """Return base angular velocity in the base frame."""
    return env.base_ang_vel


def projected_gravity(env) -> torch.Tensor:
    """Return gravity direction in the base frame."""
    return env.projected_gravity


def velocity_command(env) -> torch.Tensor:
    """Return the commanded ``(vx, vy, wz)`` values."""
    return env.command_term.command


def joint_pos_rel(env) -> torch.Tensor:
    """Return joint positions relative to the default pose."""
    return env.dof_pos - env.default_joint_pos


def joint_vel(env) -> torch.Tensor:
    """Return joint velocities."""
    return env.dof_vel


def last_action(env) -> torch.Tensor:
    """Return the previous clipped policy action."""
    return env.action_term.raw_actions


_OBSERVATION_FUNCTIONS = {
    "base_lin_vel": base_lin_vel,
    "base_ang_vel": base_ang_vel,
    "projected_gravity": projected_gravity,
    "velocity_command": velocity_command,
    "joint_pos_rel": joint_pos_rel,
    "joint_vel": joint_vel,
    "last_action": last_action,
}


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------
class ObservationManager:
    """Compute configured observation groups by concatenating their terms."""

    def __init__(self, env, groups: dict) -> None:
        self.env = env
        # {group: [term names]}, {group: [funcs]}, {group: [params]}
        self.term_names: dict[str, list[str]] = {}
        self.term_funcs: dict[str, list] = {}
        self.term_params: dict[str, list[dict]] = {}

        for group, terms in groups.items():
            if not terms:
                raise ValueError(f"Observation group '{group}' has no terms")
            self.term_names[group], self.term_funcs[group], self.term_params[group] = [], [], []

            # Parse {name: {func, params}}; `func` defaults to the term name.
            for name, term_cfg in terms.items():
                term_cfg = dict(term_cfg or {})
                func_name = term_cfg.pop("func", name)
                if func_name not in _OBSERVATION_FUNCTIONS:
                    raise ValueError(
                        f"Unknown observation function '{func_name}' (term '{group}/{name}'), "
                        f"expected one of {sorted(_OBSERVATION_FUNCTIONS)}"
                    )
                params = dict(term_cfg.pop("params", None) or {})
                if term_cfg:
                    raise ValueError(f"Unknown keys in observation term '{group}/{name}': {list(term_cfg)}")
                self.term_names[group].append(name)
                self.term_funcs[group].append(_OBSERVATION_FUNCTIONS[func_name])
                self.term_params[group].append(params)

        self._term_dims: dict[str, list[int]] | None = None

    @property
    def group_names(self) -> list[str]:
        return list(self.term_names)

    def compute_group(self, group: str) -> torch.Tensor:
        """Return one group with shape ``(num_envs, group_dim)``."""
        return torch.cat(
            [f(self.env, **p) for f, p in zip(self.term_funcs[group], self.term_params[group])], dim=-1
        )

    def compute(self) -> dict[str, torch.Tensor]:
        """Return all groups and their concatenated tensors."""
        return {group: self.compute_group(group) for group in self.term_names}

    @property
    def term_dims(self) -> dict[str, list[int]]:
        """Return the width of each term, computed lazily."""
        if self._term_dims is None:
            self._term_dims = {
                group: [f(self.env, **p).shape[-1] for f, p in zip(self.term_funcs[group], self.term_params[group])]
                for group in self.term_names
            }
        return self._term_dims

    @property
    def group_dims(self) -> dict[str, int]:
        return {group: sum(dims) for group, dims in self.term_dims.items()}

    def __repr__(self) -> str:
        lines = ["ObservationManager("]
        for group in self.term_names:
            lines.append(f"  [{group}] ({self.group_dims[group]})")
            for name, f, d in zip(self.term_names[group], self.term_funcs[group], self.term_dims[group]):
                lines.append(f"    {name:<20} {f.__name__:<20} {d:>3}")
        return "\n".join(lines + [")"])