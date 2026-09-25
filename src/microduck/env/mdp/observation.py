import torch

__all__ = [
    "ObservationGroup",
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
    """Base linear velocity in the base frame."""
    return env.base_lin_vel


def base_ang_vel(env) -> torch.Tensor:
    """Base angular velocity in the base frame (gyroscope)."""
    return env.base_ang_vel


def projected_gravity(env) -> torch.Tensor:
    """Gravity direction in the base frame; (0, 0, -1) when upright."""
    return env.projected_gravity


def velocity_command(env) -> torch.Tensor:
    """Commanded (vx, vy, wz) in the base frame."""
    return env.command_term.command


def joint_pos_rel(env) -> torch.Tensor:
    """Joint positions relative to the default pose."""
    return env.dof_pos - env.default_joint_pos


def joint_vel(env) -> torch.Tensor:
    """Joint velocities."""
    return env.dof_vel


def last_action(env) -> torch.Tensor:
    """Clipped policy output of the previous step."""
    return env.action_term.raw_actions


# ----------------------------------------------------------------------
# Observation group: concatenates terms in the order listed in the config.
# ----------------------------------------------------------------------
class ObservationGroup:
    """A named list of observation terms, concatenated along the last dim.

    Config format (key = term name, `func` defaults to the key):

        terms:
          base_ang_vel: {}
          joint_pos: {func: joint_pos_rel}
          some_term: {func: some_fn, params: {k: v}}
    """

    def __init__(self, env, terms: dict):
        self.env = env
        self.term_names: list[str] = []
        self.term_funcs = []
        self.term_params: list[dict] = []

        for name, term_cfg in terms.items():
            term_cfg = dict(term_cfg or {})
            func_name = term_cfg.pop("func", name)
            params = dict(term_cfg.pop("params", {}) or {})
            if term_cfg:
                raise ValueError(f"Unknown keys in observation term '{name}': {list(term_cfg)}")
            func = globals().get(func_name)
            if func is None or func_name not in __all__:
                raise ValueError(f"Unknown observation function '{func_name}' (term '{name}')")
            self.term_names.append(name)
            self.term_funcs.append(func)
            self.term_params.append(params)

        self._term_dims: list[int] | None = None

    def compute(self) -> torch.Tensor:
        return torch.cat(
            [f(self.env, **p) for f, p in zip(self.term_funcs, self.term_params)], dim=-1
        )

    @property
    def term_dims(self) -> list[int]:
        if self._term_dims is None:
            self._term_dims = [
                f(self.env, **p).shape[-1] for f, p in zip(self.term_funcs, self.term_params)
            ]
        return self._term_dims

    @property
    def dim(self) -> int:
        return sum(self.term_dims)

    def __repr__(self) -> str:
        rows = [f"  {n:<20} {d:>3}" for n, d in zip(self.term_names, self.term_dims)]
        return "ObservationGroup(\n" + "\n".join(rows) + f"\n  {'total':<20} {self.dim:>3}\n)"