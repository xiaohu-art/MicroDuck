"""MLP actor and critic models.

    obs dict --select obs_groups--> concat --normalizer--> MLP --> output
                                                               \--> distribution (actor only)
"""

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from . import distribution as distribution_module
from .normalizer import EmpiricalNormalization

__all__ = ["MLP", "MLPModel"]

_ACTIVATIONS = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def resolve_activation(name: str) -> nn.Module:
    """Create an activation module from its name."""
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{name}', expected one of {list(_ACTIVATIONS)}")


def resolve_class(cfg) -> tuple[type, dict]:
    """Resolve a distribution class and constructor arguments.

    Args:
        cfg: Mapping or OmegaConf config containing ``class_name`` and any
            distribution constructor arguments.

    Returns:
        A ``(class, kwargs)`` pair.

    Raises:
        KeyError: If ``class_name`` is missing.
        ValueError: If the class name is not exported by ``distribution.py``.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dict(cfg)
    if "class_name" not in cfg:
        raise KeyError(f"distribution_cfg needs a 'class_name' key, got {cfg}")
    name = cfg.pop("class_name")
    if name not in distribution_module.__all__:
        raise ValueError(f"Unknown distribution '{name}', expected one of {distribution_module.__all__}")
    return getattr(distribution_module, name), cfg


class MLP(nn.Sequential):
    """Feed-forward network with no activation on the output layer.

    Args:
        input_dim: Width of the input feature vector.
        output_dim: Width of the output feature vector.
        hidden_dims: Width of each hidden layer.
        activation: Activation name accepted by :func:`resolve_activation`.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims, activation: str = "elu") -> None:
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), resolve_activation(activation)]
        layers.append(nn.Linear(dims[-1], output_dim))
        super().__init__(*layers)


class MLPModel(nn.Module):
    """MLP on a selected set of observation groups, with optional normalization
    and an optional output distribution.

    Args:
        obs: An example observation dict {group: (num_envs, dim)}, used to infer
            the input dim.
        obs_groups: {model_name: [group, ...]}, e.g. {"actor": ["policy"],
            "critic": ["critic"]}.
        obs_set: Which entry of `obs_groups` this model uses ("actor"/"critic").
        output_dim: Action dim for the actor, 1 for the critic.
        hidden_dims: Hidden layer sizes.
        activation: Activation name, see `_ACTIVATIONS`.
        obs_normalization: Whether to normalize the input with running stats.
        distribution_cfg: {"class_name": ..., **kwargs}; None = deterministic.

    Observation groups use shape ``(batch, dim)``; output uses
    ``(batch, output_dim)``.
    """

    def __init__(
        self,
        obs: dict[str, torch.Tensor],
        obs_groups,
        obs_set: str,
        output_dim: int,
        hidden_dims=(256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg=None,
    ) -> None:
        super().__init__()

        # Input: which observation groups, and their total width.
        self.obs_groups = list(obs_groups[obs_set])
        for group in self.obs_groups:
            if group not in obs:
                raise KeyError(f"Observation group '{group}' ({obs_set}) not in env obs: {list(obs)}")
            if obs[group].ndim != 2:
                raise ValueError(f"Group '{group}' must be (num_envs, dim), got {tuple(obs[group].shape)}")
        self.obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)

        # Normalization.
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()

        # Distribution.
        if distribution_cfg is not None:
            dist_class, dist_kwargs = resolve_class(distribution_cfg)
            self.distribution = dist_class(output_dim, **dist_kwargs)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLP(self.obs_dim, mlp_output_dim, list(hidden_dims), activation)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def get_latent(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate and normalize selected observation groups."""
        latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def forward(self, obs: dict[str, torch.Tensor], stochastic_output: bool = False) -> torch.Tensor:
        """Compute deterministic or stochastic output for observations."""
        mlp_output = self.mlp(self.get_latent(obs))
        if self.distribution is None:
            return mlp_output
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    # ------------------------------------------------------------------
    # Distribution accessors (valid after a stochastic forward)
    # ------------------------------------------------------------------
    @property
    def output_mean(self) -> torch.Tensor:
        """Mean from the latest stochastic forward."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Standard deviation from the latest stochastic forward."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Per-sample entropy from the latest stochastic forward."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Parameters of the latest output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return log probabilities for outputs."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        """Return per-sample KL divergence between two policies."""
        return self.distribution.kl_divergence(old_params, new_params)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def update_normalization(self, obs: dict[str, torch.Tensor]) -> None:
        """Update running statistics with rollout observations."""
        if self.obs_normalization:
            latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(latent)