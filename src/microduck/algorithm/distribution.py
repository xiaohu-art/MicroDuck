"""Output distributions for stochastic policy models.

    dist.update(mlp_output)     # build the distribution for this batch
    a = dist.sample()
    dist.log_prob(a), dist.entropy, dist.params
"""

import math

import torch
from torch import nn
from torch.distributions import Normal

__all__ = ["Distribution", "GaussianDistribution"]


class Distribution(nn.Module):
    """Interface for policy output distributions.

    Args:
        output_dim: Action dimension.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    @property
    def input_dim(self) -> int:
        """Number of values the policy network must output."""
        raise NotImplementedError

    def update(self, mlp_output: torch.Tensor) -> None:
        """Build the distribution from raw policy output ``(batch, D)``."""
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Draw one action sample for each item in the current batch."""
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the deterministic output for raw policy output."""
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return log probabilities summed over actions, shape ``(batch,)``."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Mean action for each item in the current batch."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation for each action and batch item."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy summed over actions, shape ``(batch,)``."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Parameters saved for KL-divergence calculation."""
        raise NotImplementedError

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Return ``KL(old || new)`` summed over actions."""
        raise NotImplementedError


class GaussianDistribution(Distribution):
    """Diagonal Gaussian with state-independent standard deviations.

    Args:
        output_dim: Action dimension.
        init_std: Initial standard deviation for every action dimension.
        std_range: ``(min, max)`` clamp range for standard deviation.
        std_type: ``scalar`` for direct std or ``log`` for log std.
        learn_std: Whether the standard deviation parameter is trainable.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__(output_dim)
        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown std_type '{std_type}', expected 'scalar' or 'log'")
        self.std_type = std_type

        init = torch.full((output_dim,), float(init_std))
        if std_type == "scalar":
            self.std_param = nn.Parameter(init, requires_grad=learn_std)
        else:
            self.log_std_param = nn.Parameter(init.log(), requires_grad=learn_std)

        lo, hi = float(std_range[0]), float(std_range[1])
        lo = max(lo, 1e-6)
        self.std_range = (lo, hi)
        self.log_std_range = (math.log(lo), math.log(hi))

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def input_dim(self) -> int:
        return self.output_dim

    def current_std(self) -> torch.Tensor:
        """Clamped standard deviation, shape ``(output_dim,)``."""
        if self.std_type == "scalar":
            return self.std_param.clamp(*self.std_range)
        return self.log_std_param.clamp(*self.log_std_range).exp()

    def update(self, mlp_output: torch.Tensor) -> None:
        """Create a Normal distribution from means ``(batch, output_dim)``."""
        self._distribution = Normal(mlp_output, self.current_std())

    def sample(self) -> torch.Tensor:
        """Draw one action sample per batch item."""
        return self._distribution.sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the policy mean without sampling."""
        return mlp_output

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return summed log probabilities for a batch of actions."""
        return self._distribution.log_prob(outputs).sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Return the current distribution mean."""
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        """Return the current distribution standard deviation."""
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Return entropy summed over action dimensions."""
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current mean and standard deviation."""
        return (self.mean, self.std)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Return ``KL(old || new)`` summed over action dimensions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        ).sum(dim=-1)