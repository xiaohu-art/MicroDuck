import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Running mean / std normalization of observations."""

    def __init__(self, shape: int, eps: float = 1e-2, until: int | None = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge a batch (..., shape) into the running statistics (Chan et al.)."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return
        x = x.reshape(-1, x.shape[-1])
        n_b = x.shape[0]
        if n_b == 0:
            return
        n_a = self.count.item()
        n = n_a + n_b

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean

        # Combine sums of squared deviations of the two sets.
        m2 = self.var * n_a + batch_var * n_b + delta.square() * (n_a * n_b / n)
        self.mean += delta * (n_b / n)
        self.var.copy_(m2 / n)
        self.count += n_b