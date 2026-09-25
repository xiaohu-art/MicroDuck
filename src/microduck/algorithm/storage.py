"""Rollout storage for on-policy reinforcement learning.

Buffers use shape ``(T, N, ...)``. Scalar values use ``(T, N, 1)``.
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import torch

__all__ = ["RolloutStorage"]


class RolloutStorage:
    @dataclass
    class Transition:
        """Data collected from one environment step."""

        observations: dict[str, torch.Tensor] | None = None
        actions: torch.Tensor | None = None
        rewards: torch.Tensor | None = None
        dones: torch.Tensor | None = None
        values: torch.Tensor | None = None
        actions_log_prob: torch.Tensor | None = None
        distribution_params: tuple[torch.Tensor, ...] | None = None

        def clear(self) -> None:
            self.__init__()

    @dataclass
    class Batch:
        """Mini-batch flattened over the time and environment dimensions."""

        observations: dict[str, torch.Tensor]
        actions: torch.Tensor
        values: torch.Tensor
        advantages: torch.Tensor
        returns: torch.Tensor
        old_actions_log_prob: torch.Tensor
        old_distribution_params: tuple[torch.Tensor, ...]

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: dict[str, torch.Tensor],
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        """Allocate rollout buffers.

        Args:
            num_envs: Number of parallel environments.
            num_transitions_per_env: Transitions stored per environment.
            obs: Example observation groups used to infer buffer shapes.
            action_dim: Number of action dimensions.
            device: Device for all buffers.
        """
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device
        T, N = num_transitions_per_env, num_envs

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(T, N, *shape, dtype=dtype, device=device)

        self.observations = {
            key: zeros(*value.shape[1:], dtype=value.dtype) for key, value in obs.items()
        }
        self.actions = zeros(action_dim)
        self.rewards = zeros(1)
        self.dones = zeros(1, dtype=torch.bool)
        self.values = zeros(1)
        self.actions_log_prob = zeros(1)
        self.returns = zeros(1)
        self.advantages = zeros(1)
        # Allocated on the first transition, when the number/shape of the
        # distribution parameters is known (Gaussian: (mean, std)).
        self.distribution_params: tuple[torch.Tensor, ...] | None = None

        self.step = 0

    def add_transition(self, t: Transition) -> None:
        """Store a transition at the current buffer index."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout storage is full, call clear() before adding transitions")

        for key, value in t.observations.items():
            self.observations[key][self.step].copy_(value)
        self.actions[self.step].copy_(t.actions)
        self.rewards[self.step].copy_(t.rewards.view(-1, 1))
        self.dones[self.step].copy_(t.dones.view(-1, 1))
        self.values[self.step].copy_(t.values.view(-1, 1))
        self.actions_log_prob[self.step].copy_(t.actions_log_prob.view(-1, 1))

        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in t.distribution_params
            )
        for buffer, p in zip(self.distribution_params, t.distribution_params):
            buffer[self.step].copy_(p)

        self.step += 1

    def clear(self) -> None:
        """Start a new rollout (buffers are overwritten, not freed)."""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int) -> Iterator[Batch]:
        """Yield shuffled mini-batches for the requested number of epochs.

        The same permutation is reused for every epoch. Incomplete remainders
        are dropped when the batch size is not divisible by ``num_mini_batches``.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size * num_mini_batches != batch_size:
            warnings.warn(
                f"batch size {batch_size} is not divisible by num_mini_batches={num_mini_batches}; "
                f"{batch_size - mini_batch_size * num_mini_batches} samples are dropped per epoch"
            )
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        # Flatten (T, N, ...) -> (T * N, ...)
        observations = {key: value.flatten(0, 1) for key, value in self.observations.items()}
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield RolloutStorage.Batch(
                    observations={key: value[idx] for key, value in observations.items()},
                    actions=actions[idx],
                    values=values[idx],
                    advantages=advantages[idx],
                    returns=returns[idx],
                    old_actions_log_prob=old_actions_log_prob[idx],
                    old_distribution_params=tuple(p[idx] for p in old_distribution_params),
                )