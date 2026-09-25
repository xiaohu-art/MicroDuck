"""Generalized Advantage Estimation and advantage normalization."""

import torch

__all__ = ["compute_gae", "normalize_advantages"]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and value targets with GAE.

    Args:
        rewards: Rewards with shape ``(T, N, 1)``.
        values: Critic values with shape ``(T, N, 1)``.
        dones: Episode-end flags with shape ``(T, N, 1)``.
        last_values: Final critic values with shape ``(N, 1)``.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        ``(advantages, returns)``, both with shape ``(T, N, 1)``.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)}, values {tuple(values.shape)} and "
            f"dones {tuple(dones.shape)} must have the same shape (T, N, 1)"
        )
    if last_values.shape != values.shape[1:]:
        raise ValueError(f"last_values must be {tuple(values.shape[1:])}, got {tuple(last_values.shape)}")

    num_steps = rewards.shape[0]
    not_done = 1.0 - dones.float()
    advantages = torch.zeros_like(values)

    advantage = torch.zeros_like(last_values)
    for t in reversed(range(num_steps)):
        next_values = last_values if t == num_steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * not_done[t] * next_values - values[t]
        advantage = delta + gamma * lam * not_done[t] * advantage
        advantages[t] = advantage

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages over all entries of the input tensor."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)