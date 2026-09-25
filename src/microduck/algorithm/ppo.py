"""Proximal Policy Optimization with rollout collection and updates."""

import os
import time
from itertools import chain

import torch
from torch import nn

from .gae import compute_gae, normalize_advantages
from .logger import Logger
from .models import MLPModel
from .storage import RolloutStorage

__all__ = ["PPO"]


class PPO:
    """PPO runner and actor-critic optimizer.

    Args:
        env: Environment providing observations, actions, device, and episode state.
        cfg: Algorithm configuration.
        log_dir: TensorBoard and checkpoint directory, or ``None``.
    """

    def __init__(self, env, cfg, log_dir: str | None = None) -> None:
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.log_dir = log_dir
        self.num_steps_per_env = int(cfg.num_steps_per_env)
        self.save_interval = int(cfg.save_interval)

        obs = env.get_observations()
        self.actor = MLPModel(obs, cfg.obs_groups, "actor", env.action_dim, **cfg.actor).to(self.device)
        self.critic = MLPModel(obs, cfg.obs_groups, "critic", 1, **cfg.critic).to(self.device)
        print(f"Actor: {self.actor}\nCritic: {self.critic}")

        self._init_hparams(**cfg.algorithm)
        self.optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate
        )

        self.storage = RolloutStorage(
            env.num_envs, self.num_steps_per_env, obs, env.action_dim, self.device
        )
        self.transition = RolloutStorage.Transition()

        self.logger = Logger(log_dir, env.num_envs, self.num_steps_per_env, self.device)
        self.current_learning_iteration = 0

    def _init_hparams(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_param: float = 0.2,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        learning_rate: float = 1e-3,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 1.0,
        use_clipped_value_loss: bool = True,
        max_grad_norm: float = 1.0,
        normalize_advantage_per_mini_batch: bool = False,
    ) -> None:
        """Initialize PPO hyperparameters."""
        if schedule not in ("adaptive", "fixed"):
            raise ValueError(f"schedule must be 'adaptive' or 'fixed', got '{schedule}'")
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample actions and record rollout data."""
        t = self.transition
        t.actions = self.actor(obs, stochastic_output=True).detach()
        t.values = self.critic(obs).detach()
        t.actions_log_prob = self.actor.get_output_log_prob(t.actions).detach()
        t.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        t.observations = obs  # raw obs from *before* env.step
        return t.actions

    def process_env_step(
        self, obs: dict[str, torch.Tensor], rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Store rewards, termination flags, and the current transition."""
        t = self.transition
        t.rewards = rewards.clone()  # cloned: the bootstrap below modifies it
        t.dones = dones

        # Time-out bootstrap: an episode cut by the time limit did not fail, so
        # add the value it would still have collected. The terminal obs is not
        # available (the env already reset), so V(s_t) stands in for V(s_{t+1}).
        if "time_outs" in extras:
            t.rewards += self.gamma * (t.values * extras["time_outs"].unsqueeze(1)).squeeze(1)

        self.storage.add_transition(t)
        t.clear()

    def compute_returns(self, obs: dict[str, torch.Tensor]) -> None:
        """Compute GAE advantages and returns for the stored rollout."""
        st = self.storage
        last_values = self.critic(obs).detach()
        advantages, returns = compute_gae(
            st.rewards, st.values, st.dones, last_values, self.gamma, self.lam
        )
        if not self.normalize_advantage_per_mini_batch:
            advantages = normalize_advantages(advantages)
        # Write into the existing buffers (normal tensors, usable for autograd later).
        st.advantages.copy_(advantages)
        st.returns.copy_(returns)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        mean_value_loss = mean_surrogate_loss = mean_entropy = mean_kl = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            advantages = batch.advantages.squeeze(-1)
            if self.normalize_advantage_per_mini_batch:
                advantages = normalize_advantages(advantages)

            # Re-evaluate the stored actions under the current policy.
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            entropy = self.actor.output_entropy
            values = self.critic(batch.observations)

            # Adaptive learning rate from KL(old || new).
            with torch.no_grad():
                kl_mean = self.actor.get_kl_divergence(
                    batch.old_distribution_params, self.actor.output_distribution_params
                ).mean()
            if self.schedule == "adaptive" and self.desired_kl is not None:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            # Clipped surrogate objective.
            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss (optionally clipped around the rollout values).
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)
                ).mean()
            else:
                value_loss = (values - batch.returns).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_kl += kl_mean.item()

        # Normalizer statistics change only here, after the policy update.
        obs = {key: value.flatten(0, 1) for key, value in self.storage.observations.items()}
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates,
        }

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int | None = None, init_at_random_ep_len: bool = False) -> None:
        if num_learning_iterations is None:
            num_learning_iterations = int(self.cfg.max_iterations)

        # Spread episode ends so that not all envs time out in the same step.
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        self.train_mode()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions)
                    self.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)
                collect_time = time.time() - start

                start = time.time()
                self.compute_returns(obs)

            loss_dict = self.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.learning_rate,
                action_std=self.actor.output_std,
            )
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.close()

    def save(self, path: str, infos: dict | None = None) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True) -> dict | None:
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(loaded["actor_state_dict"])
        self.critic.load_state_dict(loaded["critic_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = loaded["iter"]
        return loaded["infos"]

    def get_inference_policy(self) -> MLPModel:
        """Set evaluation mode and return the actor."""
        self.eval_mode()
        return self.actor

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()