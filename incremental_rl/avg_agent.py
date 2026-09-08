"""Reusable Action Value Gradient (AVG) agent.

This module keeps the learning equations and default hyperparameters from the
repository's original ``avg.py`` entry point while separating agent state from
the experiment loop.  In particular, an MDP termination and an episode time
limit are represented separately: only a true termination suppresses TD
bootstrapping, while either event closes the running return used by the TD-error
scaler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


@dataclass(frozen=True)
class AVGConfig:
    """Hyperparameters used by the official HalfCheetah AVG configuration."""

    actor_lr: float = 0.0063
    critic_lr: float = 0.0087
    beta1: float = 0.0
    beta2: float = 0.999
    gamma: float = 0.99
    alpha_lr: float = 0.07
    nhid_actor: int = 256
    nhid_critic: int = 256
    l2_actor: float = 0.0
    l2_critic: float = 0.0

    @property
    def betas(self) -> tuple[float, float]:
        return self.beta1, self.beta2

    def __post_init__(self) -> None:
        if self.actor_lr <= 0 or self.critic_lr <= 0:
            raise ValueError("actor_lr and critic_lr must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1)")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0, 1]")
        if self.alpha_lr < 0:
            raise ValueError("alpha_lr must be non-negative")
        if self.nhid_actor <= 0 or self.nhid_critic <= 0:
            raise ValueError("hidden layer widths must be positive")
        if self.l2_actor < 0 or self.l2_critic < 0:
            raise ValueError("L2 regularization values must be non-negative")


def orthogonal_weight_init(module: nn.Module) -> None:
    """Apply the original AVG orthogonal initialization to linear layers."""

    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)


class RunningStats:
    """Serializable running moments matching the original AVG implementation."""

    def __init__(
        self,
        n: float = 0.0,
        mean: np.ndarray | np.generic | float | None = None,
        sum_squared_deviations: np.ndarray | np.generic | float | None = None,
    ) -> None:
        self.n = float(n)
        self.m = None if mean is None else np.asarray(mean, dtype=np.float32).copy()
        self.s = (
            None
            if sum_squared_deviations is None
            else np.asarray(sum_squared_deviations, dtype=np.float32).copy()
        )

    def update(self, value: Any) -> None:
        x = np.asarray(value, dtype=np.float32).copy()
        self.n += 1.0
        if self.n == 1.0:
            self.m = x
            self.s = np.zeros_like(x, dtype=np.float32)
            return

        assert self.m is not None and self.s is not None
        previous_mean = self.m.copy()
        self.m += (x - self.m) / self.n
        self.s += (x - previous_mean) * (x - self.m)

    @property
    def mean(self) -> np.ndarray | np.generic | float:
        return self.m if self.n else 0.0

    @property
    def variance(self) -> np.ndarray | np.generic | float:
        if not self.n:
            return 0.0
        assert self.s is not None
        return self.s / self.n

    @property
    def std(self) -> np.ndarray | np.generic | float:
        return np.sqrt(self.variance)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "m": None if self.m is None else self.m.copy(),
            "s": None if self.s is None else self.s.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.n = float(state["n"])
        self.m = (
            None
            if state["m"] is None
            else np.asarray(state["m"], dtype=np.float32).copy()
        )
        self.s = (
            None
            if state["s"] is None
            else np.asarray(state["s"], dtype=np.float32).copy()
        )


class TDErrorScaler:
    """Return-based TD-error scaling used by AVG."""

    def __init__(self) -> None:
        self.gamma_rms = RunningStats()
        self.return_sq_rms = RunningStats()
        self.reward_rms = RunningStats()
        self.return_rms = RunningStats()

    def update(self, reward: float, gamma: float, episode_return: float | None) -> None:
        if episode_return is not None:
            self.return_sq_rms.update(episode_return**2)
            self.return_rms.update(episode_return)
        self.reward_rms.update(reward)
        self.gamma_rms.update(gamma)

    @property
    def sigma(self) -> float:
        variance = max(
            float(self.reward_rms.variance)
            + float(self.gamma_rms.variance) * float(self.return_sq_rms.mean),
            1e-4,
        )
        if variance <= 0.01 and self.return_sq_rms.n == 0:
            return 1.0
        return float(np.sqrt(variance))

    def state_dict(self) -> dict[str, Any]:
        return {
            "gamma_rms": self.gamma_rms.state_dict(),
            "return_sq_rms": self.return_sq_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
            "return_rms": self.return_rms.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.gamma_rms.load_state_dict(state["gamma_rms"])
        self.return_sq_rms.load_state_dict(state["return_sq_rms"])
        self.reward_rms.load_state_dict(state["reward_rms"])
        self.return_rms.load_state_dict(state["return_rms"])


class AVGActor(nn.Module):
    """Two-layer squashed Gaussian actor used by AVG."""

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -20.0

    def __init__(self, obs_dim: int, action_dim: int, n_hid: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )
        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)
        self.apply(orthogonal_weight_init)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        phi = self.phi(obs)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        mu = self.mu(phi)
        log_std = torch.clamp(
            self.log_std(phi), self.LOG_STD_MIN, self.LOG_STD_MAX
        )

        # This intentionally matches the original AVG code, which passes
        # exp(log_std) as the diagonal covariance to MultivariateNormal.
        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        action_pre = mu if deterministic else dist.rsample()
        log_prob = dist.log_prob(action_pre)
        log_prob -= (
            2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))
        ).sum(axis=1)
        action = torch.tanh(action_pre)
        info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": log_prob,
            "action_pre": action_pre,
        }
        return action, info


class AVGQ(nn.Module):
    """Two-layer action-value network used by AVG."""

    def __init__(self, obs_dim: int, action_dim: int, n_hid: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(n_hid, 1)
        self.apply(orthogonal_weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        phi = self.phi(torch.cat((obs, action), dim=-1))
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        return self.q(phi).view(-1)


class AVGAgent:
    """Incremental AVG learner with explicit adaptation-state controls."""

    STATE_VERSION = 1

    def __init__(
        self,
        config: AVGConfig,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")

        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.actor = AVGActor(self.obs_dim, self.action_dim, config.nhid_actor).to(
            self.device
        )
        self.Q = AVGQ(self.obs_dim, self.action_dim, config.nhid_critic).to(
            self.device
        )
        self.popt = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
            betas=config.betas,
            weight_decay=config.l2_actor,
        )
        self.qopt = torch.optim.Adam(
            self.Q.parameters(),
            lr=config.critic_lr,
            betas=config.betas,
            weight_decay=config.l2_critic,
        )

        self.alpha = config.alpha_lr
        self.gamma = config.gamma
        self.td_error_scaler = TDErrorScaler()
        self.G = 0.0
        self.steps = 0

    def _observation_tensor(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            tensor = obs.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(
                np.asarray(obs, dtype=np.float32), device=self.device
            )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.obs_dim:
            raise ValueError(
                f"expected observations shaped ({self.obs_dim},) or "
                f"(batch, {self.obs_dim}); got {tuple(tensor.shape)}"
            )
        return tensor

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a batched torch action and the tensors needed by ``update``."""

        obs_tensor = self._observation_tensor(obs)
        if deterministic:
            with torch.no_grad():
                return self.actor(obs_tensor, deterministic=True)
        return self.actor(obs_tensor, deterministic=False)

    # Compatibility with the original experiment scripts.
    compute_action = act

    def update(
        self,
        obs: np.ndarray | torch.Tensor,
        action: torch.Tensor,
        next_obs: np.ndarray | torch.Tensor,
        reward: float,
        terminated: bool,
        episode_end: bool,
        **info: Any,
    ) -> dict[str, float]:
        """Apply one AVG update.

        ``terminated`` controls the Bellman bootstrap mask. ``episode_end``
        closes the return-scaling accumulator and must also be true for time
        limits/truncations.
        """

        terminated = bool(terminated)
        episode_end = bool(episode_end)
        if terminated and not episode_end:
            raise ValueError("a terminated transition must also end the episode")
        if "lprob" not in info:
            raise KeyError("update requires the `lprob` returned by act")

        obs_tensor = self._observation_tensor(obs)
        next_obs_tensor = self._observation_tensor(next_obs)
        action_tensor = action.to(self.device)
        log_prob = info["lprob"].to(self.device)

        entropy_adjusted_reward = float(reward) - self.alpha * log_prob.detach().item()
        self.G += entropy_adjusted_reward
        if episode_end:
            self.td_error_scaler.update(
                reward=entropy_adjusted_reward,
                gamma=0.0,
                episode_return=self.G,
            )
            self.G = 0.0
        else:
            self.td_error_scaler.update(
                reward=entropy_adjusted_reward,
                gamma=self.gamma,
                episode_return=None,
            )

        q_value = self.Q(obs_tensor, action_tensor.detach())
        with torch.no_grad():
            next_action, next_action_info = self.actor(next_obs_tensor)
            next_q = self.Q(next_obs_tensor, next_action)
            target_value = next_q - self.alpha * next_action_info["lprob"]

        bootstrap_mask = 1.0 - float(terminated)
        unscaled_delta = (
            float(reward) + bootstrap_mask * self.gamma * target_value - q_value
        )
        td_scale = self.td_error_scaler.sigma
        scaled_delta = unscaled_delta / td_scale
        critic_loss = scaled_delta**2

        actor_loss = self.alpha * log_prob - self.Q(obs_tensor, action_tensor)
        self.popt.zero_grad()
        actor_loss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        critic_loss.backward()
        self.qopt.step()
        self.steps += 1

        metrics = {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "td_error": float(unscaled_delta.detach().item()),
            "td_scale": float(td_scale),
            "entropy_adjusted_reward": float(entropy_adjusted_reward),
            "bootstrap_mask": float(bootstrap_mask),
            "episode_end": float(episode_end),
            "updates": float(self.steps),
        }
        distribution = info.get("dist")
        if distribution is not None:
            metrics["entropy"] = float(distribution.entropy().detach().mean().item())
        return metrics

    def reset_optimizer_state(self) -> None:
        """Clear both Adam moment histories without modifying network weights."""

        self.popt.state.clear()
        self.qopt.state.clear()

    def reset_td_scaler_state(self) -> None:
        """Replace the TD-error scaler and return accumulator with fresh ones."""

        self.td_error_scaler = TDErrorScaler()
        self.G = 0.0

    def reset_transient_components(
        self, *, optimizer: bool = False, td_scaler: bool = False
    ) -> None:
        """Reset only the explicitly configured transient components.

        Actor/critic weights are always preserved by this method.  Individual
        flags make experiments such as "optimizer only" or "TD scaler only"
        expressible without touching any other learner state.
        """

        if optimizer:
            self.reset_optimizer_state()
        if td_scaler:
            self.reset_td_scaler_state()

    def reset_transient_state(self, reset_optimizers: bool = True) -> None:
        """Reset short-term adaptation state while preserving learned weights."""

        self.reset_transient_components(
            optimizer=reset_optimizers, td_scaler=True
        )

    @staticmethod
    def _numpy_rng_state() -> tuple[Any, ...]:
        name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        return name, keys.copy(), position, has_gauss, cached_gaussian

    def state_dict(self) -> dict[str, Any]:
        """Return complete trainable, transient, and RNG state."""

        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
        return {
            "version": self.STATE_VERSION,
            "config": asdict(self.config),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
            "td_error_scaler": self.td_error_scaler.state_dict(),
            "return_accumulator": self.G,
            "steps": self.steps,
            "rng": {
                "torch": torch.get_rng_state().clone(),
                "torch_cuda": cuda_rng_state,
                "numpy": self._numpy_rng_state(),
            },
        }

    def _move_optimizer_state_to_device(self) -> None:
        for optimizer in (self.popt, self.qopt):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)

    def load_state_dict(
        self, state: Mapping[str, Any], restore_rng: bool = True
    ) -> None:
        """Restore a state produced by :meth:`state_dict`.

        The actor/critic and optimizer key names remain compatible with the
        original AVG checkpoint dictionaries.
        """

        if "obs_dim" in state and int(state["obs_dim"]) != self.obs_dim:
            raise ValueError("checkpoint observation dimension does not match agent")
        if "action_dim" in state and int(state["action_dim"]) != self.action_dim:
            raise ValueError("checkpoint action dimension does not match agent")

        if "config" in state:
            saved_config = AVGConfig(**state["config"])
            if (
                saved_config.nhid_actor != self.config.nhid_actor
                or saved_config.nhid_critic != self.config.nhid_critic
            ):
                raise ValueError("checkpoint network architecture does not match agent")
            self.config = saved_config
            self.alpha = saved_config.alpha_lr
            self.gamma = saved_config.gamma

        self.actor.load_state_dict(state["actor"])
        self.Q.load_state_dict(state["critic"])
        if "policy_opt" in state:
            self.popt.load_state_dict(state["policy_opt"])
        if "critic_opt" in state:
            self.qopt.load_state_dict(state["critic_opt"])
        self._move_optimizer_state_to_device()

        if "td_error_scaler" in state:
            self.td_error_scaler.load_state_dict(state["td_error_scaler"])
        self.G = float(state.get("return_accumulator", 0.0))
        self.steps = int(state.get("steps", 0))

        if restore_rng and "rng" in state:
            rng_state = state["rng"]
            torch.set_rng_state(rng_state["torch"].cpu())
            np.random.set_state(rng_state["numpy"])
            cuda_states = rng_state.get("torch_cuda")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)


# Convenient names for code migrated from the original standalone script.
Actor = AVGActor
Q = AVGQ
AVG = AVGAgent


__all__ = [
    "AVG",
    "AVGActor",
    "AVGAgent",
    "AVGConfig",
    "AVGQ",
    "Actor",
    "Q",
    "RunningStats",
    "TDErrorScaler",
]
