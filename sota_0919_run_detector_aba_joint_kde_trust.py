import torch
import time
import pickle
import argparse
import os
import traceback

import numpy as np
from scipy.spatial import cKDTree
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from gymnasium.wrappers import NormalizeObservation, ClipAction
from datetime import datetime
from incremental_rl.experiment_tracker import record_video
from incremental_rl.td_error_scaler import TDErrorScaler


import copy

# =========================================================
# Utilities
# =========================================================

def orthogonal_weight_init(m):
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def human_format_numbers(num, use_float=False):
    magnitude = 0

    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    if use_float:
        return "%.2f%s" % (
            num,
            ["", "K", "M", "G", "T", "P"][magnitude],
        )

    return "%d%s" % (
        num,
        ["", "K", "M", "G", "T", "P"][magnitude],
    )


def set_one_thread():
    """
    N.B: PyTorch over-allocates CPU resources.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


# =========================================================
# Raw observation passthrough
# =========================================================

class RawObservationInfo(gym.Wrapper):
    """
    Preserve the flattened raw observation in info["raw_obs"] while
    outer wrappers (e.g. NormalizeObservation) may transform the
    observation seen by AVG.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["raw_obs"] = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_obs"] = np.asarray(obs, dtype=np.float32).copy()
        return obs, reward, terminated, truncated, info


# =========================================================
# Running statistics
# =========================================================

class RunningStats:
    """
    Online mean/std using Welford's algorithm.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        x = float(x)

        self.n += 1

        delta = x - self.mean
        self.mean += delta / self.n

        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 1.0

        variance = self.M2 / (self.n - 1)

        return max(variance, 1e-8) ** 0.5


class RunningVectorStats:
    """Per-dimension Welford mean/std for detector normalization."""

    def __init__(self, dim):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return np.ones(self.dim, dtype=np.float64)
        var = self.M2 / (self.n - 1)
        return np.sqrt(np.maximum(var, 1e-8))


class KernelCompetenceMemory:
    """
    Fixed-size recent-memory kernel estimator.

    It stores normalized (state, action) points and PRE-UPDATE prediction
    errors. A compact Epanechnikov kernel gives local visitation support,
    while a temporal half-life makes the local error estimate emphasize
    newer evidence. cKDTree is only an indexing accelerator: trust is based
    on the exact normalized Euclidean distance, not hash-cell membership.
    """

    def __init__(
        self,
        dim,
        capacity=20000,
        bandwidth=0.6,
        error_half_life=5000.0,
        tree_rebuild_interval=512,
    ):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.bandwidth = float(bandwidth)
        self.error_half_life = float(error_half_life)
        self.tree_rebuild_interval = int(tree_rebuild_interval)
        self._radius = self.bandwidth * np.sqrt(self.dim)
        self.reset()

    def reset(self):
        self.x = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self.error = np.zeros(self.capacity, dtype=np.float32)
        self.step = np.zeros(self.capacity, dtype=np.int64)
        self.version = np.zeros(self.capacity, dtype=np.int64)
        self.size = 0
        self.next_slot = 0
        self.version_counter = 0
        self.tree = None
        self.tree_slots = None
        self.tree_versions = None
        self.pending_slots = []
        self.updates_since_tree = 0

    def _rebuild_tree(self):
        if self.size == 0:
            self.tree = None
            self.tree_slots = None
            self.tree_versions = None
        else:
            if self.size < self.capacity:
                slots = np.arange(self.size, dtype=np.int64)
            else:
                slots = np.arange(self.capacity, dtype=np.int64)

            data = self.x[slots].astype(np.float64, copy=True)
            self.tree = cKDTree(data, copy_data=True)
            self.tree_slots = slots.copy()
            self.tree_versions = self.version[slots].copy()

        self.pending_slots = []
        self.updates_since_tree = 0

    def add(self, x, error, step):
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        if self.size < self.capacity:
            slot = self.size
            self.size += 1
        else:
            slot = self.next_slot

        self.next_slot = (slot + 1) % self.capacity
        self.version_counter += 1

        self.x[slot] = x
        self.error[slot] = float(error)
        self.step[slot] = int(step)
        self.version[slot] = self.version_counter

        self.pending_slots.append(slot)
        self.updates_since_tree += 1

        if self.updates_since_tree >= self.tree_rebuild_interval:
            self._rebuild_tree()

    def query(self, x, current_step):
        if self.size == 0:
            return {
                "neighbor_count": 0,
                "support": 0.0,
                "effective_n": 0.0,
                "mean_error": float("nan"),
                "std_error": float("nan"),
            }

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        candidate_slots = []

        if self.tree is None:
            candidate_slots.extend(range(self.size))
        else:
            tree_ids = self.tree.query_ball_point(x, r=self._radius)
            for tree_id in tree_ids:
                slot = int(self.tree_slots[tree_id])
                if self.version[slot] == self.tree_versions[tree_id]:
                    candidate_slots.append(slot)

            candidate_slots.extend(self.pending_slots)

        if not candidate_slots:
            return {
                "neighbor_count": 0,
                "support": 0.0,
                "effective_n": 0.0,
                "mean_error": float("nan"),
                "std_error": float("nan"),
            }

        slots = np.unique(np.asarray(candidate_slots, dtype=np.int64))
        dx = self.x[slots].astype(np.float64) - x[None, :]
        rms_d = np.sqrt(np.mean(dx * dx, axis=1))
        keep = rms_d < self.bandwidth

        if not np.any(keep):
            return {
                "neighbor_count": 0,
                "support": 0.0,
                "effective_n": 0.0,
                "mean_error": float("nan"),
                "std_error": float("nan"),
            }

        slots = slots[keep]
        rms_d = rms_d[keep]

        q2 = (rms_d / self.bandwidth) ** 2
        spatial_w = np.maximum(0.0, 1.0 - q2)

        support = float(np.sum(spatial_w))
        denom2 = float(np.sum(spatial_w ** 2))
        effective_n = (support ** 2 / denom2) if denom2 > 0 else 0.0

        ages = np.maximum(0, int(current_step) - self.step[slots])
        if self.error_half_life > 0:
            temporal_w = np.power(0.5, ages / self.error_half_life)
        else:
            temporal_w = np.ones_like(spatial_w)

        error_w = spatial_w * temporal_w
        error_w_sum = float(np.sum(error_w))

        if error_w_sum <= 1e-12:
            mean_error = float("nan")
            std_error = float("nan")
        else:
            errors = self.error[slots].astype(np.float64)
            mean_error = float(np.sum(error_w * errors) / error_w_sum)
            var_error = float(
                np.sum(error_w * (errors - mean_error) ** 2)
                / error_w_sum
            )
            std_error = float(np.sqrt(max(var_error, 0.0)))

        return {
            "neighbor_count": int(len(slots)),
            "support": support,
            "effective_n": float(effective_n),
            "mean_error": mean_error,
            "std_error": std_error,
        }


# =========================================================
# Actor
# =========================================================

class Actor(nn.Module):
    """Continuous MLP Actor for Soft Actor-Critic."""

    def __init__(self, obs_dim, action_dim, device, n_hid):
        super(Actor, self).__init__()

        self.device = device

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),

            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):
        phi = self.phi(obs.to(self.device))

        phi = phi / torch.norm(
            phi,
            dim=1,
        ).view((-1, 1))

        mu = self.mu(phi)

        log_std = self.log_std(phi)

        log_std = torch.clamp(
            log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX,
        )

        dist = MultivariateNormal(
            mu,
            torch.diag_embed(log_std.exp()),
        )

        action_pre = dist.rsample()

        lprob = dist.log_prob(action_pre)

        lprob -= (
            2
            * (
                np.log(2)
                - action_pre
                - F.softplus(-2 * action_pre)
            )
        ).sum(axis=1)

        action = torch.tanh(action_pre)

        action_info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": lprob,
            "action_pre": action_pre,
        }

        return action, action_info


# =========================================================
# Critic
# =========================================================

class Q(nn.Module):

    def __init__(self, obs_dim, action_dim, device, n_hid):
        super(Q, self).__init__()

        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),

            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.q = nn.Linear(n_hid, 1)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        x = torch.cat(
            (obs, action),
            dim=-1,
        ).to(self.device)

        phi = self.phi(x)

        phi = phi / torch.norm(
            phi,
            dim=1,
        ).view((-1, 1))

        return self.q(phi).view(-1)


# =========================================================
# Probabilistic environment predictor
#
# p_n(Delta S_{t+1}, R_{t+1} | S_t, A_t)
#   = N(mu_n(S_t, A_t), Sigma_n(S_t, A_t))
#
# Sigma_n is diagonal and represented by a log-variance vector.
# =========================================================

class Predictor(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        n_hid=128,
        logvar_min=-10.0,
        logvar_max=5.0,
    ):
        super().__init__()

        self.device = device
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.out_dim = obs_dim + 1  # state delta + reward

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mean_head = nn.Linear(
            n_hid,
            self.out_dim,
        )

        self.logvar_head = nn.Linear(
            n_hid,
            self.out_dim,
        )

        self.apply(orthogonal_weight_init)
        self.to(device)

    def forward(self, raw_obs, action):

        x = torch.cat(
            (raw_obs, action),
            dim=-1,
        ).to(self.device)

        h = self.net(x)

        mean = self.mean_head(h)

        logvar = torch.clamp(
            self.logvar_head(h),
            self.logvar_min,
            self.logvar_max,
        )

        return mean, logvar


def diagonal_gaussian_log_prob(target, mean, var):
    """
    Log p(target) for a diagonal Gaussian.

    target, mean, var: [batch, dimension]
    returns: [batch]
    """
    var = torch.clamp(var, min=1e-8)

    return -0.5 * (
        torch.log(
            2.0
            * torch.pi
            * var
        )
        + (
            (target - mean) ** 2
            / var
        )
    ).sum(dim=-1)


# =========================================================
# AVG
# =========================================================

class AVG:

    def __init__(self, cfg):

        self.cfg = cfg
        self.steps = 0

        self.device = cfg.device

        # -------------------------------------------------
        # Detector logging
        # -------------------------------------------------

        os.makedirs(
            cfg.results_dir,
            exist_ok=True,
        )

        self.regime_log_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_regime_changes_aba_joint_kde_trust.log",
        )

        # Periodic detector trace for debugging/plotting.
        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_detector_trace_aba_joint_kde_trust.csv",
        )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,current_error,kde_neighbors,kde_support,kde_neff,"
                "kde_mean_error,kde_std_error,trusted,z_t,L_t,W_t,"
                "mean_total_var,mean_epistemic_var,warning,frozen,"
                "warmup_remaining,regime_change,predictor_nll\n"
            )

        # -------------------------------------------------
        # Actor / critic
        # -------------------------------------------------

        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_actor,
        )

        self.Q = Q(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_critic,
        )

        # -------------------------------------------------
        # Probabilistic predictor ensemble
        # -------------------------------------------------

        self.predictors = nn.ModuleList(
            [
                Predictor(
                    obs_dim=cfg.obs_dim,
                    action_dim=cfg.action_dim,
                    device=cfg.device,
                    n_hid=cfg.nhid_predictor,
                    logvar_min=cfg.pred_logvar_min,
                    logvar_max=cfg.pred_logvar_max,
                )
                for _ in range(cfg.num_predictors)
            ]
        )

        self.pred_opts = [
            torch.optim.Adam(
                predictor.parameters(),
                lr=cfg.predictor_lr,
            )
            for predictor in self.predictors
        ]

        # -------------------------------------------------
        # Kernel support + local competence detector
        # -------------------------------------------------

        self.num_predictors = cfg.num_predictors
        self.detector_h = cfg.detector_h
        self.detector_h_warn = cfg.detector_h_warn
        self.detector_trust_error = cfg.detector_trust_error
        self.detector_surprise_z = cfg.detector_surprise_z
        self.detector_min_support = cfg.detector_min_support
        self.detector_min_effective_n = cfg.detector_min_effective_n
        self.detector_norm_clip = cfg.detector_norm_clip

        self.initial_detector_warmup = cfg.initial_detector_warmup
        self.post_change_warmup = cfg.post_change_warmup
        self.detector_log_interval = cfg.detector_log_interval

        self.detector_x_stats = RunningVectorStats(
            cfg.obs_dim + cfg.action_dim
        )
        self.detector_y_stats = RunningVectorStats(
            cfg.obs_dim + 1
        )

        self.kernel_memory = KernelCompetenceMemory(
            dim=cfg.obs_dim + cfg.action_dim,
            capacity=cfg.detector_kde_capacity,
            bandwidth=cfg.detector_kde_bandwidth,
            error_half_life=cfg.detector_error_half_life,
            tree_rebuild_interval=cfg.detector_tree_rebuild_interval,
        )

        self.change_score = 0.0
        self.warmup_remaining = self.initial_detector_warmup

        # -------------------------------------------------
        # Actor / critic optimizers
        # -------------------------------------------------

        self.popt = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
        )

        self.qopt = torch.optim.Adam(
            self.Q.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.betas,
        )

        # -------------------------------------------------
        # AVG state
        # -------------------------------------------------

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma

        self.td_error_scaler = TDErrorScaler()

        self.G = 0


    # =====================================================
    # Action
    # =====================================================

    def compute_action(self, obs):

        obs = torch.Tensor(
            obs.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        action, action_info = self.actor(obs)

        return action, action_info


    # =====================================================
    # Update
    # =====================================================

    def update(
        self,
        obs,
        action,
        next_obs,
        reward,
        done,
        raw_obs,
        raw_next_obs,
        **kwargs,
    ):

        # AVG actor/critic observations remain normalized by
        # Gymnasium's NormalizeObservation wrapper.
        obs = torch.tensor(
            obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        next_obs = torch.tensor(
            next_obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # The detector gets RAW flattened environment observations.
        raw_obs = torch.tensor(
            raw_obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        raw_next_obs = torch.tensor(
            raw_next_obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = action.to(self.device)
        detector_action = action.detach()

        lprob = kwargs["lprob"]

        reward_tensor = torch.tensor(
            [[reward]],
            dtype=torch.float32,
            device=self.device,
        )

        # Joint detector target:
        # [Delta S_{t+1}, R_{t+1}]. Given S_t, predicting Delta S is
        # equivalent to predicting S_{t+1}, but avoids the trivial identity
        # component of next-state prediction.
        detector_target = torch.cat(
            (
                raw_next_obs - raw_obs,
                reward_tensor,
            ),
            dim=-1,
        )

        detector_x_np = np.concatenate(
            (
                raw_obs.detach().cpu().numpy().reshape(-1),
                detector_action.detach().cpu().numpy().reshape(-1),
            )
        ).astype(np.float64)

        detector_y_np = (
            detector_target.detach().cpu().numpy().reshape(-1)
            .astype(np.float64)
        )

        if self.steps < self.initial_detector_warmup:
            self.detector_x_stats.update(detector_x_np)
            self.detector_y_stats.update(detector_y_np)

        # =================================================
        # 1. Score the transition BEFORE training on it
        # =================================================

        predictor_outputs = []

        for predictor in self.predictors:
            mean_n, logvar_n = predictor(
                raw_obs,
                detector_action,
            )

            var_n = torch.exp(logvar_n)

            predictor_outputs.append(
                (
                    mean_n,
                    logvar_n,
                    var_n,
                )
            )

        with torch.no_grad():

            means = torch.stack(
                [
                    output[0].detach()
                    for output in predictor_outputs
                ],
                dim=0,
            )

            variances = torch.stack(
                [
                    output[2].detach()
                    for output in predictor_outputs
                ],
                dim=0,
            )

            # Ensemble mean:
            # mu_* = 1/N sum_n mu_n
            mu_star = means.mean(dim=0)

            # Total ensemble covariance diagonal:
            # E[var_n + mu_n^2] - E[mu_n]^2
            var_star = (
                (
                    variances
                    + means.pow(2)
                ).mean(dim=0)
                - mu_star.pow(2)
            )

            var_star = torch.clamp(
                var_star,
                min=1e-8,
            )

            # Pure between-model disagreement, useful for debugging.
            epistemic_var = (
                (
                    means
                    - mu_star.unsqueeze(0)
                ).pow(2)
            ).mean(dim=0)

            y_scale = np.maximum(
                self.detector_y_stats.std,
                1e-3,
            )
            mu_star_np = (
                mu_star.detach().cpu().numpy().reshape(-1)
                .astype(np.float64)
            )
            normalized_residual = (
                detector_y_np - mu_star_np
            ) / y_scale
            current_error = float(
                np.mean(normalized_residual ** 2)
            )

            mean_total_var = var_star.mean().item()
            mean_epistemic_var = epistemic_var.mean().item()

        # =================================================
        # 2. Kernel support + competence-gated CUSUM
        # =================================================

        regime_change = False
        warning = False
        freeze_predictors = False
        predictor_update = True
        add_to_kernel_memory = False

        kde_neighbors = 0
        kde_support = 0.0
        kde_neff = 0.0
        kde_mean_error = float("nan")
        kde_std_error = float("nan")
        trusted = False
        z_t = float("nan")
        L_t = 0.0
        W_for_log = self.change_score

        x_scale = np.maximum(
            self.detector_x_stats.std,
            1e-3,
        )
        x_norm = (
            detector_x_np - self.detector_x_stats.mean
        ) / x_scale
        x_norm = np.clip(
            x_norm,
            -self.detector_norm_clip,
            self.detector_norm_clip,
        )

        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            predictor_update = True
            add_to_kernel_memory = (
                self.steps >= self.initial_detector_warmup
            )

        else:
            local = self.kernel_memory.query(
                x_norm,
                current_step=self.steps,
            )

            kde_neighbors = local["neighbor_count"]
            kde_support = local["support"]
            kde_neff = local["effective_n"]
            kde_mean_error = local["mean_error"]
            kde_std_error = local["std_error"]

            enough_support = (
                kde_support >= self.detector_min_support
                and kde_neff >= self.detector_min_effective_n
            )

            competent = (
                np.isfinite(kde_mean_error)
                and kde_mean_error <= self.detector_trust_error
            )

            trusted = bool(enough_support and competent)

            if trusted:
                std_floor = max(
                    0.25 * self.detector_trust_error,
                    1e-6,
                )
                z_t = (
                    current_error - kde_mean_error
                ) / max(kde_std_error, std_floor)

                L_t = float(
                    np.clip(
                        z_t - self.detector_surprise_z,
                        -5.0,
                        5.0,
                    )
                )

                self.change_score = max(
                    0.0,
                    self.change_score + L_t,
                )

                positive_evidence = L_t > 0.0

                if positive_evidence:
                    freeze_predictors = True
                    predictor_update = False
                    add_to_kernel_memory = False
                else:
                    predictor_update = True
                    add_to_kernel_memory = True

            else:
                self.change_score = 0.0
                L_t = 0.0
                predictor_update = True
                add_to_kernel_memory = True

            W_for_log = self.change_score

            if self.change_score > self.detector_h:
                regime_change = True

                with open(
                    self.regime_log_path,
                    "a",
                ) as f:
                    f.write(
                        f"step={self.steps}, "
                        f"current_error={current_error:.6f}, "
                        f"neighbors={kde_neighbors}, "
                        f"support={kde_support:.6f}, "
                        f"neff={kde_neff:.6f}, "
                        f"mean_error={kde_mean_error:.6f}, "
                        f"std_error={kde_std_error:.6f}, "
                        f"z_t={z_t:.6f}, "
                        f"L_t={L_t:.6f}, "
                        f"W_t={self.change_score:.6f}\n"
                    )

                self.change_score = 0.0
                self.kernel_memory.reset()
                self.warmup_remaining = self.post_change_warmup
                predictor_update = True
                freeze_predictors = False
                add_to_kernel_memory = False

            elif self.change_score >= self.detector_h_warn:
                warning = True
                freeze_predictors = True
                predictor_update = False
                add_to_kernel_memory = False

        if add_to_kernel_memory:
            self.kernel_memory.add(
                x_norm,
                current_error,
                self.steps,
            )

        # =================================================
        # 3. Online Poisson bootstrap predictor update
        # =================================================

        predictor_nll_values = []

        if predictor_update:

            for n, (
                mean_n,
                logvar_n,
                var_n,
            ) in enumerate(predictor_outputs):

                # One streaming transition, one possible optimizer
                # update for this predictor. k=0 means this bootstrap
                # member skips the sample. k>0 weights the NLL.
                k_n_t = np.random.poisson(1.0)

                if k_n_t == 0:
                    continue

                log_prob_n = diagonal_gaussian_log_prob(
                    detector_target,
                    mean_n,
                    var_n,
                )

                nll_n = -log_prob_n.mean()

                loss_n = (
                    float(k_n_t)
                    * nll_n
                )

                self.pred_opts[n].zero_grad()

                loss_n.backward()

                self.pred_opts[n].step()

                predictor_nll_values.append(
                    nll_n.detach().item()
                )

        if predictor_nll_values:
            predictor_nll = float(
                np.mean(predictor_nll_values)
            )
        else:
            predictor_nll = float("nan")

        # =================================================
        # 4. Detector logging
        # =================================================

        if self.steps % self.detector_log_interval == 0:

            with open(
                self.detector_trace_path,
                "a",
            ) as f:

                f.write(
                    f"{self.steps},"
                    f"{current_error:.8f},"
                    f"{kde_neighbors},"
                    f"{kde_support:.8f},"
                    f"{kde_neff:.8f},"
                    f"{kde_mean_error:.8f},"
                    f"{kde_std_error:.8f},"
                    f"{int(trusted)},"
                    f"{z_t:.8f},"
                    f"{L_t:.8f},"
                    f"{W_for_log:.8f},"
                    f"{mean_total_var:.8f},"
                    f"{mean_epistemic_var:.8f},"
                    f"{int(warning)},"
                    f"{int(freeze_predictors)},"
                    f"{self.warmup_remaining},"
                    f"{int(regime_change)},"
                    f"{predictor_nll:.8f}\n"
                )

        # =================================================
        # 5. Original AVG update
        # =================================================

        # -------------------------------------------------
        # Return scaling
        # -------------------------------------------------

        r_ent = (
            reward
            - self.alpha
            * lprob.detach().item()
        )

        self.G += r_ent

        if done:

            self.td_error_scaler.update(
                reward=r_ent,
                gamma=0,
                G=self.G,
            )

            self.G = 0

        else:

            self.td_error_scaler.update(
                reward=r_ent,
                gamma=self.cfg.gamma,
                G=None,
            )

        # -------------------------------------------------
        # Q loss
        # -------------------------------------------------

        q = self.Q(
            obs,
            action.detach(),
        )

        with torch.no_grad():

            next_action, action_info = self.actor(
                next_obs
            )

            next_lprob = action_info["lprob"]

            q2 = self.Q(
                next_obs,
                next_action,
            )

            target_V = (
                q2
                - self.alpha
                * next_lprob
            )

        delta = (
            reward
            + (1 - done)
            * self.gamma
            * target_V
            - q
        )

        delta /= self.td_error_scaler.sigma

        qloss = delta ** 2

        # -------------------------------------------------
        # Policy loss
        # -------------------------------------------------

        ploss = (
            self.alpha * lprob
            - self.Q(obs, action)
        )

        self.popt.zero_grad()

        ploss.backward()

        self.popt.step()

        self.qopt.zero_grad()

        qloss.backward()

        self.qopt.step()

        self.steps += 1

        return {
            "current_error": current_error,
            "kde_neighbors": kde_neighbors,
            "kde_support": kde_support,
            "kde_neff": kde_neff,
            "kde_mean_error": kde_mean_error,
            "kde_std_error": kde_std_error,
            "trusted": trusted,
            "z_t": z_t,
            "L_t": L_t,
            "W_t": W_for_log,
            "mean_total_var": mean_total_var,
            "mean_epistemic_var": mean_epistemic_var,
            "warning": warning,
            "predictors_frozen": freeze_predictors,
            "warmup_remaining": self.warmup_remaining,
            "regime_change": regime_change,
            "predictor_nll": predictor_nll,
        }


    # =====================================================
    # Save
    # =====================================================

    def save(self, model_dir, unique_str):

        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),

            "predictors": [
                predictor.state_dict()
                for predictor in self.predictors
            ],

            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),

            "predictor_opts": [
                opt.state_dict()
                for opt in self.pred_opts
            ],
        }

        torch.save(
            model,
            "%s/%s.pt" % (
                model_dir,
                unique_str,
            ),
        )


# =========================================================
# Experiment
# =========================================================

def main(args):

    tic = time.time()

    run_id = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        +
        f"-joint"
        f"-{args.algo}"
        f"-{args.env}"
        f"_seed-{args.seed}"
    )

    # IMPORTANT:
    # AVG.__init__ needs this for regime log path.
    args.run_id = run_id

    # =====================================================
    # Environment
    # =====================================================

    env = gym.make(args.env)

    env = gym.wrappers.FlattenObservation(env)

    # Preserve raw flattened observations for the detector.
    env = RawObservationInfo(env)

    # AVG still receives its usual normalized observations.
    env = NormalizeObservation(env)

    env = ClipAction(env)

    base_env = env.unwrapped

    # =====================================================
    # A -> B -> A joint-malfunction regime
    # =====================================================

    malfunction_actuator = 0

    original_gear = (
        base_env.model.actuator_gear[
            malfunction_actuator,
            0,
        ].copy()
    )

    # =====================================================
    # Reproducibility
    # =====================================================

    env.action_space.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # =====================================================
    # Learner
    # =====================================================

    args.obs_dim = (
        env.observation_space.shape[0]
    )

    args.action_dim = (
        env.action_space.shape[0]
    )

    agent = AVG(args)

    # =====================================================
    # Interaction
    # =====================================================

    rets = []
    ep_steps = []

    ret = 0
    step = 0

    terminated = False
    truncated = False

    obs, info = env.reset(seed=args.seed)
    raw_obs = info["raw_obs"]

    ep_tic = time.time()

    try:

        for t in range(args.N):

            # ---------------------------------------------
            # Action
            # ---------------------------------------------

            action, action_info = (
                agent.compute_action(obs)
            )

            sim_action = (
                action.detach()
                .cpu()
                .view(-1)
                .numpy()
            )

            # =============================================
            # A -> B -> A joint-malfunction regime
            # =============================================

            if t == 50_000:
                # A -> B: reverse actuator-0 torque polarity.
                base_env.model.actuator_gear[
                    malfunction_actuator,
                    0,
                ] = -original_gear

                print(
                    f"A -> B at t={t}: "
                    f"actuator {malfunction_actuator} gear "
                    f"{original_gear} -> {-original_gear}"
                )

            elif t == 100_000:
                # B -> A: restore the original actuator gear.
                base_env.model.actuator_gear[
                    malfunction_actuator,
                    0,
                ] = original_gear

                print(
                    f"B -> A at t={t}: "
                    f"actuator {malfunction_actuator} gear "
                    f"{-original_gear} -> {original_gear}"
                )

            # =============================================
            # Environment transition
            # =============================================

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(sim_action)

            raw_next_obs = info["raw_obs"]

            detector_info = agent.update(
                obs,
                action,
                next_obs,
                reward,
                terminated,
                raw_obs=raw_obs,
                raw_next_obs=raw_next_obs,
                **action_info,
            )

            ret += reward
            step += 1

            obs = next_obs
            raw_obs = raw_next_obs

            # =============================================
            # Checkpoint
            # =============================================

            if (
                t % args.checkpoint == 0
                and args.save_model
            ):

                agent.save(
                    model_dir=args.results_dir,
                    unique_str=(
                        f"{run_id}"
                        f"_model_"
                        f"{human_format_numbers(t)}"
                    ),
                )

            # =============================================
            # Episode termination
            # =============================================

            if terminated or truncated:

                rets.append(ret)
                ep_steps.append(step)

                print(
                    "E: {}| D: {:.3f}| "
                    "S: {}| R: {:.2f}| T: {}".format(
                        len(rets),
                        time.time() - ep_tic,
                        step,
                        ret,
                        t,
                    )
                )

                ep_tic = time.time()

                obs, info = env.reset()
                raw_obs = info["raw_obs"]

                ret = 0
                step = 0

    except Exception as e:

        print(e)

        print(
            "Exiting this run, storing partial "
            "logs for debugging..."
        )

        traceback.print_exc()

    # =====================================================
    # Partial episode
    # =====================================================

    if not (terminated or truncated):

        print(
            "Appending partial episode #{}, "
            "length: {}, Total Steps: {}".format(
                len(rets),
                step,
                t + 1,
            )
        )

        rets.append(ret)
        ep_steps.append(step)

    # =====================================================
    # Save model
    # =====================================================

    if args.save_model:

        agent.save(
            model_dir=args.results_dir,
            unique_str=f"{run_id}_model",
        )

    print(
        "Run with id: {} took {:.3f}s!".format(
            run_id,
            time.time() - tic,
        )
    )

    # =====================================================
    # Eval
    # =====================================================

    if args.n_eval:

        record_video(
            env,
            agent,
            num_episodes=args.n_eval,
            video_filename=(
                f"{args.results_dir}"
                f"/{run_id}.avi"
            ),
        )

    return ep_steps, rets


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="HalfCheetah-v4",
        type=str,
        help="e.g., 'HalfCheetah-v4'",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for random number generator",
    )

    parser.add_argument(
        "--N",
        default=150_000,
        type=int,
        help="# timesteps for the run",
    )

    # =====================================================
    # AVG parameters
    # =====================================================

    parser.add_argument(
        "--actor_lr",
        default=0.0063,
        type=float,
        help="Actor step size",
    )

    parser.add_argument(
        "--critic_lr",
        default=0.0087,
        type=float,
        help="Critic step size",
    )

    parser.add_argument(
        "--beta1",
        default=0.0,
        type=float,
        help="Beta1 parameter of Adam optimizer",
    )

    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="Discount factor",
    )

    parser.add_argument(
        "--alpha_lr",
        default=0.07,
        type=float,
        help="Entropy Coefficient for AVG",
    )

    parser.add_argument(
        "--l2_actor",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--l2_critic",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--nhid_actor",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--nhid_critic",
        default=256,
        type=int,
    )

    # =====================================================
    # Predictor parameters
    # =====================================================

    parser.add_argument(
        "--nhid_predictor",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--predictor_lr",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--num_predictors",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--pred_logvar_min",
        default=-10.0,
        type=float,
    )

    parser.add_argument(
        "--pred_logvar_max",
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--detector_delta",
        default=2.0,
        type=float,
        help="Legacy argument retained for old launch scripts; unused by KDE detector",
    )

    parser.add_argument(
        "--detector_kde_capacity",
        default=20000,
        type=int,
        help="Fixed-size recent transition memory used by the kernel estimator",
    )

    parser.add_argument(
        "--detector_kde_bandwidth",
        default=0.6,
        type=float,
        help="Compact-kernel radius in RMS frozen-normalized (state, action) distance",
    )

    parser.add_argument(
        "--detector_min_support",
        default=10.0,
        type=float,
        help="Minimum kernel-weighted local visitation support required for trust",
    )

    parser.add_argument(
        "--detector_min_effective_n",
        default=10.0,
        type=float,
        help="Minimum effective number of local samples required for trust",
    )

    parser.add_argument(
        "--detector_trust_error",
        default=0.01,
        type=float,
        help="Maximum recency-weighted local normalized MSE for a neighborhood to be trusted",
    )

    parser.add_argument(
        "--detector_error_half_life",
        default=5000.0,
        type=float,
        help="Half-life in environment steps for local historical prediction errors",
    )

    parser.add_argument(
        "--detector_surprise_z",
        default=3.0,
        type=float,
        help="Local surprise threshold before a trusted sample contributes positive CUSUM evidence",
    )

    parser.add_argument(
        "--detector_norm_clip",
        default=5.0,
        type=float,
        help="Clip frozen-normalized detector inputs before kernel distance",
    )

    parser.add_argument(
        "--detector_tree_rebuild_interval",
        default=512,
        type=int,
        help="How often to rebuild the cKDTree index; exact pending points are checked between rebuilds",
    )

    parser.add_argument(
        "--detector_h",
        default=15.0,
        type=float,
    )

    parser.add_argument(
        "--detector_h_warn",
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--initial_detector_warmup",
        default=10_000,
        type=int,
        help="Initial predictor calibration period before CUSUM is enabled",
    )

    parser.add_argument(
        "--post_change_warmup",
        default=1_000,
        type=int,
        help="Predictor adaptation period after a detected regime change",
    )

    parser.add_argument(
        "--detector_log_interval",
        default=100,
        type=int,
        help="Write detector trace every N environment steps",
    )

    # =====================================================
    # Misc.
    # =====================================================

    parser.add_argument(
        "--checkpoint",
        default=50000,
        type=int,
    )

    parser.add_argument(
        "--results_dir",
        default="./results",
        type=str,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--n_eval",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    # Adam
    args.betas = [
        args.beta1,
        0.999,
    ]

    # Device
    if (
        torch.cuda.is_available()
        and "cuda" in args.device
    ):

        args.device = torch.device(
            args.device
        )

    else:

        args.device = torch.device(
            "cpu"
        )

    args.algo = "AVG"

    # =====================================================
    # Run
    # =====================================================

    set_one_thread()

    ep_steps, rets = main(args)

    # =====================================================
    # Save results
    # =====================================================

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    pkl_fpath = os.path.join(
        args.results_dir,
        (
            f"{args.env}"
            f"_aba_joint_detector_kde_trust"
            f"_seed-{args.seed}.pkl"
        ),
    )

    with open(
        pkl_fpath,
        "wb",
    ) as f:

        pickle.dump(
            (
                ep_steps,
                rets,
                args.env,
            ),
            f,
        )