import torch
import time
import pickle
import argparse
import os
import traceback

import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from torch.distributions import MultivariateNormal
from gymnasium.wrappers import NormalizeObservation, ClipAction
from datetime import datetime
from incremental_rl.experiment_tracker import record_video
from incremental_rl.td_error_scaler import TDErrorScaler



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


# =========================================================
# Predictor input / target normalization
# =========================================================

class OnlineFrozenVectorNormalizer:
    """
    Fixed-memory per-dimension Welford normalizer.

    Statistics are updated strictly online and are frozen after
    warmup_steps samples.  A transition is always normalized using the
    statistics available *before* that transition is inserted into the
    normalizer, preserving predict -> score -> learn ordering.

    These statistics are scale/coordinate information for the detector
    predictor, not historical regime memory, so they are preserved when
    the predictor ensemble is reset after an alarm.
    """

    def __init__(self, dim, warmup_steps=10_000, eps=1e-6):
        self.dim = int(dim)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)

        if self.dim <= 0:
            raise ValueError("Require normalizer dim > 0")
        if self.warmup_steps < 1:
            raise ValueError("Require normalizer warmup_steps >= 1")

        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

        self.frozen_mean = None
        self.frozen_std = None

    @property
    def ready(self):
        return self.frozen_mean is not None

    def _current_mean_std_numpy(self):
        if self.ready:
            return self.frozen_mean, self.frozen_std

        if self.n == 0:
            mean = np.zeros(self.dim, dtype=np.float64)
            std = np.ones(self.dim, dtype=np.float64)
            return mean, std

        mean = self.mean

        if self.n < 2:
            std = np.ones(self.dim, dtype=np.float64)
        else:
            var = self.M2 / float(self.n - 1)
            std = np.sqrt(np.maximum(var, self.eps))

        return mean, std

    def mean_std_tensors(self, device, dtype=torch.float32):
        mean, std = self._current_mean_std_numpy()

        mean_t = torch.as_tensor(
            mean,
            dtype=dtype,
            device=device,
        ).unsqueeze(0)

        std_t = torch.as_tensor(
            std,
            dtype=dtype,
            device=device,
        ).unsqueeze(0)

        return mean_t, std_t

    def observe(self, x):
        """
        Insert one vector after the transition has already been scored/trained.
        """
        if self.ready:
            return

        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected normalizer dimension {self.dim}, got {x.shape[0]}"
            )

        self.n += 1

        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        if self.n >= self.warmup_steps:
            if self.n < 2:
                var = np.ones(self.dim, dtype=np.float64)
            else:
                var = self.M2 / float(self.n - 1)

            self.frozen_mean = self.mean.copy()
            self.frozen_std = np.sqrt(
                np.maximum(var, self.eps)
            )

    def state_dict(self):
        return {
            "n": self.n,
            "mean": self.mean,
            "M2": self.M2,
            "frozen_mean": self.frozen_mean,
            "frozen_std": self.frozen_std,
            "warmup_steps": self.warmup_steps,
        }


# =========================================================
# Fixed-memory state-action familiarity
# =========================================================

class StateActionFamiliarityTracker:
    """
    Fixed-memory state-action locality sketch with local outcome statistics.

    A small Euclidean-LSH sketch maps continuous x_t = [S_t, A_t] into one
    bucket per table.  Each bucket stores fixed-memory Welford statistics for
    the normalized outcome

        y_t = [Delta S_t, R_{t+1}].

    Detection therefore compares the current outcome against outcomes
    previously observed at approximately the same state-action location,
    rather than against a continually changing neural predictor.

    For every transition, scoring happens BEFORE the current outcome updates
    its bucket statistics.  A table contributes a local normalized innovation
    score only after its bucket has at least min_outcome_count prior samples.
    The detector uses the median score across valid tables for robustness to
    individual LSH collisions.

    Memory is O(num_tables * num_buckets * outcome_dim) and does not grow with
    stream length or the number of regimes.
    """

    def __init__(
        self,
        dim,
        outcome_dim,
        seed,
        norm_warmup_steps=10_000,
        num_tables=4,
        projections_per_table=4,
        num_buckets=4096,
        hash_width=1.0,
        count_scale=10.0,
        input_clip=5.0,
        min_outcome_count=20,
        min_valid_tables=2,
        outcome_var_floor=0.01,
    ):
        self.dim = int(dim)
        self.outcome_dim = int(outcome_dim)
        self.norm_warmup_steps = int(norm_warmup_steps)
        self.num_tables = int(num_tables)
        self.projections_per_table = int(projections_per_table)
        self.num_buckets = int(num_buckets)
        self.hash_width = float(hash_width)
        self.count_scale = float(count_scale)
        self.input_clip = float(input_clip)
        self.min_outcome_count = int(min_outcome_count)
        self.min_valid_tables = int(min_valid_tables)
        self.outcome_var_floor = float(outcome_var_floor)

        if self.dim <= 0:
            raise ValueError("Require familiarity dim > 0")
        if self.outcome_dim <= 0:
            raise ValueError("Require outcome_dim > 0")
        if self.norm_warmup_steps < 1:
            raise ValueError("Require familiarity_norm_warmup >= 1")
        if self.num_tables < 1 or self.projections_per_table < 1:
            raise ValueError("Require positive LSH table/projection counts")
        if self.num_buckets < 2:
            raise ValueError("Require familiarity_num_buckets >= 2")
        if self.hash_width <= 0.0:
            raise ValueError("Require familiarity_hash_width > 0")
        if self.count_scale <= 0.0:
            raise ValueError("Require familiarity_count_scale > 0")
        if self.input_clip <= 0.0:
            raise ValueError("Require familiarity_input_clip > 0")
        if self.min_outcome_count < 2:
            raise ValueError("Require local_min_count >= 2")
        if not (1 <= self.min_valid_tables <= self.num_tables):
            raise ValueError(
                "Require 1 <= local_min_valid_tables <= familiarity_num_tables"
            )
        if self.outcome_var_floor <= 0.0:
            raise ValueError("Require local_variance_floor > 0")

        rng = np.random.RandomState(int(seed))

        self.projections = rng.normal(
            size=(
                self.num_tables,
                self.projections_per_table,
                self.dim,
            )
        ).astype(np.float64)

        norms = np.linalg.norm(
            self.projections,
            axis=-1,
            keepdims=True,
        )
        self.projections /= np.maximum(norms, 1e-12)

        self.offsets = rng.uniform(
            0.0,
            self.hash_width,
            size=(self.num_tables, self.projections_per_table),
        ).astype(np.float64)

        # Fixed coefficients for stable bucket mixing.
        self.mix_coeffs = rng.randint(
            1,
            2**31 - 1,
            size=(self.num_tables, self.projections_per_table),
            dtype=np.int64,
        )

        # Regime-local bucket statistics. Counts are also used for familiarity.
        self.counts = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.uint32,
        )
        self.outcome_mean = np.zeros(
            (self.num_tables, self.num_buckets, self.outcome_dim),
            dtype=np.float32,
        )
        self.outcome_M2 = np.zeros(
            (self.num_tables, self.num_buckets, self.outcome_dim),
            dtype=np.float32,
        )

        # State-action coordinates are normalized once and then frozen.  These
        # are coordinate statistics, not regime-specific memory.
        self.norm_count = 0
        self.norm_mean = np.zeros(self.dim, dtype=np.float64)
        self.norm_M2 = np.zeros(self.dim, dtype=np.float64)
        self.frozen_mean = None
        self.frozen_std = None

    @property
    def ready(self):
        return self.frozen_mean is not None

    @property
    def warmup_remaining(self):
        return max(0, self.norm_warmup_steps - self.norm_count)

    def _update_normalizer(self, x):
        self.norm_count += 1

        delta = x - self.norm_mean
        self.norm_mean += delta / self.norm_count
        delta2 = x - self.norm_mean
        self.norm_M2 += delta * delta2

        if self.norm_count == self.norm_warmup_steps:
            if self.norm_count > 1:
                var = self.norm_M2 / (self.norm_count - 1)
            else:
                var = np.ones(self.dim, dtype=np.float64)

            self.frozen_mean = self.norm_mean.copy()
            self.frozen_std = np.sqrt(np.maximum(var, 1e-6))

    def _standardize(self, x):
        z = (x - self.frozen_mean) / self.frozen_std
        return np.clip(z, -self.input_clip, self.input_clip)

    def _bucket_indices(self, z):
        projected = np.einsum(
            'tpd,d->tp',
            self.projections,
            z,
        )

        cells = np.floor(
            (projected + self.offsets) / self.hash_width
        ).astype(np.int64)

        buckets = []
        for table_idx in range(self.num_tables):
            # Stable integer mixing; no Python hash/randomization involved.
            mixed = np.int64(1469598103934665603 & 0x7FFFFFFFFFFFFFFF)
            for projection_idx in range(self.projections_per_table):
                term = (
                    cells[table_idx, projection_idx]
                    * self.mix_coeffs[table_idx, projection_idx]
                )
                mixed ^= np.int64(term)
                mixed = np.int64(
                    (int(mixed) * 1099511628211)
                    & 0x7FFFFFFFFFFFFFFF
                )

            buckets.append(int(mixed % self.num_buckets))

        return buckets

    def score_and_observe(self, x, outcome):
        """
        Score one transition against prior outcomes in matching LSH buckets,
        then insert the current outcome.

        Returns:
            pseudo_count, familiarity, ready,
            local_surprise, matched, valid_table_count

        local_surprise is the median, across valid hash tables, of

            mean_j (y_j - mu_{b,j})^2 / max(var_{b,j}, variance_floor).

        The score is computed before the current sample updates its bucket.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        outcome = np.asarray(outcome, dtype=np.float64).reshape(-1)

        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected state-action dimension {self.dim}, got {x.shape[0]}"
            )
        if outcome.shape[0] != self.outcome_dim:
            raise ValueError(
                f"Expected outcome dimension {self.outcome_dim}, "
                f"got {outcome.shape[0]}"
            )

        if not self.ready:
            self._update_normalizer(x)
            return 0, 0.0, self.ready, float("nan"), False, 0

        z = self._standardize(x)
        buckets = self._bucket_indices(z)

        table_counts = [
            int(self.counts[t, bucket])
            for t, bucket in enumerate(buckets)
        ]

        pseudo_count = int(np.median(table_counts))
        familiarity = float(
            1.0 - np.exp(-float(pseudo_count) / self.count_scale)
        )

        table_scores = []
        for table_idx, bucket in enumerate(buckets):
            n = int(self.counts[table_idx, bucket])
            if n < self.min_outcome_count:
                continue

            mean = self.outcome_mean[table_idx, bucket].astype(
                np.float64,
                copy=False,
            )
            var = (
                self.outcome_M2[table_idx, bucket].astype(
                    np.float64,
                    copy=False,
                )
                / float(max(1, n - 1))
            )
            var = np.maximum(var, self.outcome_var_floor)

            score = np.mean(((outcome - mean) ** 2) / var)
            if np.isfinite(score):
                table_scores.append(float(score))

        valid_table_count = len(table_scores)
        matched = valid_table_count >= self.min_valid_tables
        local_surprise = (
            float(np.median(table_scores))
            if matched
            else float("nan")
        )

        # Insert the transition after scoring it.
        max_uint32 = np.iinfo(np.uint32).max
        outcome32 = outcome.astype(np.float32, copy=False)

        for table_idx, bucket in enumerate(buckets):
            n = int(self.counts[table_idx, bucket])
            if n >= max_uint32:
                continue

            new_n = n + 1
            old_mean = self.outcome_mean[table_idx, bucket].copy()
            delta = outcome32 - old_mean
            new_mean = old_mean + delta / float(new_n)
            delta2 = outcome32 - new_mean

            self.outcome_mean[table_idx, bucket] = new_mean
            self.outcome_M2[table_idx, bucket] += delta * delta2
            self.counts[table_idx, bucket] = new_n

        return (
            pseudo_count,
            familiarity,
            True,
            local_surprise,
            matched,
            valid_table_count,
        )

    def reset_counts(self):
        """
        Forget regime-local state-action/outcome statistics while preserving
        fixed LSH projections and frozen state-action normalization.
        """
        self.counts.fill(0)
        self.outcome_mean.fill(0.0)
        self.outcome_M2.fill(0.0)

    def state_dict(self):
        return {
            "norm_count": self.norm_count,
            "norm_mean": self.norm_mean,
            "norm_M2": self.norm_M2,
            "frozen_mean": self.frozen_mean,
            "frozen_std": self.frozen_std,
            "counts": self.counts,
            "outcome_mean": self.outcome_mean,
            "outcome_M2": self.outcome_M2,
            "outcome_dim": self.outcome_dim,
            "min_outcome_count": self.min_outcome_count,
            "min_valid_tables": self.min_valid_tables,
            "outcome_var_floor": self.outcome_var_floor,
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
# p_n(Delta S_t, R_{t+1} | S_t, A_t)
#   = N(mu_n(S_t, A_t), Sigma_n(S_t, A_t))
#
# Delta S_t = S_{t+1} - S_t.
# Predictor inputs and targets are normalized online using fixed-memory
# statistics frozen after the initial normalization period.
#
# Sigma_n is diagonal and represented by a log-variance vector.
# =========================================================

class Predictor(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        n_hid=256,
        num_layers=4,
        logvar_min=-10.0,
        logvar_max=5.0,
    ):
        super().__init__()

        self.device = device
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.out_dim = obs_dim + 1  # next observation + reward
        self.n_hid = int(n_hid)
        self.num_layers = int(num_layers)

        if self.n_hid < 1:
            raise ValueError("Require nhid_predictor >= 1")
        if self.num_layers < 1:
            raise ValueError("Require predictor_num_layers >= 1")

        # Capacity stress test:
        # baseline predictor was 2 x 128.  The default here is 4 x 256.
        # The experiment is intended to test whether stationary surprise
        # spikes are partly caused by limited capacity / representation
        # interference in the continually trained predictor.
        layers = []
        in_dim = obs_dim + action_dim

        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.n_hid))
            layers.append(nn.LeakyReLU())
            in_dim = self.n_hid

        self.net = nn.Sequential(*layers)

        self.mean_head = nn.Linear(
            self.n_hid,
            self.out_dim,
        )

        self.logvar_head = nn.Linear(
            self.n_hid,
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
            f"{cfg.run_id}_regime_changes_aba_joint_surprise_familiarity_gradient.log",
        )

        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_surprise_familiarity_gradient_trace_aba_joint.csv",
        )

        self.detector_block_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_surprise_familiarity_gradient_blocks_aba_joint.csv",
        )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,surprise_raw,surprise_clipped,"
                "state_action_pseudo_count,familiarity,e_t,"
                "local_conditional_surprise,local_matched,local_valid_tables,"
                "block_E_k,W_k,block_completed,warmup_remaining,"
                "regime_change,mean_total_var,mean_epistemic_var,"
                "raw_residual_mse,raw_delta_state_mse,raw_reward_sq_error,"
                "mean_total_var_raw,mean_epistemic_var_raw,"
                "gradient_coherence,gradient_novelty,predictor_nll\n"
            )

        with open(self.detector_block_path, "w") as f:
            f.write(
                "block_index,step,E_k,W_k,"
                "mean_familiarity,mean_predictor_surprise,"
                "mean_gradient_coherence,mean_gradient_novelty,"
                "regime_change,local_surprise_mean,matched_fraction,"
                "mean_valid_tables,block_change_evidence\n"
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
                    num_layers=cfg.predictor_num_layers,
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

        self.num_predictors = cfg.num_predictors

        # -------------------------------------------------
        # Predictor normalization
        #
        # Input:
        #   x_t = [raw S_t, A_t]
        #
        # Target:
        #   y_t = [raw S_{t+1} - raw S_t, R_{t+1}]
        #
        # Welford statistics are fixed-memory and frozen after the
        # initial normalization period.  They are intentionally kept
        # across detector-predictor resets: they define coordinates,
        # not historical regime-specific predictive knowledge.
        # -------------------------------------------------
        self.predictor_input_normalizer = OnlineFrozenVectorNormalizer(
            dim=cfg.obs_dim + cfg.action_dim,
            warmup_steps=cfg.predictor_norm_warmup,
        )

        self.predictor_target_normalizer = OnlineFrozenVectorNormalizer(
            dim=cfg.obs_dim + 1,
            warmup_steps=cfg.predictor_norm_warmup,
        )

        predictor_params_each = sum(
            p.numel() for p in self.predictors[0].parameters()
        )
        predictor_params_total = sum(
            p.numel()
            for predictor in self.predictors
            for p in predictor.parameters()
        )
        print(
            "Predictor ensemble: "
            f"{cfg.predictor_num_layers} hidden layers x "
            f"{cfg.nhid_predictor} units, "
            f"{predictor_params_each:,} params/model, "
            f"{predictor_params_total:,} params total"
        )

        # -------------------------------------------------
        # Predictor-gradient direction diagnostics
        #
        # For each ensemble member, track an EMA reference direction for
        # the gradient of the unweighted predictor NLL with respect to the
        # predictor mean head.  The diagnostic does NOT affect learning or
        # regime-change decisions.
        #
        #   c_t = cos(g_t, m_{t-1})          (coherence)
        #   d_t = 1 - c_t                    (novelty)
        #
        # Each predictor keeps its own reference because independently
        # initialized predictors do not share a common parameter basis.
        # -------------------------------------------------
        self.gradient_reference_beta = cfg.gradient_reference_beta
        self.gradient_reference_eps = 1e-12
        self.gradient_direction_refs = [
            None for _ in range(self.num_predictors)
        ]

        # Number of detector-predictor resets performed after alarms.
        self.predictor_reset_count = 0

        # -------------------------------------------------
        # Local conditional-dynamics detector
        #
        # The detector no longer uses neural-predictor surprise as its change
        # statistic.  Instead, each fixed LSH state-action bucket stores
        # regime-local outcome statistics for
        #
        #     y_t = [Delta S_t, R_{t+1}].
        #
        # For a sufficiently visited bucket, score the new outcome against the
        # bucket's PRIOR mean/variance, before updating that bucket:
        #
        #     q_t = median_tables mean_j
        #           (y_{t,j} - mu_{b,j})^2 / max(var_{b,j}, v_min).
        #
        # This conditions the comparison on approximately matched (S_t,A_t),
        # removing neural-predictor learning as a source of detector surprise.
        # -------------------------------------------------

        self.detector_h = cfg.detector_h
        self.local_reference = cfg.local_reference
        self.local_margin = cfg.local_margin
        self.local_score_cap = cfg.local_score_cap
        self.local_min_block_fraction = cfg.local_min_block_fraction

        self.initial_detector_warmup = cfg.initial_detector_warmup
        self.detector_log_interval = cfg.detector_log_interval

        self.change_score = 0.0
        self.warmup_remaining = self.initial_detector_warmup

        # Block statistics for local conditional evidence.
        self.block_evidence_sum = 0.0
        self.block_matched_count = 0
        self.block_valid_tables_sum = 0.0
        self.block_familiarity_sum = 0.0
        self.block_surprise_sum = 0.0  # neural predictor diagnostic only
        self.block_gradient_coherence_sum = 0.0
        self.block_gradient_novelty_sum = 0.0
        self.block_gradient_count = 0
        self.block_count = 0
        self.block_index = 0

        self.surprise_cap = cfg.surprise_cap
        self.block_size = cfg.block_size

        self.state_action_familiarity = StateActionFamiliarityTracker(
            dim=cfg.obs_dim + cfg.action_dim,
            outcome_dim=cfg.obs_dim + 1,
            seed=cfg.seed + 7919,
            norm_warmup_steps=cfg.familiarity_norm_warmup,
            num_tables=cfg.familiarity_num_tables,
            projections_per_table=(
                cfg.familiarity_projections_per_table
            ),
            num_buckets=cfg.familiarity_num_buckets,
            hash_width=cfg.familiarity_hash_width,
            count_scale=cfg.familiarity_count_scale,
            input_clip=cfg.familiarity_input_clip,
            min_outcome_count=cfg.local_min_count,
            min_valid_tables=cfg.local_min_valid_tables,
            outcome_var_floor=cfg.local_variance_floor,
        )


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
    # Regime-local predictor reset
    # =====================================================

    def _reset_predictor_ensemble(self):
        """Reinitialize detector predictors and their Adam states.

        The detector predictor is intentionally regime-local rather than a
        lifelong world model.  After a trusted alarm, old-regime predictive
        knowledge is discarded so recurrence of an earlier regime can still
        produce surprise relative to the immediately preceding regime.

        Predictor initialization uses an isolated RNG seed and then restores
        PyTorch RNG state.  This prevents a detector reset from changing the
        actor's subsequent stochastic action samples merely by consuming RNG.
        """
        self.predictor_reset_count += 1
        reset_seed = (
            int(self.cfg.seed)
            + 104729 * self.predictor_reset_count
        )

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        try:
            torch.manual_seed(reset_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(reset_seed)

            self.predictors = nn.ModuleList(
                [
                    Predictor(
                        obs_dim=self.cfg.obs_dim,
                        action_dim=self.cfg.action_dim,
                        device=self.cfg.device,
                        n_hid=self.cfg.nhid_predictor,
                        num_layers=self.cfg.predictor_num_layers,
                        logvar_min=self.cfg.pred_logvar_min,
                        logvar_max=self.cfg.pred_logvar_max,
                    )
                    for _ in range(self.num_predictors)
                ]
            )

            self.pred_opts = [
                torch.optim.Adam(
                    predictor.parameters(),
                    lr=self.cfg.predictor_lr,
                )
                for predictor in self.predictors
            ]

        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

        # Gradient references belong to the old parameterization.
        self.gradient_direction_refs = [
            None for _ in range(self.num_predictors)
        ]

        print(
            f"Reset detector predictor ensemble "
            f"#{self.predictor_reset_count} at step {self.steps}"
        )

    # =====================================================
    # Predictor-gradient direction diagnostic
    # =====================================================

    def _gradient_direction_diagnostic(self, predictor_idx, grad_vector):
        """Return (coherence, novelty) and update the predictor's EMA direction.

        grad_vector is the gradient of the current unweighted predictor NLL
        with respect to that predictor's mean-head parameters.  References are
        unit-normalized so this statistic tracks direction rather than gradient
        magnitude.  The reference is updated only after scoring the current
        gradient, so c_t compares g_t against m_{t-1}.
        """
        grad_vector = grad_vector.detach()

        if not torch.isfinite(grad_vector).all():
            return float("nan"), float("nan")

        grad_norm = torch.linalg.vector_norm(grad_vector)
        if grad_norm.item() <= self.gradient_reference_eps:
            return float("nan"), float("nan")

        grad_unit = grad_vector / grad_norm
        reference = self.gradient_direction_refs[predictor_idx]

        if reference is None:
            coherence = float("nan")
            novelty = float("nan")
            new_reference = grad_unit.clone()
        else:
            reference_norm = torch.linalg.vector_norm(reference)
            if reference_norm.item() <= self.gradient_reference_eps:
                coherence = float("nan")
                novelty = float("nan")
                reference_unit = grad_unit
            else:
                reference_unit = reference / reference_norm
                coherence = float(
                    torch.clamp(
                        torch.dot(grad_unit, reference_unit),
                        min=-1.0,
                        max=1.0,
                    ).item()
                )
                novelty = 1.0 - coherence

            new_reference = (
                (1.0 - self.gradient_reference_beta) * reference_unit
                + self.gradient_reference_beta * grad_unit
            )

            new_reference_norm = torch.linalg.vector_norm(new_reference)
            if new_reference_norm.item() > self.gradient_reference_eps:
                new_reference = new_reference / new_reference_norm
            else:
                new_reference = grad_unit.clone()

        self.gradient_direction_refs[predictor_idx] = new_reference.detach()
        return coherence, novelty


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

        # -------------------------------------------------
        # Predictor input and target
        #
        # Predict local state change rather than the full next state:
        #
        #   x_t = [S_t, A_t]
        #   y_t = [Delta S_t, R_{t+1}]
        #   Delta S_t = S_{t+1} - S_t
        #
        # Normalization uses statistics from BEFORE this transition.
        # The current sample is inserted only after scoring/training.
        # -------------------------------------------------
        predictor_input_raw = torch.cat(
            (
                raw_obs,
                detector_action,
            ),
            dim=-1,
        )

        raw_state_delta = raw_next_obs - raw_obs

        detector_target_raw = torch.cat(
            (
                raw_state_delta,
                reward_tensor,
            ),
            dim=-1,
        )

        (
            predictor_input_mean,
            predictor_input_std,
        ) = self.predictor_input_normalizer.mean_std_tensors(
            device=self.device,
            dtype=predictor_input_raw.dtype,
        )

        (
            predictor_target_mean,
            predictor_target_std,
        ) = self.predictor_target_normalizer.mean_std_tensors(
            device=self.device,
            dtype=detector_target_raw.dtype,
        )

        predictor_input = (
            predictor_input_raw - predictor_input_mean
        ) / predictor_input_std

        detector_target = (
            detector_target_raw - predictor_target_mean
        ) / predictor_target_std

        predictor_obs = predictor_input[:, :self.cfg.obs_dim]
        predictor_action = predictor_input[:, self.cfg.obs_dim:]

        # =================================================
        # 1. Score the transition BEFORE training on it
        # =================================================

        predictor_outputs = []

        for predictor in self.predictors:
            mean_n, logvar_n = predictor(
                predictor_obs,
                predictor_action,
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

            # Moment-matched ensemble predictive distribution.
            mu_star = means.mean(dim=0)

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

            epistemic_var = (
                (
                    means
                    - mu_star.unsqueeze(0)
                ).pow(2)
            ).mean(dim=0)

            # Dimension-normalized Gaussian innovation surprise:
            #   s_t = mean_j (y_j - mu_j)^2 / var_j
            surprise_raw = (
                (
                    detector_target
                    - mu_star
                ).pow(2)
                / var_star
            ).mean().item()

            surprise_clipped = float(
                np.clip(
                    surprise_raw,
                    0.0,
                    self.surprise_cap,
                )
            )

            # Variances above are in normalized target coordinates.
            mean_total_var = var_star.mean().item()
            mean_epistemic_var = epistemic_var.mean().item()

            # Convert the ensemble prediction and uncertainty back to
            # physical/raw target coordinates for diagnostics.
            mu_star_raw = (
                mu_star * predictor_target_std
                + predictor_target_mean
            )

            target_scale_sq = predictor_target_std.pow(2)

            var_star_raw = (
                var_star * target_scale_sq
            )

            epistemic_var_raw = (
                epistemic_var * target_scale_sq
            )

            raw_residual = (
                detector_target_raw - mu_star_raw
            )

            raw_residual_mse = (
                raw_residual.pow(2).mean().item()
            )

            raw_delta_state_mse = (
                raw_residual[:, :self.cfg.obs_dim]
                .pow(2)
                .mean()
                .item()
            )

            raw_reward_sq_error = (
                raw_residual[:, self.cfg.obs_dim:]
                .pow(2)
                .mean()
                .item()
            )

            mean_total_var_raw = (
                var_star_raw.mean().item()
            )

            mean_epistemic_var_raw = (
                epistemic_var_raw.mean().item()
            )

        # =================================================
        # 1b. Predictor gradient coherence / novelty diagnostic
        #
        # Compute the gradient of each predictor's *unweighted* NLL with
        # respect to its mean head.  This is diagnostic-only: autograd.grad
        # does not populate .grad or modify optimizer state.  The same NLL
        # tensors are reused later for the actual Poisson-weighted update.
        # =================================================

        predictor_nll_tensors = []
        predictor_coherences = []
        predictor_novelties = []

        for n, (mean_n, logvar_n, var_n) in enumerate(predictor_outputs):
            log_prob_n = diagonal_gaussian_log_prob(
                detector_target,
                mean_n,
                var_n,
            )
            nll_n = -log_prob_n.mean()
            predictor_nll_tensors.append(nll_n)

            mean_head_params = tuple(
                self.predictors[n].mean_head.parameters()
            )
            mean_head_grads = torch.autograd.grad(
                nll_n,
                mean_head_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )

            grad_vector = torch.cat(
                [grad.detach().reshape(-1) for grad in mean_head_grads],
                dim=0,
            )

            coherence_n, novelty_n = self._gradient_direction_diagnostic(
                n,
                grad_vector,
            )

            if np.isfinite(coherence_n):
                predictor_coherences.append(coherence_n)
                predictor_novelties.append(novelty_n)

        if predictor_coherences:
            gradient_coherence = float(np.mean(predictor_coherences))
            gradient_novelty = float(np.mean(predictor_novelties))
        else:
            gradient_coherence = float("nan")
            gradient_novelty = float("nan")

        # =================================================
        # 2. Local conditional outcome score on matched (S,A)
        # =================================================

        state_action_np = np.concatenate(
            (
                raw_obs.detach().cpu().numpy().reshape(-1),
                detector_action.detach().cpu().numpy().reshape(-1),
            )
        ).astype(np.float64)

        detector_target_np = (
            detector_target.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )

        (
            state_action_pseudo_count,
            familiarity,
            familiarity_ready,
            local_conditional_surprise,
            local_matched,
            local_valid_tables,
        ) = self.state_action_familiarity.score_and_observe(
            state_action_np,
            detector_target_np,
        )

        if local_matched and np.isfinite(local_conditional_surprise):
            local_surprise_clipped = float(
                np.clip(
                    local_conditional_surprise,
                    0.0,
                    self.local_score_cap,
                )
            )
            # e_t now means detector evidence before subtracting the reference.
            e_t = local_surprise_clipped
        else:
            local_surprise_clipped = float("nan")
            e_t = 0.0


        # =================================================
        # 3. Block evidence and CUSUM
        # =================================================

        regime_change = False
        block_completed = False
        block_E_k = float("nan")
        W_for_log = self.change_score

        if self.warmup_remaining > 0:
            # Neural predictor and hash normalizer keep learning during warmup;
            # only detector accumulation is disabled.
            self.warmup_remaining -= 1

            self.block_evidence_sum = 0.0
            self.block_matched_count = 0
            self.block_valid_tables_sum = 0.0
            self.block_familiarity_sum = 0.0
            self.block_surprise_sum = 0.0
            self.block_gradient_coherence_sum = 0.0
            self.block_gradient_novelty_sum = 0.0
            self.block_gradient_count = 0
            self.block_count = 0

        elif familiarity_ready:
            # Every transition contributes to the block denominator.  Only
            # transitions with enough prior observations in enough LSH tables
            # contribute local conditional evidence.
            if local_matched and np.isfinite(local_surprise_clipped):
                self.block_evidence_sum += local_surprise_clipped
                self.block_matched_count += 1
                self.block_valid_tables_sum += float(local_valid_tables)

            self.block_familiarity_sum += familiarity
            self.block_surprise_sum += surprise_clipped

            if np.isfinite(gradient_coherence):
                self.block_gradient_coherence_sum += gradient_coherence
                self.block_gradient_novelty_sum += gradient_novelty
                self.block_gradient_count += 1

            self.block_count += 1

            if self.block_count >= self.block_size:
                block_completed = True

                matched_fraction = (
                    self.block_matched_count
                    / float(self.block_count)
                )

                if self.block_matched_count > 0:
                    local_surprise_mean = (
                        self.block_evidence_sum
                        / float(self.block_matched_count)
                    )
                    mean_valid_tables = (
                        self.block_valid_tables_sum
                        / float(self.block_matched_count)
                    )
                else:
                    local_surprise_mean = float("nan")
                    mean_valid_tables = 0.0

                # E_k is the block-average local score with unmatched
                # transitions contributing zero.  It is retained for compact
                # continuity with previous logs.
                block_E_k = (
                    self.block_evidence_sum
                    / float(self.block_count)
                )

                mean_block_familiarity = (
                    self.block_familiarity_sum
                    / float(self.block_count)
                )

                mean_block_surprise = (
                    self.block_surprise_sum
                    / float(self.block_count)
                )

                if self.block_gradient_count > 0:
                    mean_block_gradient_coherence = (
                        self.block_gradient_coherence_sum
                        / float(self.block_gradient_count)
                    )
                    mean_block_gradient_novelty = (
                        self.block_gradient_novelty_sum
                        / float(self.block_gradient_count)
                    )
                else:
                    mean_block_gradient_coherence = float("nan")
                    mean_block_gradient_novelty = float("nan")

                # Local conditional CUSUM.  The normalized innovation has a
                # nominal reference near one when bucket outcome statistics are
                # well estimated.  Reliability is the fraction of transitions
                # in the block that had enough matched local history.
                block_change_evidence = 0.0
                if (
                    matched_fraction >= self.local_min_block_fraction
                    and np.isfinite(local_surprise_mean)
                ):
                    block_change_evidence = (
                        matched_fraction
                        * (
                            local_surprise_mean
                            - self.local_reference
                            - self.local_margin
                        )
                    )

                    self.change_score = max(
                        0.0,
                        self.change_score + block_change_evidence,
                    )
                else:
                    # Do not carry stale positive evidence through long periods
                    # with insufficient matched state-action support.
                    self.change_score = 0.0

                W_for_log = self.change_score

                if self.change_score > self.detector_h:
                    regime_change = True

                with open(
                    self.detector_block_path,
                    "a",
                ) as f:
                    f.write(
                        f"{self.block_index},"
                        f"{self.steps},"
                        f"{block_E_k:.8f},"
                        f"{self.change_score:.8f},"
                        f"{mean_block_familiarity:.8f},"
                        f"{mean_block_surprise:.8f},"
                        f"{mean_block_gradient_coherence:.8f},"
                        f"{mean_block_gradient_novelty:.8f},"
                        f"{int(regime_change)},"
                        f"{local_surprise_mean:.8f},"
                        f"{matched_fraction:.8f},"
                        f"{mean_valid_tables:.8f},"
                        f"{block_change_evidence:.8f}\n"
                    )

                self.block_index += 1
                self.block_evidence_sum = 0.0
                self.block_matched_count = 0
                self.block_valid_tables_sum = 0.0
                self.block_familiarity_sum = 0.0
                self.block_surprise_sum = 0.0
                self.block_gradient_coherence_sum = 0.0
                self.block_gradient_novelty_sum = 0.0
                self.block_gradient_count = 0
                self.block_count = 0

                if regime_change:
                    with open(
                        self.regime_log_path,
                        "a",
                    ) as f:
                        f.write(
                            f"step={self.steps}, "
                            f"E_k={block_E_k:.8f}, "
                            f"W_k={self.change_score:.8f}, "
                            f"mean_familiarity={mean_block_familiarity:.8f}, "
                            f"mean_predictor_surprise={mean_block_surprise:.8f}, "
                            f"local_surprise_mean={local_surprise_mean:.8f}, "
                            f"matched_fraction={matched_fraction:.8f}, "
                            f"mean_valid_tables={mean_valid_tables:.8f}, "
                            f"mean_gradient_coherence="
                            f"{mean_block_gradient_coherence:.8f}, "
                            f"mean_gradient_novelty="
                            f"{mean_block_gradient_novelty:.8f}\n"
                        )

                    # Keep the existing regime-local neural predictor reset for
                    # diagnostics / downstream use.  It no longer participates
                    # in the detector statistic.
                    self._reset_predictor_ensemble()

                    # Start fresh current-regime local conditional statistics
                    # while preserving the fixed LSH mapping/normalization.
                    self.state_action_familiarity.reset_counts()

                    # Insert the alarm transition once as the first sample of
                    # the newly detected regime.  Its score is intentionally
                    # ignored because it triggered the boundary.
                    if self.state_action_familiarity.ready:
                        self.state_action_familiarity.score_and_observe(
                            state_action_np,
                            detector_target_np,
                        )

                    self.change_score = 0.0

                    # No fixed post-change warmup.  The local detector becomes
                    # eligible naturally when matched buckets accumulate enough
                    # samples in the new regime.
                    self.warmup_remaining = 0

        # =================================================
        # 4. Online Poisson bootstrap predictor update
        #
        # Always learn after scoring the current transition.
        # If an alarm occurred above, the predictor ensemble was replaced,
        # so rebuild the forward pass/NLL tensors for the fresh predictor
        # before training it on this transition.
        # =================================================

        if regime_change:
            predictor_outputs = []
            predictor_nll_tensors = []

            for predictor in self.predictors:
                mean_n, logvar_n = predictor(
                    predictor_obs,
                    predictor_action,
                )
                var_n = torch.exp(logvar_n)

                predictor_outputs.append(
                    (
                        mean_n,
                        logvar_n,
                        var_n,
                    )
                )

                log_prob_n = diagonal_gaussian_log_prob(
                    detector_target,
                    mean_n,
                    var_n,
                )
                predictor_nll_tensors.append(
                    -log_prob_n.mean()
                )

        # Temporary post-change predictor plasticity.
        # if self.predictor_adapt_remaining > 0:
        #     progress = (
        #         self.predictor_adapt_remaining
        #         / float(self.cfg.predictor_adapt_steps)
        #     )

        #     predictor_lr = (
        #         self.cfg.predictor_lr
        #         + (
        #             self.cfg.predictor_fast_lr
        #             - self.cfg.predictor_lr
        #         )
        #         * progress
        #     )

        #     self.predictor_adapt_remaining -= 1

        # else:
        #     predictor_lr = self.cfg.predictor_lr

        # for opt in self.pred_opts:
        #     for group in opt.param_groups:
        #         group["lr"] = predictor_lr


        predictor_nll_values = []

        for n, (
            mean_n,
            logvar_n,
            var_n,
        ) in enumerate(predictor_outputs):

            k_n_t = np.random.poisson(1.0)

            if k_n_t == 0:
                continue

            nll_n = predictor_nll_tensors[n]

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
        # 4b. Update predictor normalization statistics
        #
        # Do this only after predict -> score -> predictor update so
        # the current outcome cannot influence its own normalization.
        # =================================================

        self.predictor_input_normalizer.observe(
            predictor_input_raw.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        self.predictor_target_normalizer.observe(
            detector_target_raw.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        # =================================================
        # 5. Detector logging
        # =================================================

        if self.steps % self.detector_log_interval == 0:

            with open(
                self.detector_trace_path,
                "a",
            ) as f:

                f.write(
                    f"{self.steps},"
                    f"{surprise_raw:.8f},"
                    f"{surprise_clipped:.8f},"
                    f"{state_action_pseudo_count},"
                    f"{familiarity:.8f},"
                    f"{e_t:.8f},"
                    f"{local_conditional_surprise:.8f},"
                    f"{int(local_matched)},"
                    f"{local_valid_tables},"
                    f"{block_E_k:.8f},"
                    f"{W_for_log:.8f},"
                    f"{int(block_completed)},"
                    f"{self.warmup_remaining},"
                    f"{int(regime_change)},"
                    f"{mean_total_var:.8f},"
                    f"{mean_epistemic_var:.8f},"
                    f"{raw_residual_mse:.8f},"
                    f"{raw_delta_state_mse:.8f},"
                    f"{raw_reward_sq_error:.8f},"
                    f"{mean_total_var_raw:.8f},"
                    f"{mean_epistemic_var_raw:.8f},"
                    f"{gradient_coherence:.8f},"
                    f"{gradient_novelty:.8f},"
                    f"{predictor_nll:.8f}\n"
                )

        # =================================================
        # 6. Original AVG update
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
            "surprise_raw": surprise_raw,
            "surprise_clipped": surprise_clipped,
            "state_action_pseudo_count": state_action_pseudo_count,
            "familiarity": familiarity,
            "e_t": e_t,
            "local_conditional_surprise": local_conditional_surprise,
            "local_matched": local_matched,
            "local_valid_tables": local_valid_tables,
            "E_k": block_E_k,
            "W_k": W_for_log,
            "block_completed": block_completed,
            "warmup_remaining": self.warmup_remaining,
            "regime_change": regime_change,
            "mean_total_var": mean_total_var,
            "mean_epistemic_var": mean_epistemic_var,
            "raw_residual_mse": raw_residual_mse,
            "raw_delta_state_mse": raw_delta_state_mse,
            "raw_reward_sq_error": raw_reward_sq_error,
            "mean_total_var_raw": mean_total_var_raw,
            "mean_epistemic_var_raw": mean_epistemic_var_raw,
            "gradient_coherence": gradient_coherence,
            "gradient_novelty": gradient_novelty,
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

            "detector": {
                "change_score": self.change_score,
                "warmup_remaining": self.warmup_remaining,
                "block_evidence_sum": self.block_evidence_sum,
                "block_matched_count": self.block_matched_count,
                "block_valid_tables_sum": self.block_valid_tables_sum,
                "block_familiarity_sum": self.block_familiarity_sum,
                "block_surprise_sum": self.block_surprise_sum,
                "block_gradient_coherence_sum": (
                    self.block_gradient_coherence_sum
                ),
                "block_gradient_novelty_sum": (
                    self.block_gradient_novelty_sum
                ),
                "block_gradient_count": self.block_gradient_count,
                "gradient_direction_refs": [
                    None if ref is None else ref.detach().cpu()
                    for ref in self.gradient_direction_refs
                ],
                "predictor_reset_count": self.predictor_reset_count,
                "predictor_input_normalizer": (
                    self.predictor_input_normalizer.state_dict()
                ),
                "predictor_target_normalizer": (
                    self.predictor_target_normalizer.state_dict()
                ),
                "block_count": self.block_count,
                "block_index": self.block_index,
                "state_action_familiarity": (
                    self.state_action_familiarity.state_dict()
                ),
            },
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
        f"_pred-{args.predictor_num_layers}x{args.nhid_predictor}"
        f"_predreset-delta-norm-localcond"
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
        default=256,
        type=int,
        help=(
            "Hidden width of each environment predictor. "
            "Baseline implementation used 128; this capacity test uses 256."
        ),
    )

    parser.add_argument(
        "--predictor_num_layers",
        default=4,
        type=int,
        help=(
            "Number of hidden layers in each environment predictor. "
            "Baseline implementation used 2; this capacity test uses 4."
        ),
    )

    parser.add_argument(
        "--predictor_lr",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--predictor_norm_warmup",
        default=10_000,
        type=int,
        help=(
            "Samples used to estimate then freeze predictor input/target "
            "normalization statistics"
        ),
    )

    parser.add_argument(
        "--predictor_fast_lr",
        default=3e-4,
        type=float,
        help="Temporary predictor LR immediately after detected regime change",
    )

    parser.add_argument(
        "--predictor_adapt_steps",
        default=5_000,
        type=int,
        help="Steps over which predictor LR decays back to predictor_lr",
    )

    parser.add_argument(
        "--num_predictors",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--gradient_reference_beta",
        default=0.001,
        type=float,
        help=(
            "EMA rate for each predictor's mean-head gradient direction "
            "reference; diagnostic only"
        ),
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

    # =====================================================
    # Local conditional detector parameters
    # =====================================================

    parser.add_argument(
        "--detector_h",
        default=5.0,
        type=float,
        help=(
            "CUSUM alarm threshold applied to local conditional block evidence"
        ),
    )

    parser.add_argument(
        "--local_reference",
        default=1.0,
        type=float,
        help="Nominal mean normalized innovation for matched local outcomes",
    )

    parser.add_argument(
        "--local_margin",
        default=0.5,
        type=float,
        help="One-sided CUSUM dead-zone above the local reference",
    )

    parser.add_argument(
        "--local_score_cap",
        default=20.0,
        type=float,
        help="Clip per-transition local conditional surprise before blocking",
    )

    parser.add_argument(
        "--local_min_count",
        default=20,
        type=int,
        help="Prior samples required in an LSH bucket before it can score",
    )

    parser.add_argument(
        "--local_min_valid_tables",
        default=2,
        type=int,
        help="Minimum LSH tables with enough local history to score a transition",
    )

    parser.add_argument(
        "--local_variance_floor",
        default=0.01,
        type=float,
        help=(
            "Per-dimension variance floor in normalized outcome coordinates"
        ),
    )

    parser.add_argument(
        "--local_min_block_fraction",
        default=0.25,
        type=float,
        help=(
            "Minimum fraction of transitions in a block with valid local matches"
        ),
    )

    # Legacy z-CUSUM arguments are retained for CLI compatibility but are no
    # longer used by the local conditional detector.
    parser.add_argument(
        "--baseline_alpha",
        default=0.01,
        type=float,
        help="Legacy/unused: retained for CLI compatibility",
    )

    parser.add_argument(
        "--baseline_margin",
        default=0.5,
        type=float,
        help="Legacy/unused: retained for CLI compatibility",
    )

    parser.add_argument(
        "--baseline_min_weight",
        default=0.5,
        type=float,
        help="Legacy/unused: retained for CLI compatibility",
    )

    parser.add_argument(
        "--baseline_init_blocks",
        default=20,
        type=int,
        help="Legacy/unused: retained for CLI compatibility",
    )

    parser.add_argument(
        "--baseline_freeze_score",
        default=4,
        type=float,
        help="Legacy/unused: retained for CLI compatibility",
    )

    parser.add_argument(
        "--surprise_cap",
        default=20.0,
        type=float,
        help="Clip transition surprise before familiarity weighting",
    )

    parser.add_argument(
        "--familiarity_gamma",
        default=4.0,
        type=float,
        help="Legacy/unused by local conditional detector",
    )

    parser.add_argument(
        "--block_size",
        default=100,
        type=int,
        help="Transitions per block E_k",
    )

    parser.add_argument(
        "--initial_detector_warmup",
        default=10_000,
        type=int,
        help="Initial steps with predictor learning but no CUSUM accumulation",
    )

    parser.add_argument(
        "--detector_log_interval",
        default=100,
        type=int,
        help="Write transition-level detector trace every N environment steps",
    )

    parser.add_argument(
        "--familiarity_norm_warmup",
        default=10_000,
        type=int,
        help="Samples used to estimate/freeze state-action hash normalization",
    )

    parser.add_argument(
        "--familiarity_num_tables",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--familiarity_projections_per_table",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--familiarity_num_buckets",
        default=65536,
        type=int,
    )

    parser.add_argument(
        "--familiarity_hash_width",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--familiarity_count_scale",
        default=10.0,
        type=float,
    )

    parser.add_argument(
        "--familiarity_input_clip",
        default=5.0,
        type=float,
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
            f"_aba_joint_local_conditional_diagnostics"
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