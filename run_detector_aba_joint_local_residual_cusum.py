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


# =========================================================
# Fixed-memory local residual calibration
# =========================================================

class LocalResidualHashTracker:
    """
    Fixed-memory LSH statistics over x_t = [S_t, A_t].

    For each hash table and bucket, store only
        n_h, m_h, M2_h
    for the scalar normalized prediction surprise ell_t.

    The input normalization is estimated during predictor warmup and then
    frozen. Residual statistics are collected only during the separate
    residual-calibration phase, after the reference predictor is frozen.
    """

    def __init__(
        self,
        dim,
        seed,
        num_tables=4,
        projections_per_table=4,
        num_buckets=4096,
        hash_width=1.0,
        input_clip=5.0,
        residual_std_floor=0.1,
    ):
        self.dim = int(dim)
        self.num_tables = int(num_tables)
        self.projections_per_table = int(projections_per_table)
        self.num_buckets = int(num_buckets)
        self.hash_width = float(hash_width)
        self.input_clip = float(input_clip)
        self.residual_std_floor = float(residual_std_floor)

        if self.dim <= 0:
            raise ValueError("Require dim > 0")
        if self.num_tables <= 0:
            raise ValueError("Require num_tables > 0")
        if self.projections_per_table <= 0:
            raise ValueError("Require projections_per_table > 0")
        if self.num_buckets <= 0:
            raise ValueError("Require num_buckets > 0")
        if self.hash_width <= 0.0:
            raise ValueError("Require hash_width > 0")
        if self.input_clip <= 0.0:
            raise ValueError("Require input_clip > 0")
        if self.residual_std_floor <= 0.0:
            raise ValueError("Require residual_std_floor > 0")

        rng = np.random.RandomState(int(seed))

        self.projections = rng.normal(
            size=(
                self.num_tables,
                self.projections_per_table,
                self.dim,
            )
        ).astype(np.float64)

        projection_norms = np.linalg.norm(
            self.projections,
            axis=-1,
            keepdims=True,
        )
        projection_norms = np.maximum(projection_norms, 1e-12)
        self.projections /= projection_norms

        self.offsets = rng.uniform(
            low=0.0,
            high=self.hash_width,
            size=(
                self.num_tables,
                self.projections_per_table,
            ),
        ).astype(np.float64)

        self.hash_coeffs = rng.randint(
            low=1,
            high=2_147_483_647,
            size=(
                self.num_tables,
                self.projections_per_table,
            ),
        ).astype(np.int64)

        # Input normalization, accumulated only during predictor warmup.
        self.norm_count = 0
        self.norm_mean = np.zeros(self.dim, dtype=np.float64)
        self.norm_M2 = np.zeros(self.dim, dtype=np.float64)
        self.frozen_mean = np.zeros(self.dim, dtype=np.float64)
        self.frozen_std = np.ones(self.dim, dtype=np.float64)
        self.normalizer_frozen = False

        # Fixed-size per-bucket residual statistics.
        self.counts = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.uint32,
        )
        self.means = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.float64,
        )
        self.M2 = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.float64,
        )

    def update_input_normalizer(self, x):
        if self.normalizer_frozen:
            return

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, got {x.shape[0]}"
            )

        self.norm_count += 1
        delta = x - self.norm_mean
        self.norm_mean += delta / self.norm_count
        delta2 = x - self.norm_mean
        self.norm_M2 += delta * delta2

    def freeze_input_normalizer(self):
        if self.normalizer_frozen:
            return

        if self.norm_count >= 2:
            variance = self.norm_M2 / (self.norm_count - 1)
            self.frozen_std = np.sqrt(np.maximum(variance, 1e-8))
            self.frozen_mean = self.norm_mean.copy()
        elif self.norm_count == 1:
            self.frozen_mean = self.norm_mean.copy()
            self.frozen_std = np.ones(self.dim, dtype=np.float64)
        else:
            self.frozen_mean = np.zeros(self.dim, dtype=np.float64)
            self.frozen_std = np.ones(self.dim, dtype=np.float64)

        self.normalizer_frozen = True

    def _standardize(self, x):
        if not self.normalizer_frozen:
            raise RuntimeError(
                "Hash input normalizer must be frozen before hashing"
            )

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        z = (x - self.frozen_mean) / self.frozen_std
        return np.clip(z, -self.input_clip, self.input_clip)

    def bucket_indices(self, x):
        z = self._standardize(x)
        buckets = []

        for table_idx in range(self.num_tables):
            projected = self.projections[table_idx] @ z
            cells = np.floor(
                (
                    projected
                    + self.offsets[table_idx]
                )
                / self.hash_width
            ).astype(np.int64)

            mixed = int(
                np.dot(
                    cells,
                    self.hash_coeffs[table_idx],
                )
            )
            buckets.append(mixed % self.num_buckets)

        return tuple(buckets)

    def update_residual(self, x, residual_surprise):
        """Update n_h, m_h, M2_h for each table bucket."""
        ell = float(residual_surprise)
        buckets = self.bucket_indices(x)
        max_uint32 = np.iinfo(np.uint32).max

        for table_idx, bucket in enumerate(buckets):
            n_old = int(self.counts[table_idx, bucket])
            n_new = n_old + 1

            mean_old = self.means[table_idx, bucket]
            delta = ell - mean_old
            mean_new = mean_old + delta / n_new
            delta2 = ell - mean_new

            self.means[table_idx, bucket] = mean_new
            self.M2[table_idx, bucket] += delta * delta2

            if n_old < max_uint32:
                self.counts[table_idx, bucket] = n_new

        return buckets

    def query_residual(self, x, residual_surprise, min_bucket_count):
        """
        Return local z-scores from sufficiently calibrated hash buckets.

        Each returned item is
            (table_idx, bucket_idx, bucket_count, z_score).
        """
        ell = float(residual_surprise)
        min_bucket_count = int(min_bucket_count)
        buckets = self.bucket_indices(x)
        supported = []

        for table_idx, bucket in enumerate(buckets):
            count = int(self.counts[table_idx, bucket])
            if count < min_bucket_count:
                continue

            mean = float(self.means[table_idx, bucket])

            if count >= 2:
                variance = float(
                    self.M2[table_idx, bucket]
                    / (count - 1)
                )
            else:
                variance = 0.0

            std = max(
                variance ** 0.5,
                self.residual_std_floor,
            )

            z_score = (ell - mean) / std
            supported.append(
                (
                    table_idx,
                    bucket,
                    count,
                    float(z_score),
                )
            )

        return buckets, supported

    def state_dict(self):
        return {
            "norm_count": self.norm_count,
            "norm_mean": self.norm_mean,
            "norm_M2": self.norm_M2,
            "frozen_mean": self.frozen_mean,
            "frozen_std": self.frozen_std,
            "normalizer_frozen": self.normalizer_frozen,
            "counts": self.counts,
            "means": self.means,
            "M2": self.M2,
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
# p_n(S_{t+1}, R_{t+1} | S_t, A_t)
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
        self.out_dim = obs_dim + 1  # next observation + reward

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
            f"{cfg.run_id}_regime_changes_aba_joint_local_residual.log",
        )

        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_local_residual_trace_aba_joint.csv",
        )

        self.detector_block_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_local_residual_blocks_aba_joint.csv",
        )

        with open(self.regime_log_path, "w") as f:
            f.write(
                "step,detector_epoch,block_evidence,block_coverage,"
                "num_regions,W_t,transition_surprise,z_score,"
                "supported_tables\n"
            )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,detector_epoch,phase,transition_surprise,z_score,"
                "transition_evidence,valid_local_residual,supported_tables,"
                "median_bucket_count,block_completed,block_evidence,"
                "block_coverage,num_regions,W_t,log_p_hat,"
                "mean_total_var,mean_epistemic_var,"
                "predictor_warmup_remaining,residual_calibration_remaining,"
                "regime_change,predictor_nll\n"
            )

        with open(self.detector_block_path, "w") as f:
            f.write(
                "step,detector_epoch,block_evidence,block_coverage,"
                "num_regions,W_t,regime_change\n"
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
        # Detector configuration
        # -------------------------------------------------

        self.num_predictors = cfg.num_predictors
        self.predictor_warmup_steps = cfg.predictor_warmup_steps
        self.residual_calibration_steps = cfg.residual_calibration_steps
        self.min_bucket_count = cfg.min_bucket_count
        self.min_supported_tables = cfg.min_supported_tables
        self.evidence_z_threshold = cfg.evidence_z_threshold
        self.evidence_clip = cfg.evidence_clip
        self.block_size = cfg.block_size
        self.min_block_coverage = cfg.min_block_coverage
        self.cusum_drift = cfg.cusum_drift
        self.detector_h = cfg.detector_h
        self.detector_log_interval = cfg.detector_log_interval

        if self.predictor_warmup_steps <= 0:
            raise ValueError("Require predictor_warmup_steps > 0")
        if self.residual_calibration_steps <= 0:
            raise ValueError("Require residual_calibration_steps > 0")
        if self.min_bucket_count < 2:
            raise ValueError("Require min_bucket_count >= 2")
        if not (1 <= self.min_supported_tables <= cfg.hash_num_tables):
            raise ValueError(
                "Require 1 <= min_supported_tables <= hash_num_tables"
            )
        if self.evidence_z_threshold < 0.0:
            raise ValueError("Require evidence_z_threshold >= 0")
        if self.evidence_clip <= 0.0:
            raise ValueError("Require evidence_clip > 0")
        if self.block_size <= 0:
            raise ValueError("Require block_size > 0")
        if not (0.0 <= self.min_block_coverage <= 1.0):
            raise ValueError("Require 0 <= min_block_coverage <= 1")
        if self.cusum_drift < 0.0:
            raise ValueError("Require cusum_drift >= 0")
        if self.detector_h <= 0.0:
            raise ValueError("Require detector_h > 0")

        # Epoch 0 is the initial detector. Every trusted alarm creates a
        # completely fresh detector epoch without touching actor/critic state.
        self.detector_epoch = 0

        # Detector-local bootstrap RNG keeps predictor bootstrap draws
        # independent of all other NumPy use.
        self.predictor_bootstrap_rng = np.random.RandomState(
            int(cfg.seed)
        )

        self.predictors = self._make_predictor_ensemble(
            init_seed=None,
        )
        self.pred_opts = self._make_predictor_optimizers()
        self.local_residual = self._make_local_residual_tracker()

        # Detector epoch lifecycle:
        #   predictor_warmup -> residual_calibration -> monitoring
        #
        # Predictor warmup:
        #   train the Poisson-bootstrap ensemble and estimate/freeze the
        #   state-action hash normalization.
        # Residual calibration:
        #   freeze the predictor; estimate n_h, m_h, M2_h of prediction
        #   surprise for each state-action hash bucket.
        # Monitoring:
        #   freeze both predictor and residual statistics; accumulate only
        #   conditional residual violations into a blockwise CUSUM.
        self.detector_phase = "predictor_warmup"
        self.predictor_warmup_remaining = self.predictor_warmup_steps
        self.residual_calibration_remaining = self.residual_calibration_steps

        # W_t (updated at block boundaries while monitoring).
        self.change_score = 0.0
        self._reset_monitoring_block()

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
    # Detector helpers
    # =====================================================

    def _make_predictor_ensemble(self, init_seed=None):
        def build():
            return nn.ModuleList(
                [
                    Predictor(
                        obs_dim=self.cfg.obs_dim,
                        action_dim=self.cfg.action_dim,
                        device=self.cfg.device,
                        n_hid=self.cfg.nhid_predictor,
                        logvar_min=self.cfg.pred_logvar_min,
                        logvar_max=self.cfg.pred_logvar_max,
                    )
                    for _ in range(self.cfg.num_predictors)
                ]
            )

        if init_seed is None:
            return build()

        # Predictor initialization must not perturb the actor's global
        # PyTorch RNG stream after an alarm.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(int(init_seed))
            return build()

    def _make_predictor_optimizers(self):
        return [
            torch.optim.Adam(
                predictor.parameters(),
                lr=self.cfg.predictor_lr,
            )
            for predictor in self.predictors
        ]

    def _make_local_residual_tracker(self):
        tracker_seed = (
            int(self.cfg.seed)
            + 7919
            + 1_000_003 * self.detector_epoch
        )

        return LocalResidualHashTracker(
            dim=(
                self.cfg.obs_dim
                + self.cfg.action_dim
            ),
            seed=tracker_seed,
            num_tables=self.cfg.hash_num_tables,
            projections_per_table=(
                self.cfg.hash_projections_per_table
            ),
            num_buckets=self.cfg.hash_num_buckets,
            hash_width=self.cfg.hash_width,
            input_clip=self.cfg.hash_input_clip,
            residual_std_floor=self.cfg.residual_std_floor,
        )

    def _reset_monitoring_block(self):
        self.block_steps = 0
        self.block_valid_samples = 0

        # For each hash table, maintain per-bucket evidence sums/counts for
        # only the current fixed-size block. Equal averaging over buckets
        # reduces sensitivity to a policy merely changing visitation counts.
        num_tables = self.cfg.hash_num_tables
        self.block_region_evidence_sum = [
            {} for _ in range(num_tables)
        ]
        self.block_region_evidence_count = [
            {} for _ in range(num_tables)
        ]

    def _hard_reset_detector(self):
        """
        Trust the alarm and start a fresh detector epoch.

        Reset every detector-specific component:
          - predictor weights,
          - predictor Adam states,
          - Poisson bootstrap RNG,
          - state-action hash projections and normalization,
          - local residual n_h/m_h/M2_h statistics,
          - calibration phases,
          - block statistics,
          - CUSUM W_t.

        Actor, critic, their optimizers, TD-error scaler, and environment
        observation normalization are deliberately untouched.
        """
        self.detector_epoch += 1

        predictor_seed = (
            int(self.cfg.seed)
            + 1_000_003 * self.detector_epoch
        )

        self.predictors = self._make_predictor_ensemble(
            init_seed=predictor_seed,
        )
        self.pred_opts = self._make_predictor_optimizers()

        bootstrap_seed = (
            int(self.cfg.seed)
            + self.detector_epoch
        )
        self.predictor_bootstrap_rng = np.random.RandomState(
            bootstrap_seed
        )

        self.local_residual = self._make_local_residual_tracker()

        self.detector_phase = "predictor_warmup"
        self.predictor_warmup_remaining = self.predictor_warmup_steps
        self.residual_calibration_remaining = self.residual_calibration_steps
        self.change_score = 0.0
        self._reset_monitoring_block()

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

        # Detector receives raw flattened observations.
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

        # x_t = [S_t, A_t] is used only for local residual conditioning.
        state_action_np = torch.cat(
            (
                raw_obs.detach(),
                detector_action.detach(),
            ),
            dim=-1,
        ).squeeze(0).cpu().numpy()

        # y_t = [S_{t+1}, R_{t+1}] is the predictor target.
        detector_target = torch.cat(
            (
                raw_next_obs,
                reward_tensor,
            ),
            dim=-1,
        )

        epoch_for_log = self.detector_epoch
        phase_for_log = self.detector_phase
        predictor_update = (
            self.detector_phase == "predictor_warmup"
        )

        # =================================================
        # 1. Score transition under the CURRENT reference ensemble
        # =================================================

        predictor_outputs = []

        if predictor_update:
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
        else:
            with torch.no_grad():
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

            # Moment-matched ensemble prediction.
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

            log_p_hat = diagonal_gaussian_log_prob(
                detector_target,
                mu_star,
                var_star,
            )

            # Conditional prediction surprise:
            #
            #   ell_t = (1/d_y) sum_j
            #           (y_tj - mu_j(x_t))^2 / sigma_j^2(x_t)
            #
            # This is the quantity calibrated locally by state-action
            # hash region. It does NOT compare two learning speeds.
            transition_surprise = (
                (
                    (detector_target - mu_star).pow(2)
                    / var_star
                ).mean(dim=-1)
            ).item()

            log_p_hat_value = log_p_hat.item()
            mean_total_var = var_star.mean().item()
            mean_epistemic_var = epistemic_var.mean().item()

        # Defaults used when the detector is not yet monitoring.
        z_score_for_log = float("nan")
        transition_evidence_for_log = float("nan")
        valid_local_residual = False
        supported_tables = 0
        median_bucket_count = float("nan")
        block_completed = False
        block_evidence = float("nan")
        block_coverage = float("nan")
        num_regions = 0
        regime_change = False
        W_for_log = float(self.change_score)

        # =================================================
        # 2. Detector epoch lifecycle
        # =================================================

        if self.detector_phase == "predictor_warmup":
            # Learn a reference predictor and, in parallel, estimate the
            # normalization used ONLY by the fixed state-action hash.
            self.local_residual.update_input_normalizer(
                state_action_np
            )

        elif self.detector_phase == "residual_calibration":
            # Predictor is already frozen. Calibrate the normal conditional
            # prediction-surprise distribution for each hash bucket.
            self.local_residual.update_residual(
                state_action_np,
                transition_surprise,
            )

        elif self.detector_phase == "monitoring":
            # Predictor, hash normalization, and local residual statistics
            # are all frozen here. Policy learning may continue, but merely
            # visiting a bucket more often does not change its baseline.
            (
                _buckets,
                supported,
            ) = self.local_residual.query_residual(
                state_action_np,
                transition_surprise,
                min_bucket_count=self.min_bucket_count,
            )

            supported_tables = len(supported)

            if supported:
                median_bucket_count = float(
                    np.median(
                        [
                            item[2]
                            for item in supported
                        ]
                    )
                )

            if supported_tables >= self.min_supported_tables:
                valid_local_residual = True
                self.block_valid_samples += 1

                z_values = [
                    item[3]
                    for item in supported
                ]
                z_score_for_log = float(
                    np.median(z_values)
                )

                evidence_values = []

                for (
                    table_idx,
                    bucket,
                    _count,
                    local_z,
                ) in supported:
                    evidence = float(
                        np.clip(
                            local_z
                            - self.evidence_z_threshold,
                            0.0,
                            self.evidence_clip,
                        )
                    )
                    evidence_values.append(evidence)

                    old_sum = self.block_region_evidence_sum[
                        table_idx
                    ].get(bucket, 0.0)
                    old_count = self.block_region_evidence_count[
                        table_idx
                    ].get(bucket, 0)

                    self.block_region_evidence_sum[
                        table_idx
                    ][bucket] = old_sum + evidence
                    self.block_region_evidence_count[
                        table_idx
                    ][bucket] = old_count + 1

                transition_evidence_for_log = float(
                    np.median(evidence_values)
                )

            self.block_steps += 1

            if self.block_steps >= self.block_size:
                block_completed = True
                block_coverage = (
                    self.block_valid_samples
                    / float(self.block_size)
                )

                table_scores = []
                num_regions = 0

                for table_idx in range(
                    self.cfg.hash_num_tables
                ):
                    bucket_sums = self.block_region_evidence_sum[
                        table_idx
                    ]
                    bucket_counts = self.block_region_evidence_count[
                        table_idx
                    ]

                    region_means = []
                    for bucket, evidence_sum in bucket_sums.items():
                        count = bucket_counts[bucket]
                        if count > 0:
                            region_means.append(
                                evidence_sum / count
                            )

                    num_regions += len(region_means)

                    if region_means:
                        # Equal weight per visited bucket within this table.
                        table_scores.append(
                            float(np.mean(region_means))
                        )

                if (
                    block_coverage >= self.min_block_coverage
                    and table_scores
                ):
                    # Median across independent hash tables limits the effect
                    # of a single unfortunate hash collision.
                    block_evidence = float(
                        np.median(table_scores)
                    )

                    self.change_score = max(
                        0.0,
                        self.change_score
                        + block_evidence
                        - self.cusum_drift,
                    )

                W_for_log = float(self.change_score)

                if self.change_score > self.detector_h:
                    regime_change = True

                    with open(
                        self.regime_log_path,
                        "a",
                    ) as f:
                        f.write(
                            f"{self.steps},"
                            f"{epoch_for_log},"
                            f"{block_evidence:.8f},"
                            f"{block_coverage:.8f},"
                            f"{num_regions},"
                            f"{W_for_log:.8f},"
                            f"{transition_surprise:.8f},"
                            f"{z_score_for_log:.8f},"
                            f"{supported_tables}\n"
                        )

                with open(
                    self.detector_block_path,
                    "a",
                ) as f:
                    f.write(
                        f"{self.steps},"
                        f"{epoch_for_log},"
                        f"{block_evidence:.8f},"
                        f"{block_coverage:.8f},"
                        f"{num_regions},"
                        f"{W_for_log:.8f},"
                        f"{int(regime_change)}\n"
                    )

                # Start a fresh fixed-size monitoring block. If an alarm is
                # trusted below, this block reset is superseded by the full
                # detector reset.
                self._reset_monitoring_block()

        else:
            raise RuntimeError(
                f"Unknown detector phase: {self.detector_phase}"
            )

        # =================================================
        # 3. Online Poisson-bootstrap predictor update
        #    ONLY during predictor warmup
        # =================================================

        predictor_nll_values = []

        if predictor_update:
            for n, (
                mean_n,
                _logvar_n,
                var_n,
            ) in enumerate(predictor_outputs):
                # One transition, at most one optimizer step per ensemble
                # member. k=0 skips; k>0 weights the Gaussian NLL.
                k_n_t = self.predictor_bootstrap_rng.poisson(
                    1.0
                )

                if k_n_t == 0:
                    continue

                log_prob_n = diagonal_gaussian_log_prob(
                    detector_target,
                    mean_n,
                    var_n,
                )
                nll_n = -log_prob_n.mean()
                loss_n = float(k_n_t) * nll_n

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

        # Phase transitions happen AFTER the current transition has been
        # used by the phase it entered with.
        if phase_for_log == "predictor_warmup":
            self.predictor_warmup_remaining -= 1

            if self.predictor_warmup_remaining == 0:
                self.local_residual.freeze_input_normalizer()
                self.detector_phase = "residual_calibration"

        elif phase_for_log == "residual_calibration":
            self.residual_calibration_remaining -= 1

            if self.residual_calibration_remaining == 0:
                self.detector_phase = "monitoring"
                self.change_score = 0.0
                self._reset_monitoring_block()

        # =================================================
        # 4. Detector logging + trusted hard reset
        # =================================================

        predictor_warmup_remaining_for_log = (
            self.predictor_warmup_remaining
            if phase_for_log == "predictor_warmup"
            else 0
        )
        residual_calibration_remaining_for_log = (
            self.residual_calibration_remaining
            if phase_for_log == "residual_calibration"
            else 0
        )

        if self.steps % self.detector_log_interval == 0:
            with open(
                self.detector_trace_path,
                "a",
            ) as f:
                f.write(
                    f"{self.steps},"
                    f"{epoch_for_log},"
                    f"{phase_for_log},"
                    f"{transition_surprise:.8f},"
                    f"{z_score_for_log:.8f},"
                    f"{transition_evidence_for_log:.8f},"
                    f"{int(valid_local_residual)},"
                    f"{supported_tables},"
                    f"{median_bucket_count:.8f},"
                    f"{int(block_completed)},"
                    f"{block_evidence:.8f},"
                    f"{block_coverage:.8f},"
                    f"{num_regions},"
                    f"{W_for_log:.8f},"
                    f"{log_p_hat_value:.8f},"
                    f"{mean_total_var:.8f},"
                    f"{mean_epistemic_var:.8f},"
                    f"{predictor_warmup_remaining_for_log},"
                    f"{residual_calibration_remaining_for_log},"
                    f"{int(regime_change)},"
                    f"{predictor_nll:.8f}\n"
                )

        if regime_change:
            # The alarm is trusted. The detector has no persistent regime
            # memory: start a fresh predictor + local calibration epoch.
            self._hard_reset_detector()

        # =================================================
        # 5. Original AVG update
        # =================================================

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
            "detector_epoch": epoch_for_log,
            "detector_phase": phase_for_log,
            "transition_surprise": transition_surprise,
            "z_score": z_score_for_log,
            "transition_evidence": transition_evidence_for_log,
            "valid_local_residual": valid_local_residual,
            "supported_tables": supported_tables,
            "median_bucket_count": median_bucket_count,
            "block_completed": block_completed,
            "block_evidence": block_evidence,
            "block_coverage": block_coverage,
            "num_regions": num_regions,
            "W_t": W_for_log,
            "log_p_hat": log_p_hat_value,
            "mean_total_var": mean_total_var,
            "mean_epistemic_var": mean_epistemic_var,
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
            "detector_epoch": self.detector_epoch,
            "detector_phase": self.detector_phase,
            "predictor_warmup_remaining": (
                self.predictor_warmup_remaining
            ),
            "residual_calibration_remaining": (
                self.residual_calibration_remaining
            ),
            "change_score": self.change_score,
            "predictor_bootstrap_rng_state": (
                self.predictor_bootstrap_rng.get_state()
            ),
            "local_residual": self.local_residual.state_dict(),
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
        f"-joint-local-residual-cusum"
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

    # =====================================================
    # Conditional residual detector
    # =====================================================

    parser.add_argument(
        "--predictor_warmup_steps",
        default=10_000,
        type=int,
        help=(
            "Steps used to train the Poisson-bootstrap reference predictor "
            "and estimate the state-action hash normalization"
        ),
    )

    parser.add_argument(
        "--residual_calibration_steps",
        default=10_000,
        type=int,
        help=(
            "Steps used with the predictor frozen to calibrate local "
            "prediction-surprise statistics n_h, m_h, M2_h"
        ),
    )

    parser.add_argument(
        "--hash_num_tables",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--hash_projections_per_table",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--hash_num_buckets",
        default=4096,
        type=int,
    )

    parser.add_argument(
        "--hash_width",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--hash_input_clip",
        default=5.0,
        type=float,
    )

    parser.add_argument(
        "--min_bucket_count",
        default=5,
        type=int,
        help="Minimum calibration samples required for a local bucket",
    )

    parser.add_argument(
        "--min_supported_tables",
        default=3,
        type=int,
        help=(
            "Minimum number of LSH tables with calibrated local residual "
            "statistics before a transition may contribute evidence"
        ),
    )

    parser.add_argument(
        "--residual_std_floor",
        default=0.1,
        type=float,
        help="Lower bound on local residual standard deviation",
    )

    parser.add_argument(
        "--evidence_z_threshold",
        default=3.0,
        type=float,
        help=(
            "One-sided local z-score dead zone; only z above this value "
            "contributes change evidence"
        ),
    )

    parser.add_argument(
        "--evidence_clip",
        default=10.0,
        type=float,
        help="Maximum per-transition local residual evidence per table",
    )

    parser.add_argument(
        "--block_size",
        default=100,
        type=int,
        help="Monitoring transitions per detector block",
    )

    parser.add_argument(
        "--min_block_coverage",
        default=0.25,
        type=float,
        help=(
            "Minimum fraction of block transitions with enough calibrated "
            "local state-action support"
        ),
    )

    parser.add_argument(
        "--cusum_drift",
        default=0.05,
        type=float,
        help="Per-block drift subtracted from local residual evidence",
    )

    parser.add_argument(
        "--detector_h",
        default=5.0,
        type=float,
        help="Blockwise conditional-residual CUSUM alarm threshold",
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
            f"_aba_joint_local_residual_cusum"
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