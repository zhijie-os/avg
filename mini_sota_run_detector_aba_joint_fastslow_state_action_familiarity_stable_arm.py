import torch
import time
import pickle
import argparse
import copy
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
# Fixed-memory state-action familiarity tracker
# =========================================================

class StateActionFamiliarityTracker:
    """
    Approximate historical support for the predictor input (S_t, A_t)
    using a fixed-memory Euclidean locality-sensitive hash
    (E2LSH-style).

    This is intentionally aligned with the environment predictor,
    which models p(S_{t+1}, R_{t+1} | S_t, A_t). Familiarity is
    therefore based on the exact conditioning variables whose support
    matters for interpreting predictive surprise.

    The tracker uses only the current raw state S_t and the selected
    action A_t. It never uses S_{t+1} or R_{t+1}, so familiarity is
    determined before observing the transition outcome used as regime
    evidence.

    During an initial warmup, per-dimension mean/std of the concatenated
    [S_t, A_t] vector are estimated with Welford updates. These statistics
    are then frozen so that hash-bucket identities do not drift over the
    lifetime of the run.

    For each frozen-standardized state-action vector, several independent
    random-grid hash tables are queried. The pseudo-count is the median
    bucket count across tables. The current (S_t, A_t) pair is counted only
    AFTER its familiarity has been queried, preventing a sample from making
    itself look familiar.

    Memory is O(num_tables * num_buckets), independent of stream length
    and number of regimes.
    """

    def __init__(
        self,
        dim,
        seed,
        norm_warmup_steps=10_000,
        num_tables=4,
        projections_per_table=4,
        num_buckets=4096,
        hash_width=1.0,
        count_scale=10.0,
        input_clip=5.0,
        eps=1e-6,
    ):
        self.dim = int(dim)
        self.norm_warmup_steps = int(norm_warmup_steps)
        self.num_tables = int(num_tables)
        self.projections_per_table = int(projections_per_table)
        self.num_buckets = int(num_buckets)
        self.hash_width = float(hash_width)
        self.count_scale = float(count_scale)
        self.input_clip = float(input_clip)
        self.eps = float(eps)

        if self.dim <= 0:
            raise ValueError("Require familiarity dimension > 0")
        if self.norm_warmup_steps <= 1:
            raise ValueError("Require familiarity_norm_warmup > 1")
        if self.num_tables <= 0:
            raise ValueError("Require familiarity_num_tables > 0")
        if self.projections_per_table <= 0:
            raise ValueError("Require familiarity_projections_per_table > 0")
        if self.num_buckets <= 1:
            raise ValueError("Require familiarity_num_buckets > 1")
        if self.hash_width <= 0.0:
            raise ValueError("Require familiarity_hash_width > 0")
        if self.count_scale <= 0.0:
            raise ValueError("Require familiarity_count_scale > 0")
        if self.input_clip <= 0.0:
            raise ValueError("Require familiarity_input_clip > 0")

        rng = np.random.RandomState(int(seed))

        # Random Euclidean projections. Normalize every row so the
        # hash width has a comparable meaning across projections.
        self.projections = rng.normal(
            size=(
                self.num_tables,
                self.projections_per_table,
                self.dim,
            )
        ).astype(np.float32)

        proj_norm = np.linalg.norm(
            self.projections,
            axis=-1,
            keepdims=True,
        )
        proj_norm = np.maximum(proj_norm, 1e-8)
        self.projections /= proj_norm

        # Random grid offsets implement h(x)=floor((a^T x+b)/r).
        self.offsets = rng.uniform(
            low=0.0,
            high=self.hash_width,
            size=(
                self.num_tables,
                self.projections_per_table,
            ),
        ).astype(np.float32)

        # Deterministic integer mixing coefficients used to map each
        # short vector of quantized projections to a fixed bucket.
        self.hash_coeffs = rng.randint(
            low=1,
            high=2**31 - 1,
            size=(
                self.num_tables,
                self.projections_per_table,
            ),
        ).astype(np.int64)

        self.counts = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.uint32,
        )

        # Welford state for frozen [raw state, action] normalization.
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
        self.norm_mean += delta / float(self.norm_count)
        delta2 = x - self.norm_mean
        self.norm_M2 += delta * delta2

        if self.norm_count >= self.norm_warmup_steps:
            variance = self.norm_M2 / max(1, self.norm_count - 1)
            self.frozen_mean = self.norm_mean.astype(np.float32).copy()
            self.frozen_std = np.sqrt(
                np.maximum(variance, self.eps)
            ).astype(np.float32)

    def _standardize(self, x):
        z = (
            x.astype(np.float32)
            - self.frozen_mean
        ) / np.maximum(self.frozen_std, self.eps)

        return np.clip(
            z,
            -self.input_clip,
            self.input_clip,
        )

    def _bucket_indices(self, z):
        projected = np.einsum(
            "tpd,d->tp",
            self.projections,
            z,
        )

        quantized = np.floor(
            (projected + self.offsets)
            / self.hash_width
        ).astype(np.int64)

        buckets = []

        for table_idx in range(self.num_tables):
            # Mix the quantized coordinates directly modulo the fixed
            # table size. Python integers avoid overflow warnings.
            mixed = 0
            for value, coeff in zip(
                quantized[table_idx],
                self.hash_coeffs[table_idx],
            ):
                mixed = (
                    mixed
                    + int(value) * int(coeff)
                ) % self.num_buckets

            buckets.append(int(mixed))

        return buckets

    def observe(self, state_action):
        """
        Return (pseudo_count, familiarity_weight, ready).

        Familiarity is measured BEFORE the current (S_t, A_t) pair
        increments its hash buckets. The monotone weight is

            w(c) = 1 - exp(-c / count_scale),

        so rarely visited states contribute little and repeatedly
        visited states asymptotically receive full weight.
        """

        x = np.asarray(
            state_action,
            dtype=np.float32,
        ).reshape(-1)

        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected state-action dimension {self.dim}, got {x.shape[0]}"
            )

        if not self.ready:
            self._update_normalizer(
                x.astype(np.float64)
            )
            return 0, 0.0, self.ready

        z = self._standardize(x)
        buckets = self._bucket_indices(z)

        table_counts = [
            int(self.counts[t, bucket])
            for t, bucket in enumerate(buckets)
        ]

        # Median across independent locality-sensitive tables is robust
        # to one hash boundary making an otherwise familiar state look
        # novel, while still limiting inflation from an isolated collision.
        pseudo_count = int(np.median(table_counts))

        familiarity_weight = float(
            1.0
            - np.exp(
                -float(pseudo_count)
                / self.count_scale
            )
        )

        # Update AFTER scoring familiarity. Saturate instead of wrapping.
        max_uint32 = np.iinfo(np.uint32).max

        for table_idx, bucket in enumerate(buckets):
            if self.counts[table_idx, bucket] < max_uint32:
                self.counts[table_idx, bucket] += 1

        return pseudo_count, familiarity_weight, True

    def state_dict(self):
        return {
            "norm_count": self.norm_count,
            "norm_mean": self.norm_mean,
            "norm_M2": self.norm_M2,
            "frozen_mean": self.frozen_mean,
            "frozen_std": self.frozen_std,
            "counts": self.counts,
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
        # Fast-vs-slow state-action-familiarity detector logging
        # -------------------------------------------------

        os.makedirs(
            cfg.results_dir,
            exist_ok=True,
        )

        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_fastslow_state_action_familiarity_trace_aba_joint.csv",
        )

        self.block_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_fastslow_state_action_familiarity_blocks_aba_joint.csv",
        )

        self.regime_log_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_regime_changes_aba_joint.log",
        )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,fast_slow_gap,fast_wins,"
                "state_action_pseudo_count,state_action_familiarity_weight,"
                "familiarity_ready,familiarity_warmup_remaining,"
                "block_completed,block_valid,block_win_rate,"
                "block_familiarity,block_weight_sum,"
                "q_short,q_long,rate_gap,"
                "detector_armed,regime_change,"
                "positive_streak,initial_stabilization_streak,rearm_streak,"
                "initial_arming_complete,rate_warmup_remaining,"
                "initial_warmup_remaining,"
                "log_p_fast,log_p_slow,"
                "fast_mean_total_var,slow_mean_total_var,"
                "fast_mean_epistemic_var,slow_mean_epistemic_var,"
                "predictor_nll\n"
            )

        with open(self.block_trace_path, "w") as f:
            f.write(
                "block_end_step,block_valid,block_win_rate,"
                "block_familiarity,block_weight_sum,"
                "q_short,q_long,rate_gap,"
                "detector_armed,regime_change,"
                "positive_streak,initial_stabilization_streak,rearm_streak,"
                "initial_arming_complete,rate_warmup_remaining\n"
            )

        with open(self.regime_log_path, "w") as f:
            f.write(
                "step,block_win_rate,block_familiarity,"
                "q_short,q_long,rate_gap,positive_streak\n"
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
        # Fast predictive ensemble
        # -------------------------------------------------

        self.fast_predictors = nn.ModuleList(
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

        self.fast_pred_opts = [
            torch.optim.Adam(
                predictor.parameters(),
                lr=cfg.predictor_lr,
            )
            for predictor in self.fast_predictors
        ]

        # -------------------------------------------------
        # Slow predictive reference
        #
        # The slow ensemble starts IDENTICAL to the fast
        # ensemble. During initial warmup it is hard-synced to
        # fast; afterward it follows fast by EMA and becomes a
        # longer-timescale reference without replay data.
        # -------------------------------------------------

        self.slow_predictors = copy.deepcopy(
            self.fast_predictors
        )

        for predictor in self.slow_predictors:
            for param in predictor.parameters():
                param.requires_grad_(False)

        self.num_predictors = cfg.num_predictors
        self.slow_ema = cfg.slow_ema
        self.comparison_warmup = cfg.comparison_warmup
        self.detector_log_interval = cfg.detector_log_interval

        # -------------------------------------------------
        # State-action-familiarity-weighted short-vs-long fast-win detector
        #
        # Per transition:
        #   I_t = 1[ log p_fast > log p_slow ]
        #
        # Aggregate correlated transitions into blocks, weighting each
        # sample by explicit historical state-action familiarity w_t:
        #   b_k = sum_t w_t I_t / sum_t w_t
        #
        # Track a short and a long EMA of the BLOCK win rate:
        #   q_short <- beta_s q_short + (1-beta_s) b_k
        #   q_long  <- beta_l q_long  + (1-beta_l) b_k
        #
        # where beta_s < beta_l. The detector uses
        #   D_k = q_short - q_long.
        #
        # A change is declared only if D_k exceeds a threshold
        # for several COMPLETE blocks in a row. After detection,
        # the detector remains disarmed until D_k settles back
        # near zero for several blocks. Predictors are NOT reset
        # on an alarm, so the detector does not create its own
        # fast/slow transient.
        # -------------------------------------------------

        self.block_size = cfg.block_size
        self.short_rate_beta = cfg.short_rate_beta
        self.long_rate_beta = cfg.long_rate_beta
        self.rate_gap_threshold = cfg.rate_gap_threshold
        self.rate_persistence_blocks = cfg.rate_persistence_blocks
        self.rate_warmup_blocks = cfg.rate_warmup_blocks
        self.initial_stabilization_blocks = (
            cfg.initial_stabilization_blocks
        )
        self.rearm_gap_threshold = cfg.rearm_gap_threshold
        self.rearm_persistence_blocks = cfg.rearm_persistence_blocks
        self.rearm_block_deviation_threshold = (
            cfg.rearm_block_deviation_threshold
        )

        # Explicit count-based state-action familiarity.
        #
        # This matches the conditioning variables of the predictive model:
        # p(S_{t+1}, R_{t+1} | S_t, A_t). A state may be familiar while the
        # action taken there is not; such transitions should not receive
        # full regime-change evidence.
        self.state_action_familiarity = StateActionFamiliarityTracker(
            dim=cfg.obs_dim + cfg.action_dim,
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
        )

        self.min_block_familiarity = cfg.min_block_familiarity

        if not (0.0 <= self.min_block_familiarity <= 1.0):
            raise ValueError(
                "Require 0 <= min_block_familiarity <= 1"
            )

        if self.rearm_block_deviation_threshold < 0.0:
            raise ValueError(
                "Require rearm_block_deviation_threshold >= 0"
            )

        if self.block_size <= 0:
            raise ValueError("Require block_size > 0")

        if not (
            0.0 <= self.short_rate_beta
            < self.long_rate_beta
            < 1.0
        ):
            raise ValueError(
                "Require 0 <= short_rate_beta < long_rate_beta < 1"
            )

        if self.rate_gap_threshold <= 0.0:
            raise ValueError("Require rate_gap_threshold > 0")

        if self.rate_persistence_blocks <= 0:
            raise ValueError("Require rate_persistence_blocks > 0")

        if self.rate_warmup_blocks <= 0:
            raise ValueError("Require rate_warmup_blocks > 0")

        if self.initial_stabilization_blocks <= 0:
            raise ValueError(
                "Require initial_stabilization_blocks > 0"
            )

        if not (
            0.0 <= self.rearm_gap_threshold
            < self.rate_gap_threshold
        ):
            raise ValueError(
                "Require 0 <= rearm_gap_threshold < rate_gap_threshold"
            )

        if self.rearm_persistence_blocks <= 0:
            raise ValueError("Require rearm_persistence_blocks > 0")

        # Initial predictor warmup: slow is hard-synchronized to fast.
        self.warmup_remaining = self.comparison_warmup

        # Current block accumulator. We only start collecting blocks
        # after the predictor warmup has finished.
        self.block_weighted_wins = 0.0
        self.block_weight_sum = 0.0
        self.block_count = 0

        # Short/long block-rate estimates.
        self.q_short = None
        self.q_long = None

        # Minimum rate-estimation warmup. Reaching zero does NOT arm
        # the detector. Initial arming additionally requires the same
        # empirical stabilization test used for post-change rearming.
        self.rate_warmup_remaining = self.rate_warmup_blocks

        self.detector_armed = False
        self.initial_arming_complete = False
        self.initial_stabilization_streak = 0
        self.positive_streak = 0
        self.rearm_streak = 0

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
    # Predictor synchronization helpers
    # =====================================================

    def hard_sync_slow_to_fast(self):
        """Copy the fast ensemble exactly into the slow ensemble."""

        with torch.no_grad():
            for fast_predictor, slow_predictor in zip(
                self.fast_predictors,
                self.slow_predictors,
            ):
                slow_predictor.load_state_dict(
                    fast_predictor.state_dict()
                )


    def ema_update_slow_from_fast(self):
        """EMA update of the slow predictive reference."""

        with torch.no_grad():
            for fast_predictor, slow_predictor in zip(
                self.fast_predictors,
                self.slow_predictors,
            ):
                for fast_param, slow_param in zip(
                    fast_predictor.parameters(),
                    slow_predictor.parameters(),
                ):
                    slow_param.mul_(
                        self.slow_ema
                    )
                    slow_param.add_(
                        fast_param,
                        alpha=(
                            1.0
                            - self.slow_ema
                        ),
                    )


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

        # Keep immutable NumPy copies of the exact predictor inputs
        # used by the familiarity tracker. This is formed BEFORE using
        # S_{t+1} or R_{t+1}, so it cannot leak transition outcomes.
        raw_obs_for_familiarity = np.asarray(
            raw_obs,
            dtype=np.float32,
        ).reshape(-1).copy()

        action_for_familiarity = (
            action.detach()
            .cpu()
            .view(-1)
            .numpy()
            .astype(np.float32)
            .copy()
        )

        state_action_for_familiarity = np.concatenate(
            (
                raw_obs_for_familiarity,
                action_for_familiarity,
            ),
            axis=0,
        )

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
        # [S_{t+1}, R_{t+1}]
        detector_target = torch.cat(
            (
                raw_next_obs,
                reward_tensor,
            ),
            dim=-1,
        )

        # =================================================
        # 1. Score BOTH timescales BEFORE training on this
        #    transition.
        # =================================================

        fast_outputs = []
        slow_outputs = []

        for fast_predictor, slow_predictor in zip(
            self.fast_predictors,
            self.slow_predictors,
        ):

            fast_mean_n, fast_logvar_n = fast_predictor(
                raw_obs,
                detector_action,
            )

            fast_var_n = torch.exp(
                fast_logvar_n
            )

            fast_outputs.append(
                (
                    fast_mean_n,
                    fast_logvar_n,
                    fast_var_n,
                )
            )

            with torch.no_grad():

                slow_mean_n, slow_logvar_n = slow_predictor(
                    raw_obs,
                    detector_action,
                )

                slow_var_n = torch.exp(
                    slow_logvar_n
                )

            slow_outputs.append(
                (
                    slow_mean_n,
                    slow_logvar_n,
                    slow_var_n,
                )
            )

        with torch.no_grad():

            # ---------------------------------------------
            # Fast ensemble distribution
            # ---------------------------------------------

            fast_means = torch.stack(
                [
                    output[0].detach()
                    for output in fast_outputs
                ],
                dim=0,
            )

            fast_variances = torch.stack(
                [
                    output[2].detach()
                    for output in fast_outputs
                ],
                dim=0,
            )

            fast_mu = fast_means.mean(
                dim=0
            )

            fast_var = (
                (
                    fast_variances
                    + fast_means.pow(2)
                ).mean(dim=0)
                - fast_mu.pow(2)
            )

            fast_var = torch.clamp(
                fast_var,
                min=1e-8,
            )

            fast_epistemic_var = (
                (
                    fast_means
                    - fast_mu.unsqueeze(0)
                ).pow(2)
            ).mean(dim=0)

            # ---------------------------------------------
            # Slow ensemble distribution
            # ---------------------------------------------

            slow_means = torch.stack(
                [
                    output[0].detach()
                    for output in slow_outputs
                ],
                dim=0,
            )

            slow_variances = torch.stack(
                [
                    output[2].detach()
                    for output in slow_outputs
                ],
                dim=0,
            )

            slow_mu = slow_means.mean(
                dim=0
            )

            slow_var = (
                (
                    slow_variances
                    + slow_means.pow(2)
                ).mean(dim=0)
                - slow_mu.pow(2)
            )

            slow_var = torch.clamp(
                slow_var,
                min=1e-8,
            )

            slow_epistemic_var = (
                (
                    slow_means
                    - slow_mu.unsqueeze(0)
                ).pow(2)
            ).mean(dim=0)

            # ---------------------------------------------
            # Fast-vs-slow predictive evidence
            #
            # Positive gap:
            # recent/fast model explains this transition
            # better than the slow reference.
            # ---------------------------------------------

            log_p_fast = diagonal_gaussian_log_prob(
                detector_target,
                fast_mu,
                fast_var,
            )

            log_p_slow = diagonal_gaussian_log_prob(
                detector_target,
                slow_mu,
                slow_var,
            )

            fast_slow_gap = (
                log_p_fast
                - log_p_slow
            ).item()

            fast_wins = int(
                fast_slow_gap > 0.0
            )

            log_p_fast_value = (
                log_p_fast.item()
            )

            log_p_slow_value = (
                log_p_slow.item()
            )

            fast_mean_total_var = (
                fast_var.mean().item()
            )

            slow_mean_total_var = (
                slow_var.mean().item()
            )

            fast_mean_epistemic_var = (
                fast_epistemic_var.mean().item()
            )

            slow_mean_epistemic_var = (
                slow_epistemic_var.mean().item()
            )

        # =================================================
        # 2. Explicit state-action familiarity
        #
        # IMPORTANT:
        # - Uses exactly (raw S_t, A_t), matching the predictor input.
        # - Never uses S_{t+1} or R_{t+1}.
        # - Query happens before the current pair increments its count.
        # - Normalization is frozen after warmup so hash buckets do not
        #   drift as the policy changes.
        # =================================================

        (
            state_action_pseudo_count,
            state_action_familiarity_weight,
            familiarity_ready,
        ) = self.state_action_familiarity.observe(
            state_action_for_familiarity
        )

        familiarity_warmup_remaining = (
            self.state_action_familiarity.warmup_remaining
        )

        # =================================================
        # 3. Familiarity-weighted block-rate detector
        #
        # Per transition, state-action familiarity supplies w_t in [0, 1].
        # Rare/novel (S_t, A_t) pairs therefore contribute little to the
        # block evidence, while well-supported predictor inputs contribute
        # almost fully. The weighted block fast-win rate is
        #
        #   b_k = sum_t w_t I_t / sum_t w_t.
        #
        # The block is allowed to update the detector only when its
        # average familiarity (sum_t w_t / B) is sufficiently large.
        # This explicitly asks: is the fast model outperforming the
        # slow model on (S_t, A_t) INPUTS WITH HISTORICAL SUPPORT?
        # =================================================

        regime_change = False
        block_completed = False
        block_valid = False
        block_win_rate = float("nan")
        block_familiarity = float("nan")
        completed_block_weight_sum = float("nan")

        if self.q_short is None:
            q_short_for_log = float("nan")
            q_long_for_log = float("nan")
            rate_gap_for_log = float("nan")
        else:
            q_short_for_log = float(self.q_short)
            q_long_for_log = float(self.q_long)
            rate_gap_for_log = float(
                self.q_short - self.q_long
            )

        if (
            self.warmup_remaining == 0
            and familiarity_ready
        ):

            self.block_weighted_wins += (
                float(state_action_familiarity_weight)
                * float(fast_wins)
            )
            self.block_weight_sum += float(
                state_action_familiarity_weight
            )
            self.block_count += 1

            if self.block_count >= self.block_size:

                block_completed = True
                completed_block_weight_sum = float(
                    self.block_weight_sum
                )

                block_familiarity = (
                    self.block_weight_sum
                    / float(self.block_count)
                )

                if self.block_weight_sum > 1e-8:
                    block_win_rate = (
                        self.block_weighted_wins
                        / self.block_weight_sum
                    )

                block_valid = bool(
                    self.block_weight_sum > 1e-8
                    and block_familiarity
                    >= self.min_block_familiarity
                )

                self.block_weighted_wins = 0.0
                self.block_weight_sum = 0.0
                self.block_count = 0

                if block_valid:

                    # -------------------------------------
                    # Update short/long rates ONLY from
                    # sufficiently familiar blocks.
                    # -------------------------------------

                    if self.q_short is None:
                        self.q_short = block_win_rate
                        self.q_long = block_win_rate
                    else:
                        self.q_short = (
                            self.short_rate_beta
                            * self.q_short
                            + (1.0 - self.short_rate_beta)
                            * block_win_rate
                        )

                        self.q_long = (
                            self.long_rate_beta
                            * self.q_long
                            + (1.0 - self.long_rate_beta)
                            * block_win_rate
                        )

                    q_short_for_log = float(self.q_short)
                    q_long_for_log = float(self.q_long)
                    rate_gap_for_log = float(
                        self.q_short - self.q_long
                    )

                    # -------------------------------------
                    # Shared empirical stabilization condition.
                    #
                    # A small q_short-q_long gap alone is not enough:
                    # the current block itself must also lie near the
                    # long-rate baseline. This condition is used BOTH
                    # for the first arm and for post-detection rearming.
                    # -------------------------------------

                    rates_reconverged = (
                        abs(rate_gap_for_log)
                        <= self.rearm_gap_threshold
                    )

                    block_near_baseline = (
                        abs(
                            block_win_rate
                            - self.q_long
                        )
                        <= self.rearm_block_deviation_threshold
                    )

                    rates_stable = (
                        rates_reconverged
                        and block_near_baseline
                    )

                    # -------------------------------------
                    # Initial arming.
                    #
                    # First collect a minimum number of valid/familiar
                    # blocks, but DO NOT arm merely because that fixed
                    # warmup elapsed. After the minimum warmup, require
                    # several consecutive empirically stable blocks.
                    # -------------------------------------

                    if not self.initial_arming_complete:

                        self.detector_armed = False
                        self.positive_streak = 0
                        self.rearm_streak = 0

                        if self.rate_warmup_remaining > 0:
                            self.rate_warmup_remaining -= 1
                            self.initial_stabilization_streak = 0

                        else:
                            if rates_stable:
                                self.initial_stabilization_streak += 1
                            else:
                                self.initial_stabilization_streak = 0

                            if (
                                self.initial_stabilization_streak
                                >= self.initial_stabilization_blocks
                            ):
                                self.detector_armed = True
                                self.initial_arming_complete = True
                                self.initial_stabilization_streak = 0
                                self.positive_streak = 0

                                print(
                                    f"DETECTOR initially armed at "
                                    f"step={self.steps}: "
                                    f"weighted_block_rate={block_win_rate:.3f}, "
                                    f"q_short={self.q_short:.3f}, "
                                    f"q_long={self.q_long:.3f}, "
                                    f"gap={rate_gap_for_log:.3f}"
                                )

                    # -------------------------------------
                    # Armed: persistent positive elevation.
                    # -------------------------------------

                    elif self.detector_armed:
                        self.initial_stabilization_streak = 0
                        self.rearm_streak = 0

                        if (
                            rate_gap_for_log
                            > self.rate_gap_threshold
                        ):
                            self.positive_streak += 1
                        else:
                            self.positive_streak = 0

                        if (
                            self.positive_streak
                            >= self.rate_persistence_blocks
                        ):
                            regime_change = True

                            print(
                                f"REGIME CHANGE detected at step={self.steps}: "
                                f"weighted_block_rate={block_win_rate:.3f}, "
                                f"block_familiarity={block_familiarity:.3f}, "
                                f"q_short={self.q_short:.3f}, "
                                f"q_long={self.q_long:.3f}, "
                                f"gap={rate_gap_for_log:.3f}"
                            )

                            with open(
                                self.regime_log_path,
                                "a",
                            ) as f:
                                f.write(
                                    f"{self.steps},"
                                    f"{block_win_rate:.8f},"
                                    f"{block_familiarity:.8f},"
                                    f"{self.q_short:.8f},"
                                    f"{self.q_long:.8f},"
                                    f"{rate_gap_for_log:.8f},"
                                    f"{self.positive_streak}\n"
                                )

                            # Do NOT reset predictor networks or the
                            # familiarity counts.
                            self.detector_armed = False
                            self.positive_streak = 0
                            self.rearm_streak = 0

                    # -------------------------------------
                    # Post-detection rearming.
                    #
                    # Rearm only after the fast/slow rate process has
                    # empirically stabilized again. The default is now
                    # 4 consecutive stable blocks rather than 10.
                    # -------------------------------------

                    else:
                        self.initial_stabilization_streak = 0
                        self.positive_streak = 0

                        if rates_stable:
                            self.rearm_streak += 1
                        else:
                            self.rearm_streak = 0

                        if (
                            self.rearm_streak
                            >= self.rearm_persistence_blocks
                        ):
                            self.detector_armed = True
                            self.rearm_streak = 0
                            self.positive_streak = 0

                            print(
                                f"DETECTOR rearmed at step={self.steps}: "
                                f"weighted_block_rate={block_win_rate:.3f}, "
                                f"q_short={self.q_short:.3f}, "
                                f"q_long={self.q_long:.3f}, "
                                f"gap={rate_gap_for_log:.3f}"
                            )

                else:
                    # A block dominated by novel/rare states is not
                    # allowed to create or maintain change evidence.
                    if self.detector_armed:
                        self.positive_streak = 0
                    elif not self.initial_arming_complete:
                        self.initial_stabilization_streak = 0
                    else:
                        self.rearm_streak = 0

                # One row per completed raw block, including invalid
                # blocks, so we can directly inspect whether false
                # alarms coincide with low state-action familiarity.
                with open(
                    self.block_trace_path,
                    "a",
                ) as f:
                    f.write(
                        f"{self.steps},"
                        f"{int(block_valid)},"
                        f"{block_win_rate:.8f},"
                        f"{block_familiarity:.8f},"
                        f"{completed_block_weight_sum:.8f},"
                        f"{q_short_for_log:.8f},"
                        f"{q_long_for_log:.8f},"
                        f"{rate_gap_for_log:.8f},"
                        f"{int(self.detector_armed)},"
                        f"{int(regime_change)},"
                        f"{self.positive_streak},"
                        f"{self.initial_stabilization_streak},"
                        f"{self.rearm_streak},"
                        f"{int(self.initial_arming_complete)},"
                        f"{self.rate_warmup_remaining}\n"
                    )

        # =================================================
        # 4. Update ONLY the fast ensemble on the current
        #    streaming transition using online Poisson
        #    bootstrap weights.
        # =================================================

        predictor_nll_values = []

        for n, (
            mean_n,
            logvar_n,
            var_n,
        ) in enumerate(fast_outputs):

            # One transition, at most one optimizer step.
            # k=0 means this bootstrap member skips it;
            # k>0 scales the Gaussian NLL.
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

            self.fast_pred_opts[n].zero_grad()

            loss_n.backward()

            self.fast_pred_opts[n].step()

            predictor_nll_values.append(
                nll_n.detach().item()
            )

        if predictor_nll_values:
            predictor_nll = float(
                np.mean(
                    predictor_nll_values
                )
            )
        else:
            predictor_nll = float("nan")

        # =================================================
        # 4. Update the slow reference AFTER the fast update.
        #
        # Startup warmup:
        #     slow <- fast exactly
        #
        # After warmup:
        #     slow <- beta * slow + (1-beta) * fast
        #
        # IMPORTANT:
        # A detector alarm does NOT modify either predictive
        # ensemble. This keeps the detector from perturbing the
        # signal it is trying to measure.
        # =================================================

        if self.warmup_remaining > 0:

            self.hard_sync_slow_to_fast()
            self.warmup_remaining -= 1

        else:

            self.ema_update_slow_from_fast()

        # =================================================
        # 5. Diagnostic logging
        # =================================================

        if (
            block_completed
            or regime_change
            or self.steps % self.detector_log_interval == 0
        ):

            with open(
                self.detector_trace_path,
                "a",
            ) as f:

                f.write(
                    f"{self.steps},"
                    f"{fast_slow_gap:.8f},"
                    f"{fast_wins},"
                    f"{state_action_pseudo_count},"
                    f"{state_action_familiarity_weight:.8f},"
                    f"{int(familiarity_ready)},"
                    f"{familiarity_warmup_remaining},"
                    f"{int(block_completed)},"
                    f"{int(block_valid)},"
                    f"{block_win_rate:.8f},"
                    f"{block_familiarity:.8f},"
                    f"{completed_block_weight_sum:.8f},"
                    f"{q_short_for_log:.8f},"
                    f"{q_long_for_log:.8f},"
                    f"{rate_gap_for_log:.8f},"
                    f"{int(self.detector_armed)},"
                    f"{int(regime_change)},"
                    f"{self.positive_streak},"
                    f"{self.initial_stabilization_streak},"
                    f"{self.rearm_streak},"
                    f"{int(self.initial_arming_complete)},"
                    f"{self.rate_warmup_remaining},"
                    f"{self.warmup_remaining},"
                    f"{log_p_fast_value:.8f},"
                    f"{log_p_slow_value:.8f},"
                    f"{fast_mean_total_var:.8f},"
                    f"{slow_mean_total_var:.8f},"
                    f"{fast_mean_epistemic_var:.8f},"
                    f"{slow_mean_epistemic_var:.8f},"
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
            "fast_slow_gap": fast_slow_gap,
            "fast_wins": fast_wins,
            "state_action_pseudo_count": state_action_pseudo_count,
            "state_action_familiarity_weight": state_action_familiarity_weight,
            "familiarity_ready": familiarity_ready,
            "familiarity_warmup_remaining": familiarity_warmup_remaining,
            "block_completed": block_completed,
            "block_valid": block_valid,
            "block_win_rate": block_win_rate,
            "block_familiarity": block_familiarity,
            "block_weight_sum": completed_block_weight_sum,
            "q_short": q_short_for_log,
            "q_long": q_long_for_log,
            "rate_gap": rate_gap_for_log,
            "detector_armed": self.detector_armed,
            "regime_change": regime_change,
            "positive_streak": self.positive_streak,
            "initial_stabilization_streak": (
                self.initial_stabilization_streak
            ),
            "rearm_streak": self.rearm_streak,
            "initial_arming_complete": self.initial_arming_complete,
            "rate_warmup_remaining": self.rate_warmup_remaining,
            "log_p_fast": log_p_fast_value,
            "log_p_slow": log_p_slow_value,
            "fast_mean_total_var": fast_mean_total_var,
            "slow_mean_total_var": slow_mean_total_var,
            "fast_mean_epistemic_var": fast_mean_epistemic_var,
            "slow_mean_epistemic_var": slow_mean_epistemic_var,
            "warmup_remaining": self.warmup_remaining,
            "predictor_nll": predictor_nll,
        }



    # =====================================================
    # Save
    # =====================================================

    def save(self, model_dir, unique_str):

        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),

            "fast_predictors": [
                predictor.state_dict()
                for predictor in self.fast_predictors
            ],

            "slow_predictors": [
                predictor.state_dict()
                for predictor in self.slow_predictors
            ],

            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),

            "fast_predictor_opts": [
                opt.state_dict()
                for opt in self.fast_pred_opts
            ],

            "warmup_remaining": self.warmup_remaining,
            "block_weighted_wins": self.block_weighted_wins,
            "block_weight_sum": self.block_weight_sum,
            "block_count": self.block_count,
            "state_action_familiarity": self.state_action_familiarity.state_dict(),
            "q_short": self.q_short,
            "q_long": self.q_long,
            "rate_warmup_remaining": self.rate_warmup_remaining,
            "detector_armed": self.detector_armed,
            "initial_arming_complete": self.initial_arming_complete,
            "initial_stabilization_streak": (
                self.initial_stabilization_streak
            ),
            "positive_streak": self.positive_streak,
            "rearm_streak": self.rearm_streak,
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
        f"-joint-fastslow-state-action-familiarity-stable-arm"
        f"-{args.algo}"
        f"-{args.env}"
        f"_seed-{args.seed}"
    )

    # IMPORTANT:
    # AVG.__init__ needs this for trace-file naming.
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

            if t == args.shift1:
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

            elif t == args.shift2:
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
    # Fast/slow predictor parameters
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
        "--slow_ema",
        default=0.999,
        type=float,
        help="EMA coefficient for the slow predictive reference",
    )

    parser.add_argument(
        "--comparison_warmup",
        default=10_000,
        type=int,
        help=(
            "Initial steps during which slow is hard-synced "
            "to fast and block-rate detection is disabled"
        ),
    )

    # =====================================================
    # Explicit count-based state-action familiarity
    # =====================================================

    parser.add_argument(
        "--familiarity_norm_warmup",
        default=10_000,
        type=int,
        help=(
            "State-action samples used to estimate and then freeze "
            "the normalization used by the familiarity hash"
        ),
    )

    parser.add_argument(
        "--familiarity_num_tables",
        default=4,
        type=int,
        help="Number of independent locality-sensitive count tables",
    )

    parser.add_argument(
        "--familiarity_projections_per_table",
        default=4,
        type=int,
        help="Random Euclidean hash projections per count table",
    )

    parser.add_argument(
        "--familiarity_num_buckets",
        default=4096,
        type=int,
        help="Fixed number of counters in each familiarity table",
    )

    parser.add_argument(
        "--familiarity_hash_width",
        default=1.0,
        type=float,
        help=(
            "Random-grid hash width in frozen standardized "
            "state-action space"
        ),
    )

    parser.add_argument(
        "--familiarity_count_scale",
        default=10.0,
        type=float,
        help=(
            "Visit-count scale tau in w(c)=1-exp(-c/tau)"
        ),
    )

    parser.add_argument(
        "--familiarity_input_clip",
        "--familiarity_state_clip",
        dest="familiarity_input_clip",
        default=5.0,
        type=float,
        help=(
            "Clip each frozen-standardized state/action dimension "
            "before hashing. --familiarity_state_clip is retained as "
            "a backwards-compatible alias."
        ),
    )

    parser.add_argument(
        "--min_block_familiarity",
        default=0.25,
        type=float,
        help=(
            "Minimum average state-action-familiarity weight required "
            "for a block to update the regime detector"
        ),
    )

    parser.add_argument(
        "--block_size",
        default=100,
        type=int,
        help=(
            "Number of transitions aggregated into one fast-win "
            "rate observation"
        ),
    )

    parser.add_argument(
        "--short_rate_beta",
        default=0.80,
        type=float,
        help=(
            "EMA coefficient for the short-timescale block win rate"
        ),
    )

    parser.add_argument(
        "--long_rate_beta",
        default=0.99,
        type=float,
        help=(
            "EMA coefficient for the long-timescale block win rate"
        ),
    )

    parser.add_argument(
        "--rate_warmup_blocks",
        default=20,
        type=int,
        help=(
            "Minimum number of valid/familiar completed blocks used "
            "to establish short/long rate estimates before initial "
            "stabilization checking begins"
        ),
    )

    parser.add_argument(
        "--initial_stabilization_blocks",
        default=4,
        type=int,
        help=(
            "After the minimum rate warmup, require this many "
            "consecutive empirically stable familiar blocks before "
            "the detector is armed for the first time"
        ),
    )

    parser.add_argument(
        "--rate_gap_threshold",
        default=0.15,
        type=float,
        help=(
            "Detectable elevation in short-minus-long block fast-win rate"
        ),
    )

    parser.add_argument(
        "--rate_persistence_blocks",
        default=3,
        type=int,
        help=(
            "Number of consecutive completed blocks for which the "
            "rate gap must exceed the threshold"
        ),
    )

    parser.add_argument(
        "--rearm_gap_threshold",
        default=0.05,
        type=float,
        help=(
            "After a detection, short and long rates must reconverge "
            "within this absolute gap before rearming"
        ),
    )

    parser.add_argument(
        "--rearm_block_deviation_threshold",
        default=0.10,
        type=float,
        help=(
            "After a detection, the current weighted block win rate must "
            "also be within this distance of q_long before rearming"
        ),
    )

    parser.add_argument(
        "--rearm_persistence_blocks",
        default=4,
        type=int,
        help=(
            "Number of consecutive empirically stable blocks required "
            "to rearm after a detection"
        ),
    )

    parser.add_argument(
        "--detector_log_interval",
        default=100,
        type=int,
        help="Write detector trace every N environment steps",
    )

    parser.add_argument(
        "--shift1",
        default=50_000,
        type=int,
        help="A -> B regime-change step",
    )

    parser.add_argument(
        "--shift2",
        default=100_000,
        type=int,
        help="B -> A regime-change step",
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
            f"_aba_joint_fastslow_state_action_familiarity"
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