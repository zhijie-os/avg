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
# Fixed-memory state-action familiarity
# =========================================================

class StateActionFamiliarityTracker:
    """
    Fixed-memory approximate visitation counts for x_t = [S_t, A_t].

    A small Euclidean-LSH sketch maps every continuous state-action pair
    into one bucket per table. The number of buckets is fixed, so memory
    does not grow with stream length.

    The first norm_warmup_steps inputs are used only to estimate a frozen
    mean/std. Thereafter, familiarity is queried BEFORE incrementing the
    current buckets:

        F_t = 1 - exp(-c_t / count_scale),

    where c_t is the median pseudo-count across independent tables.
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
    ):
        self.dim = int(dim)
        self.norm_warmup_steps = int(norm_warmup_steps)
        self.num_tables = int(num_tables)
        self.projections_per_table = int(projections_per_table)
        self.num_buckets = int(num_buckets)
        self.hash_width = float(hash_width)
        self.count_scale = float(count_scale)
        self.input_clip = float(input_clip)

        if self.dim <= 0:
            raise ValueError("Require familiarity dim > 0")
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

        self.counts = np.zeros(
            (self.num_tables, self.num_buckets),
            dtype=np.uint32,
        )

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

    def observe(self, x):
        """
        Return (pseudo_count, familiarity, ready).
        Query happens before the current x increments its buckets.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.shape[0] != self.dim:
            raise ValueError(
                f"Expected state-action dimension {self.dim}, got {x.shape[0]}"
            )

        if not self.ready:
            self._update_normalizer(x)
            return 0, 0.0, self.ready

        z = self._standardize(x)
        buckets = self._bucket_indices(z)

        table_counts = [
            int(self.counts[t, bucket])
            for t, bucket in enumerate(buckets)
        ]

        pseudo_count = int(np.median(table_counts))

        familiarity = float(
            1.0
            - np.exp(
                -float(pseudo_count)
                / self.count_scale
            )
        )

        max_uint32 = np.iinfo(np.uint32).max
        for table_idx, bucket in enumerate(buckets):
            if self.counts[table_idx, bucket] < max_uint32:
                self.counts[table_idx, bucket] += 1

        return pseudo_count, familiarity, True

    def reset_counts(self):
        """
        Forget state-action familiarity from the previous regime while
        preserving the fixed LSH projections and frozen normalization.
        """
        self.counts.fill(0)

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
        # Detector logging
        # -------------------------------------------------

        os.makedirs(
            cfg.results_dir,
            exist_ok=True,
        )

        self.regime_log_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_regime_changes_aba_joint_surprise_familiarity.log",
        )

        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_surprise_familiarity_trace_aba_joint.csv",
        )

        self.detector_block_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_surprise_familiarity_blocks_aba_joint.csv",
        )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,surprise_raw,surprise_clipped,"
                "state_action_pseudo_count,familiarity,e_t,"
                "block_E_k,W_k,block_completed,warmup_remaining,"
                "regime_change,mean_total_var,mean_epistemic_var,"
                "predictor_nll\n"
            )

        with open(self.detector_block_path, "w") as f:
            f.write(
                "block_index,step,E_k,W_k,"
                "mean_familiarity,mean_surprise,"
                "regime_change\n"
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

        self.num_predictors = cfg.num_predictors

        # -------------------------------------------------
        # Simple surprise / familiarity block CUSUM
        #
        # Transition evidence:
        #   s_t = mean_j (y_j - mu_j)^2 / var_j
        #   e_t = clip(s_t, 0, s_max) * F_t^gamma
        #
        # Block evidence:
        #   E_k = (1/B) sum_{t in block k} e_t
        #
        # CUSUM:
        #   W_k = max(0, W_{k-1} + E_k - kappa)
        #
        # Predictors ALWAYS keep learning. There is no fast/slow pair
        # and no warning freeze. Low-familiarity inputs are suppressed
        # by F_t^gamma while still training the predictor normally.
        # -------------------------------------------------

        self.detector_h = cfg.detector_h

        # Baseline is now the nominal prediction surprise, NOT E_k.
        self.baseline_alpha = cfg.baseline_alpha
        self.baseline_margin = cfg.baseline_margin
        self.baseline_min_weight = cfg.baseline_min_weight
        self.surprise_baseline = None

        self.surprise_cap = cfg.surprise_cap
        self.familiarity_gamma = cfg.familiarity_gamma
        self.block_size = cfg.block_size

        self.initial_detector_warmup = cfg.initial_detector_warmup
        self.post_change_warmup = cfg.post_change_warmup
        self.detector_log_interval = cfg.detector_log_interval

        self.change_score = 0.0
        self.warmup_remaining = self.initial_detector_warmup

        # Block statistics
        self.block_evidence_sum = 0.0       # sum w_t * s_t
        self.block_weight_sum = 0.0         # sum w_t
        self.block_familiarity_sum = 0.0
        self.block_surprise_sum = 0.0
        self.block_count = 0
        self.block_index = 0

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
        # [S_{t+1}, R_{t+1}]
        detector_target = torch.cat(
            (
                raw_next_obs,
                reward_tensor,
            ),
            dim=-1,
        )

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

            mean_total_var = var_star.mean().item()
            mean_epistemic_var = epistemic_var.mean().item()

        # =================================================
        # 2. State-action familiarity
        # =================================================

        state_action_np = np.concatenate(
            (
                raw_obs.detach().cpu().numpy().reshape(-1),
                detector_action.detach().cpu().numpy().reshape(-1),
            )
        ).astype(np.float64)

        (
            state_action_pseudo_count,
            familiarity,
            familiarity_ready,
        ) = self.state_action_familiarity.observe(
            state_action_np
        )

        # Less-familiar inputs are strongly suppressed rather than
        # creating change evidence. With gamma=4, for example,
        # F=0.3 contributes only 0.3^4 = 0.0081 of its raw surprise.
        if familiarity_ready:
            familiarity_weight = (
                familiarity ** self.familiarity_gamma
            )

            # Keep this for diagnostics:
            # e_t = w_t * s_t
            e_t = (
                familiarity_weight
                * surprise_clipped
            )
        else:
            familiarity_weight = 0.0
            e_t = 0.0

        # =================================================
        # 3. Block evidence and CUSUM
        # =================================================

        regime_change = False
        block_completed = False
        block_E_k = float("nan")
        W_for_log = self.change_score

        if self.warmup_remaining > 0:
            # Predictor and familiarity keep learning during warmup;
            # only detector accumulation is disabled.
            self.warmup_remaining -= 1

            self.block_evidence_sum = 0.0
            self.block_weight_sum = 0.0
            self.block_familiarity_sum = 0.0
            self.block_surprise_sum = 0.0
            self.block_count = 0

        elif familiarity_ready:
            self.block_evidence_sum += e_t
            self.block_weight_sum += familiarity_weight
            self.block_familiarity_sum += familiarity
            self.block_surprise_sum += surprise_clipped
            self.block_count += 1

            if self.block_count >= self.block_size:
                block_completed = True

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

                # Average gated evidence, kept mainly for logging:
                # E_k = (1/B) sum_t w_t s_t
                block_E_k = (
                    self.block_evidence_sum
                    / float(self.block_count)
                )

                # Average familiarity weight:
                # w_bar = (1/B) sum_t w_t
                mean_block_weight = (
                    self.block_weight_sum
                    / float(self.block_count)
                )

                # Familiarity-weighted estimate of raw surprise:
                #
                #   s_hat_k = sum w_t s_t / sum w_t
                #
                if self.block_weight_sum > 1e-8:
                    weighted_surprise_mean = (
                        self.block_evidence_sum
                        / self.block_weight_sum
                    )
                else:
                    weighted_surprise_mean = float("nan")


                # -------------------------------------------------
                # Surprise baseline + familiarity-gated CUSUM
                # -------------------------------------------------

                block_change_evidence = 0.0

                if self.surprise_baseline is None:

                    # Do not establish a nominal surprise level until
                    # familiarity is sufficiently strong.
                    if (
                        mean_block_weight
                        >= self.baseline_min_weight
                    ):
                        self.surprise_baseline = weighted_surprise_mean

                    self.change_score = 0.0

                else:

                    # G_k =
                    #   (1/B) sum_t w_t [s_t - baseline - margin]
                    #
                    # equivalently:
                    #   E_k - (baseline + margin) * mean(w_t)
                    block_change_evidence = (
                        block_E_k
                        - (
                            self.surprise_baseline
                            + self.baseline_margin
                        )
                        * mean_block_weight
                    )

                    self.change_score = max(
                        0.0,
                        self.change_score
                        + block_change_evidence,
                    )

                    # Update baseline only when detector is fully stable.
                    if (
                        self.change_score == 0.0
                        and mean_block_weight
                        >= self.baseline_min_weight
                    ):
                        self.surprise_baseline = (
                            (1.0 - self.baseline_alpha)
                            * self.surprise_baseline
                            + self.baseline_alpha
                            * weighted_surprise_mean
                        )

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
                        f"{int(regime_change)}\n"
                    )

                self.block_index += 1
                self.block_evidence_sum = 0.0
                self.block_weight_sum = 0.0
                self.block_familiarity_sum = 0.0
                self.block_surprise_sum = 0.0
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
                            f"mean_familiarity="
                            f"{mean_block_familiarity:.8f}, "
                            f"mean_surprise="
                            f"{mean_block_surprise:.8f}\n"
                        )
                    # New regime: forget old-regime state-action familiarity.
                    self.state_action_familiarity.reset_counts()

                    # Reset sequential detector state.
                    self.change_score = 0.0

                    # Re-estimate nominal surprise for the new regime.
                    self.surprise_baseline = None

                    # Predictor keeps learning normally during this period,
                    # but detector accumulation is disabled.
                    self.warmup_remaining = self.post_change_warmup
        # =================================================
        # 4. Online Poisson bootstrap predictor update
        #
        # Always learn after scoring the current transition. There is no
        # warning-stage freeze in this simple detector.
        # =================================================

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
                    f"{block_E_k:.8f},"
                    f"{W_for_log:.8f},"
                    f"{int(block_completed)},"
                    f"{self.warmup_remaining},"
                    f"{int(regime_change)},"
                    f"{mean_total_var:.8f},"
                    f"{mean_epistemic_var:.8f},"
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
            "E_k": block_E_k,
            "W_k": W_for_log,
            "block_completed": block_completed,
            "warmup_remaining": self.warmup_remaining,
            "regime_change": regime_change,
            "mean_total_var": mean_total_var,
            "mean_epistemic_var": mean_epistemic_var,
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
                "block_familiarity_sum": self.block_familiarity_sum,
                "block_surprise_sum": self.block_surprise_sum,
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
    # Surprise / familiarity detector parameters
    # =====================================================

    parser.add_argument(
        "--detector_h",
        default=3.0,
        type=float,
        help="CUSUM alarm threshold applied to block-level W_k",
    )

    parser.add_argument(
        "--baseline_alpha",
        default=0.01,
        type=float,
        help="EMA rate for the stable-regime block evidence baseline",
    )

    parser.add_argument(
        "--baseline_margin",
        default=0.10,
        type=float,
        help="Required excess above nominal block evidence before CUSUM grows",
    )

    parser.add_argument(
        "--baseline_min_weight",
        default=0.5,
        type=float,
        help="Minimum mean F^gamma required to initialize/update surprise baseline",
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
        help="Exponent in e_t = surprise * familiarity^gamma",
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
        "--post_change_warmup",
        default=1_000,
        type=int,
        help="Steps after an alarm with learning but no CUSUM accumulation",
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
            f"_aba_joint_surprise_familiarity_cusum"
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