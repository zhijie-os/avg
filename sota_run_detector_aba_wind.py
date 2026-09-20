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


class P2Quantile:
    """Constant-memory P² estimator for one streaming quantile."""

    def __init__(self, quantile):
        self.quantile = float(quantile)
        if not (0.0 < self.quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        self.reset()

    def reset(self):
        self.count = 0
        self.initial = []
        self.marker_heights = None
        self.marker_positions = None
        self.desired_positions = None
        self.position_increments = np.asarray(
            [
                0.0,
                self.quantile / 2.0,
                self.quantile,
                (1.0 + self.quantile) / 2.0,
                1.0,
            ],
            dtype=np.float64,
        )

    def update(self, value):
        value = float(value)
        if not np.isfinite(value):
            return

        self.count += 1

        if self.marker_heights is None:
            self.initial.append(value)
            if len(self.initial) < 5:
                return

            self.initial.sort()
            self.marker_heights = np.asarray(
                self.initial,
                dtype=np.float64,
            )
            self.marker_positions = np.asarray(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                dtype=np.float64,
            )
            q = self.quantile
            self.desired_positions = np.asarray(
                [
                    1.0,
                    1.0 + 2.0 * q,
                    1.0 + 4.0 * q,
                    3.0 + 2.0 * q,
                    5.0,
                ],
                dtype=np.float64,
            )
            return

        h = self.marker_heights
        n = self.marker_positions
        np_des = self.desired_positions

        if value < h[0]:
            h[0] = value
            k = 0
        elif value < h[1]:
            k = 0
        elif value < h[2]:
            k = 1
        elif value < h[3]:
            k = 2
        elif value <= h[4]:
            k = 3
        else:
            h[4] = value
            k = 3

        n[k + 1 :] += 1.0
        np_des += self.position_increments

        for i in range(1, 4):
            d = np_des[i] - n[i]
            if (
                (d >= 1.0 and n[i + 1] - n[i] > 1.0)
                or (d <= -1.0 and n[i - 1] - n[i] < -1.0)
            ):
                direction = 1.0 if d > 0.0 else -1.0
                ip = i + int(direction)
                im = i - int(direction)

                parabolic = h[i] + direction / (
                    n[i + 1] - n[i - 1]
                ) * (
                    (n[i] - n[i - 1] + direction)
                    * (h[i + 1] - h[i])
                    / (n[i + 1] - n[i])
                    + (n[i + 1] - n[i] - direction)
                    * (h[i] - h[i - 1])
                    / (n[i] - n[i - 1])
                )

                if h[i - 1] < parabolic < h[i + 1]:
                    h[i] = parabolic
                else:
                    h[i] = h[i] + direction * (
                        h[ip] - h[i]
                    ) / (
                        n[ip] - n[i]
                    )

                n[i] += direction

    @property
    def value(self):
        if self.count == 0:
            return float("nan")
        if self.marker_heights is not None:
            return float(self.marker_heights[2])
        return float(np.quantile(self.initial, self.quantile))


class LocalWeightedAffineMemory:
    """
    Fixed-size recent-memory local weighted affine predictor.

    For a query x, historical transitions are first collected inside a maximum
    RMS radius. The actual kernel radius is then chosen adaptively: use the
    smallest observed-neighbor radius that provides the target effective sample
    size, falling back to the maximum radius when necessary.

    The same adaptive local neighborhood provides:
      1. visitation support / effective sample size;
      2. a local affine prediction for the current x;
      3. a recency-weighted estimate of historical predictive competence;
      4. query leverage, measuring whether the query is geometrically
         supported by the local cloud or requires extrapolation.

    Historical competence uses leave-one-out residuals from the local weighted
    ridge fit so that a neighborhood is not declared accurate merely because
    the affine model interpolates the samples used to fit it.

    Query leverage is reported as both the raw local-linear leverage and a
    dimensionless leverage inflation ratio:

        leverage_ratio = support * leverage_query.

    The absolute scale of leverage_ratio depends on the local design and ridge
    regularization, so it is not compared with a universal fixed threshold.
    The detector instead compares it with an online regime-local upper-tail
    quantile learned from nominal queries.

    cKDTree is only an indexing accelerator. Locality is always decided by the
    exact RMS Euclidean distance in frozen-normalized x-space.
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        capacity=20000,
        bandwidth=0.6,
        target_support=0.0,
        target_effective_n=0.0,
        error_half_life=5000.0,
        ridge=1e-3,
        tree_rebuild_interval=512,
    ):
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.capacity = int(capacity)

        # This is now the MAXIMUM allowed radius. Each query can use a
        # smaller radius when the local visitation density is high.
        self.bandwidth = float(bandwidth)
        self.target_support = float(target_support)
        self.target_effective_n = float(target_effective_n)

        self.error_half_life = float(error_half_life)
        self.ridge = float(ridge)
        self.tree_rebuild_interval = int(tree_rebuild_interval)
        self._radius = self.bandwidth * np.sqrt(self.x_dim)
        self.reset()

    def reset(self):
        self.x = np.zeros(
            (self.capacity, self.x_dim),
            dtype=np.float32,
        )
        self.y = np.zeros(
            (self.capacity, self.y_dim),
            dtype=np.float32,
        )
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

    def add(self, x, y, step):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if self.size < self.capacity:
            slot = self.size
            self.size += 1
        else:
            slot = self.next_slot

        self.next_slot = (slot + 1) % self.capacity
        self.version_counter += 1

        self.x[slot] = x
        self.y[slot] = y
        self.step[slot] = int(step)
        self.version[slot] = self.version_counter

        self.pending_slots.append(slot)
        self.updates_since_tree += 1

        if self.updates_since_tree >= self.tree_rebuild_interval:
            self._rebuild_tree()

    def _candidate_slots(self, x):
        if self.size == 0:
            return np.empty(0, dtype=np.int64)

        candidate_slots = []

        if self.tree is None:
            candidate_slots.extend(range(self.size))
        else:
            tree_ids = self.tree.query_ball_point(
                x,
                r=self._radius,
            )
            for tree_id in tree_ids:
                slot = int(self.tree_slots[tree_id])
                if self.version[slot] == self.tree_versions[tree_id]:
                    candidate_slots.append(slot)

            candidate_slots.extend(self.pending_slots)

        if not candidate_slots:
            return np.empty(0, dtype=np.int64)

        return np.unique(
            np.asarray(candidate_slots, dtype=np.int64)
        )

    @staticmethod
    def _kernel_stats_for_radius(rms_d, radius):
        """Return Epanechnikov weights, support, and effective N."""
        if radius <= 0.0 or rms_d.size == 0:
            return (
                np.empty(0, dtype=np.float64),
                0.0,
                0.0,
            )

        q2 = (rms_d / radius) ** 2
        spatial_w = np.maximum(0.0, 1.0 - q2)

        support = float(np.sum(spatial_w))
        denom2 = float(np.sum(spatial_w ** 2))
        effective_n = (
            support ** 2 / denom2
            if denom2 > 1e-12
            else 0.0
        )

        return spatial_w, support, effective_n

    def _adaptive_bandwidth(self, rms_d):
        """
        Choose a query-local radius from observed neighbor distances.

        We use the smallest observed-neighbor boundary whose Epanechnikov
        weights provide both the requested weighted support and effective
        sample size. This avoids a
        hand-chosen radius grid. If no smaller radius is sufficient, use the
        configured maximum radius.
        """
        if rms_d.size == 0:
            return self.bandwidth

        order = np.argsort(rms_d)
        d_sorted = rms_d[order]

        # N_eff cannot exceed the number of included points, so there is no
        # reason to inspect boundaries with fewer than ceil(target) points.
        start_j = max(
            1,
            int(np.ceil(self.target_effective_n)),
        )

        if start_j <= len(d_sorted):
            d2 = d_sorted ** 2
            d4 = d2 ** 2
            cum_d2 = np.cumsum(d2)
            cum_d4 = np.cumsum(d4)

            for j in range(start_j, len(d_sorted) + 1):
                boundary = float(d_sorted[j - 1])

                # At exactly d == h the Epanechnikov weight is zero. Move to
                # the next representable float so the boundary point enters
                # with an infinitesimal positive weight.
                radius = float(
                    np.nextafter(
                        max(boundary, 1e-12),
                        np.inf,
                    )
                )

                if radius >= self.bandwidth:
                    break

                h2 = radius * radius
                n = float(j)
                sum_d2 = float(cum_d2[j - 1])
                sum_d4 = float(cum_d4[j - 1])

                support = n - sum_d2 / h2
                denom2 = (
                    n
                    - 2.0 * sum_d2 / h2
                    + sum_d4 / (h2 * h2)
                )
                effective_n = (
                    support ** 2 / denom2
                    if denom2 > 1e-12
                    else 0.0
                )

                if (
                    support >= self.target_support
                    and effective_n >= self.target_effective_n
                ):
                    return radius

        # Sparse region: use the full allowed radius. The caller will still
        # check whether the target effective sample size was actually reached.
        return self.bandwidth

    def query(self, x, y, current_step):
        """
        Query BEFORE inserting the current transition.

        x and y are already frozen-normalized.
        """
        empty = {
            "neighbor_count": 0,
            "support": 0.0,
            "effective_n": 0.0,
            "bandwidth_used": float("nan"),
            "prediction": None,
            "current_error": float("nan"),
            "mean_error": float("nan"),
            "std_error": float("nan"),
            "query_leverage": float("nan"),
            "query_leverage_ratio": float("nan"),
            "residual_sq": None,
            "max_residual_dim": -1,
            "max_residual_sq": float("nan"),
            "state_residual_mean_sq": float("nan"),
            "reward_residual_sq": float("nan"),
            "current_residual": None,
        }

        if self.size == 0:
            return empty

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        slots = self._candidate_slots(x)
        if slots.size == 0:
            return empty

        dx = self.x[slots].astype(np.float64) - x[None, :]
        rms_d = np.sqrt(np.mean(dx * dx, axis=1))

        # Candidate retrieval uses the maximum radius. Exact RMS distance is
        # checked here before selecting a smaller adaptive radius.
        keep_max = rms_d < self.bandwidth
        if not np.any(keep_max):
            return empty

        slots = slots[keep_max]
        dx = dx[keep_max]
        rms_d = rms_d[keep_max]

        bandwidth_used = self._adaptive_bandwidth(rms_d)
        keep = rms_d < bandwidth_used

        if not np.any(keep):
            return empty

        slots = slots[keep]
        dx = dx[keep]
        rms_d = rms_d[keep]

        spatial_w, support, effective_n = (
            self._kernel_stats_for_radius(
                rms_d,
                bandwidth_used,
            )
        )

        # Standard local-linear parameterization around the query:
        # y ~= b + B^T (x_i - x_query)
        # so the prediction at x_query is simply the intercept.
        phi = np.concatenate(
            (
                dx,
                np.ones((len(slots), 1), dtype=np.float64),
            ),
            axis=1,
        )
        y_hist = self.y[slots].astype(np.float64)

        weighted_phi = spatial_w[:, None] * phi
        A = phi.T @ weighted_phi

        # Ridge only the slopes; leave the intercept unregularized.
        reg = np.eye(self.x_dim + 1, dtype=np.float64)
        reg[-1, -1] = 0.0
        A = A + self.ridge * reg

        B = phi.T @ (spatial_w[:, None] * y_hist)

        try:
            coef = np.linalg.solve(A, B)
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
            coef = A_inv @ B

        prediction = coef[-1].copy()
        current_error = float(
            np.mean((y - prediction) ** 2)
        )

        # Per-output current prediction error diagnostics.
        # y = [Delta S, reward], so reward is the final output dimension.
        current_residual = y - prediction
        residual_sq = current_residual ** 2
        max_residual_dim = int(np.argmax(residual_sq))
        max_residual_sq = float(residual_sq[max_residual_dim])
        state_residual_mean_sq = float(np.mean(residual_sq[:-1]))
        reward_residual_sq = float(residual_sq[-1])

        # -----------------------------------------------------
        # Query leverage / interpolation geometry
        # -----------------------------------------------------
        # Since phi is centered on x_query, the query feature vector is
        # [0, ..., 0, 1]. Its leverage is therefore the variance factor of
        # the fitted intercept at the query itself.
        phi_query = np.zeros(self.x_dim + 1, dtype=np.float64)
        phi_query[-1] = 1.0

        query_leverage = float(
            phi_query @ A_inv @ phi_query
        )
        query_leverage = max(query_leverage, 0.0)

        # 1/support is the centered-cloud baseline for the intercept variance.
        # Multiplying by support gives an interpretable inflation factor:
        # near 1 = well centered; >>1 = one-sided / extrapolative geometry.
        query_leverage_ratio = (
            float(support * query_leverage)
            if support > 1e-12
            else float("inf")
        )

        # Historical competence: leave-one-out local prediction errors.
        fitted = phi @ coef
        residual = y_hist - fitted

        leverage = spatial_w * np.einsum(
            "ij,jk,ik->i",
            phi,
            A_inv,
            phi,
        )
        leverage = np.clip(leverage, 0.0, 1.0 - 1e-8)

        loo_residual = residual / (
            1.0 - leverage
        )[:, None]
        loo_error = np.mean(
            loo_residual ** 2,
            axis=1,
        )

        # More recent local competence evidence counts more.
        ages = np.maximum(
            0,
            int(current_step) - self.step[slots],
        )
        if self.error_half_life > 0:
            temporal_w = np.power(
                0.5,
                ages / self.error_half_life,
            )
        else:
            temporal_w = np.ones_like(spatial_w)

        error_w = spatial_w * temporal_w
        error_w_sum = float(np.sum(error_w))

        if error_w_sum <= 1e-12:
            mean_error = float("nan")
            std_error = float("nan")
        else:
            mean_error = float(
                np.sum(error_w * loo_error)
                / error_w_sum
            )
            var_error = float(
                np.sum(
                    error_w
                    * (loo_error - mean_error) ** 2
                )
                / error_w_sum
            )
            std_error = float(
                np.sqrt(max(var_error, 0.0))
            )

        return {
            "neighbor_count": int(len(slots)),
            "support": support,
            "effective_n": float(effective_n),
            "bandwidth_used": float(bandwidth_used),
            "prediction": prediction,
            "current_error": current_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "query_leverage": float(query_leverage),
            "query_leverage_ratio": float(query_leverage_ratio),
            "residual_sq": residual_sq.copy(),
            "max_residual_dim": max_residual_dim,
            "max_residual_sq": max_residual_sq,
            "state_residual_mean_sq": state_residual_mean_sq,
            "reward_residual_sq": reward_residual_sq,
            "current_residual": current_residual.copy(),
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
            f"{cfg.run_id}_regime_changes_aba_wind_local_weighted_affine.log",
        )

        # Periodic detector trace for debugging/plotting.
        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_detector_trace_aba_wind_local_weighted_affine.csv",
        )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,current_error,max_abs_x_norm,num_outside_clip,"
                "kde_neighbors,kde_support,kde_neff,kde_bandwidth_used,"
                "query_leverage,query_leverage_ratio,leverage_q99,"
                "leverage_baseline_n,geometry_ok,"
                "max_residual_dim,max_residual_sq,"
                "state_residual_mean_sq,reward_residual_sq,"
                "residual_rank1_energy,residual_positive_count,rank1_ok,"
                "kde_mean_error,kde_std_error,trusted,z_t,L_t,W_t,"
                "warning,memory_frozen,warmup_remaining,regime_change\n"
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
        # Kernel-local weighted affine detector
        # -------------------------------------------------

        self.detector_h = cfg.detector_h
        self.detector_h_warn = cfg.detector_h_warn
        self.detector_trust_error = cfg.detector_trust_error
        self.detector_surprise_z = cfg.detector_surprise_z
        self.detector_min_support = cfg.detector_min_support
        self.detector_min_effective_n = cfg.detector_min_effective_n
        self.detector_norm_clip = cfg.detector_norm_clip

        self.detector_leverage_quantile = float(
            cfg.detector_leverage_quantile
        )
        self.detector_leverage_min_samples = int(
            np.ceil(1.0 / (1.0 - self.detector_leverage_quantile))
        )
        self.leverage_quantile = P2Quantile(
            self.detector_leverage_quantile
        )

        auto_target_neff = 2.0 * (
            cfg.obs_dim + cfg.action_dim + 1
        )
        requested_target_neff = (
            cfg.detector_adaptive_target_neff
            if cfg.detector_adaptive_target_neff > 0.0
            else auto_target_neff
        )
        self.detector_target_effective_n = max(
            float(self.detector_min_effective_n),
            float(requested_target_neff),
        )

        self.initial_detector_warmup = cfg.initial_detector_warmup
        self.post_change_warmup = cfg.post_change_warmup
        self.detector_log_interval = cfg.detector_log_interval

        self.detector_x_stats = RunningVectorStats(
            cfg.obs_dim + cfg.action_dim
        )
        self.detector_y_stats = RunningVectorStats(
            cfg.obs_dim + 1
        )

        self.kernel_memory = LocalWeightedAffineMemory(
            x_dim=cfg.obs_dim + cfg.action_dim,
            y_dim=cfg.obs_dim + 1,
            capacity=cfg.detector_kde_capacity,
            bandwidth=cfg.detector_kde_bandwidth,
            target_support=self.detector_min_support,
            target_effective_n=self.detector_target_effective_n,
            error_half_life=cfg.detector_error_half_life,
            ridge=cfg.detector_local_ridge,
            tree_rebuild_interval=cfg.detector_tree_rebuild_interval,
        )

        self.change_score = 0.0
        self.warmup_remaining = self.initial_detector_warmup

        # Sign-invariant residual-subspace coherence for the current CUSUM
        # excursion. Only positive-evidence residuals contribute. The scatter
        # matrix is sufficient to compute
        #   rho = lambda_max(S) / trace(S),  S = sum r r^T,
        # without storing past residual vectors.
        self.detector_rank1_threshold = float(
            cfg.detector_rank1_threshold
        )
        self.detector_rank1_min_positive = int(
            cfg.detector_rank1_min_positive
        )
        residual_dim = cfg.obs_dim + 1
        self.residual_scatter = np.zeros(
            (residual_dim, residual_dim),
            dtype=np.float64,
        )
        self.residual_positive_count = 0

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
        current_residual = None
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
        # 1. Frozen normalization for local detector
        # =================================================

        x_scale = np.maximum(
            self.detector_x_stats.std,
            1e-3,
        )
        y_scale = np.maximum(
            self.detector_y_stats.std,
            1e-3,
        )

        x_norm = (
            detector_x_np - self.detector_x_stats.mean
        ) / x_scale

        max_abs_x_norm = float(np.max(np.abs(x_norm)))
        num_outside_clip = int(
            np.sum(np.abs(x_norm) > self.detector_norm_clip)
        )
        # x_norm = np.clip(
        #     x_norm,
        #     -self.detector_norm_clip,
        #     self.detector_norm_clip,
        # )

        y_norm = (
            detector_y_np - self.detector_y_stats.mean
        ) / y_scale

        # =================================================
        # 2. Query-local weighted affine prediction
        # =================================================

        regime_change = False
        warning = False
        freeze_memory = False
        add_to_kernel_memory = False

        kde_neighbors = 0
        kde_support = 0.0
        kde_neff = 0.0
        kde_bandwidth_used = float("nan")
        query_leverage = float("nan")
        query_leverage_ratio = float("nan")
        leverage_q99 = float("nan")
        leverage_baseline_n = self.leverage_quantile.count
        geometry_ok = False
        residual_sq = None
        max_residual_dim = -1
        max_residual_sq = float("nan")
        state_residual_mean_sq = float("nan")
        reward_residual_sq = float("nan")
        residual_rank1_energy = float("nan")
        residual_positive_count_for_log = self.residual_positive_count
        rank1_ok = False
        kde_mean_error = float("nan")
        kde_std_error = float("nan")
        current_error = float("nan")
        trusted = False
        z_t = float("nan")
        L_t = 0.0
        W_for_log = self.change_score

        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1

            # The initial 10k steps are used only to freeze x/y
            # normalization. Post-change warmup does collect the new
            # regime's local transitions.
            add_to_kernel_memory = (
                self.steps >= self.initial_detector_warmup
            )

        else:
            # IMPORTANT: query before inserting the current transition.
            local = self.kernel_memory.query(
                x=x_norm,
                y=y_norm,
                current_step=self.steps,
            )

            kde_neighbors = local["neighbor_count"]
            kde_support = local["support"]
            kde_neff = local["effective_n"]
            kde_bandwidth_used = local["bandwidth_used"]
            query_leverage = local["query_leverage"]
            query_leverage_ratio = local["query_leverage_ratio"]
            residual_sq = local["residual_sq"]
            max_residual_dim = local["max_residual_dim"]
            max_residual_sq = local["max_residual_sq"]
            state_residual_mean_sq = local["state_residual_mean_sq"]
            reward_residual_sq = local["reward_residual_sq"]
            kde_mean_error = local["mean_error"]
            kde_std_error = local["std_error"]
            current_error = local["current_error"]
            current_residual = local["current_residual"]

            enough_support = (
                kde_support >= self.detector_min_support
                and kde_neff >= self.detector_target_effective_n
            )

            competent = (
                np.isfinite(kde_mean_error)
                and kde_mean_error <= self.detector_trust_error
            )

            base_trusted = bool(
                enough_support
                and competent
                and np.isfinite(current_error)
            )

            leverage_baseline_n = self.leverage_quantile.count
            leverage_q99 = self.leverage_quantile.value
            leverage_ready = (
                leverage_baseline_n
                >= self.detector_leverage_min_samples
            )

            # Before the regime-local 99th-percentile baseline is ready,
            # do not gate queries by leverage. Once ready, reject only
            # unusually extrapolative geometry relative to nominal history.
            geometry_ok = bool(
                np.isfinite(query_leverage_ratio)
                and (
                    not leverage_ready
                    or (
                        np.isfinite(leverage_q99)
                        and query_leverage_ratio <= leverage_q99
                    )
                )
            )

            trusted = bool(
                base_trusted
                and geometry_ok
            )

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
                    # Do not let a suspicious transition immediately
                    # redefine the local affine reference model.
                    freeze_memory = True
                    add_to_kernel_memory = False

                    # Accumulate the residual direction only for positive
                    # change evidence. r r^T is sign-invariant, so opposite
                    # residual signs along the same changed-dynamics axis are
                    # still treated as coherent.
                    if (
                        current_residual is not None
                        and np.all(np.isfinite(current_residual))
                    ):
                        r = np.asarray(
                            current_residual,
                            dtype=np.float64,
                        ).reshape(-1)
                        self.residual_scatter += np.outer(r, r)
                        self.residual_positive_count += 1
                else:
                    add_to_kernel_memory = True

                # A CUSUM excursion ends only when the score returns to zero.
                # Until then, retain the positive residuals even across
                # negative-evidence or untrusted steps.
                if self.change_score <= 0.0:
                    self.residual_scatter.fill(0.0)
                    self.residual_positive_count = 0

            else:
                # An unfamiliar or historically inaccurate location is
                # training data, not change evidence.
                # self.change_score = 0.0
                L_t = 0.0
                add_to_kernel_memory = True

            # Rank-1 residual energy for the current CUSUM excursion.
            # rho near 1 means the surprising residuals lie on one common
            # axis/subspace (up to sign); low rho means their directions are
            # diffuse, as in unrelated local prediction failures.
            if self.residual_positive_count > 0:
                total_residual_energy = float(
                    np.trace(self.residual_scatter)
                )
                if total_residual_energy > 1e-12:
                    largest_eigenvalue = float(
                        np.linalg.eigvalsh(self.residual_scatter)[-1]
                    )
                    residual_rank1_energy = (
                        largest_eigenvalue / total_residual_energy
                    )

            rank1_ok = bool(
                self.residual_positive_count
                >= self.detector_rank1_min_positive
                and np.isfinite(residual_rank1_energy)
                and residual_rank1_energy
                >= self.detector_rank1_threshold
            )

            W_for_log = self.change_score
            residual_positive_count_for_log = self.residual_positive_count

            if (
                self.change_score > self.detector_h
                and rank1_ok
            ):
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
                        f"bandwidth={kde_bandwidth_used:.6f}, "
                        f"query_leverage_ratio={query_leverage_ratio:.6f}, "
                        f"leverage_q99={leverage_q99:.6f}, "
                        f"leverage_baseline_n={leverage_baseline_n}, "
                        f"max_residual_dim={max_residual_dim}, "
                        f"max_residual_sq={max_residual_sq:.6f}, "
                        f"state_residual_mean_sq={state_residual_mean_sq:.6f}, "
                        f"reward_residual_sq={reward_residual_sq:.6f}, "
                        f"residual_rank1_energy={residual_rank1_energy:.6f}, "
                        f"residual_positive_count={self.residual_positive_count}, "
                        f"rank1_ok={int(rank1_ok)}, "
                        f"mean_error={kde_mean_error:.6f}, "
                        f"std_error={kde_std_error:.6f}, "
                        f"z_t={z_t:.6f}, "
                        f"L_t={L_t:.6f}, "
                        f"W_t={self.change_score:.6f}\n"
                    )

                self.change_score = 0.0
                self.kernel_memory.reset()
                self.leverage_quantile.reset()
                self.residual_scatter.fill(0.0)
                self.residual_positive_count = 0
                self.warmup_remaining = self.post_change_warmup
                freeze_memory = False
                add_to_kernel_memory = False

            elif self.change_score >= self.detector_h_warn:
                warning = True
                freeze_memory = True
                add_to_kernel_memory = False

            # Update the regime-local leverage distribution after making the
            # current decision, so the current query cannot move its own gate.
            # Use all otherwise-supported/competent queries, including ones
            # rejected by the geometry gate, so q99 tracks gradual visitation
            # changes instead of self-censoring downward. Freeze it while
            # accumulated change evidence is in the warning region.
            if (
                not regime_change
                and not warning
                and base_trusted
                and np.isfinite(query_leverage_ratio)
            ):
                self.leverage_quantile.update(
                    query_leverage_ratio
                )

        if add_to_kernel_memory:
            self.kernel_memory.add(
                x=x_norm,
                y=y_norm,
                step=self.steps,
            )

        # =================================================
        # 3. Detector logging
        # =================================================

        if self.steps % self.detector_log_interval == 0:

            with open(
                self.detector_trace_path,
                "a",
            ) as f:

                f.write(
                    f"{self.steps},"
                    f"{current_error:.8f},"
                    f"{max_abs_x_norm:.8f},"
                    f"{num_outside_clip},"
                    f"{kde_neighbors},"
                    f"{kde_support:.8f},"
                    f"{kde_neff:.8f},"
                    f"{kde_bandwidth_used:.8f},"
                    f"{query_leverage:.8f},"
                    f"{query_leverage_ratio:.8f},"
                    f"{leverage_q99:.8f},"
                    f"{leverage_baseline_n},"
                    f"{int(geometry_ok)},"
                    f"{max_residual_dim},"
                    f"{max_residual_sq:.8f},"
                    f"{state_residual_mean_sq:.8f},"
                    f"{reward_residual_sq:.8f},"
                    f"{residual_rank1_energy:.8f},"
                    f"{residual_positive_count_for_log},"
                    f"{int(rank1_ok)},"
                    f"{kde_mean_error:.8f},"
                    f"{kde_std_error:.8f},"
                    f"{int(trusted)},"
                    f"{z_t:.8f},"
                    f"{L_t:.8f},"
                    f"{W_for_log:.8f},"
                    f"{int(warning)},"
                    f"{int(freeze_memory)},"
                    f"{self.warmup_remaining},"
                    f"{int(regime_change)}\n"
                )

        # =================================================
        # 4. Original AVG update
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
            "kde_bandwidth_used": kde_bandwidth_used,
            "query_leverage": query_leverage,
            "query_leverage_ratio": query_leverage_ratio,
            "leverage_q99": leverage_q99,
            "leverage_baseline_n": leverage_baseline_n,
            "geometry_ok": geometry_ok,
            "max_residual_dim": max_residual_dim,
            "max_residual_sq": max_residual_sq,
            "state_residual_mean_sq": state_residual_mean_sq,
            "reward_residual_sq": reward_residual_sq,
            "residual_rank1_energy": residual_rank1_energy,
            "residual_positive_count": residual_positive_count_for_log,
            "rank1_ok": rank1_ok,
            "kde_mean_error": kde_mean_error,
            "kde_std_error": kde_std_error,
            "trusted": trusted,
            "z_t": z_t,
            "L_t": L_t,
            "W_t": W_for_log,
            "warning": warning,
            "memory_frozen": freeze_memory,
            "warmup_remaining": self.warmup_remaining,
            "regime_change": regime_change,
        }


    # =====================================================
    # Save
    # =====================================================

    def save(self, model_dir, unique_str):

        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
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
        f"-wind"
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
    # A -> B -> A wind regime
    # =====================================================

    torso_id = base_env.model.body("torso").id
    wind_force_x = -50.0

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
            # A -> B -> A wind regime
            # =============================================

            # External forces persist in MuJoCo unless overwritten, so clear
            # them every step and apply wind only during regime B.
            base_env.data.xfrc_applied[:] = 0.0

            if t == 25_000:
                print(
                    f"A -> B wind regime at t={t}: "
                    f"torso x-force 0.0 -> {wind_force_x}"
                )

            elif t == 50_000:
                print(
                    f"B -> A wind regime at t={t}: "
                    f"torso x-force {wind_force_x} -> 0.0"
                )

            if 25_000 <= t < 50_000:
                base_env.data.xfrc_applied[
                    torso_id,
                    0,
                ] = wind_force_x

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
        default=75_000,
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
    # Legacy predictor arguments (unused; retained for launch compatibility)
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
        help="Legacy argument retained for old launch scripts; unused by local-affine detector",
    )

    parser.add_argument(
        "--detector_kde_capacity",
        default=20000,
        type=int,
        help="Fixed-size recent transition memory used by the kernel estimator",
    )

    parser.add_argument(
        "--detector_kde_bandwidth",
        default=0.8,
        type=float,
        help="Maximum compact-kernel radius in RMS frozen-normalized (state, action) distance; each query adapts downward",
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
        help="Floor on effective local sample size; adaptive target defaults to max(this, 2*(x_dim+1))",
    )

    parser.add_argument(
        "--detector_adaptive_target_neff",
        default=0.0,
        type=float,
        help="Adaptive-radius target N_eff; <=0 uses 2*(x_dim+1)",
    )

    parser.add_argument(
        "--detector_leverage_quantile",
        default=0.99,
        type=float,
        help="Regime-local upper-tail leverage quantile used as the geometry gate; 0.99 rejects only unusually extrapolative queries",
    )

    parser.add_argument(
        "--detector_max_query_leverage_ratio",
        default=2.0,
        type=float,
        help="Legacy argument retained for launch compatibility; unused when the online leverage quantile gate is enabled",
    )

    parser.add_argument(
        "--detector_trust_error",
        default=0.05,
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
        "--detector_local_ridge",
        default=1e-3,
        type=float,
        help="Ridge regularization for local affine slopes; intercept is unregularized",
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
        help="Diagnostic threshold for reporting extreme normalized coordinates; detector inputs are not clipped",
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
        "--detector_rank1_threshold",
        default=0.8,
        type=float,
        help="Minimum rank-1 residual energy required to confirm a regime change",
    )

    parser.add_argument(
        "--detector_rank1_min_positive",
        default=4,
        type=int,
        help="Minimum number of positive-evidence residuals required before the rank-1 gate can confirm a change",
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
        default=1,
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
            f"_aba_wind_detector_local_weighted_affine"
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