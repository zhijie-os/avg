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
    Fixed-memory approximate visitation counts for x_t = [S_t, A_t].

    A small Euclidean-LSH sketch maps every continuous state-action pair
    into one bucket per table. The number of buckets is fixed, so memory
    does not grow with stream length.

    The first norm_warmup_steps inputs are used only to estimate a frozen
    mean/std. Thereafter, familiarity is queried BEFORE incrementing the
    current buckets.

    Let c_t be the median pseudo-count across independent tables and let u_t
    be the running mean of previous regime-local pseudo-counts. Define

        r_t = c_t / (u_t + eps).

    Relative familiarity is

        F_t = 0,                                      r_t <= 1
        F_t = 1 - exp(-((r_t - 1) / lambda)^p),       r_t > 1

    so merely being above the regime-average count is not enough to be
    considered familiar. Only state-action regions with counts substantially
    larger than the current regime-local mean approach F_t = 1.

    The running count mean is updated only AFTER F_t is computed, preserving
    score-before-update ordering. It is reset after each trusted alarm along
    with the LSH counts.
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
        relative_scale=2.0,
        relative_power=2.0,
        input_clip=5.0,
    ):
        self.dim = int(dim)
        self.norm_warmup_steps = int(norm_warmup_steps)
        self.num_tables = int(num_tables)
        self.projections_per_table = int(projections_per_table)
        self.num_buckets = int(num_buckets)
        self.hash_width = float(hash_width)
        self.relative_scale = float(relative_scale)
        self.relative_power = float(relative_power)
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
        if self.relative_scale <= 0.0:
            raise ValueError("Require familiarity_relative_scale > 0")
        if self.relative_power <= 0.0:
            raise ValueError("Require familiarity_relative_power > 0")
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

        # Regime-local running mean of queried pseudo-counts. This is used as
        # the reference support level u_t for relative familiarity.
        self.regime_count_n = 0
        self.regime_count_mean = 0.0
        self.relative_eps = 1e-8

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
            return 0, 0.0, self.ready, 0.0, 0.0

        z = self._standardize(x)
        buckets = self._bucket_indices(z)

        table_counts = [
            int(self.counts[t, bucket])
            for t, bucket in enumerate(buckets)
        ]

        pseudo_count = int(np.median(table_counts))

        # Compare current support against the regime-local mean support from
        # previous transitions. The current count must not influence its own
        # familiarity score.
        count_mean_before = float(self.regime_count_mean)

        if self.regime_count_n == 0 or count_mean_before <= self.relative_eps:
            count_ratio = 0.0
            familiarity = 0.0
        else:
            count_ratio = (
                float(pseudo_count)
                / (count_mean_before + self.relative_eps)
            )

            if count_ratio <= 1.0:
                familiarity = 0.0
            else:
                scaled_excess = (
                    (count_ratio - 1.0)
                    / self.relative_scale
                )
                familiarity = float(
                    1.0
                    - np.exp(
                        -(scaled_excess ** self.relative_power)
                    )
                )

        # Update u_t only after scoring this transition.
        self.regime_count_n += 1
        count_delta = float(pseudo_count) - self.regime_count_mean
        self.regime_count_mean += count_delta / float(self.regime_count_n)

        max_uint32 = np.iinfo(np.uint32).max
        for table_idx, bucket in enumerate(buckets):
            if self.counts[table_idx, bucket] < max_uint32:
                self.counts[table_idx, bucket] += 1

        return (
            pseudo_count,
            familiarity,
            True,
            count_mean_before,
            count_ratio,
        )

    def reset_counts(self):
        """
        Forget state-action familiarity from the previous regime while
        preserving the fixed LSH projections and frozen normalization.
        """
        self.counts.fill(0)
        self.regime_count_n = 0
        self.regime_count_mean = 0.0

    def state_dict(self):
        return {
            "norm_count": self.norm_count,
            "norm_mean": self.norm_mean,
            "norm_M2": self.norm_M2,
            "frozen_mean": self.frozen_mean,
            "frozen_std": self.frozen_std,
            "counts": self.counts,
            "regime_count_n": self.regime_count_n,
            "regime_count_mean": self.regime_count_mean,
            "relative_scale": self.relative_scale,
            "relative_power": self.relative_power,
        }


# =========================================================
# Fixed-memory local conditional outcome models
# =========================================================

class StateActionConditionalOutcomeTracker:
    """
    Fixed-memory local conditional models for

        x_t = [S_t, A_t]
        y_t = [Delta S_t, R_{t+1}].

    The existing StateActionFamiliarityTracker selects one LSH bucket per
    table.  Instead of treating every x in a bucket as having the same
    outcome mean, each selected bucket owns an independent affine model

        y_hat = phi(x)^T W_b,
        phi(x) = [z(x), 1],

    where z(x) is the same frozen standardized/clipped state-action vector
    used by the familiarity tracker.

    W_b is learned online with ridge-initialized recursive least squares
    (RLS).  Buckets share no learned parameters, so training in one region
    cannot overwrite predictions in another region.

    After an initial model warmup, every bucket also keeps Welford statistics
    of its own PRE-UPDATE prediction residuals.

    A bucket is allowed to contribute change evidence only when its local
    model is trustworthy at the CURRENT x.  Trust requires both:

        n_b >= max(min_bucket_count, feature_dim)

    and low RLS leverage

        l_b(x) = phi(x)^T P_b phi(x) <= max_leverage.

    Thus a large residual from a poorly covered/extrapolative local model is
    treated as "I do not know" rather than as evidence of a regime change.

    A trusted transition is scored by

        q_t^(l) =
            mean_j
            (e_{t,j} - mean_e[b,j])^2
            / (var_e[b,j] + floor),

        e_t = y_t - y_hat_t,

    using only historical residual statistics.  The final diagnostic is the
    median q_t^(l) across sufficiently supported LSH tables.

    Ordering is strictly:

        predict -> score -> update residual statistics -> update local model.

    No transitions are retained.  Memory is fixed by the number of tables,
    buckets, input dimensions, and output dimensions.
    """

    def __init__(
        self,
        familiarity_tracker,
        target_dim,
        min_bucket_count=20,
        min_tables=2,
        variance_floor=1e-3,
        surprise_cap=20.0,
        ridge=1.0,
        max_leverage=1.0,
    ):
        self.familiarity_tracker = familiarity_tracker
        self.target_dim = int(target_dim)
        self.min_bucket_count = int(min_bucket_count)
        self.min_tables = int(min_tables)
        self.variance_floor = float(variance_floor)
        self.surprise_cap = float(surprise_cap)
        self.ridge = float(ridge)
        self.max_leverage = float(max_leverage)

        if self.target_dim <= 0:
            raise ValueError("Require conditional target_dim > 0")
        if self.min_bucket_count < 2:
            raise ValueError("Require conditional_min_bucket_count >= 2")
        if (
            self.min_tables < 1
            or self.min_tables > self.familiarity_tracker.num_tables
        ):
            raise ValueError(
                "Require 1 <= conditional_min_tables <= familiarity_num_tables"
            )
        if self.variance_floor <= 0.0:
            raise ValueError("Require conditional_variance_floor > 0")
        if self.surprise_cap <= 0.0:
            raise ValueError("Require conditional_surprise_cap > 0")
        if self.ridge <= 0.0:
            raise ValueError("Require conditional_ridge > 0")
        if self.max_leverage <= 0.0:
            raise ValueError("Require conditional_max_leverage > 0")

        self.input_dim = int(self.familiarity_tracker.dim)
        self.feature_dim = self.input_dim + 1  # affine bias

        # A local affine model has feature_dim coefficients per output.
        # Do not trust it before it has at least this many observations,
        # even if the older min_bucket_count setting is smaller.
        self.trust_min_model_count = max(
            self.min_bucket_count,
            self.feature_dim,
        )

        shape = (
            self.familiarity_tracker.num_tables,
            self.familiarity_tracker.num_buckets,
        )

        # Number of observations used to train each local model.
        self.counts = np.zeros(
            shape,
            dtype=np.uint32,
        )

        # Local affine coefficient matrix W_b:
        #     [feature_dim, target_dim]
        self.weights = np.zeros(
            shape + (self.feature_dim, self.target_dim),
            dtype=np.float32,
        )

        # RLS inverse regularized design matrix:
        #     P_b = (ridge I + sum phi phi^T)^(-1)
        #
        # Initial P = I / ridge.
        self.P = np.zeros(
            shape + (self.feature_dim, self.feature_dim),
            dtype=np.float32,
        )

        diag = np.arange(self.feature_dim)
        self.P[..., diag, diag] = np.float32(1.0 / self.ridge)

        # Historical PRE-UPDATE local-model residual statistics.
        #
        # We intentionally begin collecting these only after the local model
        # has seen min_bucket_count samples.  This prevents the very early
        # zero/undertrained model residuals from defining the nominal scale.
        self.residual_counts = np.zeros(
            shape,
            dtype=np.uint32,
        )
        self.residual_mean = np.zeros(
            shape + (self.target_dim,),
            dtype=np.float32,
        )
        self.residual_M2 = np.zeros(
            shape + (self.target_dim,),
            dtype=np.float32,
        )

    def _feature_vector(self, x):
        """
        Return the exact standardized state-action vector plus affine bias.

        This is deliberately richer than the old constant bucket model:
        two points that hash to the same bucket may still receive different
        predictions because their full standardized x vectors differ.
        """
        z = self.familiarity_tracker._standardize(x)

        phi = np.empty(
            self.feature_dim,
            dtype=np.float64,
        )
        phi[:-1] = z
        phi[-1] = 1.0

        return z, phi

    def observe(self, x, y):
        """
        Return:
            conditional_surprise_raw,
            conditional_surprise_clipped,
            median_historical_count,
            tables_used,
            median_leverage

        The same current transition is scored independently by the local model
        in each LSH table.  Only tables with both:
          1. a sufficiently trained local model, and
          2. enough historical local-model residuals
        may contribute to the score.

        The current transition NEVER influences its own prediction, residual
        baseline, or surprise score.
        """
        if not self.familiarity_tracker.ready:
            return (
                float("nan"),
                float("nan"),
                0.0,
                0,
                float("nan"),
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        ).reshape(-1)

        y = np.asarray(
            y,
            dtype=np.float64,
        ).reshape(-1)

        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected conditional x dimension "
                f"{self.input_dim}, got {x.shape[0]}"
            )

        if y.shape[0] != self.target_dim:
            raise ValueError(
                f"Expected conditional y dimension "
                f"{self.target_dim}, got {y.shape[0]}"
            )

        z, phi = self._feature_vector(x)
        buckets = self.familiarity_tracker._bucket_indices(z)

        table_scores = []
        table_counts = []
        table_leverages = []

        # Cache each table's pre-update residual/leverage because these must be
        # computed strictly before the current transition updates the model.
        preupdate_residuals = []
        preupdate_leverages = []

        # -------------------------------------------------
        # 1. Predict and score BEFORE any update.
        # -------------------------------------------------
        for table_idx, bucket in enumerate(buckets):
            n = int(
                self.counts[
                    table_idx,
                    bucket,
                ]
            )

            residual_n = int(
                self.residual_counts[
                    table_idx,
                    bucket,
                ]
            )

            table_counts.append(n)

            W = self.weights[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            prediction = phi @ W
            residual = y - prediction
            preupdate_residuals.append(residual)

            # RLS leverage measures whether the current direction in x-space
            # is actually supported by this bucket's historical design matrix.
            #
            #   leverage = phi^T P phi
            #
            # High leverage means that, despite hashing to this bucket, the
            # local linear model is extrapolating and its prediction should
            # not be trusted as regime-change evidence.
            P = self.P[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            P_phi = P @ phi
            leverage = float(phi @ P_phi)

            table_leverages.append(leverage)
            preupdate_leverages.append(leverage)

            # Trust gate:
            #   1. enough samples to constrain the affine model,
            #   2. enough historical trusted residuals to estimate its scale,
            #   3. low leverage at this exact current x.
            if (
                n < self.trust_min_model_count
                or residual_n < self.min_bucket_count
                or not np.isfinite(leverage)
                or leverage > self.max_leverage
            ):
                continue

            residual_mean = self.residual_mean[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            residual_M2 = self.residual_M2[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            residual_var = (
                residual_M2
                / float(max(residual_n - 1, 1))
            )

            residual_var = np.maximum(
                residual_var,
                self.variance_floor,
            )

            centered_residual = (
                residual - residual_mean
            )

            score = float(
                np.mean(
                    centered_residual ** 2
                    / residual_var
                )
            )

            if np.isfinite(score):
                table_scores.append(score)

        median_historical_count = (
            float(np.median(table_counts))
            if table_counts
            else 0.0
        )

        median_leverage = (
            float(np.median(table_leverages))
            if table_leverages
            else float("nan")
        )

        if len(table_scores) >= self.min_tables:
            conditional_surprise_raw = float(
                np.median(table_scores)
            )

            conditional_surprise_clipped = float(
                np.clip(
                    conditional_surprise_raw,
                    0.0,
                    self.surprise_cap,
                )
            )
        else:
            conditional_surprise_raw = float("nan")
            conditional_surprise_clipped = float("nan")

        # -------------------------------------------------
        # 2. Update historical PRE-UPDATE residual stats.
        # -------------------------------------------------
        max_uint32 = np.iinfo(np.uint32).max

        for table_idx, bucket in enumerate(buckets):
            n = int(
                self.counts[
                    table_idx,
                    bucket,
                ]
            )

            residual = preupdate_residuals[
                table_idx
            ]

            leverage = preupdate_leverages[
                table_idx
            ]

            # The residual baseline itself must also be built only from
            # trustworthy local predictions.  Otherwise extrapolation errors
            # would inflate the nominal residual distribution.
            if (
                n >= self.trust_min_model_count
                and np.isfinite(leverage)
                and leverage <= self.max_leverage
            ):
                residual_n = int(
                    self.residual_counts[
                        table_idx,
                        bucket,
                    ]
                )

                if residual_n < max_uint32:
                    new_residual_n = residual_n + 1

                    old_mean = self.residual_mean[
                        table_idx,
                        bucket,
                    ].astype(
                        np.float64,
                        copy=True,
                    )

                    old_M2 = self.residual_M2[
                        table_idx,
                        bucket,
                    ].astype(
                        np.float64,
                        copy=True,
                    )

                    delta = residual - old_mean
                    new_mean = (
                        old_mean
                        + delta / float(new_residual_n)
                    )
                    delta2 = residual - new_mean
                    new_M2 = (
                        old_M2
                        + delta * delta2
                    )

                    self.residual_counts[
                        table_idx,
                        bucket,
                    ] = new_residual_n

                    self.residual_mean[
                        table_idx,
                        bucket,
                    ] = new_mean.astype(np.float32)

                    self.residual_M2[
                        table_idx,
                        bucket,
                    ] = new_M2.astype(np.float32)

        # -------------------------------------------------
        # 3. RLS update of the local affine models.
        #
        # For one bucket:
        #
        #   P   <- P - P phi phi^T P / (1 + phi^T P phi)
        #   W   <- W + K (y - phi^T W)^T
        #   K    = P phi / (1 + phi^T P phi)
        #
        # The residual here is still the PRE-UPDATE residual.
        # -------------------------------------------------
        for table_idx, bucket in enumerate(buckets):
            n = int(
                self.counts[
                    table_idx,
                    bucket,
                ]
            )

            if n >= max_uint32:
                continue

            P = self.P[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            W = self.weights[
                table_idx,
                bucket,
            ].astype(
                np.float64,
                copy=False,
            )

            residual = preupdate_residuals[
                table_idx
            ]

            P_phi = P @ phi

            denominator = float(
                1.0 + phi @ P_phi
            )

            # ridge > 0 makes this positive in exact arithmetic.
            denominator = max(
                denominator,
                1e-12,
            )

            gain = (
                P_phi / denominator
            )

            new_W = (
                W
                + np.outer(
                    gain,
                    residual,
                )
            )

            new_P = (
                P
                - np.outer(
                    gain,
                    P_phi,
                )
            )

            # Remove tiny numerical asymmetry from repeated float32 storage.
            new_P = 0.5 * (
                new_P + new_P.T
            )

            self.weights[
                table_idx,
                bucket,
            ] = new_W.astype(np.float32)

            self.P[
                table_idx,
                bucket,
            ] = new_P.astype(np.float32)

            self.counts[
                table_idx,
                bucket,
            ] = n + 1

        return (
            conditional_surprise_raw,
            conditional_surprise_clipped,
            median_historical_count,
            len(table_scores),
            median_leverage,
        )

    def reset(self):
        """Forget all regime-local local-model and residual statistics."""
        self.counts.fill(0)
        self.weights.fill(0.0)

        self.P.fill(0.0)
        diag = np.arange(self.feature_dim)
        self.P[..., diag, diag] = np.float32(
            1.0 / self.ridge
        )

        self.residual_counts.fill(0)
        self.residual_mean.fill(0.0)
        self.residual_M2.fill(0.0)

    def state_dict(self):
        return {
            "counts": self.counts,
            "weights": self.weights,
            "P": self.P,
            "residual_counts": self.residual_counts,
            "residual_mean": self.residual_mean,
            "residual_M2": self.residual_M2,
            "input_dim": self.input_dim,
            "feature_dim": self.feature_dim,
            "target_dim": self.target_dim,
            "min_bucket_count": self.min_bucket_count,
            "min_tables": self.min_tables,
            "variance_floor": self.variance_floor,
            "surprise_cap": self.surprise_cap,
            "ridge": self.ridge,
            "max_leverage": self.max_leverage,
            "trust_min_model_count": self.trust_min_model_count,
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
                "state_action_pseudo_count,familiarity_count_mean,"
                "familiarity_count_ratio,familiarity,e_t,"
                "block_E_k,W_k,block_completed,warmup_remaining,"
                "regime_change,mean_total_var,mean_epistemic_var,"
                "raw_residual_mse,raw_delta_state_mse,raw_reward_sq_error,"
                "mean_total_var_raw,mean_epistemic_var_raw,"
                "gradient_coherence,gradient_novelty,predictor_nll,"
                "conditional_surprise_raw,conditional_surprise_clipped,"
                "conditional_historical_count,conditional_tables_used,"
                "conditional_median_leverage,conditional_trust_fraction\n"
            )

        with open(self.detector_block_path, "w") as f:
            f.write(
                "block_index,step,E_k,W_k,"
                "mean_pseudo_count,mean_count_reference,mean_count_ratio,"
                "mean_familiarity,mean_surprise,"
                "mean_conditional_surprise,mean_conditional_support,"
                "conditional_valid_fraction,"
                "conditional_z,conditional_run_length,"
                "conditional_baseline_n,conditional_baseline_mean,"
                "conditional_baseline_std,"
                "mean_conditional_leverage,"
                "mean_conditional_trust_fraction,"
                "mean_gradient_coherence,mean_gradient_novelty,"
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
        # Simple surprise / familiarity block CUSUM
        #
        # Transition evidence:
        #   s_t = mean_j (y_j - mu_j)^2 / var_j
        #   e_t = clip(s_t, 0, s_max) * F_t^gamma
        #
        # Block evidence:
        #   E_k = (1/B) sum_{t in block k} e_t
        #
        # Familiarity-gated surprise residual CUSUM:
        #   G_k = (1/B) sum_t F_t^gamma * (s_t - mu_s - delta)
        #   W_k = max(0, W_{k-1} + G_k)
        #
        # Predictors learn continuously within the currently detected
        # regime.  On a trusted alarm, the complete predictor ensemble
        # and predictor Adam states are reinitialized.  Low-familiarity
        # inputs naturally suppress detector evidence while the fresh
        # predictor learns the new regime.
        # -------------------------------------------------

        self.detector_h = cfg.detector_h

        # -------------------------------------------------
        # Nominal surprise baseline
        #
        # The baseline is regime-local:
        #   1. After startup / a detected change, collect several
        #      trustworthy blocks before initializing it.
        #   2. Slowly update it while change evidence is small.
        #   3. Freeze it once W becomes sufficiently suspicious.
        # -------------------------------------------------
        self.baseline_alpha = cfg.baseline_alpha
        self.baseline_margin = cfg.baseline_margin
        self.baseline_min_weight = cfg.baseline_min_weight
        self.baseline_init_blocks = cfg.baseline_init_blocks
        self.baseline_freeze_score = cfg.baseline_freeze_score

        self.surprise_baseline = None

        # Used only while constructing a new regime-local baseline.
        self.baseline_init_sum = 0.0
        self.baseline_init_count = 0

        self.surprise_cap = cfg.surprise_cap
        self.familiarity_gamma = cfg.familiarity_gamma
        self.block_size = cfg.block_size

        self.initial_detector_warmup = cfg.initial_detector_warmup
        self.detector_log_interval = cfg.detector_log_interval

        self.change_score = 0.0
        self.warmup_remaining = self.initial_detector_warmup

        # -------------------------------------------------
        # Local-conditional block decision rule
        #
        # Q_k = mean local conditional surprise in block k.
        #
        # A regime-local Welford baseline standardizes Q_k:
        #
        #     Z_k = (Q_k - mean_Q) / std_Q.
        #
        # The current block is ALWAYS scored against the baseline from
        # previous blocks.  Blocks with Z_k above the one-sided threshold
        # are NOT inserted into the baseline.  A change is declared only
        # after a contiguous run of abnormal blocks.
        #
        # This is a practical streaming heuristic, not a theorem-calibrated
        # false-alarm guarantee.  The purpose of Z_k is to make the decision
        # rule comparable across seeds with different raw Q_k scales.
        # -------------------------------------------------
        self.conditional_decision_init_blocks = (
            cfg.conditional_decision_init_blocks
        )
        self.conditional_decision_min_valid_fraction = (
            cfg.conditional_decision_min_valid_fraction
        )
        self.conditional_decision_z_threshold = (
            cfg.conditional_decision_z_threshold
        )
        self.conditional_decision_persistence = (
            cfg.conditional_decision_persistence
        )
        self.conditional_decision_eps = 1e-8

        self.conditional_baseline_n = 0
        self.conditional_baseline_mean = 0.0
        self.conditional_baseline_M2 = 0.0
        self.conditional_exceedance_run = 0

        # Block statistics
        self.block_evidence_sum = 0.0       # sum w_t * s_t
        self.block_weight_sum = 0.0         # sum w_t
        self.block_pseudo_count_sum = 0.0
        self.block_count_reference_sum = 0.0
        self.block_count_ratio_sum = 0.0
        self.block_familiarity_sum = 0.0
        self.block_surprise_sum = 0.0

        # Historical conditional-anchor diagnostics.  Only transitions with
        # enough pre-existing bucket support contribute.
        self.block_conditional_surprise_sum = 0.0
        self.block_conditional_support_sum = 0.0
        self.block_conditional_valid_count = 0
        self.block_conditional_leverage_sum = 0.0
        self.block_conditional_trust_fraction_sum = 0.0
        self.block_conditional_diagnostic_count = 0

        self.block_gradient_coherence_sum = 0.0
        self.block_gradient_novelty_sum = 0.0
        self.block_gradient_count = 0
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
            relative_scale=cfg.familiarity_relative_scale,
            relative_power=cfg.familiarity_relative_power,
            input_clip=cfg.familiarity_input_clip,
        )

        # Independent local affine models for the conditional mapping
        # (S_t, A_t) -> (Delta S_t, R_{t+1}).  Every LSH bucket gets its own
        # fixed-memory RLS model; buckets share no learned parameters.
        self.conditional_outcome_tracker = StateActionConditionalOutcomeTracker(
            familiarity_tracker=self.state_action_familiarity,
            target_dim=cfg.obs_dim + 1,
            min_bucket_count=cfg.conditional_min_bucket_count,
            min_tables=cfg.conditional_min_tables,
            variance_floor=cfg.conditional_variance_floor,
            surprise_cap=cfg.conditional_surprise_cap,
            ridge=cfg.conditional_ridge,
            max_leverage=cfg.conditional_max_leverage,
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
    # Local-conditional block decision
    # =====================================================

    def _conditional_baseline_std(self):
        if self.conditional_baseline_n < 2:
            return float("nan")

        variance = (
            self.conditional_baseline_M2
            / float(self.conditional_baseline_n - 1)
        )

        return float(
            np.sqrt(
                max(
                    variance,
                    self.conditional_decision_eps,
                )
            )
        )

    def _conditional_baseline_update(self, q):
        """Welford update using one accepted nominal block Q_k."""
        q = float(q)

        self.conditional_baseline_n += 1

        delta = (
            q - self.conditional_baseline_mean
        )

        self.conditional_baseline_mean += (
            delta
            / float(self.conditional_baseline_n)
        )

        delta2 = (
            q - self.conditional_baseline_mean
        )

        self.conditional_baseline_M2 += (
            delta * delta2
        )

    def _reset_conditional_decision_state(self):
        """Start a fresh regime-local block baseline after an alarm."""
        self.conditional_baseline_n = 0
        self.conditional_baseline_mean = 0.0
        self.conditional_baseline_M2 = 0.0
        self.conditional_exceedance_run = 0

    def _conditional_block_decision(
        self,
        q,
        valid_fraction,
    ):
        """
        Score one completed block and return

            z,
            run_length,
            baseline_n,
            baseline_mean,
            baseline_std,
            alarm.

        Only blocks with enough valid local-model transitions participate.
        Missing/low-support blocks break persistence because there is no
        positive evidence that the conditional mapping changed.

        Baseline construction:
          * collect conditional_decision_init_blocks eligible blocks;
          * after initialization, score before updating;
          * abnormal blocks (Z > threshold) are frozen out of the baseline;
          * ordinary blocks reset the run length and update Welford stats.

        Alarm:
          conditional_decision_persistence consecutive abnormal blocks.
        """
        q = float(q)
        valid_fraction = float(valid_fraction)

        baseline_std = self._conditional_baseline_std()

        if (
            not np.isfinite(q)
            or valid_fraction
            < self.conditional_decision_min_valid_fraction
        ):
            self.conditional_exceedance_run = 0

            return (
                float("nan"),
                self.conditional_exceedance_run,
                self.conditional_baseline_n,
                self.conditional_baseline_mean,
                baseline_std,
                False,
            )

        # Build the initial per-regime baseline before making decisions.
        if (
            self.conditional_baseline_n
            < self.conditional_decision_init_blocks
        ):
            self._conditional_baseline_update(q)
            self.conditional_exceedance_run = 0

            return (
                float("nan"),
                self.conditional_exceedance_run,
                self.conditional_baseline_n,
                self.conditional_baseline_mean,
                self._conditional_baseline_std(),
                False,
            )

        baseline_mean_before = float(
            self.conditional_baseline_mean
        )

        baseline_std_before = self._conditional_baseline_std()

        z = (
            q - baseline_mean_before
        ) / max(
            baseline_std_before,
            self.conditional_decision_eps,
        )

        if z > self.conditional_decision_z_threshold:
            # Freeze the baseline during a suspicious excursion.
            self.conditional_exceedance_run += 1
        else:
            # The excursion ended.  This is accepted as another nominal block.
            self.conditional_exceedance_run = 0
            self._conditional_baseline_update(q)

        alarm = (
            self.conditional_exceedance_run
            >= self.conditional_decision_persistence
        )

        return (
            float(z),
            self.conditional_exceedance_run,
            self.conditional_baseline_n,
            baseline_mean_before,
            baseline_std_before,
            bool(alarm),
        )


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
            familiarity_count_mean,
            familiarity_count_ratio,
        ) = self.state_action_familiarity.observe(
            state_action_np
        )

        # -------------------------------------------------
        # Local conditional outcome models
        #
        # This score is independent of the global neural predictor parameters.
        # Each LSH bucket predicts y_t from the exact standardized current
        # state-action x_t using its own local affine RLS model.
        #
        # The target normalizer must already be frozen so historical y values
        # live in one fixed coordinate system across the run.
        # -------------------------------------------------
        conditional_target_np = (
            detector_target.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )

        conditional_surprise_raw = float("nan")
        conditional_surprise_clipped = float("nan")
        conditional_historical_count = 0.0
        conditional_tables_used = 0
        conditional_median_leverage = float("nan")
        conditional_trust_fraction = 0.0

        if (
            self.state_action_familiarity.ready
            and self.predictor_target_normalizer.ready
        ):
            (
                conditional_surprise_raw,
                conditional_surprise_clipped,
                conditional_historical_count,
                conditional_tables_used,
                conditional_median_leverage,
            ) = self.conditional_outcome_tracker.observe(
                state_action_np,
                conditional_target_np,
            )

            conditional_trust_fraction = (
                float(conditional_tables_used)
                / float(self.state_action_familiarity.num_tables)
            )

        # Relative familiarity is zero when the current pseudo-count is not
        # above the regime-local mean count. It approaches one only when the
        # current support is substantially larger than that mean. The existing
        # gamma exponent then provides an additional suppression of merely
        # moderate familiarity.
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

        conditional_z_for_log = float("nan")
        conditional_run_length_for_log = (
            self.conditional_exceedance_run
        )
        conditional_baseline_n_for_log = (
            self.conditional_baseline_n
        )
        conditional_baseline_mean_for_log = (
            self.conditional_baseline_mean
        )
        conditional_baseline_std_for_log = (
            self._conditional_baseline_std()
        )

        if self.warmup_remaining > 0:
            # Predictor and familiarity keep learning during warmup;
            # only detector accumulation is disabled.
            self.warmup_remaining -= 1

            self.block_evidence_sum = 0.0
            self.block_weight_sum = 0.0
            self.block_pseudo_count_sum = 0.0
            self.block_count_reference_sum = 0.0
            self.block_count_ratio_sum = 0.0
            self.block_familiarity_sum = 0.0
            self.block_surprise_sum = 0.0
            self.block_conditional_surprise_sum = 0.0
            self.block_conditional_support_sum = 0.0
            self.block_conditional_valid_count = 0
            self.block_conditional_leverage_sum = 0.0
            self.block_conditional_trust_fraction_sum = 0.0
            self.block_conditional_diagnostic_count = 0
            self.block_gradient_coherence_sum = 0.0
            self.block_gradient_novelty_sum = 0.0
            self.block_gradient_count = 0
            self.block_count = 0

        elif familiarity_ready:
            self.block_evidence_sum += e_t
            self.block_weight_sum += familiarity_weight
            self.block_pseudo_count_sum += float(state_action_pseudo_count)
            self.block_count_reference_sum += familiarity_count_mean
            self.block_count_ratio_sum += familiarity_count_ratio
            self.block_familiarity_sum += familiarity
            self.block_surprise_sum += surprise_clipped

            if np.isfinite(conditional_surprise_clipped):
                self.block_conditional_surprise_sum += (
                    conditional_surprise_clipped
                )
                self.block_conditional_support_sum += (
                    conditional_historical_count
                )
                self.block_conditional_valid_count += 1

            if np.isfinite(conditional_median_leverage):
                self.block_conditional_leverage_sum += (
                    conditional_median_leverage
                )
                self.block_conditional_trust_fraction_sum += (
                    conditional_trust_fraction
                )
                self.block_conditional_diagnostic_count += 1

            if np.isfinite(gradient_coherence):
                self.block_gradient_coherence_sum += gradient_coherence
                self.block_gradient_novelty_sum += gradient_novelty
                self.block_gradient_count += 1

            self.block_count += 1

            if self.block_count >= self.block_size:
                block_completed = True

                block_E_k = (
                    self.block_evidence_sum
                    / float(self.block_count)
                )

                mean_block_pseudo_count = (
                    self.block_pseudo_count_sum
                    / float(self.block_count)
                )

                mean_block_count_reference = (
                    self.block_count_reference_sum
                    / float(self.block_count)
                )

                mean_block_count_ratio = (
                    self.block_count_ratio_sum
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

                if self.block_conditional_valid_count > 0:
                    mean_block_conditional_surprise = (
                        self.block_conditional_surprise_sum
                        / float(self.block_conditional_valid_count)
                    )
                    mean_block_conditional_support = (
                        self.block_conditional_support_sum
                        / float(self.block_conditional_valid_count)
                    )
                else:
                    mean_block_conditional_surprise = float("nan")
                    mean_block_conditional_support = 0.0

                conditional_valid_fraction = (
                    float(self.block_conditional_valid_count)
                    / float(self.block_count)
                )

                if self.block_conditional_diagnostic_count > 0:
                    mean_block_conditional_leverage = (
                        self.block_conditional_leverage_sum
                        / float(self.block_conditional_diagnostic_count)
                    )
                    mean_block_conditional_trust_fraction = (
                        self.block_conditional_trust_fraction_sum
                        / float(self.block_conditional_diagnostic_count)
                    )
                else:
                    mean_block_conditional_leverage = float("nan")
                    mean_block_conditional_trust_fraction = 0.0

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
                # Local conditional-surprise decision
                #
                # This is the ACTUAL regime-change decision rule.
                # The older NN-surprise CUSUM below is retained only as a
                # diagnostic so existing E_k/W_k traces remain comparable.
                # -------------------------------------------------

                (
                    conditional_z_for_log,
                    conditional_run_length_for_log,
                    conditional_baseline_n_for_log,
                    conditional_baseline_mean_for_log,
                    conditional_baseline_std_for_log,
                    conditional_alarm,
                ) = self._conditional_block_decision(
                    mean_block_conditional_surprise,
                    conditional_valid_fraction,
                )

                if conditional_alarm:
                    regime_change = True

                # -------------------------------------------------
                # Legacy NN surprise baseline + CUSUM
                # Diagnostic only: it no longer triggers an alarm.
                # -------------------------------------------------

                block_change_evidence = 0.0

                # =================================================
                # A. No baseline yet:
                #    collect several trustworthy blocks first.
                # =================================================
                if self.surprise_baseline is None:

                    if (
                        mean_block_weight
                        >= self.baseline_min_weight
                        and np.isfinite(weighted_surprise_mean)
                    ):
                        self.baseline_init_sum += weighted_surprise_mean
                        self.baseline_init_count += 1

                        # Initialize the regime-local nominal surprise
                        # only after enough eligible blocks have been seen.
                        if (
                            self.baseline_init_count
                            >= self.baseline_init_blocks
                        ):
                            self.surprise_baseline = (
                                self.baseline_init_sum
                                / float(self.baseline_init_count)
                            )

                    # Do not accumulate change evidence until
                    # a reliable baseline has been initialized.
                    self.change_score = 0.0

                # =================================================
                # B. Baseline exists:
                #    compute familiarity-gated excess surprise.
                # =================================================
                else:

                    # G_k =
                    #   (1/B) sum_t w_t [s_t - mu_s - delta]
                    #
                    # equivalently:
                    #   E_k - (mu_s + delta) * mean(w_t)
                    block_change_evidence = (
                        block_E_k
                        - (
                            self.surprise_baseline
                            + self.baseline_margin
                        )
                        * mean_block_weight
                    )

                    # Standard one-sided CUSUM.
                    self.change_score = max(
                        0.0,
                        self.change_score
                        + block_change_evidence,
                    )

                    # Slowly track ordinary within-regime changes in
                    # predictive difficulty while W is still small.
                    # Once W becomes meaningfully positive, freeze the
                    # baseline so it cannot learn away a true regime change.
                    if (
                        self.change_score
                        < self.baseline_freeze_score
                        and mean_block_weight
                        >= self.baseline_min_weight
                        and np.isfinite(weighted_surprise_mean)
                    ):
                        self.surprise_baseline = (
                            (1.0 - self.baseline_alpha)
                            * self.surprise_baseline
                            + self.baseline_alpha
                            * weighted_surprise_mean
                        )

                W_for_log = self.change_score

                # NOTE:
                # self.change_score / detector_h are now diagnostic only.
                # regime_change is decided above from standardized local
                # conditional surprise plus contiguous persistence.

                with open(
                    self.detector_block_path,
                    "a",
                ) as f:
                    f.write(
                        f"{self.block_index},"
                        f"{self.steps},"
                        f"{block_E_k:.8f},"
                        f"{self.change_score:.8f},"
                        f"{mean_block_pseudo_count:.8f},"
                        f"{mean_block_count_reference:.8f},"
                        f"{mean_block_count_ratio:.8f},"
                        f"{mean_block_familiarity:.8f},"
                        f"{mean_block_surprise:.8f},"
                        f"{mean_block_conditional_surprise:.8f},"
                        f"{mean_block_conditional_support:.8f},"
                        f"{conditional_valid_fraction:.8f},"
                        f"{conditional_z_for_log:.8f},"
                        f"{conditional_run_length_for_log},"
                        f"{conditional_baseline_n_for_log},"
                        f"{conditional_baseline_mean_for_log:.8f},"
                        f"{conditional_baseline_std_for_log:.8f},"
                        f"{mean_block_conditional_leverage:.8f},"
                        f"{mean_block_conditional_trust_fraction:.8f},"
                        f"{mean_block_gradient_coherence:.8f},"
                        f"{mean_block_gradient_novelty:.8f},"
                        f"{int(regime_change)}\n"
                    )

                self.block_index += 1
                self.block_evidence_sum = 0.0
                self.block_weight_sum = 0.0
                self.block_pseudo_count_sum = 0.0
                self.block_count_reference_sum = 0.0
                self.block_count_ratio_sum = 0.0
                self.block_familiarity_sum = 0.0
                self.block_surprise_sum = 0.0
                self.block_conditional_surprise_sum = 0.0
                self.block_conditional_support_sum = 0.0
                self.block_conditional_valid_count = 0
                self.block_conditional_leverage_sum = 0.0
                self.block_conditional_trust_fraction_sum = 0.0
                self.block_conditional_diagnostic_count = 0
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
                            f"mean_familiarity="
                            f"{mean_block_familiarity:.8f}, "
                            f"mean_surprise="
                            f"{mean_block_surprise:.8f}, "
                            f"mean_conditional_surprise="
                            f"{mean_block_conditional_surprise:.8f}, "
                            f"conditional_valid_fraction="
                            f"{conditional_valid_fraction:.8f}, "
                            f"conditional_z="
                            f"{conditional_z_for_log:.8f}, "
                            f"conditional_run_length="
                            f"{conditional_run_length_for_log}, "
                            f"conditional_baseline_mean="
                            f"{conditional_baseline_mean_for_log:.8f}, "
                            f"conditional_baseline_std="
                            f"{conditional_baseline_std_for_log:.8f}, "
                            f"mean_conditional_leverage="
                            f"{mean_block_conditional_leverage:.8f}, "
                            f"mean_conditional_trust_fraction="
                            f"{mean_block_conditional_trust_fraction:.8f}, "
                            f"mean_gradient_coherence="
                            f"{mean_block_gradient_coherence:.8f}, "
                            f"mean_gradient_novelty="
                            f"{mean_block_gradient_novelty:.8f}\n"
                        )
                    # -------------------------------------------------
                    # Start a fresh regime-local detector model.
                    # -------------------------------------------------

                    # Discard old-regime predictive knowledge and Adam state.
                    # The actor, critic, and their optimizers are untouched.
                    self._reset_predictor_ensemble()

                    # Forget old-regime state-action visitation while keeping
                    # the fixed LSH projections and frozen normalization.
                    self.state_action_familiarity.reset_counts()
                    self.conditional_outcome_tracker.reset()

                    # Count the alarm transition once as the first observed
                    # state-action input of the newly detected regime.
                    # Its returned familiarity is intentionally ignored.
                    if self.state_action_familiarity.ready:
                        self.state_action_familiarity.observe(
                            state_action_np
                        )

                        if self.predictor_target_normalizer.ready:
                            # Seed the new regime's historical conditional
                            # anchor with this transition.  No score is used.
                            self.conditional_outcome_tracker.observe(
                                state_action_np,
                                conditional_target_np,
                            )

                    # Reset sequential detector state.
                    self.change_score = 0.0
                    self._reset_conditional_decision_state()

                    # The old regime's nominal surprise level is no longer
                    # the appropriate reference.
                    self.surprise_baseline = None

                    # Restart multi-block baseline initialization.
                    self.baseline_init_sum = 0.0
                    self.baseline_init_count = 0

                    # No fixed post-change warmup.  Immediately continue
                    # streaming; low familiarity naturally suppresses
                    # evidence until the fresh predictor has learned enough.
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
                    f"{familiarity_count_mean:.8f},"
                    f"{familiarity_count_ratio:.8f},"
                    f"{familiarity:.8f},"
                    f"{e_t:.8f},"
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
                    f"{predictor_nll:.8f},"
                    f"{conditional_surprise_raw:.8f},"
                    f"{conditional_surprise_clipped:.8f},"
                    f"{conditional_historical_count:.8f},"
                    f"{conditional_tables_used},"
                    f"{conditional_median_leverage:.8f},"
                    f"{conditional_trust_fraction:.8f}\n"
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
            "raw_residual_mse": raw_residual_mse,
            "raw_delta_state_mse": raw_delta_state_mse,
            "raw_reward_sq_error": raw_reward_sq_error,
            "mean_total_var_raw": mean_total_var_raw,
            "mean_epistemic_var_raw": mean_epistemic_var_raw,
            "gradient_coherence": gradient_coherence,
            "gradient_novelty": gradient_novelty,
            "predictor_nll": predictor_nll,
            "conditional_surprise_raw": conditional_surprise_raw,
            "conditional_surprise_clipped": conditional_surprise_clipped,
            "conditional_historical_count": conditional_historical_count,
            "conditional_tables_used": conditional_tables_used,
            "conditional_median_leverage": conditional_median_leverage,
            "conditional_trust_fraction": conditional_trust_fraction,
            "conditional_z": conditional_z_for_log,
            "conditional_run_length": conditional_run_length_for_log,
            "conditional_baseline_n": conditional_baseline_n_for_log,
            "conditional_baseline_mean": conditional_baseline_mean_for_log,
            "conditional_baseline_std": conditional_baseline_std_for_log,
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
                "conditional_baseline_n": self.conditional_baseline_n,
                "conditional_baseline_mean": self.conditional_baseline_mean,
                "conditional_baseline_M2": self.conditional_baseline_M2,
                "conditional_exceedance_run": (
                    self.conditional_exceedance_run
                ),
                "surprise_baseline": self.surprise_baseline,
                "baseline_init_sum": self.baseline_init_sum,
                "baseline_init_count": self.baseline_init_count,
                "block_evidence_sum": self.block_evidence_sum,
                "block_weight_sum": self.block_weight_sum,
                "block_familiarity_sum": self.block_familiarity_sum,
                "block_surprise_sum": self.block_surprise_sum,
                "block_conditional_surprise_sum": (
                    self.block_conditional_surprise_sum
                ),
                "block_conditional_support_sum": (
                    self.block_conditional_support_sum
                ),
                "block_conditional_valid_count": (
                    self.block_conditional_valid_count
                ),
                "block_conditional_leverage_sum": (
                    self.block_conditional_leverage_sum
                ),
                "block_conditional_trust_fraction_sum": (
                    self.block_conditional_trust_fraction_sum
                ),
                "block_conditional_diagnostic_count": (
                    self.block_conditional_diagnostic_count
                ),
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
                "conditional_outcome_tracker": (
                    self.conditional_outcome_tracker.state_dict()
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
        f"_predreset-delta-norm-relcount-localbucket-trust-zpersist"
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
    # Surprise / familiarity detector parameters
    # =====================================================

    parser.add_argument(
        "--detector_h",
        default=1.3,
        type=float,
        help=(
            "Legacy NN-surprise CUSUM threshold retained for diagnostics; "
            "it no longer triggers regime-change decisions"
        ),
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
        "--baseline_init_blocks",
        default=20,
        type=int,
        help=(
            "Number of eligible blocks used to initialize "
            "the regime-local surprise baseline"
        ),
    )

    parser.add_argument(
        "--baseline_freeze_score",
        default=0.5,
        type=float,
        help=(
            "Freeze surprise-baseline adaptation once "
            "CUSUM W_k reaches this level"
        ),
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
        "--familiarity_relative_scale",
        default=0.5,
        type=float,
        help=(
            "Lambda in relative familiarity: "
            "F=1-exp(-((c/u-1)/lambda)^p) for c/u>1"
        ),
    )

    parser.add_argument(
        "--familiarity_relative_power",
        default=2.0,
        type=float,
        help=(
            "Shape exponent p in relative familiarity; larger values "
            "make the knee around c/u=1 sharper"
        ),
    )

    parser.add_argument(
        "--familiarity_input_clip",
        default=5.0,
        type=float,
    )

    # =====================================================
    # Historical conditional outcome anchor
    # =====================================================

    parser.add_argument(
        "--conditional_min_bucket_count",
        default=20,
        type=int,
        help=(
            "Local-model support threshold. Each bucket first trains its "
            "affine RLS model for this many samples, then collects this many "
            "pre-update residuals before it may emit conditional surprise"
        ),
    )

    parser.add_argument(
        "--conditional_min_tables",
        default=3,
        type=int,
        help=(
            "Minimum number of LSH tables with sufficient historical "
            "outcome support required to emit a conditional score"
        ),
    )

    parser.add_argument(
        "--conditional_variance_floor",
        default=1e-3,
        type=float,
        help=(
            "Per-dimension variance floor for the historical conditional "
            "outcome score in frozen normalized target coordinates"
        ),
    )

    parser.add_argument(
        "--conditional_surprise_cap",
        default=20.0,
        type=float,
        help=(
            "Clip local conditional surprise only for block averaging; "
            "the raw transition score is also logged"
        ),
    )

    parser.add_argument(
        "--conditional_ridge",
        default=1.0,
        type=float,
        help=(
            "Ridge prior strength for each bucket-local recursive "
            "least-squares affine model"
        ),
    )

    parser.add_argument(
        "--conditional_max_leverage",
        default=1.0,
        type=float,
        help=(
            "Maximum pre-update RLS leverage phi^T P phi for a bucket-local "
            "prediction to be trusted as regime-change evidence.  With the "
            "usual linear-model interpretation, leverage <= 1 means the "
            "parameter-uncertainty contribution does not exceed the residual "
            "noise contribution in the one-step predictive variance."
        ),
    )

    # =====================================================
    # Local conditional block decision
    # =====================================================

    parser.add_argument(
        "--conditional_decision_init_blocks",
        default=30,
        type=int,
        help=(
            "Eligible local-conditional blocks used to initialize the "
            "per-regime mean/std before change decisions begin"
        ),
    )

    parser.add_argument(
        "--conditional_decision_min_valid_fraction",
        default=0.5,
        type=float,
        help=(
            "Minimum fraction of transitions in a block with TRUSTED local "
            "conditional scores before that block may affect the decision"
        ),
    )

    parser.add_argument(
        "--conditional_decision_z_threshold",
        default=3.0,
        type=float,
        help=(
            "One-sided standardized local-conditional block threshold. "
            "Heuristic default; validate across seeds rather than treating "
            "it as a calibrated false-alarm probability"
        ),
    )

    parser.add_argument(
        "--conditional_decision_persistence",
        default=3,
        type=int,
        help=(
            "Number of consecutive eligible blocks above the Z threshold "
            "required to declare a regime change"
        ),
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

    if args.conditional_max_leverage <= 0.0:
        raise ValueError(
            "conditional_max_leverage must be > 0"
        )

    if args.conditional_decision_init_blocks < 2:
        raise ValueError(
            "conditional_decision_init_blocks must be >= 2"
        )

    if not (
        0.0
        <= args.conditional_decision_min_valid_fraction
        <= 1.0
    ):
        raise ValueError(
            "conditional_decision_min_valid_fraction must be in [0, 1]"
        )

    if args.conditional_decision_z_threshold <= 0.0:
        raise ValueError(
            "conditional_decision_z_threshold must be > 0"
        )

    if args.conditional_decision_persistence < 1:
        raise ValueError(
            "conditional_decision_persistence must be >= 1"
        )

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
            f"_aba_joint_surprise_familiarity_gradient_diagnostics"
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