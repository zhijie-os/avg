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
# Local frozen predictor detector
# =========================================================

class VectorRunningStats:
    """Per-dimension Welford statistics used only for warm-up normalization."""

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

        return np.sqrt(
            np.maximum(
                var,
                1e-8,
            )
        )


class EMAErrorStats:
    """
    Fixed-memory estimate of the current prediction-error distribution.

    The detector cares about predictor competence, not visitation frequency.
    These statistics track how wrong a local predictor normally is.
    """

    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.var = 0.0

    def update(self, x):
        x = float(x)

        if self.n == 0:
            self.mean = x
            self.var = 0.0
            self.n = 1
            return

        old_mean = self.mean
        delta = x - old_mean

        self.mean = (
            old_mean
            + self.alpha * delta
        )

        # Exponentially weighted analogue of a variance update.
        self.var = (
            (1.0 - self.alpha)
            * (
                self.var
                + self.alpha
                * delta * delta
            )
        )

        self.n += 1

    @property
    def std(self):
        return max(
            self.var,
            0.0,
        ) ** 0.5


class LocalBucketState:
    """
    One non-interfering local affine predictor.

    W/P are the active predictor.
    shadow_W/shadow_P adapt only while this trusted predictor is surprising.
    """

    def __init__(
        self,
        feature_dim,
        target_dim,
        ridge,
        residual_alpha,
    ):
        self.feature_dim = int(feature_dim)
        self.target_dim = int(target_dim)

        self.W = np.zeros(
            (
                self.feature_dim,
                self.target_dim,
            ),
            dtype=np.float64,
        )

        self.P = (
            np.eye(
                self.feature_dim,
                dtype=np.float64,
            )
            / float(ridge)
        )

        self.error_stats = EMAErrorStats(
            residual_alpha
        )

        self.learning_samples = 0
        self.trusted = False

        # Normalized state-action point that earned trust for this bucket.
        # Used only as a diagnostic anchor; it does not affect detection.
        self.trust_x = None

        self.shadow_W = None
        self.shadow_P = None

    def clear_shadow(self):
        self.shadow_W = None
        self.shadow_P = None

    def reset_trust(self):
        self.trusted = False
        self.learning_samples = 0
        self.error_stats.reset()
        self.trust_x = None
        self.clear_shadow()


class LocalFrozenPredictorDetector:
    """
    Detector built around one simple idea:

        only use a prediction error as change evidence after
        that local predictor has demonstrated low error there.

    State-action space is partitioned by one fixed SimHash table.
    Every bucket owns its own affine RLS model, so an update in one
    bucket cannot interfere with any other bucket.

    Once a bucket is trusted, its active predictor is frozen.
    A surprising trusted transition updates a SHADOW copy instead.
    If the global CUSUM falls back to zero, all shadows are discarded.
    If a regime change is confirmed, shadows are promoted and every
    bucket's prediction-error baseline is reset to "unknown".
    """

    def __init__(
        self,
        input_dim,
        target_dim,
        num_buckets=512,
        ridge=1.0,
        trust_error_delta=0.01,
        trust_min_samples=20,
        residual_alpha=0.05,
        residual_std_floor=1e-3,
        detector_delta=3.0,
        detector_h=15.0,
        detector_h_warn=5.0,
        detector_increment_clip=5.0,
        hash_seed=12345,
    ):
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)

        self.num_buckets = int(num_buckets)
        self.ridge = float(ridge)

        self.trust_error_delta = float(
            trust_error_delta
        )
        self.trust_min_samples = int(
            trust_min_samples
        )

        self.residual_alpha = float(
            residual_alpha
        )
        self.residual_std_floor = float(
            residual_std_floor
        )

        self.detector_delta = float(
            detector_delta
        )
        self.detector_h = float(
            detector_h
        )
        self.detector_h_warn = float(
            detector_h_warn
        )
        self.detector_increment_clip = float(
            detector_increment_clip
        )

        if self.num_buckets < 2:
            raise ValueError(
                "detector_num_buckets must be >= 2"
            )

        if self.ridge <= 0:
            raise ValueError(
                "detector_ridge must be > 0"
            )

        if self.trust_error_delta <= 0:
            raise ValueError(
                "trust_error_delta must be > 0"
            )

        if self.trust_min_samples < 2:
            raise ValueError(
                "trust_min_samples must be >= 2"
            )

        if not (0.0 < self.residual_alpha <= 1.0):
            raise ValueError(
                "residual_alpha must be in (0, 1]"
            )

        if self.residual_std_floor <= 0:
            raise ValueError(
                "residual_std_floor must be > 0"
            )

        self.x_stats = VectorRunningStats(
            self.input_dim
        )
        self.y_stats = VectorRunningStats(
            self.target_dim
        )

        self.normalizers_frozen = False
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.hash_bits = int(
            np.ceil(
                np.log2(
                    self.num_buckets
                )
            )
        )

        rng = np.random.default_rng(
            int(hash_seed)
        )

        self.hash_projection = rng.standard_normal(
            (
                self.hash_bits,
                self.input_dim,
            )
        )

        # Local affine feature = normalized x plus a bias.
        self.feature_dim = (
            self.input_dim
            + 1
        )

        # At most num_buckets states can exist.
        self.buckets = {}

        self.change_score = 0.0

    def observe_warmup(
        self,
        x,
        y,
    ):
        if self.normalizers_frozen:
            return

        self.x_stats.update(x)
        self.y_stats.update(y)

    def freeze_normalizers(self):
        if self.normalizers_frozen:
            return

        self.x_mean = self.x_stats.mean.copy()
        self.x_std = self.x_stats.std.copy()

        self.y_mean = self.y_stats.mean.copy()
        self.y_std = self.y_stats.std.copy()

        self.x_std = np.maximum(
            self.x_std,
            1e-4,
        )

        self.y_std = np.maximum(
            self.y_std,
            1e-4,
        )

        self.normalizers_frozen = True

    def _normalize_x(self, x):
        z = (
            np.asarray(
                x,
                dtype=np.float64,
            ).reshape(-1)
            - self.x_mean
        ) / self.x_std

        return np.clip(
            z,
            -5.0,
            5.0,
        )

    def _normalize_y(self, y):
        z = (
            np.asarray(
                y,
                dtype=np.float64,
            ).reshape(-1)
            - self.y_mean
        ) / self.y_std

        return np.clip(
            z,
            -10.0,
            10.0,
        )

    def _bucket_id(self, x_norm):
        projected = (
            self.hash_projection
            @ x_norm
        )

        bits = projected >= 0.0

        signature = 0

        for bit in bits:
            signature = (
                (signature << 1)
                | int(bit)
            )

        return (
            signature
            % self.num_buckets
        )

    def _get_bucket(self, bucket_id):
        bucket = self.buckets.get(
            bucket_id
        )

        if bucket is None:
            bucket = LocalBucketState(
                feature_dim=self.feature_dim,
                target_dim=self.target_dim,
                ridge=self.ridge,
                residual_alpha=self.residual_alpha,
            )

            self.buckets[bucket_id] = bucket

        return bucket

    @staticmethod
    def _predict(
        W,
        phi,
    ):
        return (
            phi
            @ W
        )

    @staticmethod
    def _rls_update(
        W,
        P,
        phi,
        target,
    ):
        """
        One recursive least-squares update.

        This uses only the current transition.
        """
        P_phi = (
            P
            @ phi
        )

        denom = (
            1.0
            + float(
                phi
                @ P_phi
            )
        )

        K = (
            P_phi
            / max(
                denom,
                1e-12,
            )
        )

        pred = (
            phi
            @ W
        )

        residual = (
            target
            - pred
        )

        W_new = (
            W
            + np.outer(
                K,
                residual,
            )
        )

        phi_T_P = (
            phi
            @ P
        )

        P_new = (
            P
            - np.outer(
                K,
                phi_T_P,
            )
        )

        # Numerical symmetry.
        P_new = 0.5 * (
            P_new
            + P_new.T
        )

        return W_new, P_new

    def _start_or_update_shadow(
        self,
        bucket,
        phi,
        target,
    ):
        if bucket.shadow_W is None:
            bucket.shadow_W = (
                bucket.W.copy()
            )

            bucket.shadow_P = (
                bucket.P.copy()
            )

        (
            bucket.shadow_W,
            bucket.shadow_P,
        ) = self._rls_update(
            bucket.shadow_W,
            bucket.shadow_P,
            phi,
            target,
        )

    def _discard_all_shadows(self):
        for bucket in self.buckets.values():
            bucket.clear_shadow()

    def _promote_shadows_and_reset_trust(self):
        for bucket in self.buckets.values():

            if bucket.shadow_W is not None:
                bucket.W = (
                    bucket.shadow_W
                )

                bucket.P = (
                    bucket.shadow_P
                )

            # Regardless of whether this bucket had direct evidence,
            # the old residual distribution no longer certifies the
            # predictor under the newly detected regime.
            bucket.reset_trust()
            
    def process(
        self,
        x,
        y,
    ):
        if not self.normalizers_frozen:
            raise RuntimeError(
                "Detector normalizers must be frozen before process()."
            )

        x_norm = self._normalize_x(x)
        y_norm = self._normalize_y(y)

        bucket_id = self._bucket_id(
            x_norm
        )

        bucket = self._get_bucket(
            bucket_id
        )

        phi = np.concatenate(
            (
                x_norm,
                np.array(
                    [1.0],
                    dtype=np.float64,
                ),
            )
        )

        # -------------------------------------------------
        # Predict BEFORE any update.
        # -------------------------------------------------

        pred = self._predict(
            bucket.W,
            phi,
        )

        error = float(
            np.mean(
                (
                    y_norm
                    - pred
                ) ** 2
            )
        )

        baseline_mean = (
            bucket.error_stats.mean
            if bucket.error_stats.n > 0
            else float("nan")
        )

        baseline_std = (
            bucket.error_stats.std
            if bucket.error_stats.n > 0
            else float("nan")
        )

        z_score = float("nan")
        L_t = 0.0
        warning = False
        regime_change = False
        became_trusted = False

        # RMS distance in frozen-normalized x-space from the point that
        # originally earned trust for this bucket. Diagnostic only.
        trust_distance = float("nan")

        shadow_active = (
            bucket.shadow_W is not None
        )

        W_before = self.change_score
        W_for_log = self.change_score

        # =================================================
        # 1. LEARNING STATE
        # =================================================

        if not bucket.trusted:

            bucket.learning_samples += 1

            # -------------------------------------------------
            # The CURRENT PRE-UPDATE prediction is already good.
            #
            # Freeze exactly this predictor. Do NOT update it
            # using the current transition.
            # -------------------------------------------------

            if error <= self.trust_error_delta:

                bucket.trusted = True
                became_trusted = True

                # Store the exact normalized state-action point that earned
                # trust. This is diagnostic only and does not gate detection.
                bucket.trust_x = x_norm.copy()
                trust_distance = 0.0

                # Throw away errors collected while the model
                # was still learning. They do not describe the
                # now-frozen predictor.
                bucket.error_stats.reset()

                # The current successful prediction becomes the
                # first residual observation for the frozen model.
                bucket.error_stats.update(
                    error
                )

            else:

                # -------------------------------------------------
                # Predictor is not accurate enough yet.
                #
                # Record its PRE-UPDATE error, then continue
                # learning using this transition.
                # -------------------------------------------------

                bucket.error_stats.update(
                    error
                )

                (
                    bucket.W,
                    bucket.P,
                ) = self._rls_update(
                    bucket.W,
                    bucket.P,
                    phi,
                    y_norm,
                )

            baseline_mean = (
                bucket.error_stats.mean
            )

            baseline_std = (
                bucket.error_stats.std
            )

        # =================================================
        # 2. TRUSTED / FROZEN STATE
        # =================================================

        else:

            # Diagnostic only: how far is the current normalized state-action
            # from the normalized point that originally earned trust?
            if bucket.trust_x is not None:
                trust_distance = float(
                    np.linalg.norm(x_norm - bucket.trust_x)
                    / np.sqrt(self.input_dim)
                )

            # -------------------------------------------------
            # Active predictor stays frozen.
            #
            # Therefore its old demonstrated accuracy remains
            # meaningful unless the environment changes.
            # -------------------------------------------------

            baseline_mean = (
                bucket.error_stats.mean
            )

            baseline_std = max(
                bucket.error_stats.std,
                self.residual_std_floor,
            )

            # Current error relative to what this frozen
            # predictor normally produced.
            z_score = (
                error
                - baseline_mean
            ) / baseline_std

            L_t_raw = (
                z_score
                - self.detector_delta
            )

            L_t = float(
                np.clip(
                    L_t_raw,
                    -self.detector_increment_clip,
                    self.detector_increment_clip,
                )
            )

            # -------------------------------------------------
            # Sequential accumulation of surprise.
            # -------------------------------------------------

            self.change_score = max(
                0.0,
                self.change_score
                + L_t,
            )

            W_for_log = self.change_score

            # -------------------------------------------------
            # Suspicious transition.
            #
            # Do NOT modify the trusted reference predictor.
            # Adapt a shadow copy instead.
            # -------------------------------------------------

            if L_t > 0.0:

                self._start_or_update_shadow(
                    bucket,
                    phi,
                    y_norm,
                )

                shadow_active = True

            # -------------------------------------------------
            # Normal transition.
            #
            # Predictor remains frozen, but refine the scalar
            # residual distribution for this frozen predictor.
            # -------------------------------------------------

            else:

                bucket.error_stats.update(
                    error
                )

                baseline_mean = (
                    bucket.error_stats.mean
                )

                baseline_std = max(
                    bucket.error_stats.std,
                    self.residual_std_floor,
                )

            warning = (
                self.change_score
                >= self.detector_h_warn
            )

            # =================================================
            # 3. REGIME CHANGE CONFIRMED
            # =================================================

            if (
                self.change_score
                > self.detector_h
            ):

                regime_change = True

                # Shadows have already been adapting to the
                # suspected new regime.
                #
                # Promote them and invalidate all old trust
                # certificates.
                self._promote_shadows_and_reset_trust()

                shadow_active = False

                self.change_score = 0.0

            # =================================================
            # 4. SURPRISE DIED OUT
            # =================================================

            elif (
                self.change_score == 0.0
                and W_before > 0.0
            ):

                # Evidence disappeared, so provisional shadow
                # adaptation was caused by noise/outliers.
                self._discard_all_shadows()

                shadow_active = False

        # =================================================
        # Logging
        # =================================================

        num_trusted_buckets = sum(
            int(b.trusted)
            for b in self.buckets.values()
        )

        return {
            "bucket_id": bucket_id,
            "bucket_trusted": bucket.trusted,
            "became_trusted": became_trusted,
            "learning_samples": bucket.learning_samples,
            "prediction_error": error,
            "trust_distance": trust_distance,
            "baseline_error_mean": baseline_mean,
            "baseline_error_std": baseline_std,
            "z_score": z_score,
            "L_t": L_t,
            "W_t": W_for_log,
            "warning": warning,
            "shadow_active": shadow_active,
            "regime_change": regime_change,
            "num_trusted_buckets": num_trusted_buckets,
        }
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
            f"{cfg.run_id}_regime_changes_aba_joint_local_frozen.log",
        )

        self.detector_trace_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_detector_trace_aba_joint_local_frozen.csv",
        )

        self.trusted_visit_path = os.path.join(
            cfg.results_dir,
            f"{cfg.run_id}_trusted_visits.csv",
        )

        with open(self.trusted_visit_path, "w") as f:
            f.write(
                "step,bucket_id,prediction_error,trust_distance,"
                "baseline_error_mean,baseline_error_std,"
                "z_score,L_t\n"
            )

        with open(self.detector_trace_path, "w") as f:
            f.write(
                "step,bucket_id,bucket_trusted,became_trusted,"
                "learning_samples,prediction_error,trust_distance,"
                "baseline_error_mean,baseline_error_std,"
                "z_score,L_t,W_t,warning,shadow_active,"
                "warmup_remaining,regime_change,"
                "num_trusted_buckets\n"
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
        # Local non-interfering predictor detector
        # -------------------------------------------------

        self.detector = LocalFrozenPredictorDetector(
            input_dim=(
                cfg.obs_dim
                + cfg.action_dim
            ),
            target_dim=(
                cfg.obs_dim
                + 1
            ),
            num_buckets=cfg.detector_num_buckets,
            ridge=cfg.detector_ridge,
            trust_error_delta=cfg.trust_error_delta,
            trust_min_samples=cfg.trust_min_samples,
            residual_alpha=cfg.detector_residual_alpha,
            residual_std_floor=cfg.detector_residual_std_floor,
            detector_delta=cfg.detector_delta,
            detector_h=cfg.detector_h,
            detector_h_warn=cfg.detector_h_warn,
            detector_increment_clip=cfg.detector_increment_clip,
            hash_seed=cfg.detector_hash_seed,
        )

        self.initial_detector_warmup = (
            cfg.initial_detector_warmup
        )

        self.warmup_remaining = (
            self.initial_detector_warmup
        )

        self.detector_log_interval = (
            cfg.detector_log_interval
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

        # Detector uses RAW flattened observations.
        raw_obs_t = torch.tensor(
            raw_obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        raw_next_obs_t = torch.tensor(
            raw_next_obs.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = action.to(self.device)
        detector_action = (
            action.detach()
            .cpu()
            .view(-1)
            .numpy()
            .astype(np.float64)
        )

        lprob = kwargs["lprob"]

        raw_obs_np = (
            raw_obs_t.detach()
            .cpu()
            .view(-1)
            .numpy()
            .astype(np.float64)
        )

        raw_next_obs_np = (
            raw_next_obs_t.detach()
            .cpu()
            .view(-1)
            .numpy()
            .astype(np.float64)
        )

        # Conditional detector input x = (S_t, A_t).
        detector_x = np.concatenate(
            (
                raw_obs_np,
                detector_action,
            )
        )

        # Predict displacement rather than the identity-heavy next state:
        # y = (Delta S_t, R_{t+1}).
        detector_y = np.concatenate(
            (
                raw_next_obs_np
                - raw_obs_np,
                np.array(
                    [reward],
                    dtype=np.float64,
                ),
            )
        )

        # =================================================
        # 1. Detector
        # =================================================

        if self.warmup_remaining > 0:

            # Warm-up is used ONLY to establish fixed coordinate scales.
            self.detector.observe_warmup(
                detector_x,
                detector_y,
            )

            self.warmup_remaining -= 1

            if self.warmup_remaining == 0:
                self.detector.freeze_normalizers()

            detector_info = {
                "bucket_id": -1,
                "bucket_trusted": False,
                "became_trusted": False,
                "learning_samples": 0,
                "prediction_error": float("nan"),
                "trust_distance": float("nan"),
                "baseline_error_mean": float("nan"),
                "baseline_error_std": float("nan"),
                "z_score": float("nan"),
                "L_t": 0.0,
                "W_t": self.detector.change_score,
                "warning": False,
                "shadow_active": False,
                "regime_change": False,
                "num_trusted_buckets": 0,
            }

        else:

            detector_info = self.detector.process(
                detector_x,
                detector_y,
            )

            if (
                np.isfinite(detector_info["trust_distance"])
                and not detector_info["became_trusted"]
            ):
                with open(
                    self.trusted_visit_path,
                    "a",
                ) as f:
                    f.write(
                        f"{self.steps},"
                        f"{detector_info['bucket_id']},"
                        f"{detector_info['prediction_error']:.8f},"
                        f"{detector_info['trust_distance']:.8f},"
                        f"{detector_info['baseline_error_mean']:.8f},"
                        f"{detector_info['baseline_error_std']:.8f},"
                        f"{detector_info['z_score']:.8f},"
                        f"{detector_info['L_t']:.8f}\n"
                    )

        regime_change = detector_info[
            "regime_change"
        ]

        # =================================================
        # 2. Detector logging
        # =================================================

        if regime_change:

            with open(
                self.regime_log_path,
                "a",
            ) as f:

                f.write(
                    f"step={self.steps}, "
                    f"bucket={detector_info['bucket_id']}, "
                    f"prediction_error="
                    f"{detector_info['prediction_error']:.8f}, "
                    f"trust_distance="
                    f"{detector_info['trust_distance']:.8f}, "
                    f"baseline_mean="
                    f"{detector_info['baseline_error_mean']:.8f}, "
                    f"baseline_std="
                    f"{detector_info['baseline_error_std']:.8f}, "
                    f"z="
                    f"{detector_info['z_score']:.8f}, "
                    f"L_t="
                    f"{detector_info['L_t']:.8f}, "
                    f"W_t="
                    f"{detector_info['W_t']:.8f}\n"
                )

        if self.steps % self.detector_log_interval == 0:

            with open(
                self.detector_trace_path,
                "a",
            ) as f:

                f.write(
                    f"{self.steps},"
                    f"{detector_info['bucket_id']},"
                    f"{int(detector_info['bucket_trusted'])},"
                    f"{int(detector_info['became_trusted'])},"
                    f"{detector_info['learning_samples']},"
                    f"{detector_info['prediction_error']:.8f},"
                    f"{detector_info['trust_distance']:.8f},"
                    f"{detector_info['baseline_error_mean']:.8f},"
                    f"{detector_info['baseline_error_std']:.8f},"
                    f"{detector_info['z_score']:.8f},"
                    f"{detector_info['L_t']:.8f},"
                    f"{detector_info['W_t']:.8f},"
                    f"{int(detector_info['warning'])},"
                    f"{int(detector_info['shadow_active'])},"
                    f"{self.warmup_remaining},"
                    f"{int(regime_change)},"
                    f"{detector_info['num_trusted_buckets']}\n"
                )

        # =================================================
        # 3. Original AVG update
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

        return detector_info

    # =====================================================
    # Save
    # =====================================================

    def save(self, model_dir, unique_str):

        detector_buckets = {}

        for bucket_id, bucket in self.detector.buckets.items():
            detector_buckets[int(bucket_id)] = {
                "W": bucket.W,
                "P": bucket.P,
                "error_n": bucket.error_stats.n,
                "error_mean": bucket.error_stats.mean,
                "error_var": bucket.error_stats.var,
                "learning_samples": bucket.learning_samples,
                "trusted": bucket.trusted,
                "trust_x": bucket.trust_x,
                "shadow_W": bucket.shadow_W,
                "shadow_P": bucket.shadow_P,
            }

        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
            "detector": {
                "normalizers_frozen": self.detector.normalizers_frozen,
                "x_mean": self.detector.x_mean,
                "x_std": self.detector.x_std,
                "y_mean": self.detector.y_mean,
                "y_std": self.detector.y_std,
                "hash_projection": self.detector.hash_projection,
                "change_score": self.detector.change_score,
                "buckets": detector_buckets,
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
        f"_local-frozen"
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
    # Local frozen predictor detector parameters
    # =====================================================

    parser.add_argument(
        "--detector_num_buckets",
        default=8192,
        type=int,
        help="Fixed number of SimHash local predictor buckets",
    )

    parser.add_argument(
        "--detector_ridge",
        default=1.0,
        type=float,
        help="RLS ridge regularization for each local predictor",
    )

    parser.add_argument(
        "--trust_error_delta",
        default=0.01,
        type=float,
        help=(
            "A local predictor becomes trusted once its recent "
            "normalized pre-update MSE is <= this value"
        ),
    )

    parser.add_argument(
        "--trust_min_samples",
        default=20,
        type=int,
        help=(
            "Minimum local prediction-error observations required "
            "before a bucket can become trusted; this is only for "
            "estimating predictor competence, not a familiarity weight"
        ),
    )

    parser.add_argument(
        "--detector_residual_alpha",
        default=0.05,
        type=float,
        help="EMA step size for each bucket's prediction-error distribution",
    )

    parser.add_argument(
        "--detector_residual_std_floor",
        default=1e-3,
        type=float,
        help="Numerical floor for local prediction-error standard deviation",
    )

    parser.add_argument(
        "--detector_delta",
        default=3.0,
        type=float,
        help=(
            "CUSUM reference value in local prediction-error standard deviations"
        ),
    )

    parser.add_argument(
        "--detector_h",
        default=15.0,
        type=float,
        help="CUSUM regime-change threshold",
    )

    parser.add_argument(
        "--detector_h_warn",
        default=5.0,
        type=float,
        help="CUSUM warning threshold used only for logging",
    )

    parser.add_argument(
        "--detector_increment_clip",
        default=5.0,
        type=float,
        help=(
            "Clip each standardized CUSUM increment so one outlier "
            "cannot declare a regime change by itself"
        ),
    )

    parser.add_argument(
        "--detector_hash_seed",
        default=12345,
        type=int,
        help="Fixed seed for the detector's SimHash projections",
    )

    parser.add_argument(
        "--initial_detector_warmup",
        default=10_000,
        type=int,
        help=(
            "Initial steps used only to estimate frozen detector "
            "input/target normalization statistics"
        ),
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
            f"_aba_joint_detector_short"
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