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
# Environment predictor
#
# F_psi(S_t, A_t)
#     -> (S_hat_{t+1}, R_hat_{t+1})
# =========================================================

class Predictor(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        n_hid=128,
    ):
        super().__init__()

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),

            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.next_obs_head = nn.Linear(
            n_hid,
            obs_dim,
        )

        self.reward_head = nn.Linear(
            n_hid,
            1,
        )

        self.apply(orthogonal_weight_init)
        self.to(device)

    def forward(self, obs, action):

        x = torch.cat(
            (obs, action),
            dim=-1,
        ).to(self.device)

        h = self.net(x)

        pred_next_obs = self.next_obs_head(h)

        pred_reward = (
            self.reward_head(h)
            .squeeze(-1)
        )

        return pred_next_obs, pred_reward


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
            f"{cfg.run_id}_regime_changes.log",
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
        # Predictor
        # -------------------------------------------------

        self.predictor = Predictor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_predictor,
        )

        # -------------------------------------------------
        # Detector statistics
        # -------------------------------------------------

        # Distribution of e_t^P and e_t^R
        # for the currently believed regime.
        self.pred_error_P_stats = RunningStats()
        self.pred_error_R_stats = RunningStats()

        # Used only to put reward error on a reasonable scale.
        # This is NOT reset at a detected regime boundary.
        self.reward_stats = RunningStats()

        self.change_score = 0.0

        # Detector hyperparameters
        self.detector_beta = 0.99
        self.detector_tau = 2.0
        self.detector_h = 3.0

        self.detector_warmup = 1000

        self.steps_since_change = 0

        # -------------------------------------------------
        # Optimizers
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

        self.pred_opt = torch.optim.Adam(
            self.predictor.parameters(),
            lr=cfg.predictor_lr,
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
        **kwargs,
    ):

        obs = torch.Tensor(
            obs.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        next_obs = torch.Tensor(
            next_obs.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        action = action.to(self.device)

        lprob = kwargs["lprob"]

        reward_tensor = torch.tensor(
            [reward],
            dtype=torch.float32,
            device=self.device,
        )

        # =================================================
        # 1. Predictor forward pass
        #
        # IMPORTANT:
        # prediction happens BEFORE updating predictor.
        # =================================================

        pred_next_obs, pred_reward = self.predictor(
            obs,
            action.detach(),
        )

        # =================================================
        # 2. Prediction errors
        # =================================================

        # The environment already uses NormalizeObservation,
        # therefore next_obs is already normalized.
        #
        # e_t^P =
        # 1 / sqrt(d) * ||S_{t+1} - S_hat_{t+1}||_2
        #
        # which is RMSE across observation dimensions.

        e_P = torch.sqrt(
            torch.mean(
                (
                    next_obs
                    - pred_next_obs.detach()
                ) ** 2
            )
        ).item()

        # Reward scale from PREVIOUS rewards only.
        reward_scale = max(
            self.reward_stats.std,
            1e-3,
        )

        # e_t^R
        e_R = torch.abs(
            (
                reward_tensor
                - pred_reward.detach()
            )
            / (reward_scale + 1e-8)
        ).mean().item()

        # =================================================
        # 3. Regime detector
        # =================================================

        # Defaults during warmup
        e_P_bar = 0.0
        e_R_bar = 0.0

        D_t = 0.0
        regime_change = False

        # -------------------------------------------------
        # Warmup / calibration period
        # -------------------------------------------------

        if (
            self.steps_since_change
            < self.detector_warmup
        ):

            self.pred_error_P_stats.update(e_P)
            self.pred_error_R_stats.update(e_R)

        # -------------------------------------------------
        # Normal detector operation
        # -------------------------------------------------

        else:

            # Standardized transition prediction error
            e_P_bar = (
                e_P
                - self.pred_error_P_stats.mean
            ) / (
                self.pred_error_P_stats.std
                + 1e-8
            )

            # Standardized reward prediction error
            e_R_bar = (
                e_R
                - self.pred_error_R_stats.mean
            ) / (
                self.pred_error_R_stats.std
                + 1e-8
            )

            # ---------------------------------------------
            # Instantaneous surprise
            #
            # D_t = 1/2 [
            #   max(0, e_P_bar)^2
            #   +
            #   max(0, e_R_bar)^2
            # ]
            # ---------------------------------------------

            D_t = 0.5 * (
                max(
                    0.0,
                    e_P_bar,
                ) ** 2
                +
                max(
                    0.0,
                    e_R_bar,
                ) ** 2
            )

            # ---------------------------------------------
            # Persistent surprise
            #
            # c_t =
            # beta c_{t-1}
            # +
            # (1-beta) max(0, D_t - tau)
            # ---------------------------------------------

            self.change_score = (
                self.detector_beta
                * self.change_score
                +
                (
                    1.0
                    - self.detector_beta
                )
                * max(
                    0.0,
                    D_t
                    - self.detector_tau,
                )
            )

            # ---------------------------------------------
            # Detect regime boundary
            # ---------------------------------------------

            regime_change = (
                self.change_score
                > self.detector_h
            )

            # ---------------------------------------------
            # Regime change detected
            # ---------------------------------------------

            if regime_change:

                # Write BEFORE resetting c_t.
                with open(
                    self.regime_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"step={self.steps}, "
                        f"D_t={D_t:.6f}, "
                        f"c_t={self.change_score:.6f}, "
                        f"e_P={e_P:.6f}, "
                        f"e_R={e_R:.6f}, "
                        f"e_P_bar={e_P_bar:.6f}, "
                        f"e_R_bar={e_R_bar:.6f}\n"
                    )

                # Reset error reference distribution.
                self.pred_error_P_stats.reset()
                self.pred_error_R_stats.reset()

                # Reset persistent surprise.
                self.change_score = 0.0

                # Start calibration for new regime.
                self.steps_since_change = 0

            # ---------------------------------------------
            # Normal-looking sample
            #
            # Only update reference statistics when the
            # transition is not currently suspicious.
            # ---------------------------------------------

            elif D_t <= self.detector_tau:

                self.pred_error_P_stats.update(e_P)
                self.pred_error_R_stats.update(e_R)

            # If:
            #
            # D_t > tau
            # but
            # c_t <= h
            #
            # we do NOT update the reference statistics.
            # This prevents suspicious samples from
            # immediately contaminating the baseline.

        # =================================================
        # 4. Train predictor
        # =================================================

        state_pred_loss = torch.mean(
            (
                next_obs
                - pred_next_obs
            ) ** 2
        )

        reward_pred_loss = torch.mean(
            (
                (
                    reward_tensor
                    - pred_reward
                )
                / (
                    reward_scale
                    + 1e-8
                )
            ) ** 2
        )

        pred_loss = (
            state_pred_loss
            + reward_pred_loss
        )

        self.pred_opt.zero_grad()

        pred_loss.backward()

        self.pred_opt.step()

        # Reward normalization statistics are updated only
        # AFTER calculating this transition's surprise.
        self.reward_stats.update(reward)

        self.steps_since_change += 1

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

        # Useful if you later want to save / plot
        # detector signals.
        return {
            "e_P": e_P,
            "e_R": e_R,
            "e_P_bar": e_P_bar,
            "e_R_bar": e_R_bar,
            "D_t": D_t,
            "c_t": self.change_score,
            "regime_change": regime_change,
            "predictor_loss": pred_loss.detach().item(),
        }


    # =====================================================
    # Save
    # =====================================================

    def save(self, model_dir, unique_str):

        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),

            "predictor": self.predictor.state_dict(),

            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
            "predictor_opt": self.pred_opt.state_dict(),
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

    env = NormalizeObservation(env)

    env = ClipAction(env)

    base_env = env.unwrapped

    # =====================================================
    # A -> B -> A combined regime
    # =====================================================

    # Control-cost reward
    original_ctrl_cost_weight = (
        base_env._ctrl_cost_weight
    )

    # Joint malfunction
    malfunction_actuator = 0

    original_gear = (
        base_env.model.actuator_gear[
            malfunction_actuator,
            0,
        ].copy()
    )

    # Wind
    torso_id = (
        base_env.model.body("torso").id
    )

    # =====================================================
    # Reproducibility
    # =====================================================

    env.reset(seed=args.seed)

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

    obs, _ = env.reset()

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
            # A -> B -> A combined regime
            # =============================================

            # Always clear external forces first.
            base_env.data.xfrc_applied[:] = 0.0

            # ---------------------------------------------
            # A -> B
            # ---------------------------------------------

            if t == 5_000_000:

                # 1. Reward change:
                # increase control-cost penalty
                base_env._ctrl_cost_weight = 1.0

                # 2. Dynamics change:
                # reverse actuator 0 torque polarity
                base_env.model.actuator_gear[
                    malfunction_actuator,
                    0,
                ] = -original_gear

                print(
                    f"A -> B combined regime "
                    f"at t={t}"
                )

            # ---------------------------------------------
            # B -> A
            # ---------------------------------------------

            elif t == 10_000_000:

                # Restore reward
                base_env._ctrl_cost_weight = (
                    original_ctrl_cost_weight
                )

                # Restore dynamics
                base_env.model.actuator_gear[
                    malfunction_actuator,
                    0,
                ] = original_gear

                print(
                    f"B -> A combined regime "
                    f"at t={t}"
                )

            # ---------------------------------------------
            # Wind only in B
            # ---------------------------------------------

            if (
                5_000_000
                <= t
                < 10_000_000
            ):

                base_env.data.xfrc_applied[
                    torso_id,
                    0,
                ] = -50.0

            # =============================================
            # Environment transition
            # =============================================

            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(sim_action)

            detector_info = agent.update(
                obs,
                action,
                next_obs,
                reward,
                terminated,
                **action_info,
            )

            ret += reward
            step += 1

            obs = next_obs

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

                obs, _ = env.reset()

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
        default=15_001_000,
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
            f"_aba_combined_detector"
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