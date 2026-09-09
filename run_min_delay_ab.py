"""Minimum-Delay Adaptation (AAMAS 2021) Half-Cheetah non-stationarities: A -> B.

Reproduces the three sources of non-stationarity from Alegre, Bazzan & da
Silva, "Minimum-Delay Adaptation in Non-Stationary Reinforcement Learning via
Online High-Confidence Change-Point Detection" (arXiv:2105.09452), Section 5
"Half-Cheetah in a Non-Stationary World":

    1. random wind:          "an external latent horizontal force, opposite to
                             the agent's movement direction, is applied"
    2. joint malfunction:    "the torque applied to a joint ... has its
                             polarity/sign changed"
    3. target velocity:      "the target velocity of the robot is sampled from
                             the interval 1.5 to 2.5"

The ``combined`` shift type applies all three perturbations simultaneously in
regime B (wind force, one sign-flipped actuator, and the target-velocity
change); regime A stays the default environment.

Matched against the released implementation (github.com/LucasAlegre/mbcd,
mbcd/envs/non_stationary_wrapper.py):

    * wind   = constant [-4, 0, 0, 0, 0, 0] force vector assigned to
               data.xfrc_applied for every body each step.  Half-Cheetah runs
               toward +x, so -4 N in x opposes movement.  (The paper gives no
               magnitude; the released code uses 4 N.  Our wrapper applies the
               same vector to the robot bodies; the released code also assigns
               it to the world body, which is a physics no-op.)
    * joint malfunction = action mask with -1 entries (released code flips
               actuators 0 and 1 together; here exactly one actuator is
               configurable so each index can be swept independently).
    * target velocity = released code uses fixed 1.5 (default) -> 2.0
               (velocity task); the paper says it is sampled from [1.5, 2.5].
               Defaults here follow the released code; --sample-target-velocity
               reproduces the paper's sampling (seeded Uniform[1.5, 2.5]).

IMPORTANT - reward: the released code replaces the Half-Cheetah reward with
the velocity-tracking reward in EVERY task, including the normal/default one:
reward = -|forward_velocity - target_velocity| - control_cost.  So regime A
here is the paper's default environment (tracking reward, target 1.5), not
the raw Gymnasium Half-Cheetah reward.  This differs deliberately from
run_actuator_ab.py, which uses the unmodified Gymnasium reward.

One single continuous AVG run across the switch: no network, optimizer,
normalization, or scaler resets; the regime ID is never given to the agent.

Usage:
    python3 run_min_delay_ab.py --shift-type wind
    python3 run_min_delay_ab.py --shift-type actuator --actuator-index 3
    python3 run_min_delay_ab.py --shift-type target_velocity --target-velocity-b 2.5
    python3 run_min_delay_ab.py --shift-type target_velocity --sample-target-velocity
    python3 run_min_delay_ab.py --shift-type combined
    python3 run_min_delay_ab.py --shift-type wind --smoke-test
"""

import numpy as np

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEP = 5_000_000

# Perturbation defaults matched to the released mbcd implementation.
DEFAULT_WIND_FORCE = -4.0        # [-4, 0, 0, 0, 0, 0] N (x-axis), opposes +x motion
DEFAULT_ACTUATOR_INDEX = 0      # released code flips indices 0 and 1 together
DEFAULT_TARGET_VELOCITY_A = 1.5  # released code's default_target_vel
DEFAULT_TARGET_VELOCITY_B = 2.0  # released code's velocity task
TARGET_VELOCITY_SAMPLE_RANGE = (1.5, 2.5)  # paper Section 5

SHIFT_TYPES = ("wind", "actuator", "target_velocity", "combined")

# Half-Cheetah actuator order (HalfCheetah-v4 model):
# 0=back thigh, 1=back shin, 2=back foot, 3=front thigh, 4=front shin, 5=front foot
ACTUATOR_NAMES = (
    "back thigh", "back shin", "back foot",
    "front thigh", "front shin", "front foot",
)


def _target_velocity_b(args):
    """Regime B target velocity: released-code default, or the paper's sampling."""
    if args.sample_target_velocity:
        value = float(
            np.random.RandomState(args.seed).uniform(*TARGET_VELOCITY_SAMPLE_RANGE)
        )
        print(
            f"TARGET VELOCITY: B sampled {value:.3f} from "
            f"{TARGET_VELOCITY_SAMPLE_RANGE} (seed {args.seed})",
            flush=True,
        )
        return value
    return args.target_velocity_b


def make_schedule(args):
    """Build the A -> B schedule for the selected shift type."""

    if args.shift_type == "wind":
        regime_a = RegimeSpec(name="A", target_velocity=args.target_velocity_a)
        regime_b = RegimeSpec(
            name="B",
            target_velocity=args.target_velocity_a,  # reward unchanged: only wind changes
            wind_force_x=args.wind_force,
        )
    elif args.shift_type == "actuator":
        regime_a = RegimeSpec(name="A", target_velocity=args.target_velocity_a)
        regime_b = RegimeSpec(
            name="B",
            target_velocity=args.target_velocity_a,  # reward unchanged: only the flip changes
            action_sign_flips=(args.actuator_index,),
        )
    elif args.shift_type == "target_velocity":
        regime_a = RegimeSpec(name="A", target_velocity=args.target_velocity_a)
        regime_b = RegimeSpec(name="B", target_velocity=_target_velocity_b(args))
    elif args.shift_type == "combined":
        regime_a = RegimeSpec(name="A", target_velocity=args.target_velocity_a)
        regime_b = RegimeSpec(
            name="B",
            target_velocity=_target_velocity_b(args),
            wind_force_x=args.wind_force,
            action_sign_flips=(args.actuator_index,),
        )
    else:
        raise ValueError(f"Unknown shift type {args.shift_type!r}")

    return (
        schedule_from_steps(
            args.total_steps,
            [
                {"start_step": 0, "regime": regime_a},
                {"start_step": args.switch_step, "regime": regime_b},
            ],
        ),
        regime_a,
        regime_b,
    )


def _smoke_verify(args, regime_a, regime_b) -> None:
    """Numerically verify baseline, switch timing, and perturbation effect.

    Checks, for the (short) smoke schedule:
    1. regime A dynamics are identical to the plain environment and its
       reward matches the paper's tracking formula at target velocity A;
    2. the perturbation is absent before the switch and present exactly from
       the switch step onward;
    3. regime B differs from a perturbation-off reference in the intended
       dimension only (dynamics for wind/actuator, reward for target
       velocity).
    """

    import gymnasium as gym

    from incremental_rl.envs.nonstationary_half_cheetah import (
        NonStationaryHalfCheetah,
        RegimeSpec,
        schedule_from_steps,
    )

    check_wind = args.shift_type in ("wind", "combined")
    check_actuator = args.shift_type in ("actuator", "combined")
    check_target = args.shift_type in ("target_velocity", "combined")

    # Perturbation-off reference: regime A extended over the whole horizon.
    regime_b_off = RegimeSpec(name="B_off", target_velocity=args.target_velocity_a)

    schedule_on = schedule_from_steps(
        args.total_steps,
        [
            {"start_step": 0, "regime": regime_a},
            {"start_step": args.switch_step, "regime": regime_b},
        ],
    )
    schedule_off = schedule_from_steps(
        args.total_steps,
        [
            {"start_step": 0, "regime": regime_a},
            {"start_step": args.switch_step, "regime": regime_b_off},
        ],
    )

    env_on = NonStationaryHalfCheetah(gym.make(args.env_id), schedule=schedule_on)
    env_off = NonStationaryHalfCheetah(gym.make(args.env_id), schedule=schedule_off)
    plain = gym.make(args.env_id)
    try:
        obs_on, _ = env_on.reset(seed=7)
        obs_off, _ = env_off.reset(seed=7)
        obs_plain, _ = plain.reset(seed=7)
        assert np.allclose(obs_on, obs_plain, atol=1e-10), "regime A reset differs from plain env"

        action = np.full((6,), 0.3)
        b_rewards_on, b_rewards_off = [], []
        for t in range(args.total_steps):
            next_on, reward_on, _, _, info_on = env_on.step(action)
            next_off, reward_off, _, _, info_off = env_off.step(action)

            if t < args.switch_step:
                # (1) Regime A = plain env dynamics, paper tracking reward.
                obs_plain, _, _, _, _ = plain.step(action)
                assert np.allclose(next_on, obs_plain, atol=1e-10), (
                    f"regime A dynamics differ from plain env at t={t}"
                )
                assert np.allclose(next_on, next_off, atol=1e-12)
                assert np.isclose(
                    info_on["x_velocity"],
                    info_off["x_velocity"],
                    atol=1e-10,
                )
                expected = -abs(
                    info_on["x_velocity"] - args.target_velocity_a
                ) - 0.1 * float(np.square(info_on["executed_action"]).sum())
                assert np.isclose(reward_on, expected, atol=1e-6), (
                    f"regime A reward != tracking formula at t={t}"
                )
                assert np.allclose(
                    env_on.unwrapped.data.xfrc_applied, 0.0, atol=1e-12
                ), "unexpected external forces in regime A"
            else:
                # (2)+(3) Perturbation active from the switch step onward.
                assert info_on["context_changed"] is False or t >= args.switch_step
                if check_wind:
                    assert np.isclose(
                        info_on["wind_force_x"], args.wind_force
                    ), f"wind force wrong at t={t}"
                    assert np.allclose(
                        env_on.unwrapped.data.xfrc_applied[1:, 0],
                        args.wind_force,
                        atol=1e-12,
                    ), f"xfrc_applied not set at t={t}"
                if check_actuator:
                    executed = info_on["executed_action"]
                    assert np.isclose(
                        executed[args.actuator_index], -action[args.actuator_index]
                    ), f"actuator {args.actuator_index} not flipped at t={t}"
                    others = [
                        i for i in range(6) if i != args.actuator_index
                    ]
                    assert np.allclose(
                        executed[others], action[others]
                    ), f"non-flipped actuators changed at t={t}"
                if check_target:
                    assert np.isclose(
                        info_on["target_velocity"], regime_b.target_velocity
                    ), f"target velocity wrong at t={t}"
                    expected = -abs(
                        info_on["x_velocity"] - regime_b.target_velocity
                    ) - 0.1 * float(np.square(info_on["executed_action"]).sum())
                    assert np.isclose(reward_on, expected, atol=1e-6), (
                        f"regime B reward != tracking formula at t={t}"
                    )
                    b_rewards_on.append(reward_on)
                    b_rewards_off.append(reward_off)
                if args.shift_type == "target_velocity":
                    # A pure target-velocity change must not alter dynamics.
                    assert np.allclose(next_on, next_off, atol=1e-10), (
                        f"target-velocity shift must not change dynamics at t={t}"
                    )
                else:
                    # Wind and/or actuator flip must alter the dynamics.
                    assert not np.allclose(next_on, next_off, atol=1e-8), (
                        f"{args.shift_type} produced no dynamics change at t={t}"
                    )

        if check_target:
            assert not np.allclose(b_rewards_on, b_rewards_off, atol=1e-6), (
                "target-velocity change produced no reward change in regime B"
            )
    finally:
        env_on.close()
        env_off.close()
        plain.close()

    print(
        f"PASS smoke env: regime A matches plain Half-Cheetah dynamics with "
        f"tracking reward at v={args.target_velocity_a}; perturbation is off "
        f"before step {args.switch_step} and active from step {args.switch_step} on.",
        flush=True,
    )


def _smoke_verify_continuity(output_dir, total_steps) -> None:
    """After the short run: no resets, no new learners, one continuous policy."""

    import csv
    import json

    import torch

    with open(output_dir / "switches.csv", newline="", encoding="utf-8") as stream:
        switches = list(csv.DictReader(stream))
    boundary = next(
        (row for row in switches if row["from_regime"] and row["to_regime"] != ""),
        None,
    )
    assert boundary is not None, "no boundary row in switches.csv"
    assert boundary["from_regime"] == "A" and boundary["to_regime"] == "B"
    assert boundary["intervention"] == "preserve_all_state", (
        f"unexpected intervention at boundary: {boundary['intervention']}"
    )

    final = torch.load(output_dir / "final.pt", weights_only=False)
    manager_state = final["manager"]
    assert manager_state["baseline"] == "continuous_avg"
    assert set(manager_state["bundles"]) == {"shared"}, (
        "more than one learner bundle: the policy was not continuous"
    )
    agent_steps = manager_state["bundles"]["shared"]["agent"]["steps"]
    assert agent_steps == total_steps, (
        f"learner steps {agent_steps} != total {total_steps}: training restarted"
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"

    print(
        "PASS smoke continuity: single shared learner, 'preserve_all_state' at the "
        f"switch, {agent_steps} updates over {total_steps} steps -- the same "
        "policy/network trained continuously across A -> B.",
        flush=True,
    )


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument(
        "--shift-type", choices=SHIFT_TYPES, required=True,
        help="Which single non-stationarity source to apply in regime B",
    )
    parser.add_argument("--switch-step", type=int, default=DEFAULT_SWITCH_STEP)
    parser.add_argument(
        "--wind-force", type=float, default=DEFAULT_WIND_FORCE,
        help="Signed x-axis wind force in N (negative = opposite to +x motion)",
    )
    parser.add_argument(
        "--actuator-index", type=int, default=DEFAULT_ACTUATOR_INDEX,
        help=f"Actuator index negated in regime B (0-5): {ACTUATOR_NAMES}",
    )
    parser.add_argument(
        "--target-velocity-a", type=float, default=DEFAULT_TARGET_VELOCITY_A,
    )
    parser.add_argument(
        "--target-velocity-b", type=float, default=DEFAULT_TARGET_VELOCITY_B,
    )
    parser.add_argument(
        "--sample-target-velocity", action="store_true", default=False,
        help=(
            "Sample regime B's target velocity from "
            f"{TARGET_VELOCITY_SAMPLE_RANGE} (paper Section 5), seeded by --seed"
        ),
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help=(
            "Run a short (300-step) experiment and numerically verify the "
            "baseline, switch timing, perturbation effect, and policy continuity"
        ),
    )
    args = parser.parse_args()

    if args.actuator_index < 0:
        parser.error("--actuator-index must be non-negative")
    if args.sample_target_velocity and args.shift_type not in (
        "target_velocity",
        "combined",
    ):
        parser.error(
            "--sample-target-velocity requires --shift-type "
            "target_velocity or combined"
        )

    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "continuous_avg"
    args.experiment_name = f"min_delay_{args.shift_type}_ab"

    smoke_total = 0
    if args.smoke_test:
        smoke_total = 300
        args.total_steps = smoke_total
        args.switch_step = 150
        args.eval_interval = 0
        args.progress_interval = 0
        args.diagnostic_interval = 10

    print(f"SHIFT TYPE: {args.shift_type}", flush=True)
    print(f"A -> B at step {args.switch_step}", flush=True)
    print(f"total steps {args.total_steps}", flush=True)
    if args.shift_type in ("wind", "combined"):
        print(
            f"WIND FORCE: {args.wind_force} N (x-axis; negative opposes +x motion)",
            flush=True,
        )
    if args.shift_type in ("actuator", "combined"):
        print(
            f"ACTUATOR INDEX: {args.actuator_index} "
            f"({ACTUATOR_NAMES[args.actuator_index]})",
            flush=True,
        )
    if args.shift_type in ("target_velocity", "combined"):
        print(
            f"TARGET VELOCITY: A={args.target_velocity_a}, "
            f"B={args.target_velocity_b}",
            flush=True,
        )

    args.schedule, regime_a, regime_b = make_schedule(args)

    if args.smoke_test:
        _smoke_verify(args, regime_a, regime_b)

    output_dir = run(args)

    if args.smoke_test:
        _smoke_verify_continuity(output_dir, smoke_total)


if __name__ == "__main__":
    main()
