"""Minimum-Delay Adaptation Half-Cheetah non-stationarity, recurring: A -> B -> A.

Same MBCD-style environments and reward definitions as run_min_delay_ab.py
(reused through the shared ``min_delay_regime_spec`` helper), but regime A
returns after B: the schedule is A [0, 5M), B [5M, 10M), A [10M, 15M).

One single continuous AVG learner trains for the entire 15M steps: no
network, optimizer, normalization, or TD-scaler resets at either switch, no
new learner, no replay, and the regime ID is never given to the agent.  This
experiment measures forgetting when a previously encountered regime returns.

Shift types (same as run_min_delay_ab.py):

    wind            - B adds the released code's -4.0 N x-wind
    actuator        - B sign-flips one configurable actuator
    target_velocity - B raises the target velocity 1.5 -> 2.0
    combined        - B applies all three simultaneously

When B -> A at 10M, all B perturbations are removed and the environment
exactly returns to the original A regime (pristine physics restored from the
wrapper's base snapshot; the same RegimeSpec instance is used for both A
segments).

Usage:
    python3 run_min_delay_aba.py --shift-type wind
    python3 run_min_delay_aba.py --shift-type actuator --actuator-index 3
    python3 run_min_delay_aba.py --shift-type target_velocity --sample-target-velocity
    python3 run_min_delay_aba.py --shift-type combined --smoke-test
"""

import numpy as np

from incremental_rl.envs.nonstationary_half_cheetah import schedule_from_steps
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run
from run_min_delay_ab import (
    ACTUATOR_NAMES,
    DEFAULT_ACTUATOR_INDEX,
    DEFAULT_TARGET_VELOCITY_A,
    DEFAULT_TARGET_VELOCITY_B,
    DEFAULT_WIND_FORCE,
    SHIFT_TYPES,
    TARGET_VELOCITY_SAMPLE_RANGE,
    _B_SHIFT_REGIME,
    _target_velocity_b,
    min_delay_regime_spec,
)

DEFAULT_TOTAL_STEPS = 15_000_000
DEFAULT_SWITCH_STEPS = (5_000_000, 10_000_000)


def make_schedule(args):
    """Build the A -> B -> A schedule for the selected shift type."""

    # The SAME RegimeSpec instance is used for both A segments, so the
    # returned A is identical to the original A by construction.
    regime_a = min_delay_regime_spec("A", target_velocity_a=args.target_velocity_a)
    regime_b = min_delay_regime_spec(
        _B_SHIFT_REGIME[args.shift_type],
        as_name="B",
        wind_force=args.wind_force,
        actuator_index=args.actuator_index,
        target_velocity_a=args.target_velocity_a,
        target_velocity_b=_target_velocity_b(args),
    )
    return (
        schedule_from_steps(
            args.total_steps,
            [
                {"start_step": 0, "regime": regime_a},
                {"start_step": args.switch_step_1, "regime": regime_b},
                {"start_step": args.switch_step_2, "regime": regime_a},
            ],
        ),
        regime_a,
        regime_b,
    )


def _smoke_verify(args, regime_a, regime_b) -> None:
    """Numerically verify the A -> B -> A boundaries and perturbation removal."""

    import gymnasium as gym

    from incremental_rl.envs.nonstationary_half_cheetah import NonStationaryHalfCheetah

    switch_1, switch_2 = args.switch_step_1, args.switch_step_2
    # (6) The returned A is the same spec as the original A.
    assert args.schedule.regimes[0] == args.schedule.regimes[2], (
        "the two A segments use different regime specs"
    )
    assert regime_a == min_delay_regime_spec(
        "A", target_velocity_a=args.target_velocity_a
    )
    # B matches the corresponding A -> B experiment's regime B.  The
    # reference target velocity is regime_b's own (sampled or configured),
    # so --sample-target-velocity comparisons also hold.
    assert regime_b == min_delay_regime_spec(
        _B_SHIFT_REGIME[args.shift_type],
        as_name="B",
        wind_force=args.wind_force,
        actuator_index=args.actuator_index,
        target_velocity_a=args.target_velocity_a,
        target_velocity_b=regime_b.target_velocity,
    )

    check_wind = args.shift_type in ("wind", "combined")
    check_actuator = args.shift_type in ("actuator", "combined")
    check_target = args.shift_type in ("target_velocity", "combined")
    flips = regime_b.action_sign_flips

    env = NonStationaryHalfCheetah(gym.make(args.env_id), schedule=args.schedule)
    try:
        env.reset(seed=7)
        action = np.full((6,), 0.3)
        for t in range(args.total_steps):
            _, reward, _, _, info = env.step(action)
            in_b = switch_1 <= t < switch_2

            # (1)(2)(3) Exact regime identity and boundary timing.
            assert info["regime_id"] == ("B" if in_b else "A"), (
                f"wrong regime at t={t}: {info['regime_id']}"
            )
            assert bool(info["context_changed"]) == (t in (switch_1, switch_2)), (
                f"context_changed wrong at t={t}: {info['context_changed']}"
            )

            if not in_b:
                # (4)(5) Outside B: wind absent, no flip, target velocity A.
                assert np.isclose(info["wind_force_x"], 0.0), f"wind at t={t}"
                assert np.allclose(
                    env.unwrapped.data.xfrc_applied, 0.0, atol=1e-12
                ), f"external forces at t={t}"
                assert np.allclose(
                    info["executed_action"], action, atol=1e-12
                ), f"action modified at t={t}"
                assert np.isclose(
                    info["target_velocity"], args.target_velocity_a
                ), f"target velocity wrong at t={t}"
                expected = -abs(
                    info["x_velocity"] - args.target_velocity_a
                ) - 0.1 * float(np.square(info["executed_action"]).sum())
                assert np.isclose(reward, expected, atol=1e-6), (
                    f"reward != A tracking formula at t={t}"
                )
            else:
                # (4) During B: exactly the configured perturbations.
                if check_wind:
                    assert np.isclose(
                        info["wind_force_x"], args.wind_force
                    ), f"wind force wrong at t={t}"
                    assert np.allclose(
                        env.unwrapped.data.xfrc_applied[1:, 0],
                        args.wind_force,
                        atol=1e-12,
                    ), f"xfrc_applied not set at t={t}"
                if check_actuator:
                    executed = info["executed_action"]
                    for index in flips:
                        assert np.isclose(
                            executed[index], -action[index]
                        ), f"actuator {index} not flipped at t={t}"
                    for index in [i for i in range(6) if i not in flips]:
                        assert np.isclose(
                            executed[index], action[index]
                        ), f"actuator {index} wrongly flipped at t={t}"
                if check_target:
                    assert np.isclose(
                        info["target_velocity"], regime_b.target_velocity
                    ), f"target velocity wrong at t={t}"
                    expected = -abs(
                        info["x_velocity"] - regime_b.target_velocity
                    ) - 0.1 * float(np.square(info["executed_action"]).sum())
                    assert np.isclose(reward, expected, atol=1e-6), (
                        f"reward != B tracking formula at t={t}"
                    )
    finally:
        env.close()

    print(
        f"PASS smoke env: A active on [0, {switch_1}) and [{switch_2}, "
        f"{args.total_steps}) with no wind, no flips, target "
        f"{args.target_velocity_a}; B active exactly on [{switch_1}, "
        f"{switch_2}) with the configured perturbations; context changes "
        f"occur only at steps {switch_1} and {switch_2}.",
        flush=True,
    )


def _smoke_verify_continuity(output_dir, total_steps) -> None:
    """After the short run: no resets, no new learners, one continuous policy."""

    import csv

    import torch

    with open(output_dir / "switches.csv", newline="", encoding="utf-8") as stream:
        switches = list(csv.DictReader(stream))
    # Initial row plus exactly two boundary rows, both preserving all state.
    assert len(switches) == 3, f"expected 2 switch events, got {switches}"
    first, second = switches[1], switches[2]
    assert (first["from_regime"], first["to_regime"]) == ("A", "B")
    assert (second["from_regime"], second["to_regime"]) == ("B", "A")
    for row in (first, second):
        assert row["intervention"] == "preserve_all_state", (
            f"unexpected intervention at step {row['step']}: {row['intervention']}"
        )

    final = torch.load(output_dir / "final.pt", weights_only=False)
    manager_state = final["manager"]
    # (7) One continuous AVG learner; (8) nothing reset at either switch.
    assert manager_state["baseline"] == "continuous_avg"
    assert set(manager_state["bundles"]) == {"shared"}, (
        "more than one learner bundle: a new learner was created at a switch"
    )
    agent_steps = manager_state["bundles"]["shared"]["agent"]["steps"]
    assert agent_steps == total_steps, (
        f"learner steps {agent_steps} != total {total_steps}: training restarted"
    )

    print(
        f"PASS smoke continuity: single shared learner, 'preserve_all_state' at "
        f"both switches, {agent_steps} updates over {total_steps} steps -- no "
        f"network/optimizer/normalization reset at A -> B or B -> A.",
        flush=True,
    )


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument(
        "--shift-type", choices=SHIFT_TYPES, required=True,
        help="Which single non-stationarity source to apply in regime B",
    )
    parser.add_argument("--switch-step-1", type=int, default=DEFAULT_SWITCH_STEPS[0])
    parser.add_argument("--switch-step-2", type=int, default=DEFAULT_SWITCH_STEPS[1])
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
            "Run a short (450-step) A -> B -> A experiment and numerically "
            "verify the boundaries, perturbation removal, and learner continuity"
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
    args.experiment_name = f"min_delay_{args.shift_type}_aba"

    smoke_total = 0
    if args.smoke_test:
        smoke_total = 450
        args.total_steps = smoke_total
        args.switch_step_1 = 150
        args.switch_step_2 = 300
        args.eval_interval = 0
        args.progress_interval = 0
        args.diagnostic_interval = 10

    print("MIN-DELAY ABA EXPERIMENT", flush=True)
    print(f"SHIFT TYPE: {args.shift_type}", flush=True)
    print(f"A -> B at step {args.switch_step_1}", flush=True)
    print(f"B -> A at step {args.switch_step_2}", flush=True)
    print(f"TOTAL STEPS: {args.total_steps}", flush=True)
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
