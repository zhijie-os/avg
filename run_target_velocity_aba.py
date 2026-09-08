"""Experiment D: HalfCheetah target velocity v_A -> v_B -> v_A, continuous AVG.

The reward is the target-velocity tracking reward
``-|forward_velocity - v_target| - 0.1 * sum(action^2)``, i.e. the standard
HalfCheetah control-cost term plus an absolute velocity-error term.  Defaults:
v_A = 1.0, v_B = 3.0.

Usage:
    python3 run_target_velocity_aba.py
    python3 run_target_velocity_aba.py --va 1.0 --vb 3.0 --switch-step-1 1000 --switch-step-2 2000 --total-steps 3000
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEPS = (5_000_000, 7_500_000)
DEFAULT_VA = 1.0
DEFAULT_VB = 3.0


def make_schedule(total_steps, switch_steps, va, vb):
    regime_a = RegimeSpec(
        name="A", use_original_reward=False, target_velocity=va
    )
    regime_b = RegimeSpec(
        name="B", use_original_reward=False, target_velocity=vb
    )
    return schedule_from_steps(
        total_steps,
        [
            {"start_step": 0, "regime": regime_a},
            {"start_step": switch_steps[0], "regime": regime_b},
            {"start_step": switch_steps[1], "regime": regime_a},
        ],
    )


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument("--switch-step-1", type=int, default=DEFAULT_SWITCH_STEPS[0])
    parser.add_argument("--switch-step-2", type=int, default=DEFAULT_SWITCH_STEPS[1])
    parser.add_argument("--va", type=float, default=DEFAULT_VA)
    parser.add_argument("--vb", type=float, default=DEFAULT_VB)
    args = parser.parse_args()
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "continuous_avg"
    args.experiment_name = "target_velocity_aba"
    args.schedule = make_schedule(
        args.total_steps,
        (args.switch_step_1, args.switch_step_2),
        args.va,
        args.vb,
    )
    run(args)


if __name__ == "__main__":
    main()
