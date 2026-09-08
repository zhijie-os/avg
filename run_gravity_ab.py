"""Experiment A: HalfCheetah gravity 1.0 -> 0.7, continuous AVG.

One AVG learner that does nothing at the MDP change (the main baseline).
The regime ID is never exposed to the policy.

Usage:
    python3 run_gravity_ab.py
    python3 run_gravity_ab.py --switch-step 100000 --total-steps 200000 --seed 1
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEP = 5_000_000


def make_schedule(total_steps, switch_step, gravity_scale_b):
    regime_a = RegimeSpec(name="A", use_original_reward=True, gravity_scale=1.0)
    regime_b = RegimeSpec(
        name="B", use_original_reward=True, gravity_scale=gravity_scale_b
    )
    return schedule_from_steps(
        total_steps,
        [
            {"start_step": 0, "regime": regime_a},
            {"start_step": switch_step, "regime": regime_b},
        ],
    )


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument("--switch-step", type=int, default=DEFAULT_SWITCH_STEP)
    parser.add_argument("--gravity-scale-b", type=float, default=0.7)
    args = parser.parse_args()
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "continuous_avg"
    args.experiment_name = "gravity_ab"
    args.schedule = make_schedule(
        args.total_steps, args.switch_step, args.gravity_scale_b
    )
    run(args)


if __name__ == "__main__":
    main()
