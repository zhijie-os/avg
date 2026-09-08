"""Experiment C: HalfCheetah action scale 1.0 -> 0.4 -> 1.0, continuous AVG.

Actuator effectiveness drops to 40% during regime B: the torque MuJoCo
receives is ``action_scale * agent_action``.  One continuous AVG learner.

Usage:
    python3 run_action_aba.py
    python3 run_action_aba.py --action-scale-b 0.6 --switch-step-1 1000 --switch-step-2 2000 --total-steps 3000
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEPS = (5_000_000, 7_500_000)


def make_schedule(total_steps, switch_steps, action_scale_b):
    regime_a = RegimeSpec(name="A", use_original_reward=True, action_scale=1.0)
    regime_b = RegimeSpec(
        name="B", use_original_reward=True, action_scale=action_scale_b
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
    parser.add_argument("--action-scale-b", type=float, default=0.4)
    args = parser.parse_args()
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "continuous_avg"
    args.experiment_name = "action_aba"
    args.schedule = make_schedule(
        args.total_steps,
        (args.switch_step_1, args.switch_step_2),
        args.action_scale_b,
    )
    run(args)


if __name__ == "__main__":
    main()
