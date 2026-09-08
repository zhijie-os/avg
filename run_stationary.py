"""Stationary AVG on HalfCheetah (regime A, original reward).

The reference run: one AVG learner on the unchanged HalfCheetah MDP with the
official hyperparameters.  The regime machinery is disabled via a
single-regime schedule, so this reproduces the original repository behavior.

Usage:
    python3 run_stationary.py
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    args = parser.parse_args()
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "stationary"
    args.experiment_name = "stationary"
    regime_a = RegimeSpec(
        name="A", use_original_reward=True, gravity_scale=args.gravity_scale
    )
    args.schedule = schedule_from_steps(
        args.total_steps, [{"start_step": 0, "regime": regime_a}]
    )
    run(args)


if __name__ == "__main__":
    main()
