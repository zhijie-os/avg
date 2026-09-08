"""Oracle optimizer reset: HalfCheetah gravity 1.0 -> 0.7 -> 1.0.

The experiment knows the true changepoints.  At each regime switch the
actor/critic weights are kept and only the configured transient components
are reset (default: Adam moments, observation normalization, TD-error
scaler).  The policy still never sees the regime ID.

Usage:
    python3 run_oracle_optimizer_reset.py
    python3 run_oracle_optimizer_reset.py --reset-components optimizer
    python3 run_oracle_optimizer_reset.py --reset-components obs_norm td_scaler
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEPS = (5_000_000, 7_500_000)


def make_schedule(total_steps, switch_steps, gravity_scale_b):
    regime_a = RegimeSpec(name="A", use_original_reward=True, gravity_scale=1.0)
    regime_b = RegimeSpec(
        name="B", use_original_reward=True, gravity_scale=gravity_scale_b
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
    parser.add_argument("--gravity-scale-b", type=float, default=0.7)
    args = parser.parse_args()
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "oracle_optimizer_reset"
    args.experiment_name = "gravity_aba"
    args.schedule = make_schedule(
        args.total_steps,
        (args.switch_step_1, args.switch_step_2),
        args.gravity_scale_b,
    )
    run(args)


if __name__ == "__main__":
    main()
