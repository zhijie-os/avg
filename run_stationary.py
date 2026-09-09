"""Train stationary AVG on a chosen HalfCheetah gravity regime."""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--regime-name", type=str, default="A")

    args = parser.parse_args()

    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "stationary"

    if args.experiment_name is None:
        args.experiment_name = f"stationary_{args.regime_name}"

    regime = RegimeSpec(
        name=args.regime_name,
        use_original_reward=True,
        gravity_scale=args.gravity_scale,
    )

    args.schedule = schedule_from_steps(
        args.total_steps,
        [{"start_step": 0, "regime": regime}],
    )

    run(args)


if __name__ == "__main__":
    main()