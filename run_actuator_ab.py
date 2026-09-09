"""Experiment: HalfCheetah actuator sign flip A -> B, continuous AVG.

Regime A is the default HalfCheetah environment, untouched.  At the switch
step, exactly one actuator's sign is flipped at the environment interface:
before ``env.step(action)``, ``modified_action[flipped_actuator_idx] *= -1``.
All other action dimensions are unchanged and the actor keeps outputting its
normal action, so the agent is never told about the regime change.

This is a single continuous AVG training run across A -> B: no policy,
optimizer, normalization, or scaler resets at the switch.

HalfCheetah has 6 actuators:
    0 = back thigh, 1 = back shin, 2 = back foot,
    3 = front thigh, 4 = front shin, 5 = front foot

Usage:
    python3 run_actuator_ab.py
    python3 run_actuator_ab.py --flipped-actuator-idx 3
    python3 run_actuator_ab.py --switch-step 100000 --total-steps 200000 --seed 1
"""

from incremental_rl.envs.nonstationary_half_cheetah import (
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import build_parser, run

DEFAULT_TOTAL_STEPS = 10_000_000
DEFAULT_SWITCH_STEP = 5_000_000

# Which actuator is negated in regime B (0-5 for HalfCheetah).
# Overridable via --flipped-actuator-idx.
FLIPPED_ACTUATOR_IDX = 0


def make_schedule(total_steps, switch_step, flipped_actuator_idx):
    regime_a = RegimeSpec(name="A", use_original_reward=True)
    regime_b = RegimeSpec(
        name="B",
        use_original_reward=True,
        action_sign_flips=(flipped_actuator_idx,),
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
    parser.add_argument(
        "--flipped-actuator-idx",
        type=int,
        default=FLIPPED_ACTUATOR_IDX,
        help="Actuator index negated in regime B (0-5 for HalfCheetah)",
    )
    args = parser.parse_args()
    if args.flipped_actuator_idx < 0:
        parser.error("--flipped-actuator-idx must be non-negative")
    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "continuous_avg"
    args.experiment_name = "actuator_ab"
    print(
        f"ACTUATOR FLIP CONFIG: regime B negates actuator index "
        f"{args.flipped_actuator_idx} "
        f"(A -> B at step {args.switch_step}, total {args.total_steps})",
        flush=True,
    )
    args.schedule = make_schedule(
        args.total_steps, args.switch_step, args.flipped_actuator_idx
    )
    run(args)


if __name__ == "__main__":
    main()
