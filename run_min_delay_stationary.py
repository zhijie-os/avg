"""Stationary pretraining for the oracle mixture baseline (min-delay regimes).

Trains one AVG agent from scratch, entirely within a single min-delay regime
for 5,000,000 steps, with no regime switch.  The resulting checkpoints
(``model.pt``, saved by default) are meant to be loaded later by an oracle
regime-switching experiment that jumps directly to the pretrained model of
the current regime.

Regimes (identical RegimeSpecs to run_min_delay_ab.py via the shared
``min_delay_regime_spec`` helper):

    A            - no wind, no malfunction, target velocity 1.5
    B_wind       - wind force -4.0 N, target velocity 1.5
    B_joint      - one sign-flipped actuator (default 0), target velocity 1.5
    B_velocity   - target velocity 2.0
    B_combined   - wind -4.0 N + actuator flip + target velocity 2.0

This is "B_x from scratch -> train for 5M steps", NOT "A -> B_x": a freshly
initialized AVG agent trains continuously for the whole run; nothing is
loaded, reset, or switched midway, and no replay is used.

Usage:
    python3 run_min_delay_stationary.py --regime A
    python3 run_min_delay_stationary.py --regime B_joint --actuator-index 3
    python3 run_min_delay_stationary.py --regime B_combined --smoke-test
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
    min_delay_regime_spec,
)

DEFAULT_TOTAL_STEPS = 5_000_000

REGIME_NAMES = ("A", "B_wind", "B_joint", "B_velocity", "B_combined")

# Physics/reward fields that must match the A -> B experiment's regimes.
REGIME_PARAM_FIELDS = (
    "target_velocity",
    "action_sign_flips",
    "disabled_action_indices",
    "action_scale",
    "wind_force_x",
    "wind_body_ids",
    "gravity_scale",
    "mass_scale",
    "use_original_reward",
)


def _smoke_verify(args, spec) -> None:
    """Env-level check: the selected regime is active from step 0 onward."""

    import gymnasium as gym

    from incremental_rl.envs.nonstationary_half_cheetah import NonStationaryHalfCheetah

    assert len(args.schedule.regimes) == 1, "stationary schedule must have exactly one regime"

    # (3) The spec must match the corresponding regime of run_min_delay_ab.py,
    # up to the display name (the A -> B experiment labels every B "B").
    ab_ref = min_delay_regime_spec(
        args.regime,
        as_name=None if args.regime == "A" else "B",
        wind_force=args.wind_force,
        actuator_index=args.actuator_index,
        target_velocity_a=args.target_velocity_a,
        target_velocity_b=args.target_velocity_b,
    )
    for field in REGIME_PARAM_FIELDS:
        assert getattr(spec, field) == getattr(ab_ref, field), (
            f"field {field} diverged from run_min_delay_ab: "
            f"{getattr(spec, field)!r} != {getattr(ab_ref, field)!r}"
        )

    env = NonStationaryHalfCheetah(gym.make(args.env_id), schedule=args.schedule)
    try:
        env.reset(seed=7)
        action = np.full((6,), 0.3)
        flips = spec.action_sign_flips
        for t in range(60):
            _, reward, _, _, info = env.step(action)
            # (1) Selected regime active from step 0, no changes during the run.
            assert info["regime_id"] == spec.name, f"wrong regime at t={t}"
            assert np.isclose(
                info["target_velocity"], spec.target_velocity
            ), f"target velocity wrong at t={t}"
            assert np.isclose(
                info["wind_force_x"], spec.wind_force_x
            ), f"wind force wrong at t={t}"
            if spec.wind_force_x != 0.0:
                assert np.allclose(
                    env.unwrapped.data.xfrc_applied[1:, 0],
                    spec.wind_force_x,
                    atol=1e-12,
                ), f"xfrc_applied not set at t={t}"
            else:
                assert np.allclose(
                    env.unwrapped.data.xfrc_applied, 0.0, atol=1e-12
                ), f"unexpected external forces at t={t}"
            executed = info["executed_action"]
            for index in flips:
                assert np.isclose(
                    executed[index], -action[index]
                ), f"actuator {index} not flipped at t={t}"
            for index in [i for i in range(6) if i not in flips]:
                assert np.isclose(
                    executed[index], action[index]
                ), f"actuator {index} should not be flipped at t={t}"
            expected = -abs(
                info["x_velocity"] - spec.target_velocity
            ) - 0.1 * float(np.square(executed).sum())
            assert np.isclose(reward, expected, atol=1e-6), (
                f"reward != tracking formula at t={t}"
            )
    finally:
        env.close()

    print(
        f"PASS smoke env: regime {args.regime} active from step 0 for all 60 "
        f"probe steps; parameters identical to run_min_delay_ab's corresponding "
        f"regime (wind={spec.wind_force_x}, flips={spec.action_sign_flips}, "
        f"target_velocity={spec.target_velocity}).",
        flush=True,
    )


def _smoke_verify_artifacts(output_dir, total_steps) -> None:
    """After the short run: no switch, one learner, checkpoint loadable."""

    import csv
    import json

    import torch

    from incremental_rl.avg_agent import AVGAgent, AVGConfig
    from incremental_rl.normalization import ObservationNormalizer

    with open(output_dir / "switches.csv", newline="", encoding="utf-8") as stream:
        switches = list(csv.DictReader(stream))
    # (2) No regime switch: only the initial row exists.
    assert len(switches) == 1, f"expected no switch events, got {switches}"
    assert switches[0]["step"] == "0" and switches[0]["intervention"] == "initialize"

    final = torch.load(output_dir / "final.pt", weights_only=False)
    manager_state = final["manager"]
    # (4) One continuous AVG learner.
    assert manager_state["baseline"] == "stationary"
    assert set(manager_state["bundles"]) == {"shared"}, (
        "more than one learner bundle: training was not continuous"
    )
    agent_steps = manager_state["bundles"]["shared"]["agent"]["steps"]
    assert agent_steps == total_steps, (
        f"learner steps {agent_steps} != total {total_steps}"
    )

    # (5) The saved checkpoint loads into a fresh AVGAgent.
    ckpt = torch.load(output_dir / "model.pt", weights_only=False)
    agent = AVGAgent(AVGConfig(), obs_dim=17, action_dim=6, device="cpu")
    agent.load_state_dict(ckpt["model"]["agent"], restore_rng=False)
    normalizer = ObservationNormalizer(17)
    normalizer.load_state_dict(ckpt["model"]["normalizer"])
    assert agent.steps == total_steps

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"

    print(
        f"PASS smoke artifacts: no switch events; one shared learner over "
        f"{agent_steps} steps; model.pt reloads into a fresh AVGAgent "
        f"(steps={agent.steps}, normalizer restored).",
        flush=True,
    )


def main():
    parser = build_parser(description=__doc__)
    parser.add_argument("--regime", choices=REGIME_NAMES, required=True)
    parser.add_argument(
        "--wind-force", type=float, default=DEFAULT_WIND_FORCE,
        help="Signed x-axis wind force in N (negative = opposite to +x motion)",
    )
    parser.add_argument(
        "--actuator-index", type=int, default=DEFAULT_ACTUATOR_INDEX,
        help=f"Actuator index negated (0-5): {ACTUATOR_NAMES}",
    )
    parser.add_argument(
        "--target-velocity-a", type=float, default=DEFAULT_TARGET_VELOCITY_A,
    )
    parser.add_argument(
        "--target-velocity-b", type=float, default=DEFAULT_TARGET_VELOCITY_B,
    )
    parser.add_argument(
        "--no-save-model", action="store_true", default=False,
        help="Skip writing model.pt (the checkpoint is saved by default)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=False,
        help=(
            "Run a short (300-step) experiment and verify the regime is active "
            "from step 0, matches run_min_delay_ab.py, never switches, uses one "
            "continuous learner, and produces a loadable checkpoint"
        ),
    )
    args = parser.parse_args()

    if args.actuator_index < 0:
        parser.error("--actuator-index must be non-negative")

    args.total_steps = args.total_steps or DEFAULT_TOTAL_STEPS
    args.baseline = "stationary"
    args.experiment_name = f"stationary_{args.regime}"
    args.save_model = not args.no_save_model  # pretraining saves by default

    smoke_total = 0
    if args.smoke_test:
        smoke_total = 300
        args.total_steps = smoke_total
        args.eval_interval = 0
        args.progress_interval = 0
        args.diagnostic_interval = 10

    spec = min_delay_regime_spec(
        args.regime,
        wind_force=args.wind_force,
        actuator_index=args.actuator_index,
        target_velocity_a=args.target_velocity_a,
        target_velocity_b=args.target_velocity_b,
    )
    args.schedule = schedule_from_steps(
        args.total_steps, [{"start_step": 0, "regime": spec}]
    )

    print("STATIONARY ORACLE PRETRAINING", flush=True)
    print(f"REGIME: {args.regime}", flush=True)
    print(f"TOTAL STEPS: {args.total_steps}", flush=True)
    print(f"WIND FORCE: {spec.wind_force_x} N", flush=True)
    print(
        f"ACTUATOR INDEX: {spec.action_sign_flips[0] if spec.action_sign_flips else 'none'}",
        flush=True,
    )
    print(f"TARGET VELOCITY: {spec.target_velocity}", flush=True)

    if args.smoke_test:
        _smoke_verify(args, spec)

    output_dir = run(args)

    if args.smoke_test:
        _smoke_verify_artifacts(output_dir, smoke_total)


if __name__ == "__main__":
    main()
