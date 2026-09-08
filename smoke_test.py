"""Small smoke test: verify the regime changes at the requested global steps.

Two parts, both using the real MuJoCo HalfCheetah:

1. Environment-level: step a scheduled A -> B -> A wrapper by hand and check
   that the gravity/action-scale actually flips exactly at the configured
   global steps, that rewards pass through unchanged for original-reward
   regimes, and that the observation (the only thing the agent sees) never
   carries the regime ID.

2. Baseline ladder: run every baseline for a few hundred steps through the
   full experiment runner and check the CHANGE/RESET/RESTORE event lines and
   the produced log artifacts.

Usage:
    python3 smoke_test.py
"""

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np

from incremental_rl.envs.nonstationary_half_cheetah import (
    NonStationaryHalfCheetah,
    RegimeSpec,
    schedule_from_steps,
)
from incremental_rl.experiments.halfcheetah_changing_mdp import (
    build_parser,
    run,
)
from incremental_rl.utils import set_one_thread

TOTAL_STEPS = 120
SWITCH_1 = 40
SWITCH_2 = 80

REGIME_A = RegimeSpec(name="A", use_original_reward=True, gravity_scale=1.0)
REGIME_B = RegimeSpec(name="B", use_original_reward=True, gravity_scale=0.7)


def make_schedule():
    return schedule_from_steps(
        TOTAL_STEPS,
        [
            {"start_step": 0, "regime": REGIME_A},
            {"start_step": SWITCH_1, "regime": REGIME_B},
            {"start_step": SWITCH_2, "regime": REGIME_A},
        ],
    )


def part1_environment_check() -> None:
    import gymnasium as gym

    schedule = make_schedule()
    raw = gym.make("HalfCheetah-v4")
    env = NonStationaryHalfCheetah(raw, schedule=schedule)
    obs_dim = int(env.observation_space.shape[0])

    expected_gravity = [1.0] * SWITCH_1 + [0.7] * (SWITCH_2 - SWITCH_1) + [
        1.0
    ] * (TOTAL_STEPS - SWITCH_2)
    expected_regime = (
        ["A"] * SWITCH_1
        + ["B"] * (SWITCH_2 - SWITCH_1)
        + ["A"] * (TOTAL_STEPS - SWITCH_2)
    )

    obs, _ = env.reset(seed=0)
    gravity_seen, regime_seen, reward_passthrough = [], [], []
    for t in range(TOTAL_STEPS):
        # The regime for step t is installed inside step(); read it from the
        # step info, which reflects the context actually used for that step.
        action = np.full(env.action_space.shape, 0.5)
        next_obs, reward, _, _, info = env.step(action)
        gravity_seen.append(float(info["gravity_scale"]))
        regime_seen.append(info["regime_id"])
        reward_passthrough.append(bool(np.isclose(reward, info["base_env_reward"])))
        obs = next_obs
    env.close()

    assert gravity_seen == expected_gravity, (
        f"gravity_scale schedule wrong:\n{gravity_seen}"
    )
    assert regime_seen == expected_regime, f"regime schedule wrong:\n{regime_seen}"
    assert all(reward_passthrough), "original reward not passed through"
    assert obs.shape[0] == obs_dim == 17, "observation shape changed"
    print(
        f"PASS part 1: gravity follows [1.0 x{SWITCH_1}, 0.7 x{SWITCH_2 - SWITCH_1}, "
        f"1.0 x{TOTAL_STEPS - SWITCH_2}] exactly; regime IDs matched; "
        f"original rewards passed through; agent-facing obs is 17-D "
        f"(no regime ID leaks into the observation)."
    )


def part1_action_scale_check() -> None:
    import gymnasium as gym

    a_scale = 0.4
    schedule = schedule_from_steps(
        10,
        [
            {
                "start_step": 0,
                "regime": RegimeSpec(
                    name="S", use_original_reward=True, action_scale=a_scale
                ),
            }
        ],
    )
    raw = gym.make("HalfCheetah-v4")
    env = NonStationaryHalfCheetah(raw, schedule=schedule)
    env.reset(seed=0)
    commanded = np.full(env.action_space.shape, 0.5)
    _, _, _, _, info = env.step(commanded)
    env.close()
    assert np.allclose(info["executed_action"], commanded * a_scale), (
        f"action_scale not applied: {info['executed_action']}"
    )
    print(
        f"PASS part 1b: action_scale {a_scale} applied in-place "
        f"(0.5 commanded -> {info['executed_action'][0]} executed)."
    )


def part1_normalizer_equivalence_check() -> None:
    """ObservationNormalizer must match gymnasium NormalizeObservation, which
    the original stationary runner wraps the environment with."""

    import gymnasium as gym
    from gymnasium.wrappers import NormalizeObservation
    from incremental_rl.normalization import ObservationNormalizer

    raw = gym.make("HalfCheetah-v4")
    env = gym.make("HalfCheetah-v4")
    env = NormalizeObservation(env)
    own = ObservationNormalizer(raw.observation_space.shape[0])

    obs, _ = raw.reset(seed=3)
    wrapped_obs, _ = env.reset(seed=3)
    matched = True
    for _ in range(50):
        own_normalized = own.normalize(obs, update=True)
        if not np.allclose(own_normalized, wrapped_obs, atol=1e-5):
            matched = False
            break
        action = env.action_space.sample()
        obs, _, term, trunc, _ = raw.step(action)
        wrapped_obs, _, _, _, _ = env.step(action)
        if term or trunc:
            obs, _ = raw.reset()
            wrapped_obs, _ = env.reset()
    raw.close()
    env.close()
    assert matched, "ObservationNormalizer differs from gymnasium NormalizeObservation"
    print(
        "PASS part 1c: ObservationNormalizer matches gymnasium NormalizeObservation, "
        "so stationary mode follows the original normalization stream."
    )


def part2_baseline_ladder_check() -> None:
    results_root = Path("results_smoke_test")
    if results_root.exists():
        shutil.rmtree(results_root)

    expected_lines = {
        "continuous_avg": ["CHANGE: A -> B at step 40", "CHANGE: B -> A at step 80"],
        "oracle_optimizer_reset": [
            "CHANGE: A -> B at step 40",
            "CHANGE: B -> A at step 80",
            "RESET optimizer",
            "RESET observation normalization",
            "RESET TD scaler",
        ],
        "oracle_full_reset": [
            "CHANGE: A -> B at step 40",
            "CHANGE: B -> A at step 80",
            "RESET critic/actor",
        ],
        "oracle_mixture": [
            "CHANGE: A -> B at step 40",
            "CHANGE: B -> A at step 80",
            "CREATE learner B",
            "RESTORE learner A",
        ],
    }

    for baseline, required in expected_lines.items():
        parser = build_parser()
        args = parser.parse_args([])
        args.baseline = baseline
        args.experiment_name = "smoke_gravity_aba"
        args.total_steps = TOTAL_STEPS
        args.schedule = make_schedule()
        args.results_dir = results_root
        args.output_dir = None
        args.seed = 0
        args.eval_interval = 0
        args.diagnostic_interval = 10
        args.progress_interval = 0

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            output_dir = run(args)
        console = buffer.getvalue()

        missing = [line for line in required if line not in console]
        assert not missing, f"[{baseline}] missing console events: {missing}"
        for name in ("metadata.json", "episodes.csv", "switches.csv",
                     "diagnostics.csv", "steps.npz", "final.pt"):
            assert (output_dir / name).exists(), f"[{baseline}] missing {name}"
        print(f"PASS part 2 [{baseline}] -> {output_dir.relative_to(Path.cwd())}")
        for line in console.splitlines():
            if "CHANGE" in line or "RESET" in line or "learner" in line:
                print(f"    {line}")

    print(f"(smoke-test artifacts left in {results_root}/ — safe to delete)")


if __name__ == "__main__":
    set_one_thread()
    part1_environment_check()
    part1_action_scale_check()
    part1_normalizer_equivalence_check()
    part2_baseline_ladder_check()
    print("\nALL SMOKE TESTS PASSED")
