"""Run the AVG baseline ladder on piecewise-stationary Half-Cheetah MDPs.

The environment follows a ``Schedule`` of regimes (gravity scale, action
scale, target velocity, ...) driven purely by global environment steps.  The
AVG agent never observes the regime ID: the wrapper reports it only in the
step ``info`` dictionary, where it is logged for evaluation.

Baselines:

* ``stationary`` / ``continuous_avg`` - one AVG learner, no intervention at
  regime boundaries (``continuous_avg`` is the main baseline).
* ``oracle_optimizer_reset`` - keep actor/critic weights; reset the
  explicitly configured transient components (Adam moments, observation
  normalization, TD-error scaler) at true boundaries.
* ``oracle_full_reset`` - reinitialize actor, critic, optimizers, and all
  normalization/TD statistics at true boundaries, continuing the same global
  step counter.
* ``oracle_mixture`` - one independent AVG learner per regime; inactive
  learners are frozen and resumed exactly when their regime returns (the
  upper-reference oracle).

A run writes ``metadata.json``, ``episodes.csv``, ``switches.csv``,
``diagnostics.csv``, ``evaluations.csv``, ``steps.npz``, and ``final.pt`` to
its output directory.  ``plot_results.py`` aggregates these for side-by-side
comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch

from incremental_rl.avg_agent import AVGAgent, AVGConfig
from incremental_rl.envs.nonstationary_half_cheetah import (
    NonStationaryHalfCheetah,
    RegimeSpec,
    Schedule,
    schedule_from_steps,
)
from incremental_rl.normalization import ObservationNormalizer


BASELINES = (
    "stationary",
    "continuous_avg",
    "oracle_optimizer_reset",
    "oracle_full_reset",
    "oracle_mixture",
)

# Components that oracle_optimizer_reset can reset independently.  Any
# combination is valid, e.g. ``--reset-components optimizer`` for an
# "optimizer only" experiment.
RESET_COMPONENTS = ("optimizer", "obs_norm", "td_scaler")


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class LearnerBundle:
    """One AVG learner plus its observation normalizer."""

    agent: AVGAgent
    normalizer: ObservationNormalizer
    initialization_seed: int

    def state_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "initialization_seed": self.initialization_seed,
        }


class LearnerManager:
    """Implement the baseline ladder with one explicit routing surface."""

    def __init__(
        self,
        *,
        baseline: str,
        agent_config: AVGConfig,
        obs_dim: int,
        action_dim: int,
        seed: int,
        regimes: Iterable[RegimeSpec],
        device: torch.device,
        reset_components: Sequence[str] = RESET_COMPONENTS,
    ) -> None:
        if baseline not in BASELINES:
            raise ValueError(f"Unknown baseline {baseline!r}")
        unknown = set(reset_components) - set(RESET_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown reset components: {sorted(unknown)}")
        self.baseline = baseline
        self.agent_config = agent_config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.device = device
        self.regimes = list(dict.fromkeys(regimes))
        self.reset_components = tuple(dict.fromkeys(reset_components))
        self.bundles: dict[str, LearnerBundle] = {}
        self.current_key: Optional[str] = None
        self.current_regime: Optional[RegimeSpec] = None

    @property
    def current(self) -> LearnerBundle:
        if self.current_key is None:
            raise RuntimeError("LearnerManager has not been initialized")
        return self.bundles[self.current_key]

    def _fresh_seed(self, token: int) -> int:
        return self.seed + 1_000_003 * token

    def _new_bundle(self, initialization_seed: int) -> LearnerBundle:
        torch.manual_seed(initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initialization_seed)
        agent = AVGAgent(
            self.agent_config,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        return LearnerBundle(
            agent=agent,
            normalizer=ObservationNormalizer(self.obs_dim),
            initialization_seed=initialization_seed,
        )

    def initialize(self, regime: RegimeSpec) -> tuple[str, ...]:
        """Create the first learner.  Returns human-readable event lines."""

        if self.current_key is not None:
            raise RuntimeError("LearnerManager is already initialized")
        key = regime.name if self.baseline == "oracle_mixture" else "shared"
        self.bundles[key] = self._new_bundle(self.seed)
        self.current_key = key
        self.current_regime = regime
        return ()

    def switch(self, regime: RegimeSpec, segment_index: int) -> tuple[str, ...]:
        """Apply a known true-boundary intervention before the next action.

        Returns the console event lines (e.g. ``"RESET optimizer"``) to print,
        or an empty tuple when the baseline does nothing at boundaries.
        """

        if self.current_key is None:
            return self.initialize(regime)
        self.current_regime = regime

        if self.baseline in ("stationary", "continuous_avg"):
            return ()

        if self.baseline == "oracle_optimizer_reset":
            lines = []
            if "optimizer" in self.reset_components:
                self.current.agent.reset_transient_components(optimizer=True)
                lines.append("RESET optimizer")
            if "obs_norm" in self.reset_components:
                self.current.normalizer.reset()
                lines.append("RESET observation normalization")
            if "td_scaler" in self.reset_components:
                self.current.agent.reset_transient_components(td_scaler=True)
                lines.append("RESET TD scaler")
            return tuple(lines)

        if self.baseline == "oracle_full_reset":
            self.bundles["shared"] = self._new_bundle(
                self._fresh_seed(segment_index)
            )
            self.current_key = "shared"
            return (
                "RESET critic/actor (fresh full learner)",
                "RESET optimizer",
                "RESET observation normalization",
                "RESET TD scaler",
            )

        if self.baseline == "oracle_mixture":
            if regime.name in self.bundles:
                self.current_key = regime.name
                return (f"RESTORE learner {regime.name}",)
            self.bundles[regime.name] = self._new_bundle(
                self._fresh_seed(len(self.bundles))
            )
            self.current_key = regime.name
            return (f"CREATE learner {regime.name}",)

        raise AssertionError(self.baseline)

    def bundle_for_regime(self, regime: RegimeSpec) -> Optional[LearnerBundle]:
        if self.baseline == "oracle_mixture":
            return self.bundles.get(regime.name)
        return self.current

    def state_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "current_key": self.current_key,
            "bundles": {
                key: bundle.state_dict() for key, bundle in self.bundles.items()
            },
        }


def _classify_change(reward_change: bool, transition_change: bool) -> str:
    if reward_change and transition_change:
        return "reward_and_transition"
    if reward_change:
        return "reward"
    if transition_change:
        return "transition"
    return "none"


def boundary_change_kind(previous: RegimeSpec, current: RegimeSpec) -> str:
    """Classify the MDP dimensions that changed at one boundary."""

    def reward_differs(a: RegimeSpec, b: RegimeSpec) -> bool:
        if a.use_original_reward or b.use_original_reward:
            return a.use_original_reward != b.use_original_reward
        return a.target_velocity != b.target_velocity

    transition_change = any(
        (
            previous.action_sign_flips != current.action_sign_flips,
            previous.disabled_action_indices != current.disabled_action_indices,
            previous.action_scale != current.action_scale,
            previous.wind_force_x != current.wind_force_x,
            previous.wind_body_ids != current.wind_body_ids,
            previous.gravity_scale != current.gravity_scale,
            previous.mass_scale != current.mass_scale,
        )
    )
    return _classify_change(
        reward_differs(previous, current), transition_change
    )


def schedule_metadata(schedule: Schedule) -> list[dict[str, Any]]:
    result = []
    start = 0
    visits: dict[str, int] = {}
    for index, (regime, duration) in enumerate(
        zip(schedule.regimes, schedule.durations)
    ):
        visit = visits.get(regime.name, 0)
        result.append(
            {
                "segment_index": index,
                "visit_index": visit,
                "regime_id": regime.name,
                "start_step": start,
                "end_step": start + duration,
                "spec": asdict(regime),
            }
        )
        visits[regime.name] = visit + 1
        start += duration
    return result


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*command: str) -> Optional[str]:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    revision = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--short")
    return {"revision": revision, "dirty": bool(status), "status_short": status}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    collisions = [
        name
        for name in (
            "metadata.json",
            "episodes.csv",
            "steps.npz",
            "final.pt",
            "model.pt",
        )
        if (path / name).exists()
    ]
    if collisions:
        raise FileExistsError(
            f"Refusing to overwrite completed artifacts in {path}: {collisions}"
        )


def make_environment(env_id: str, schedule: Schedule) -> NonStationaryHalfCheetah:
    raw = gym.make(env_id)
    if tuple(raw.observation_space.shape) != (17,):
        raw.close()
        raise ValueError(
            f"Expected the paper's 17-D HalfCheetah observation, got "
            f"{raw.observation_space.shape} from {env_id}"
        )
    if tuple(raw.action_space.shape) != (6,):
        raw.close()
        raise ValueError(
            f"Expected the paper's 6-D HalfCheetah action, got "
            f"{raw.action_space.shape} from {env_id}"
        )
    return NonStationaryHalfCheetah(raw, schedule=schedule)


def evaluate_bundle(
    bundle: LearnerBundle,
    regime: RegimeSpec,
    *,
    env_id: str,
    seed: int,
    episodes: int,
    max_episode_steps: int = 1_000,
) -> tuple[float, float, list[float]]:
    """Evaluate deterministically without updating learner or normalization state."""

    schedule = Schedule((regime,), (episodes * max_episode_steps,))
    env = make_environment(env_id, schedule)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            for _ in range(max_episode_steps):
                normalized = bundle.normalizer.normalize(observation, update=False)
                action, _ = bundle.agent.act(normalized, deterministic=True)
                simulation_action = action.detach().cpu().reshape(-1).numpy()
                observation, reward, terminated, truncated, _ = env.step(
                    simulation_action
                )
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    values = np.asarray(returns, dtype=np.float64)
    return float(values.mean()), float(values.std()), returns


def record_evaluations(
    rows: list[dict[str, Any]],
    manager: LearnerManager,
    regimes: Iterable[RegimeSpec],
    *,
    step: int,
    env_id: str,
    seed: int,
    episodes: int,
) -> None:
    seen: set[str] = set()
    for target in regimes:
        if target.name in seen:
            continue
        seen.add(target.name)
        bundle = manager.bundle_for_regime(target)
        base = {
            "step": step,
            "evaluated_regime": target.name,
            "policy_regime": target.name
            if manager.baseline == "oracle_mixture"
            else manager.current_regime.name,  # type: ignore[union-attr]
            "episodes": episodes,
        }
        if bundle is None:
            rows.append(
                {
                    **base,
                    "available": False,
                    "mean_return": np.nan,
                    "std_return": np.nan,
                }
            )
            continue
        mean, std, _ = evaluate_bundle(
            bundle,
            target,
            env_id=env_id,
            # Keep evaluation initial states paired across checkpoints and
            # baselines.  The evaluation environments own their RNGs, so this
            # does not perturb online training.
            seed=seed + 100_000,
            episodes=episodes,
        )
        rows.append(
            {**base, "available": True, "mean_return": mean, "std_return": std}
        )


def optimizer_moment_norm(agent: AVGAgent) -> float:
    total = 0.0
    for optimizer in (agent.popt, agent.qopt):
        for state in optimizer.state.values():
            moment = state.get("exp_avg")
            if isinstance(moment, torch.Tensor):
                total += float(torch.sum(moment.detach() ** 2).item())
    return float(np.sqrt(total))


def load_schedule_from_json(path: Path) -> Schedule:
    """Load the generic nonstationary config format from a JSON file::

        {
          "enabled": true,
          "total_steps": 10000000,
          "schedule": [
            {"start_step": 0, "regime": {"name": "A", "gravity_scale": 1.0}},
            {"start_step": 5000000, "regime": {"name": "B", "gravity_scale": 0.7}}
          ]
        }

    ``regime`` entries may be inline ``RegimeSpec`` dicts or names resolved
    against an optional ``"regimes"`` table.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("enabled", True):
        raise ValueError(
            f"{path}: schedule config has 'enabled': false; run a stationary "
            "experiment instead of passing a schedule"
        )
    regimes = data.get("regimes")
    return schedule_from_steps(
        int(data["total_steps"]), data["schedule"], regimes=regimes
    )


def build_metadata(
    args: argparse.Namespace, schedule: Schedule, config: AVGConfig
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    arguments = {key: value for key, value in vars(args).items() if key != "schedule"}
    metadata = {
        "status": "running",
        "experiment": args.experiment_name,
        "baseline": args.baseline,
        "seed": args.seed,
        "environment_id": args.env_id,
        "total_steps": schedule.total_steps,
        "schedule": schedule_metadata(schedule),
        "avg_config": asdict(config),
        "context_in_observation": False,
        "online_action_mode": "stochastic",
        "time_limit_bootstrap": True,
        "arguments": arguments,
        "command": [sys.executable, *sys.argv],
        "git": git_metadata(root),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "mujoco": getattr(__import__("mujoco"), "__version__", "unknown"),
        },
        "started_unix_time": time.time(),
    }
    if args.baseline == "oracle_optimizer_reset":
        metadata["reset_components"] = list(args.reset_components)
    return metadata


def persist_logs(
    output_dir: Path,
    *,
    episodes: list[dict[str, Any]],
    switches: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    completed_steps: int,
    step_arrays: Mapping[str, np.ndarray],
) -> None:
    write_csv(output_dir / "episodes.csv", episodes)
    write_csv(output_dir / "switches.csv", switches)
    write_csv(output_dir / "diagnostics.csv", diagnostics)
    write_csv(output_dir / "evaluations.csv", evaluations)
    np.savez_compressed(
        output_dir / "steps.npz",
        **{name: values[:completed_steps] for name, values in step_arrays.items()},
    )


def save_model_checkpoint(
    output_dir: Path, manager: LearnerManager, completed_steps: int
) -> Path:
    """Export the trained AVG learner(s) to ``<output_dir>/model.pt``.

    The saved agent states are directly reloadable with
    ``AVGAgent.load_state_dict``; each entry also carries the observation
    normalizer required to feed the model.  ``oracle_mixture`` exports every
    regime learner, keyed by regime name.
    """

    checkpoint = {
        "format": "avg_model_v1",
        "completed_steps": completed_steps,
        "baseline": manager.baseline,
    }
    if manager.baseline == "oracle_mixture":
        checkpoint["models"] = {
            key: {
                "agent": bundle.agent.state_dict(),
                "normalizer": bundle.normalizer.state_dict(),
            }
            for key, bundle in manager.bundles.items()
        }
    else:
        checkpoint["model"] = {
            "agent": manager.current.agent.state_dict(),
            "normalizer": manager.current.normalizer.state_dict(),
        }
    path = output_dir / "model.pt"
    torch.save(checkpoint, path)
    return path


def run(args: argparse.Namespace) -> Path:
    """Run one baseline on one schedule; ``args.schedule`` must be a Schedule."""

    schedule: Schedule = args.schedule
    if not isinstance(schedule, Schedule):
        raise TypeError("args.schedule must be a Schedule")
    if args.total_steps is not None and int(args.total_steps) != schedule.total_steps:
        raise ValueError(
            f"--total-steps {args.total_steps} does not match the schedule "
            f"budget {schedule.total_steps}"
        )
    args.total_steps = schedule.total_steps

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            Path(args.results_dir)
            / f"{args.experiment_name}_{args.baseline}_{stamp}_seed-{args.seed}"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(args.output_dir)

    unique_regimes = list(dict.fromkeys(schedule.regimes))
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    config = AVGConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        gamma=args.gamma,
        alpha_lr=args.alpha,
        nhid_actor=args.actor_hidden,
        nhid_critic=args.critic_hidden,
        l2_actor=args.l2_actor,
        l2_critic=args.l2_critic,
    )
    args.reset_components = tuple(
        RESET_COMPONENTS if args.reset_components is None else args.reset_components
    )
    metadata = build_metadata(args, schedule, config)
    write_json(args.output_dir / "metadata.json", metadata)

    env: Optional[NonStationaryHalfCheetah] = None
    try:
        env = make_environment(args.env_id, schedule)
        env.action_space.seed(args.seed)
        observation, _ = env.reset(seed=args.seed)
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        manager = LearnerManager(
            baseline=args.baseline,
            agent_config=config,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=args.seed,
            regimes=unique_regimes,
            device=device,
            reset_components=args.reset_components,
        )
        first_position = schedule.position_at(0)
        manager.initialize(first_position.regime)
        print(
            f"START: regime {first_position.regime.name} at step 0 "
            f"(baseline={args.baseline})",
            flush=True,
        )
    except Exception as exc:
        if env is not None:
            env.close()
        metadata.update(
            {
                "status": "failed",
                "completed_steps": 0,
                "error": repr(exc),
                "finished_unix_time": time.time(),
            }
        )
        write_json(args.output_dir / "metadata.json", metadata)
        raise

    try:
        episodes: list[dict[str, Any]] = []
        switches: list[dict[str, Any]] = [
            {
                "step": 0,
                "segment_index": 0,
                "visit_index": 0,
                "from_regime": "",
                "to_regime": first_position.regime.name,
                "change_kind": "initial",
                "intervention": "initialize",
            }
        ]
        diagnostics: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []

        total_steps = schedule.total_steps
        regime_codebook = {
            regime.name: index for index, regime in enumerate(unique_regimes)
        }
        step_arrays: dict[str, np.ndarray] = {
            "reward": np.empty(total_steps, dtype=np.float32),
            "regime_code": np.empty(total_steps, dtype=np.int16),
            "x_velocity": np.empty(total_steps, dtype=np.float32),
            "target_velocity": np.empty(total_steps, dtype=np.float32),
            "velocity_error": np.empty(total_steps, dtype=np.float32),
            "control_cost": np.empty(total_steps, dtype=np.float32),
            "gravity_scale": np.empty(total_steps, dtype=np.float32),
            "action_scale": np.empty(total_steps, dtype=np.float32),
            "terminated": np.empty(total_steps, dtype=np.bool_),
            "truncated": np.empty(total_steps, dtype=np.bool_),
        }
        metadata["regime_codebook"] = {
            str(code): regime.name
            for regime, code in (
                (regime, regime_codebook[regime.name]) for regime in unique_regimes
            )
        }
        write_json(args.output_dir / "metadata.json", metadata)

        current_segment = 0
        episode_index = 0
        episode_start_step = 0
        episode_start_regime = first_position.regime.name
        episode_return = 0.0
        episode_length = 0
        episode_crossed_context = False
        completed_steps = 0
        last_evaluation_step: Optional[int] = None
        cached_normalized_observation: Optional[np.ndarray] = None

        if args.eval_interval > 0:
            record_evaluations(
                evaluations,
                manager,
                unique_regimes,
                step=0,
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.eval_episodes,
            )
            last_evaluation_step = 0
    except Exception as exc:
        assert env is not None
        env.close()
        metadata.update(
            {
                "status": "failed",
                "completed_steps": 0,
                "error": repr(exc),
                "finished_unix_time": time.time(),
            }
        )
        write_json(args.output_dir / "metadata.json", metadata)
        raise

    try:
        for step in range(total_steps):
            position = schedule.position_at(step)
            changed = position.segment_index != current_segment
            intervention_lines: tuple[str, ...] = ()
            if changed:
                previous = schedule.regimes[current_segment]
                # Apply the oracle intervention before selecting the boundary
                # action.  The wrapper installs the new MDP immediately before
                # executing that action, preserving its context-change event.
                intervention_lines = manager.switch(
                    position.regime, position.segment_index
                )
                print(
                    f"CHANGE: {previous.name} -> {position.regime.name} "
                    f"at step {step}",
                    flush=True,
                )
                print(
                    f"SWITCH: regime {previous.name} -> regime "
                    f"{position.regime.name} at step {step}",
                    flush=True,
                )
                for line in intervention_lines:
                    print(f"    {line}", flush=True)
                switches.append(
                    {
                        "step": step,
                        "segment_index": position.segment_index,
                        "visit_index": position.visit_index,
                        "from_regime": previous.name,
                        "to_regime": position.regime.name,
                        "change_kind": boundary_change_kind(
                            previous, position.regime
                        ),
                        "intervention": "; ".join(intervention_lines)
                        if intervention_lines
                        else "preserve_all_state",
                    }
                )
                current_segment = position.segment_index
                # A shared, unreset learner has already normalized this state
                # as the preceding transition's next observation.  Every other
                # intervention resets or changes the active normalizer.
                if intervention_lines:
                    cached_normalized_observation = None
                if episode_length:
                    episode_crossed_context = True

            if args.eval_interval > 0 and step > 0 and step % args.eval_interval == 0:
                record_evaluations(
                    evaluations,
                    manager,
                    unique_regimes,
                    step=step,
                    env_id=args.env_id,
                    seed=args.seed,
                    episodes=args.eval_episodes,
                )
                last_evaluation_step = step

            bundle = manager.current
            if cached_normalized_observation is None:
                normalized_observation = bundle.normalizer.normalize(
                    observation, update=True
                )
            else:
                normalized_observation = cached_normalized_observation
            action, action_info = bundle.agent.act(normalized_observation)
            simulation_action = action.detach().cpu().reshape(-1).numpy()
            next_observation, reward, terminated, truncated, info = env.step(
                simulation_action
            )
            episode_end = bool(terminated or truncated)
            normalized_next_observation = bundle.normalizer.normalize(
                next_observation, update=True
            )

            update_metrics: dict[str, float] = {}
            update_metrics = bundle.agent.update(
                normalized_observation,
                action,
                normalized_next_observation,
                reward,
                terminated=terminated,
                episode_end=episode_end,
                **action_info,
            )

            regime = position.regime
            step_arrays["reward"][step] = reward
            step_arrays["regime_code"][step] = regime_codebook[regime.name]
            step_arrays["x_velocity"][step] = info["x_velocity"]
            step_arrays["target_velocity"][step] = info["target_velocity"]
            step_arrays["velocity_error"][step] = info["velocity_error"]
            step_arrays["control_cost"][step] = -info["reward_control"]
            step_arrays["gravity_scale"][step] = info["gravity_scale"]
            step_arrays["action_scale"][step] = info["action_scale"]
            step_arrays["terminated"][step] = terminated
            step_arrays["truncated"][step] = truncated
            completed_steps = step + 1

            episode_return += float(reward)
            episode_length += 1
            if (
                step % args.diagnostic_interval == 0
                or changed
                or episode_end
                or (args.eval_interval > 0 and step % args.eval_interval == 0)
            ):
                diagnostics.append(
                    {
                        "step": step,
                        "regime_id": regime.name,
                        "segment_index": position.segment_index,
                        "visit_index": position.visit_index,
                        "reward": float(reward),
                        "episode_return_so_far": episode_return,
                        "x_velocity": info["x_velocity"],
                        "target_velocity": info["target_velocity"],
                        "velocity_error": info["velocity_error"],
                        "control_cost": -info["reward_control"],
                        "commanded_action_norm": float(
                            np.linalg.norm(info["commanded_action"])
                        ),
                        "executed_action_norm": float(
                            np.linalg.norm(info["executed_action"])
                        ),
                        "observation_count": bundle.normalizer.count,
                        "observation_variance_mean": float(
                            bundle.normalizer.var.mean()
                        ),
                        "optimizer_moment_norm": optimizer_moment_norm(
                            bundle.agent
                        ),
                        "td_scaler_reward_var": float(
                            bundle.agent.td_error_scaler.reward_rms.variance
                        ),
                        **update_metrics,
                    }
                )

            episode_return_log_value = 0
            if episode_end:
                episodes.append(
                    {
                        "episode": episode_index,
                        "start_step": episode_start_step,
                        "end_step": step + 1,
                        "length": episode_length,
                        "episode_return": episode_return,
                        "start_regime": episode_start_regime,
                        "regime_id": regime.name,
                        "crossed_context": episode_crossed_context,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "partial": False,
                    }
                )
                episode_index += 1
                episode_start_step = step + 1
                episode_return_log_value = episode_return
                episode_return = 0.0
                episode_length = 0
                episode_crossed_context = False
                if step + 1 < total_steps:
                    observation, _ = env.reset()
                    cached_normalized_observation = None
                    episode_start_regime = (
                        schedule.position_at(step + 1).regime.name
                    )
                else:
                    observation = next_observation
            else:
                observation = next_observation
                cached_normalized_observation = normalized_next_observation

            if args.progress_interval and completed_steps % args.progress_interval == 0:
                print(
                    f"step={completed_steps}/{total_steps} "
                    f"regime={regime.name} episodes={episode_index} "
                    f"return={episode_return_log_value:.2f}",
                    flush=True,
                )

        if episode_length:
            episodes.append(
                {
                    "episode": episode_index,
                    "start_step": episode_start_step,
                    "end_step": total_steps,
                    "length": episode_length,
                    "episode_return": episode_return,
                    "start_regime": episode_start_regime,
                    "regime_id": schedule.regime_at(total_steps - 1).name,
                    "crossed_context": episode_crossed_context,
                    "terminated": False,
                    "truncated": False,
                    "partial": True,
                }
            )

        if args.eval_interval > 0 and last_evaluation_step != total_steps:
            record_evaluations(
                evaluations,
                manager,
                unique_regimes,
                step=total_steps,
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.eval_episodes,
            )

        persist_logs(
            args.output_dir,
            episodes=episodes,
            switches=switches,
            diagnostics=diagnostics,
            evaluations=evaluations,
            completed_steps=completed_steps,
            step_arrays=step_arrays,
        )
        torch.save(
            {
                "format": "avg_halfcheetah_run_v1",
                "completed_steps": completed_steps,
                "manager": manager.state_dict(),
            },
            args.output_dir / "final.pt",
        )
        if args.save_model:
            model_path = save_model_checkpoint(
                args.output_dir, manager, completed_steps
            )
            print(f"Saved AVG model to {model_path}", flush=True)
        metadata.update(
            {
                "status": "completed",
                "completed_steps": completed_steps,
                "episodes": len(episodes),
                "finished_unix_time": time.time(),
            }
        )
        write_json(args.output_dir / "metadata.json", metadata)
        return args.output_dir
    except Exception as exc:
        persist_logs(
            args.output_dir,
            episodes=episodes,
            switches=switches,
            diagnostics=diagnostics,
            evaluations=evaluations,
            completed_steps=completed_steps,
            step_arrays=step_arrays,
        )
        metadata.update(
            {
                "status": "failed",
                "completed_steps": completed_steps,
                "error": repr(exc),
                "finished_unix_time": time.time(),
            }
        )
        write_json(args.output_dir / "metadata.json", metadata)
        raise
    finally:
        assert env is not None
        env.close()


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    """Common CLI surface shared by the per-experiment launcher scripts."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--baseline", choices=BASELINES, default=None)
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total interaction budget; defaults to the schedule's budget.",
    )
    parser.add_argument(
        "--schedule-json",
        type=Path,
        default=None,
        help="Generic nonstationary config file (see load_schedule_from_json).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Parent results dir"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact run dir; defaults to a timestamped dir under --results-dir",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained AVG model(s) to <output_dir>/model.pt",
    )

    # AVG hyperparameters (the official HalfCheetah configuration).
    parser.add_argument("--actor-lr", type=float, default=0.0063)
    parser.add_argument("--critic-lr", type=float, default=0.0087)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--actor-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument("--l2-actor", type=float, default=0.0)
    parser.add_argument("--l2-critic", type=float, default=0.0)

    # Oracle optimizer reset: which transient components to reset.  Defaults
    # to all three; e.g. --reset-components optimizer for optimizer-only.
    parser.add_argument(
        "--reset-components",
        nargs="+",
        choices=RESET_COMPONENTS,
        default=None,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        help="Deterministic cross-context evaluation interval; 0 disables it.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--diagnostic-interval", type=int, default=1_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.schedule_json is not None:
        args.schedule = load_schedule_from_json(args.schedule_json)
    if getattr(args, "schedule", None) is None:
        parser.error("either --schedule-json or a programmatic schedule is required")
    if args.experiment_name is None:
        args.experiment_name = args.schedule_json.stem if args.schedule_json else "changing_mdp"
    if args.baseline is None:
        args.baseline = "continuous_avg"
    output = run(args)
    print(f"Completed {args.baseline} seed {args.seed}: {output}")


if __name__ == "__main__":
    main()
