"""Piecewise-stationary HalfCheetah environments used by AVG experiments.

The canonical regimes in this module port the released environment code for
"Minimum-Delay Adaptation in Non-Stationary Reinforcement Learning via Online
High-Confidence Change-Point Detection" to Gymnasium's HalfCheetah-v4/v5 API.

The wrapper deliberately owns only the environment/context mechanics.  It does
not expose the regime ID in the observation and it does not implement any
learner-side oracle routing or resetting.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

try:  # Keep schedule/unit tests usable when the optional MuJoCo stack is absent.
    import gymnasium as gym

    _WrapperBase = gym.Wrapper
except ImportError:  # pragma: no cover - exercised only in minimal installations.
    gym = None

    class _WrapperBase:  # type: ignore[no-redef]
        """Small delegation-compatible fallback for environments without Gymnasium."""

        def __init__(self, env: Any) -> None:
            self.env = env

        @property
        def unwrapped(self) -> Any:
            return getattr(self.env, "unwrapped", self.env)

        @property
        def action_space(self) -> Any:
            return self.env.action_space

        @property
        def observation_space(self) -> Any:
            return self.env.observation_space

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)


IndexTuple = Tuple[int, ...]


def _immutable_indices(values: Iterable[int], field_name: str) -> IndexTuple:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} must contain integers, got {value!r}")
        index = int(value)
        if index < 0:
            raise ValueError(f"{field_name} cannot contain negative index {index}")
        result.append(index)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} cannot contain duplicate indices")
    return tuple(result)


@dataclass(frozen=True)
class RegimeSpec:
    """An immutable description of one latent HalfCheetah MDP.

    ``wind_body_ids=None`` means all non-world bodies.  Action sign flips,
    disabled actions, and the action scale transform the command before it
    reaches MuJoCo.  Disabled actions take precedence semantically, so
    overlapping index sets are rejected to keep every regime specification
    unambiguous.

    ``use_original_reward=True`` returns the base environment's untouched
    reward (the standard HalfCheetah forward-velocity reward).  Otherwise the
    reward is the target-velocity tracking reward
    ``-|x_velocity - target_velocity| - control_cost_weight * sum(action^2)``.
    """

    name: str
    target_velocity: float = 1.5
    action_sign_flips: IndexTuple = ()
    disabled_action_indices: IndexTuple = ()
    action_scale: float = 1.0
    wind_force_x: float = 0.0
    wind_body_ids: Optional[IndexTuple] = None
    gravity_scale: float = 1.0
    mass_scale: float = 1.0
    use_original_reward: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A regime name must be a non-empty string")

        flips = _immutable_indices(self.action_sign_flips, "action_sign_flips")
        disabled = _immutable_indices(
            self.disabled_action_indices, "disabled_action_indices"
        )
        if set(flips).intersection(disabled):
            raise ValueError("An action index cannot be both sign-flipped and disabled")
        object.__setattr__(self, "action_sign_flips", flips)
        object.__setattr__(self, "disabled_action_indices", disabled)

        if self.wind_body_ids is not None:
            object.__setattr__(
                self,
                "wind_body_ids",
                _immutable_indices(self.wind_body_ids, "wind_body_ids"),
            )

        for field_name in ("target_velocity", "wind_force_x"):
            if not np.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        for field_name in ("gravity_scale", "mass_scale", "action_scale"):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and strictly positive")


@dataclass(frozen=True)
class SchedulePosition:
    """Resolved immutable schedule metadata for one global environment step."""

    regime: RegimeSpec
    segment_index: int
    visit_index: int
    start_step: int
    end_step: int
    step_in_segment: int


@dataclass(frozen=True)
class Schedule:
    """An immutable finite sequence of regimes and positive segment durations."""

    regimes: Tuple[RegimeSpec, ...]
    durations: Tuple[int, ...]
    _start_steps: Tuple[int, ...] = field(init=False, repr=False, compare=False)
    _end_steps: Tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        regimes = tuple(self.regimes)
        raw_durations = tuple(self.durations)
        if not regimes:
            raise ValueError("A schedule must contain at least one regime")
        if len(regimes) != len(raw_durations):
            raise ValueError("regimes and durations must have the same length")

        durations = []
        for duration in raw_durations:
            if isinstance(duration, bool) or not isinstance(duration, Integral):
                raise TypeError(f"Schedule durations must be integers, got {duration!r}")
            duration = int(duration)
            if duration <= 0:
                raise ValueError("Schedule durations must be strictly positive")
            durations.append(duration)

        # A regime name is the oracle-routing key.  Reusing it for a different
        # MDP would silently leak state between contexts, so reject that case.
        by_name: Dict[str, RegimeSpec] = {}
        for regime in regimes:
            previous = by_name.setdefault(regime.name, regime)
            if previous != regime:
                raise ValueError(
                    f"Regime name {regime.name!r} refers to multiple specifications"
                )

        starts = []
        ends = []
        cursor = 0
        for duration in durations:
            starts.append(cursor)
            cursor += duration
            ends.append(cursor)

        object.__setattr__(self, "regimes", regimes)
        object.__setattr__(self, "durations", tuple(durations))
        object.__setattr__(self, "_start_steps", tuple(starts))
        object.__setattr__(self, "_end_steps", tuple(ends))

    @property
    def total_steps(self) -> int:
        return self._end_steps[-1]

    @property
    def boundaries(self) -> Tuple[int, ...]:
        """Segment start steps, including zero."""

        return self._start_steps

    def position_at(self, global_step: int) -> SchedulePosition:
        if isinstance(global_step, bool) or not isinstance(global_step, Integral):
            raise TypeError("global_step must be an integer")
        global_step = int(global_step)
        if global_step < 0 or global_step >= self.total_steps:
            raise IndexError(
                f"global_step {global_step} is outside [0, {self.total_steps})"
            )

        segment_index = bisect_right(self._end_steps, global_step)
        regime = self.regimes[segment_index]
        visit_index = sum(
            earlier.name == regime.name for earlier in self.regimes[:segment_index]
        )
        start = self._start_steps[segment_index]
        return SchedulePosition(
            regime=regime,
            segment_index=segment_index,
            visit_index=int(visit_index),
            start_step=start,
            end_step=self._end_steps[segment_index],
            step_in_segment=global_step - start,
        )

    def regime_at(self, global_step: int) -> RegimeSpec:
        return self.position_at(global_step).regime


def schedule_from_steps(
    total_steps: int,
    entries: Iterable[Mapping[str, Any]],
    regimes: Optional[Mapping[str, RegimeSpec]] = None,
) -> Schedule:
    """Build a ``Schedule`` from a generic ``{start_step, regime}`` config.

    This mirrors the experiment-config format::

        nonstationary:
            enabled: true
            total_steps: 10000000
            schedule:
                - start_step: 0
                  regime: A
                - start_step: 5000000
                  regime: B
                - start_step: 7500000
                  regime: A

    Each entry is ``{"start_step": int, "regime": <RegimeSpec | name | dict>}``.
    ``regimes`` optionally maps names to ``RegimeSpec`` objects; a bare dict is
    passed to ``RegimeSpec(**spec)``.  Start steps must be strictly increasing,
    the first must be 0, and the final segment runs to ``total_steps``.
    """

    entry_list = [dict(entry) for entry in entries]
    if not entry_list:
        raise ValueError("A schedule config must contain at least one entry")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    start_steps = []
    specs = []
    for entry in entry_list:
        start = int(entry["start_step"])
        if start_steps and start <= start_steps[-1]:
            raise ValueError("schedule start_steps must be strictly increasing")
        start_steps.append(start)
        raw = entry["regime"]
        if isinstance(raw, RegimeSpec):
            spec = raw
        elif regimes is not None and isinstance(raw, str):
            if raw not in regimes:
                raise ValueError(f"Unknown regime name {raw!r} in schedule config")
            spec = regimes[raw]
        elif isinstance(raw, Mapping):
            spec = RegimeSpec(**dict(raw))
        else:
            raise TypeError(
                f"schedule regime must be a RegimeSpec, name, or dict; got {raw!r}"
            )
        specs.append(spec)

    if start_steps[0] != 0:
        raise ValueError("The first schedule entry must have start_step 0")
    if start_steps[-1] >= total_steps:
        raise ValueError("The last start_step must be less than total_steps")

    boundaries = start_steps + [int(total_steps)]
    durations = [
        int(end - start) for start, end in zip(boundaries, boundaries[1:])
    ]
    return Schedule(tuple(specs), tuple(durations))


PAPER_SEGMENT_STEPS = 40_000
PAPER_NORMAL = RegimeSpec(name="normal", target_velocity=1.5)
PAPER_JOINT_MALFUNCTION = RegimeSpec(
    name="joint_flip_0_1", target_velocity=1.5, action_sign_flips=(0, 1)
)
PAPER_WIND = RegimeSpec(
    name="wind_x_neg4", target_velocity=1.5, wind_force_x=-4.0
)
PAPER_VELOCITY = RegimeSpec(name="velocity_2.0", target_velocity=2.0)

# This includes the released code's deliberate reorder at step 320k.
_PAPER_REGIME_ORDER = (
    PAPER_NORMAL,
    PAPER_JOINT_MALFUNCTION,
    PAPER_WIND,
    PAPER_VELOCITY,
    PAPER_NORMAL,
    PAPER_JOINT_MALFUNCTION,
    PAPER_WIND,
    PAPER_VELOCITY,
    PAPER_WIND,
    PAPER_NORMAL,
    PAPER_VELOCITY,
    PAPER_JOINT_MALFUNCTION,
)
PAPER_SCHEDULE = Schedule(
    regimes=_PAPER_REGIME_ORDER,
    durations=(PAPER_SEGMENT_STEPS,) * len(_PAPER_REGIME_ORDER),
)


def paper_schedule() -> Schedule:
    """Return the immutable, exact 480k-step released-paper schedule."""

    return PAPER_SCHEDULE


def _readonly_copy(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


class NonStationaryHalfCheetah(_WrapperBase):
    """Gymnasium v4/v5 wrapper for scheduled HalfCheetah MDP changes.

    The context for step ``t`` is installed before the action at ``t`` is sent
    to the underlying environment.  Physics parameters are always reconstructed
    from pristine arrays captured at construction, making context changes
    idempotent, reversible, and non-compounding.
    """

    def __init__(
        self,
        env: Any,
        schedule: Optional[Schedule] = None,
        *,
        control_cost_weight: float = 0.1,
        initial_global_step: int = 0,
    ) -> None:
        super().__init__(env)
        self.schedule = PAPER_SCHEDULE if schedule is None else schedule
        if not isinstance(self.schedule, Schedule):
            raise TypeError("schedule must be a Schedule")
        if not np.isfinite(control_cost_weight) or control_cost_weight < 0:
            raise ValueError("control_cost_weight must be finite and non-negative")
        if (
            isinstance(initial_global_step, bool)
            or not isinstance(initial_global_step, Integral)
            or initial_global_step < 0
            or initial_global_step >= self.schedule.total_steps
        ):
            raise ValueError("initial_global_step must lie within the schedule")

        self.control_cost_weight = float(control_cost_weight)
        self.global_step = int(initial_global_step)
        self._pending_context_change = False

        raw_env = self.unwrapped
        try:
            self._model = raw_env.model
            self._data = raw_env.data
            self._base_gravity = _readonly_copy(self._model.opt.gravity)
            self._base_body_mass = _readonly_copy(self._model.body_mass)
            self._base_body_inertia = _readonly_copy(self._model.body_inertia)
        except AttributeError as exc:
            raise TypeError(
                "NonStationaryHalfCheetah requires a MuJoCo-style unwrapped "
                "environment exposing model, data, model.opt.gravity, "
                "model.body_mass, and model.body_inertia"
            ) from exc

        xfrc_applied = getattr(self._data, "xfrc_applied", None)
        self._base_xfrc_applied = (
            None if xfrc_applied is None else _readonly_copy(xfrc_applied)
        )

        action_shape = tuple(getattr(self.action_space, "shape", ()))
        if len(action_shape) != 1 or action_shape[0] <= 0:
            raise TypeError("HalfCheetah must expose a one-dimensional action space")
        self._action_shape = action_shape
        for regime in self.schedule.regimes:
            self._validate_regime_for_environment(regime)

        self._current_regime: Optional[RegimeSpec] = None
        initial = self.schedule.position_at(self.global_step).regime
        self.set_context(initial)

    @property
    def current_regime(self) -> RegimeSpec:
        assert self._current_regime is not None
        return self._current_regime

    def _validate_regime_for_environment(self, regime: RegimeSpec) -> None:
        action_dim = self._action_shape[0]
        for index in regime.action_sign_flips + regime.disabled_action_indices:
            if index >= action_dim:
                raise ValueError(
                    f"Action index {index} is invalid for action dimension {action_dim}"
                )

        if regime.wind_force_x != 0.0:
            if self._base_xfrc_applied is None:
                raise TypeError("This environment does not expose data.xfrc_applied")
            nbody = self._base_xfrc_applied.shape[0]
            if self._base_xfrc_applied.ndim != 2 or self._base_xfrc_applied.shape[1] != 6:
                raise TypeError("data.xfrc_applied must have shape (nbody, 6)")
            if regime.wind_body_ids is not None:
                for body_id in regime.wind_body_ids:
                    if body_id >= nbody:
                        raise ValueError(
                            f"Wind body index {body_id} is invalid for {nbody} bodies"
                        )

    def _refresh_mujoco(self) -> None:
        """Refresh derived MuJoCo constants when real bindings are available."""

        try:
            import mujoco
        except ImportError:  # Fake environments and dependency-light unit tests.
            return
        if not isinstance(self._model, mujoco.MjModel):
            return
        mujoco.mj_setConst(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def _restore_external_forces(self) -> None:
        if self._base_xfrc_applied is not None:
            self._data.xfrc_applied[...] = self._base_xfrc_applied

    def _wind_body_ids(self, regime: RegimeSpec) -> IndexTuple:
        if regime.wind_force_x == 0.0 or self._base_xfrc_applied is None:
            return ()
        if regime.wind_body_ids is not None:
            return regime.wind_body_ids
        # Body 0 is MuJoCo's world body.  The released environment applies the
        # -4 x-force to every actual/named robot body.
        return tuple(range(1, self._base_xfrc_applied.shape[0]))

    def _apply_wind_force(self, regime: RegimeSpec) -> IndexTuple:
        self._restore_external_forces()
        body_ids = self._wind_body_ids(regime)
        if body_ids:
            selected = np.asarray(body_ids, dtype=np.intp)
            # Match the released wrapper: assign the full force/torque vector
            # ``[wind_force_x, 0, 0, 0, 0, 0]`` rather than accumulating it.
            self._data.xfrc_applied[selected, :] = 0.0
            self._data.xfrc_applied[selected, 0] = regime.wind_force_x
        return body_ids

    def set_context(self, regime: RegimeSpec) -> None:
        """Install ``regime`` absolutely from the pristine physics snapshot."""

        if not isinstance(regime, RegimeSpec):
            raise TypeError("regime must be a RegimeSpec")
        self._validate_regime_for_environment(regime)

        self._model.opt.gravity[...] = self._base_gravity * regime.gravity_scale

        self._model.body_mass[...] = self._base_body_mass
        self._model.body_inertia[...] = self._base_body_inertia
        # Keep the world body untouched even if a non-unit mass scale is used.
        self._model.body_mass[1:] = self._base_body_mass[1:] * regime.mass_scale
        self._model.body_inertia[1:] = (
            self._base_body_inertia[1:] * regime.mass_scale
        )

        self._restore_external_forces()
        self._refresh_mujoco()
        self._current_regime = regime

    # "Regime" and "context" are synonyms in the paper and experiment configs.
    set_regime = set_context

    def _sync_context_for_current_step(self) -> bool:
        position = self.schedule.position_at(self.global_step)
        if position.regime != self.current_regime:
            self.set_context(position.regime)
            return True
        return False

    def _action_mask(self, regime: RegimeSpec) -> np.ndarray:
        mask = np.ones(self._action_shape, dtype=np.float64)
        if regime.action_sign_flips:
            mask[np.asarray(regime.action_sign_flips, dtype=np.intp)] = -1.0
        if regime.disabled_action_indices:
            mask[np.asarray(regime.disabled_action_indices, dtype=np.intp)] = 0.0
        # ``action_scale`` models actuator effectiveness: the torque MuJoCo
        # actually receives is the commanded, bounded action times this scale.
        mask *= regime.action_scale
        return mask

    def reset(self, **kwargs: Any) -> Any:
        # If an episode ends exactly on a boundary, install the new context
        # before reset_model runs.  Retain the event so the first step's info
        # still reports the change.
        if self.global_step < self.schedule.total_steps:
            self._pending_context_change |= self._sync_context_for_current_step()
        else:
            self.set_context(self.current_regime)

        result = self.env.reset(**kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        observation, raw_info = result
        info = dict(raw_info)
        info.update(
            {
                "regime_id": self.current_regime.name,
                "target_velocity": self.current_regime.target_velocity,
                "global_step": self.global_step,
            }
        )
        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self.global_step >= self.schedule.total_steps:
            raise RuntimeError(
                "The non-stationary schedule is exhausted; create a longer Schedule "
                "instead of stepping beyond its declared budget"
            )

        position = self.schedule.position_at(self.global_step)
        changed = self._sync_context_for_current_step() or self._pending_context_change
        self._pending_context_change = False
        regime = self.current_regime

        commanded_action = np.asarray(action, dtype=np.float64)
        if commanded_action.shape != self._action_shape:
            raise ValueError(
                f"Expected action shape {self._action_shape}, got {commanded_action.shape}"
            )
        commanded_action = commanded_action.copy()

        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        bounded_action = np.clip(commanded_action, low, high)
        action_mask = self._action_mask(regime)
        executed_action = bounded_action * action_mask

        position_before = None
        qpos = getattr(self._data, "qpos", None)
        if qpos is not None and len(qpos):
            position_before = float(qpos[0])

        wind_body_ids = self._apply_wind_force(regime)
        result = self.env.step(executed_action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError(
                "Gymnasium HalfCheetah-v4/v5 step() must return "
                "(observation, reward, terminated, truncated, info)"
            )
        next_observation, base_reward, terminated, truncated, raw_info = result
        info = dict(raw_info)

        if "x_velocity" in info:
            x_velocity = float(np.asarray(info["x_velocity"]).item())
        elif position_before is not None and qpos is not None and hasattr(self.unwrapped, "dt"):
            x_velocity = (float(qpos[0]) - position_before) / float(self.unwrapped.dt)
        else:
            raise KeyError(
                "Could not obtain x velocity from info['x_velocity'] or MuJoCo qpos/dt"
            )

        velocity_error = abs(x_velocity - regime.target_velocity)
        control_cost = self.control_cost_weight * float(np.square(executed_action).sum())
        tracking_reward = -velocity_error
        paper_reward = tracking_reward - control_cost
        reward = float(base_reward) if regime.use_original_reward else paper_reward

        info.update(
            {
                "base_env_reward": float(base_reward),
                "paper_reward": paper_reward,
                "reward_velocity_tracking": tracking_reward,
                "reward_control": -control_cost,
                "velocity_error": velocity_error,
                "x_velocity": x_velocity,
                "target_velocity": regime.target_velocity,
                "use_original_reward": regime.use_original_reward,
                "regime_id": regime.name,
                "regime_index": position.segment_index,
                "regime_visit_index": position.visit_index,
                "context_changed": bool(changed),
                "segment_boundary": position.step_in_segment == 0,
                "global_step": self.global_step,
                "step_in_segment": position.step_in_segment,
                "segment_start_step": position.start_step,
                "segment_end_step": position.end_step,
                "commanded_action": commanded_action,
                "bounded_action": bounded_action.copy(),
                "executed_action": executed_action.copy(),
                "action_mask": action_mask.copy(),
                "wind_force_x": regime.wind_force_x,
                "wind_body_ids": wind_body_ids,
                "gravity_scale": regime.gravity_scale,
                "mass_scale": regime.mass_scale,
                "action_scale": regime.action_scale,
            }
        )

        self.global_step += 1
        return (
            next_observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )


# A descriptive alias for callers that suffix Gymnasium environments with Env.
NonStationaryHalfCheetahEnv = NonStationaryHalfCheetah


__all__ = [
    "NonStationaryHalfCheetah",
    "NonStationaryHalfCheetahEnv",
    "PAPER_JOINT_MALFUNCTION",
    "PAPER_NORMAL",
    "PAPER_SCHEDULE",
    "PAPER_SEGMENT_STEPS",
    "PAPER_VELOCITY",
    "PAPER_WIND",
    "RegimeSpec",
    "Schedule",
    "SchedulePosition",
    "paper_schedule",
    "schedule_from_steps",
]
