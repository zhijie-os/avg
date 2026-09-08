"""Explicit, serializable observation normalization for changing-MDP runs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


class ObservationNormalizer:
    """Track and apply per-coordinate running mean and variance.

    The statistics are kept outside the environment so an experiment runner can
    reset them at an oracle boundary or keep one independent normalizer for each
    oracle-mixture expert.
    """

    def __init__(
        self,
        shape: int | Sequence[int],
        epsilon: float = 1e-8,
        initial_count: float = 1e-4,
    ) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(int(dimension) for dimension in shape)
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("shape must contain positive dimensions")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if initial_count < 0:
            raise ValueError("initial_count must be non-negative")
        self.epsilon = float(epsilon)
        self.initial_count = float(initial_count)
        self.reset()

    def reset(self) -> None:
        """Return running statistics to their initial neutral values."""

        self.mean = np.zeros(self.shape, dtype=np.float32)
        self.var = np.ones(self.shape, dtype=np.float32)
        self.count = self.initial_count

    def _as_batch(self, observations: np.ndarray) -> np.ndarray:
        if observations.shape == self.shape:
            return observations.reshape((1,) + self.shape)
        if observations.ndim < len(self.shape) + 1:
            raise ValueError(
                f"expected observation shape {self.shape} or a batch ending in "
                f"{self.shape}; got {observations.shape}"
            )
        if tuple(observations.shape[-len(self.shape) :]) != self.shape:
            raise ValueError(
                f"expected observation shape {self.shape} or a batch ending in "
                f"{self.shape}; got {observations.shape}"
            )
        return observations.reshape((-1,) + self.shape)

    def update(self, observations: np.ndarray | Sequence[float]) -> None:
        """Update moments from one observation or a batch of observations."""

        values = np.asarray(observations, dtype=np.float32)
        batch = self._as_batch(values)
        if batch.shape[0] == 0:
            raise ValueError("cannot update from an empty observation batch")

        batch_mean = np.mean(batch, axis=0, dtype=np.float64)
        batch_var = np.var(batch, axis=0, dtype=np.float64)
        batch_count = float(batch.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        correction = delta**2 * self.count * batch_count / total_count
        new_var = (old_m2 + batch_m2 + correction) / total_count

        self.mean = np.asarray(new_mean, dtype=np.float32)
        self.var = np.asarray(new_var, dtype=np.float32)
        self.count = total_count

    def normalize(
        self,
        observation: np.ndarray | Sequence[float],
        update: bool = True,
    ) -> np.ndarray:
        """Normalize an observation, optionally incorporating it first."""

        value = np.asarray(observation, dtype=np.float32)
        self._as_batch(value)  # Shape validation for both single and batched input.
        if update:
            self.update(value)
        return np.asarray(
            (value - self.mean) / np.sqrt(self.var + self.epsilon),
            dtype=np.float32,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "initial_count": self.initial_count,
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state_shape = tuple(int(dimension) for dimension in state["shape"])
        if state_shape != self.shape:
            raise ValueError(
                f"normalizer shape {self.shape} does not match state {state_shape}"
            )
        mean = np.asarray(state["mean"], dtype=np.float32)
        variance = np.asarray(state["var"], dtype=np.float32)
        if mean.shape != self.shape or variance.shape != self.shape:
            raise ValueError("normalizer state has invalid moment shapes")
        if np.any(variance < 0):
            raise ValueError("normalizer variance cannot be negative")

        self.epsilon = float(state["epsilon"])
        self.initial_count = float(state.get("initial_count", 1e-4))
        self.mean = mean.copy()
        self.var = variance.copy()
        self.count = float(state["count"])

    def clone(self) -> "ObservationNormalizer":
        duplicate = ObservationNormalizer(
            self.shape,
            epsilon=self.epsilon,
            initial_count=self.initial_count,
        )
        duplicate.load_state_dict(self.state_dict())
        return duplicate


__all__ = ["ObservationNormalizer"]
