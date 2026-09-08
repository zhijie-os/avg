"""Aggregate and plot AVG changing-MDP runs without altering raw data.

Point it at a results tree containing one or more runs produced by
``run_*.py`` / ``halfcheetah_changing_mdp``.  Completed runs are discovered
through their ``metadata.json``, validated to share the same environment,
budget, and schedule, and drawn on one figure (mean over seeds with
+/- one std shading).

Usage:
    python3 plot_results.py --results-root results --experiment gravity_aba

Also writes ``<output stem>_visit_metrics.csv`` with per-regime-visit
statistics: first episode return after a switch, mean return, last-20%
return (steady-state proxy), and cumulative reward, so recovery time and the
gap to oracle_mixture can be computed afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

BASELINE_ORDER = (
    "stationary",
    "continuous_avg",
    "oracle_optimizer_reset",
    "oracle_full_reset",
    "oracle_mixture",
)

# Categorical slots in fixed order (validated, colorblind-safe adjacent
# pairlist).  Color follows the baseline identity, never its rank or count.
SERIES_COLORS = {
    "stationary": "#2a78d6",            # slot 1: blue
    "continuous_avg": "#eb6834",        # slot 2: orange
    "oracle_optimizer_reset": "#1baf7a",  # slot 3: aqua
    "oracle_full_reset": "#eda100",     # slot 4: yellow
    "oracle_mixture": "#e87ba4",        # slot 5: magenta
}

# Neutral inks and regime tints (no hue competition with the series).
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
REGIME_TINTS = ("#f0efec", "#e4e2de")
BOUNDARY_COLOR = "#6b7280"


@dataclass
class Run:
    path: Path
    metadata: dict
    episodes: list[dict]

    @property
    def label(self) -> str:
        baseline = self.metadata["baseline"]
        if baseline == "stationary":
            return f"stationary:{self.metadata['schedule'][0]['regime_id']}"
        return baseline


def read_runs(root: Path, experiment: str | None) -> list[Run]:
    runs: list[Run] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        episodes_path = metadata_path.with_name("episodes.csv")
        if not episodes_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "completed":
                continue
            if experiment is not None and metadata.get("experiment") != experiment:
                continue
            with episodes_path.open(newline="", encoding="utf-8") as stream:
                episodes = [
                    row
                    for row in csv.DictReader(stream)
                    if row.get("partial", "False").lower() != "true"
                ]
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Skipping malformed run {metadata_path.parent}: {exc}")
            continue
        if episodes:
            runs.append(Run(metadata_path.parent, metadata, episodes))
    return runs


def validate_comparison_set(runs: list[Run]) -> None:
    """Reject accidental averaging of incompatible or duplicate campaigns."""

    if not runs:
        raise ValueError("No completed runs found under the results root")

    environment_horizons = {
        (
            run.metadata.get("environment_id"),
            int(run.metadata.get("total_steps", -1)),
        )
        for run in runs
    }
    if len(environment_horizons) != 1:
        raise ValueError(
            "Runs mix environment IDs or training horizons; point --results-root "
            "at one comparable campaign"
        )

    experiments = {run.metadata.get("experiment") for run in runs}
    if len(experiments) != 1:
        raise ValueError(
            f"Runs belong to different experiments {sorted(experiments)}; "
            "filter with --experiment"
        )

    nonstationary_schedules = {
        tuple(
            (
                segment.get("regime_id"),
                int(segment.get("start_step", -1)),
                int(segment.get("end_step", -1)),
                json.dumps(segment.get("spec", {}), sort_keys=True),
            )
            for segment in run.metadata.get("schedule", [])
        )
        for run in runs
        if run.metadata.get("baseline") != "stationary"
    }
    if len(nonstationary_schedules) > 1:
        raise ValueError(
            "Non-stationary runs use different schedules; analyze each schedule "
            "under a separate results root"
        )

    seen: dict[tuple[str, int], Path] = {}
    for run in runs:
        key = (run.label, int(run.metadata["seed"]))
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"Duplicate baseline/seed {key} in {previous} and {run.path}; "
                "remove one rerun to avoid pseudo-replication"
            )
        seen[key] = run.path


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def aligned_curves(
    runs: Iterable[Run], smooth_window: int
) -> tuple[np.ndarray, np.ndarray]:
    curves: list[dict[int, float]] = []
    for run in runs:
        x = np.asarray([int(row["end_step"]) for row in run.episodes], dtype=np.int64)
        y = np.asarray([float(row["episode_return"]) for row in run.episodes])
        y = moving_average(y, smooth_window)
        curves.append(dict(zip(x.tolist(), y.tolist())))
    common_steps = sorted(set.intersection(*(set(curve) for curve in curves)))
    if not common_steps:
        raise ValueError("Runs have no common episode end steps")
    values = np.asarray(
        [[curve[step] for step in common_steps] for curve in curves],
        dtype=np.float64,
    )
    return np.asarray(common_steps, dtype=np.int64), values


def write_visit_metrics(runs: list[Run], output: Path) -> Path:
    rows: list[dict] = []
    for run in runs:
        schedule = run.metadata.get("schedule", [])
        for segment in schedule:
            start = int(segment["start_step"])
            end = int(segment["end_step"])
            selected = [
                row
                for row in run.episodes
                if start <= int(row["end_step"]) - 1 < end
            ]
            if not selected:
                continue
            returns = np.asarray(
                [float(row["episode_return"]) for row in selected], dtype=np.float64
            )
            tail_count = max(1, int(np.ceil(0.2 * len(returns))))
            rows.append(
                {
                    "experiment": run.metadata.get("experiment"),
                    "baseline": run.label,
                    "seed": run.metadata["seed"],
                    "segment_index": segment["segment_index"],
                    "visit_index": segment["visit_index"],
                    "regime_id": segment["regime_id"],
                    "start_step": start,
                    "end_step": end,
                    "episodes": len(returns),
                    "first_episode_return": returns[0],
                    "mean_episode_return": returns.mean(),
                    "last_20pct_return": returns[-tail_count:].mean(),
                    "cumulative_reward_auc": returns.sum(),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "experiment", "baseline", "seed", "segment_index", "visit_index",
        "regime_id", "start_step", "end_step", "episodes",
        "first_episode_return", "mean_episode_return", "last_20pct_return",
        "cumulative_reward_auc",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _order_label(label: str) -> int:
    if label.startswith("stationary"):
        return 0
    try:
        return BASELINE_ORDER.index(label)
    except ValueError:
        return len(BASELINE_ORDER)


def plot_runs(runs: list[Run], output: Path, smooth_window: int) -> None:
    import matplotlib.pyplot as plt

    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.label].append(run)

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ordered = sorted(groups.items(), key=lambda item: _order_label(item[0]))
    for label, group in ordered:
        x, values = aligned_curves(group, smooth_window)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        color = SERIES_COLORS.get(
            label if label in SERIES_COLORS else label.split(":")[0], "#757575"
        )
        (line,) = ax.plot(
            x,
            mean,
            linewidth=2,
            color=color,
            label=f"{label} (n={len(group)})",
        )
        if len(group) > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
        # Direct label at the end of each curve (text wears ink, not color).
        ax.annotate(
            label,
            xy=(x[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )

    nonstationary = next(
        (run for run in runs if run.metadata["baseline"] != "stationary"), None
    )
    if nonstationary is not None:
        for index, segment in enumerate(nonstationary.metadata.get("schedule", [])):
            start = int(segment["start_step"])
            end = int(segment["end_step"])
            regime = segment["regime_id"]
            ax.axvspan(
                start,
                end,
                color=REGIME_TINTS[index % len(REGIME_TINTS)],
                alpha=0.5,
                zorder=0,
            )
            if index:
                ax.axvline(start, color=BOUNDARY_COLOR, linewidth=0.6, alpha=0.7)
            ax.text(
                (start + end) / 2,
                0.985,
                regime,
                ha="center",
                va="top",
                fontsize=9,
                color=TEXT_SECONDARY,
                transform=ax.get_xaxis_transform(),
            )

    experiment = runs[0].metadata.get("experiment", "")
    ax.set_title(
        f"AVG baseline ladder on changing Half-Cheetah MDPs ({experiment})",
        color=TEXT_PRIMARY,
    )
    ax.set_xlabel("Environment step", color=TEXT_SECONDARY)
    ax.set_ylabel("Online episode return (smoothed)", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d1")
    ax.grid(alpha=0.2)
    if len(ordered) > 1:
        legend = ax.legend(loc="upper left", fontsize=8, ncol=1, frameon=False)
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Only plot runs with this experiment name (e.g. gravity_aba).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Defaults to <output stem>_visit_metrics.csv.",
    )
    parser.add_argument("--smooth-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be at least 1")
    runs = read_runs(args.results_root, args.experiment)
    validate_comparison_set(runs)
    metrics_output = args.metrics_output or args.output.with_name(
        f"{args.output.stem}_visit_metrics.csv"
    )
    plot_runs(runs, args.output, args.smooth_window)
    write_visit_metrics(runs, metrics_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {metrics_output}")


if __name__ == "__main__":
    main()
