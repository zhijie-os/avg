#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def read_metadata(run_dir: Path):
    with open(run_dir / "metadata.json", "r") as f:
        return json.load(f)


def read_episodes(run_dir: Path):
    df = pd.read_csv(run_dir / "episodes.csv")

    # Drop partial episodes
    if "partial" in df.columns:
        partial = (
            df["partial"].astype(str).str.lower().isin(["true", "1"])
        )
        df = df[~partial]

    # Drop episodes crossing regime boundaries
    if "crossed_context" in df.columns:
        crossed = (
            df["crossed_context"].astype(str).str.lower().isin(["true", "1"])
        )
        df = df[~crossed]

    return df.copy()


def smooth(df: pd.DataFrame, window: int):
    df = df.sort_values("end_step").copy()
    df["smoothed_return"] = df["episode_return"].rolling(
        window=window,
        min_periods=1
    ).mean()
    return df


def plot_ab_by_segment(ax, ab_dir: Path, window: int):
    meta = read_metadata(ab_dir)
    schedule = meta["schedule"]
    episodes = read_episodes(ab_dir)

    first = True
    for seg in schedule:
        start = seg["start_step"]
        end = seg["end_step"]
        regime = seg["regime_id"]

        seg_df = episodes[
            (episodes["end_step"] > start) &
            (episodes["end_step"] <= end)
        ].copy()

        if len(seg_df) == 0:
            continue

        seg_df = smooth(seg_df, window)

        ax.plot(
            seg_df["end_step"],
            seg_df["smoothed_return"],
            linewidth=2,
            label="AB continuous AVG" if first else None,
        )
        first = False

    # Vertical regime boundaries
    for seg in schedule[1:]:
        ax.axvline(seg["start_step"], linestyle="--", linewidth=1.2, alpha=0.7)

    return schedule


def overlay_stationary_b_on_ab_b_segment(ax, stationary_dir: Path, b_start: int, b_end: int, window: int):
    episodes = read_episodes(stationary_dir)

    # Duration of AB's B segment
    b_duration = b_end - b_start

    # Keep only the first b_duration steps of stationary_B
    stationary_b = episodes[episodes["end_step"] <= b_duration].copy()

    if len(stationary_b) == 0:
        raise ValueError("No stationary_B episodes found within B-segment duration.")

    stationary_b = smooth(stationary_b, window)

    # Shift x-axis so stationary step 0 lines up with AB's B start
    shifted_x = stationary_b["end_step"] + b_start

    ax.plot(
        shifted_x,
        stationary_b["smoothed_return"],
        linewidth=2,
        linestyle=":",
        label="Stationary B (aligned to AB B segment)",
    )


def add_regime_labels(ax, schedule):
    ymin, ymax = ax.get_ylim()

    for i, seg in enumerate(schedule):
        mid = (seg["start_step"] + seg["end_step"]) / 2
        ax.text(
            mid,
            ymax - 0.04 * (ymax - ymin),
            f"Regime {seg['regime_id']}",
            ha="center",
            va="top",
            fontsize=11,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Overlay stationary_B on the B segment of an AB run."
    )
    parser.add_argument("--stationary-b", required=True,
                        help="Experiment dir name under results/ for stationary B run")
    parser.add_argument("--ab", required=True,
                        help="Experiment dir name under results/ for AB run")
    parser.add_argument("--results-dir", default="results",
                        help="Results parent directory")
    parser.add_argument("--window", type=int, default=200,
                        help="Rolling window size")
    parser.add_argument("--out", default="stationary_vs_ab.png",
                        help="Output PNG name")
    parser.add_argument("--title", default=None,
                        help="Optional custom title")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    stationary_dir = results_dir / args.stationary_b
    ab_dir = results_dir / args.ab

    if not stationary_dir.exists():
        raise FileNotFoundError(f"Missing directory: {stationary_dir}")
    if not ab_dir.exists():
        raise FileNotFoundError(f"Missing directory: {ab_dir}")

    fig, ax = plt.subplots(figsize=(12, 7))

    schedule = plot_ab_by_segment(ax, ab_dir, args.window)

    if len(schedule) < 2:
        raise ValueError("AB run does not appear to have at least two schedule segments.")

    # Assume segment 0 = A, segment 1 = B
    b_segment = schedule[1]
    b_start = b_segment["start_step"]
    b_end = b_segment["end_step"]

    overlay_stationary_b_on_ab_b_segment(
        ax=ax,
        stationary_dir=stationary_dir,
        b_start=b_start,
        b_end=b_end,
        window=args.window,
    )

    ax.set_xlabel("Environment step")
    ax.set_ylabel("Online episode return")

    if args.title is None:
        ax.set_title("Stationary B aligned to the B segment of A→B continuous AVG")
    else:
        ax.set_title(args.title)

    ax.grid(alpha=0.25)
    ax.legend()

    add_regime_labels(ax, schedule)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
