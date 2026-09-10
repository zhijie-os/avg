#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
WINDOW = 200

EXPERIMENTS = [
    {
        "name": "Actuator",
        "ab": "min_delay_actuator_ab_continuous_avg_20260909_161504_seed-0",
        "stationary_b": "stationary_B_joint_stationary_20260909_164309_seed-0",
    },
    {
        "name": "Target velocity",
        "ab": "min_delay_target_velocity_ab_continuous_avg_20260909_161504_seed-0",
        "stationary_b": "stationary_B_velocity_stationary_20260909_164343_seed-0",
    },
    {
        "name": "Wind",
        "ab": "min_delay_wind_ab_continuous_avg_20260909_161504_seed-0",
        "stationary_b": "stationary_B_wind_stationary_20260909_164309_seed-0",
    },
]


def load_metadata(run_dir):
    with open(run_dir / "metadata.json") as f:
        return json.load(f)


def load_episodes(run_dir):
    df = pd.read_csv(run_dir / "episodes.csv")

    # Remove partial episodes
    if "partial" in df.columns:
        partial = (
            df["partial"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        df = df[~partial]

    # Remove episodes crossing a regime boundary
    if "crossed_context" in df.columns:
        crossed = (
            df["crossed_context"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        df = df[~crossed]

    return df.copy()


def smooth(df, window=WINDOW):
    df = df.sort_values("end_step").copy()

    df["smoothed_return"] = (
        df["episode_return"]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    return df


def plot_comparison(name, ab_name, stationary_name):
    ab_dir = RESULTS_DIR / ab_name
    stationary_dir = RESULTS_DIR / stationary_name

    metadata = load_metadata(ab_dir)
    schedule = metadata["schedule"]

    if len(schedule) < 2:
        raise ValueError(f"{ab_name} does not contain an A -> B schedule")

    # Assume schedule[0] = A and schedule[1] = B
    A = schedule[0]
    B = schedule[1]

    b_start = B["start_step"]
    b_end = B["end_step"]
    b_duration = b_end - b_start

    print()
    print("=" * 70)
    print(name)
    print(f"A: {A['start_step']:,} -> {A['end_step']:,}")
    print(f"B: {B['start_step']:,} -> {B['end_step']:,}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load A -> B continuous AVG
    # ---------------------------------------------------------

    ab = load_episodes(ab_dir)

    ab_A = ab[
        (ab["end_step"] > A["start_step"])
        & (ab["end_step"] <= A["end_step"])
    ].copy()

    ab_B = ab[
        (ab["end_step"] > B["start_step"])
        & (ab["end_step"] <= B["end_step"])
    ].copy()

    # Smooth A and B independently so A does not contaminate B
    ab_A = smooth(ab_A)
    ab_B = smooth(ab_B)

    # ---------------------------------------------------------
    # Load stationary B
    # ---------------------------------------------------------

    stationary = load_episodes(stationary_dir)

    # Only compare the same number of training steps as the B phase
    stationary = stationary[
        stationary["end_step"] <= b_duration
    ].copy()

    stationary = smooth(stationary)

    # Shift stationary B:
    #
    # stationary step 0     -> AB step 5M
    # stationary step 5M    -> AB step 10M
    #
    stationary["aligned_step"] = (
        stationary["end_step"] + b_start
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 7))

    # Same continuous AVG = SAME COLOR in A and B
    ax.plot(
        ab_A["end_step"],
        ab_A["smoothed_return"],
        linewidth=2.2,
        color="tab:blue",
        label="A→B continuous AVG",
    )

    ax.plot(
        ab_B["end_step"],
        ab_B["smoothed_return"],
        linewidth=2.2,
        color="tab:blue",
    )

    # Fresh B learner
    ax.plot(
        stationary["aligned_step"],
        stationary["smoothed_return"],
        linewidth=2.2,
        linestyle=":",
        color="tab:green",
        label="Stationary B (fresh learner)",
    )

    # A -> B transition
    ax.axvline(
        b_start,
        linestyle="--",
        linewidth=1.4,
        color="black",
        alpha=0.7,
    )

    # ---------------------------------------------------------
    # Regime labels
    # ---------------------------------------------------------

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.text(
        (A["start_step"] + A["end_step"]) / 2,
        ymax - 0.04 * yrange,
        "Regime A",
        ha="center",
        va="top",
        fontsize=13,
    )

    ax.text(
        (B["start_step"] + B["end_step"]) / 2,
        ymax - 0.04 * yrange,
        "Regime B",
        ha="center",
        va="top",
        fontsize=13,
    )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_title(
        f"{name}: A→B transfer vs B trained from scratch",
        fontsize=15,
    )

    ax.set_xlabel("Environment step")
    ax.set_ylabel("Online episode return")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = f"{name.lower().replace(' ', '_')}_ab_vs_stationary_B.png"

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Saved: {output}")

    plt.close(fig)


def main():
    for exp in EXPERIMENTS:
        plot_comparison(
            name=exp["name"],
            ab_name=exp["ab"],
            stationary_name=exp["stationary_b"],
        )

    print()
    print("Done. Generated 3 plots.")


if __name__ == "__main__":
    main()
