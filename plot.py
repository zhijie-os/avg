#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot online episode return for one AVG experiment."
    )

    parser.add_argument(
        "experiment",
        help="Experiment directory name under results/"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Rolling smoothing window in episodes (default: 200)"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results)"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG filename"
    )

    args = parser.parse_args()

    run_dir = args.results_dir / args.experiment

    episodes_path = run_dir / "episodes.csv"
    metadata_path = run_dir / "metadata.json"

    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Could not find {episodes_path}"
        )

    # ------------------------------------------------------------
    # Load episodes
    # ------------------------------------------------------------

    df = pd.read_csv(episodes_path)

    # Ignore partial episodes
    if "partial" in df.columns:
        partial = (
            df["partial"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        df = df[~partial]

    # Ignore episodes crossing an MDP boundary
    if "crossed_context" in df.columns:
        crossed = (
            df["crossed_context"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        )
        df = df[~crossed]

    # ------------------------------------------------------------
    # Load schedule
    # ------------------------------------------------------------

    schedule = None

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

        schedule = metadata.get("schedule")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    plt.figure(figsize=(11, 6))

    if schedule:
        # Smooth each regime separately so the moving average does not
        # leak across A -> B boundaries.
        for segment in schedule:
            start = segment["start_step"]
            end = segment["end_step"]
            regime = segment["regime_id"]

            seg = df[
                (df["end_step"] > start)
                & (df["end_step"] <= end)
            ].copy()

            if len(seg) == 0:
                continue

            seg["smoothed_return"] = (
                seg["episode_return"]
                .rolling(
                    window=args.window,
                    min_periods=1
                )
                .mean()
            )

            plt.plot(
                seg["end_step"],
                seg["smoothed_return"],
                linewidth=2,
            )

        # Draw regime boundaries
        for segment in schedule[1:]:
            plt.axvline(
                segment["start_step"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

        # Add regime names
        ymin, ymax = plt.ylim()

        for segment in schedule:
            midpoint = (
                segment["start_step"]
                + segment["end_step"]
            ) / 2

            plt.text(
                midpoint,
                ymax - 0.04 * (ymax - ymin),
                f"Regime {segment['regime_id']}",
                ha="center",
                va="top",
                fontsize=11,
            )

    else:
        # Fallback if metadata.json does not exist
        df["smoothed_return"] = (
            df["episode_return"]
            .rolling(
                window=args.window,
                min_periods=1
            )
            .mean()
        )

        plt.plot(
            df["end_step"],
            df["smoothed_return"],
            linewidth=2
        )

    plt.xlabel("Environment step")
    plt.ylabel("Online episode return")
    plt.title(args.experiment)

    plt.grid(alpha=0.25)
    plt.tight_layout()

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    if args.out is None:
        output = f"{args.experiment}.png"
    else:
        output = args.out

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
