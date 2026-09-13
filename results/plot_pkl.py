#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Rolling average window in episodes"
    )
    args = parser.parse_args()

    path = Path(args.file)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    with open(path, "rb") as f:
        ep_steps, returns, env_name = pickle.load(f)

    ep_steps = np.asarray(ep_steps)
    returns = np.asarray(returns)

    # ep_steps contains episode LENGTHS.
    # Convert them to cumulative environment steps.
    env_steps = np.cumsum(ep_steps)

    print(f"Environment: {env_name}")
    print(f"Episodes: {len(returns)}")
    print(f"Total steps: {env_steps[-1]:,}")
    print(f"Return range: {returns.min():.2f} -> {returns.max():.2f}")

    # ------------------------------------------------------------
    # Smooth
    # ------------------------------------------------------------

    df = pd.DataFrame({
        "step": env_steps,
        "return": returns,
    })

    df["smoothed_return"] = (
        df["return"]
        .rolling(
            window=args.window,
            min_periods=1
        )
        .mean()
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        df["step"],
        df["smoothed_return"],
        linewidth=2,
    )

    # If this is an ABA experiment, show regime boundaries.
    if env_steps[-1] >= 10_000_000:
        ax.axvline(
            5_000_000,
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        ax.axvline(
            10_000_000,
            linestyle="--",
            color="black",
            alpha=0.7,
        )

    ax.set_xlabel("Environment step")
    ax.set_ylabel("Episode return")
    ax.set_title(path.stem)

    ax.grid(alpha=0.25)

    fig.tight_layout()

    output = path.with_suffix(".png")

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
