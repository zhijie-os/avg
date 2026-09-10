import pandas as pd
import matplotlib.pyplot as plt

path = "results/stationary_B_combined_stationary_20260909_164343_seed-0/episodes.csv"

df = pd.read_csv(path)

# Smooth episode return
window = 200
df["smoothed_return"] = df["episode_return"].rolling(
    window=window,
    min_periods=1
).mean()

plt.figure(figsize=(10, 6))

plt.plot(
    df["end_step"],
    df["smoothed_return"],
    linewidth=2
)

plt.xlabel("Environment step")
plt.ylabel("Online episode return")
plt.title("Stationary B — Combined")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("stationary_B_combined.png", dpi=200)
plt.show()
