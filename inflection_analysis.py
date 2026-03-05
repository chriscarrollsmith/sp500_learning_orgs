"""
Inflection analysis: measure which companies are trending up vs down
on learning org scores, and test whether that predicts forward returns.

Approach:
  - For each transcript, compute a trailing slope of composite scores
    over the prior 8 quarters (2 years) using OLS
  - Classify as "positive inflection" vs "negative inflection"
  - Test whether inflection predicts forward 6-month returns
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCORES_WITH_RETURNS = "data/scores_with_returns.csv"
OUTPUT_DIR = "data"
TRAILING_QUARTERS = 8  # 2 years of history to compute slope


def compute_trailing_slopes(df, window=TRAILING_QUARTERS):
    """
    For each company-quarter, fit a simple linear regression of composite
    score over the prior `window` quarters. The slope is the inflection signal.
    """
    df = df.sort_values(["symbol", "date"]).copy()

    slopes = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i in range(len(grp)):
            # Need at least `window` quarters of history
            start = max(0, i - window + 1)
            trail = grp.iloc[start:i + 1]

            if len(trail) < 4:  # require at least 4 quarters
                slopes.append(np.nan)
                continue

            # Simple OLS: y = composite scores, x = 0,1,2,...
            x = np.arange(len(trail), dtype=float)
            y = trail["composite"].values
            # slope = cov(x,y) / var(x)
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            slopes.append(slope)

    df["composite_slope"] = slopes
    return df


def analyze(df):
    """Run the inflection analysis."""
    # Drop rows without slope or return
    df = df.dropna(subset=["composite_slope", "fwd_6m_return"]).copy()
    print(f"{len(df)} observations with both slope and forward returns")

    # --- 1. Correlation ---
    corr = df[["composite_slope", "fwd_6m_return"]].corr().iloc[0, 1]
    print(f"\nCorrelation (slope vs fwd 6m return): r = {corr:+.4f}")

    # Also check: slope signal vs level signal
    corr_level = df[["composite", "fwd_6m_return"]].corr().iloc[0, 1]
    print(f"Correlation (level vs fwd 6m return):  r = {corr_level:+.4f}")

    # --- 2. Quintile analysis on slope ---
    df["slope_quintile"] = pd.qcut(
        df["composite_slope"], 5,
        labels=["Q1 (Declining)", "Q2", "Q3", "Q4", "Q5 (Rising)"]
    )

    quintile_stats = (
        df.groupby("slope_quintile", observed=True)
        .agg(
            mean_return=("fwd_6m_return", "mean"),
            median_return=("fwd_6m_return", "median"),
            mean_level=("composite", "mean"),
            mean_slope=("composite_slope", "mean"),
            count=("fwd_6m_return", "count"),
        )
    )
    print("\n=== QUINTILE ANALYSIS (Composite Slope / Inflection) ===")
    print(quintile_stats.to_string(float_format="%.4f"))
    quintile_stats.to_csv(f"{OUTPUT_DIR}/inflection_quintile_analysis.csv")

    # --- 3. 2x2: Level x Slope interaction ---
    df["high_level"] = df["composite"] >= df["composite"].median()
    df["rising"] = df["composite_slope"] >= df["composite_slope"].median()

    interaction = (
        df.groupby(["high_level", "rising"])
        .agg(
            mean_return=("fwd_6m_return", "mean"),
            median_return=("fwd_6m_return", "median"),
            count=("fwd_6m_return", "count"),
        )
    )
    interaction.index = interaction.index.map({
        (False, False): "Low & Declining",
        (False, True):  "Low & Rising",
        (True, False):  "High & Declining",
        (True, True):   "High & Rising",
    })
    print("\n=== 2x2: LEVEL x SLOPE INTERACTION ===")
    print(interaction.to_string(float_format="%.4f"))
    interaction.to_csv(f"{OUTPUT_DIR}/level_slope_interaction.csv")

    # --- 4. Current inflectors: most recent quarter ---
    latest_date = df["date"].max()
    # Use last year of data to be safe
    recent = df[df["date"] >= (latest_date - pd.Timedelta(days=180))].copy()

    top_risers = (
        recent.groupby(["symbol", "company_name"])
        .agg(
            latest_slope=("composite_slope", "last"),
            latest_composite=("composite", "last"),
        )
        .sort_values("latest_slope", ascending=False)
        .head(20)
    )
    print("\n=== TOP 20 POSITIVELY INFLECTING COMPANIES (Recent) ===")
    print(top_risers.to_string(float_format="%.4f"))
    top_risers.to_csv(f"{OUTPUT_DIR}/top_risers.csv")

    top_decliners = (
        recent.groupby(["symbol", "company_name"])
        .agg(
            latest_slope=("composite_slope", "last"),
            latest_composite=("composite", "last"),
        )
        .sort_values("latest_slope", ascending=True)
        .head(20)
    )
    print("\n=== TOP 20 NEGATIVELY INFLECTING COMPANIES (Recent) ===")
    print(top_decliners.to_string(float_format="%.4f"))
    top_decliners.to_csv(f"{OUTPUT_DIR}/top_decliners.csv")

    # --- 5. Plots ---
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot A: Slope quintile returns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    q_means = df.groupby("slope_quintile", observed=True)["fwd_6m_return"].mean()
    colors = sns.color_palette("RdYlGn", 5)
    q_means.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Mean 6-Month Forward Return\nby Learning Org Score SLOPE Quintile")
    ax.set_ylabel("Mean Forward 6-Month Return")
    ax.set_xlabel("Composite Score Slope (Trailing 8Q)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(q_means):
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=15)

    # Plot B: 2x2 interaction
    ax = axes[1]
    groups = ["Low & Declining", "Low & Rising", "High & Declining", "High & Rising"]
    vals = [interaction.loc[g, "mean_return"] for g in groups]
    bar_colors = ["#d73027", "#fee08b", "#91bfdb", "#1a9850"]
    bars = ax.bar(groups, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Mean 6-Month Forward Return\nLevel x Slope Interaction")
    ax.set_ylabel("Mean Forward 6-Month Return")
    ax.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=10)
    plt.sca(ax)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/inflection_analysis.png", dpi=150)
    plt.close()

    # Plot C: Scatter slope vs return
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample["composite_slope"], sample["fwd_6m_return"],
               alpha=0.15, s=8, color="steelblue")
    z = np.polyfit(df["composite_slope"].values, df["fwd_6m_return"].values, 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["composite_slope"].quantile(0.01),
                           df["composite_slope"].quantile(0.99), 100)
    ax.plot(x_range, p(x_range), color="red", linewidth=2, label=f"r={corr:+.3f}")
    ax.set_title("Score Slope (Inflection) vs Forward 6-Month Return")
    ax.set_xlabel("Composite Score Slope (trailing 8Q)")
    ax.set_ylabel("Forward 6-Month Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/inflection_scatter.png", dpi=150)
    plt.close()

    print(f"\nPlots saved:")
    print(f"  {OUTPUT_DIR}/inflection_analysis.png")
    print(f"  {OUTPUT_DIR}/inflection_scatter.png")


def main():
    print("Loading scores with returns...")
    df = pd.read_csv(SCORES_WITH_RETURNS, parse_dates=["date"])
    print(f"  {len(df)} rows")

    print("Computing trailing slopes...")
    df = compute_trailing_slopes(df)

    analyze(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
