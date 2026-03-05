"""
Join learning org scores with forward 6-month stock returns.

For each earnings call, compute the 6-month forward return from the
call date, then analyze whether higher learning-org scores predict
better returns.
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

SCORES_FILE = "data/quarterly_scores.csv"
PRICES_FILE = "data/stock_prices.parquet"
OUTPUT_DIR = "data"


def load_prices():
    """Load stock prices, keeping only close prices, indexed by (symbol, date)."""
    print("Loading stock prices...")
    # Read only needed columns
    df = pd.read_parquet(PRICES_FILE, columns=["symbol", "report_date", "close"])
    df["date"] = pd.to_datetime(df["report_date"])
    df["close"] = df["close"].astype(float)
    df = df.drop(columns=["report_date"]).sort_values(["symbol", "date"])
    print(f"  {len(df)} daily prices for {df['symbol'].nunique()} symbols")
    return df


def compute_forward_returns(scores_df, prices_df, forward_days=126):
    """
    For each earnings call, find the closing price on/after the call date
    and 126 trading days (~6 months) later. Compute forward return.
    """
    print(f"Computing {forward_days}-day forward returns...")
    results = []

    # Group prices by symbol for fast lookup
    price_groups = {sym: grp.set_index("date")["close"] for sym, grp in prices_df.groupby("symbol")}

    for _, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Returns"):
        sym = row["symbol"]
        call_date = row["date"]

        if sym not in price_groups:
            continue

        prices = price_groups[sym]

        # Find first available price on or after call date (within 5 business days)
        mask_start = prices.index >= call_date
        if mask_start.sum() == 0:
            continue
        start_prices = prices[mask_start]
        if len(start_prices) < forward_days + 1:
            continue

        start_price = start_prices.iloc[0]
        end_price = start_prices.iloc[min(forward_days, len(start_prices) - 1)]

        fwd_return = (end_price - start_price) / start_price

        results.append({
            **row.to_dict(),
            "start_price": float(start_price),
            "end_price": float(end_price),
            "fwd_6m_return": float(fwd_return),
        })

    result_df = pd.DataFrame(results)
    print(f"  {len(result_df)} transcript-return pairs computed")
    return result_df


def analyze_and_plot(df):
    """Generate analysis outputs and plots."""
    print("Generating analysis...")

    score_cols = ["learning_org", "lean_opex", "devops_agile", "psych_safety", "composite"]

    # --- 1. Correlation table ---
    print("\n=== CORRELATION: Learning Org Scores vs Forward 6-Month Returns ===")
    corrs = {}
    for col in score_cols:
        corr = df[[col, "fwd_6m_return"]].dropna().corr().iloc[0, 1]
        corrs[col] = corr
        print(f"  {col:15s}  r = {corr:+.4f}")

    # --- 2. Quintile analysis ---
    print("\n=== QUINTILE ANALYSIS (Composite Score) ===")
    df["composite_quintile"] = pd.qcut(df["composite"], 5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"])
    quintile_stats = (
        df.groupby("composite_quintile", observed=True)["fwd_6m_return"]
        .agg(["mean", "median", "std", "count"])
    )
    print(quintile_stats.to_string(float_format="%.4f"))

    quintile_stats.to_csv(f"{OUTPUT_DIR}/quintile_analysis.csv")

    # --- 3. Year-over-year quintile returns ---
    df["call_year"] = df["date"].dt.year
    yearly_quintile = (
        df.groupby(["call_year", "composite_quintile"], observed=True)["fwd_6m_return"]
        .mean()
        .unstack()
    )
    yearly_quintile.to_csv(f"{OUTPUT_DIR}/yearly_quintile_returns.csv")

    # --- 4. Plots ---
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot A: Quintile bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    quintile_means = df.groupby("composite_quintile", observed=True)["fwd_6m_return"].mean()
    colors = sns.color_palette("RdYlGn", 5)
    quintile_means.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Mean 6-Month Forward Return by Learning Org Score Quintile")
    ax.set_ylabel("Mean Forward 6-Month Return")
    ax.set_xlabel("Composite Learning Org Score Quintile")
    ax.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(quintile_means):
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quintile_returns.png", dpi=150)
    plt.close()

    # Plot B: Scatter with regression line
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    # Subsample for readability
    sample = df.sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample["composite"], sample["fwd_6m_return"], alpha=0.15, s=8, color="steelblue")
    z = np.polyfit(df["composite"].values, df["fwd_6m_return"].values, 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["composite"].min(), df["composite"].max(), 100)
    ax.plot(x_range, p(x_range), color="red", linewidth=2, label=f"r={corrs['composite']:+.3f}")
    ax.set_title("Composite Score vs Forward 6-Month Return")
    ax.set_xlabel("Composite Learning Org Score")
    ax.set_ylabel("Forward 6-Month Return")
    ax.legend()

    # Plot C: Per-cluster correlations bar
    ax = axes[1]
    cluster_corrs = pd.Series(corrs)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    cluster_corrs.plot.barh(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Correlation with Forward 6-Month Returns")
    ax.set_xlabel("Pearson Correlation (r)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    for i, v in enumerate(cluster_corrs):
        ax.text(v + 0.001 if v >= 0 else v - 0.001, i, f"{v:+.4f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_analysis.png", dpi=150)
    plt.close()

    # Plot D: Time-series of Q5-Q1 spread
    fig, ax = plt.subplots(figsize=(12, 5))
    if "Q5 (High)" in yearly_quintile.columns and "Q1 (Low)" in yearly_quintile.columns:
        spread = yearly_quintile["Q5 (High)"] - yearly_quintile["Q1 (Low)"]
        spread = spread.dropna()
        colors_ts = ["green" if v > 0 else "red" for v in spread]
        ax.bar(spread.index, spread.values, color=colors_ts, edgecolor="black", linewidth=0.5)
        ax.set_title("Q5 (High Score) minus Q1 (Low Score): 6-Month Forward Return Spread")
        ax.set_ylabel("Return Spread (Q5 - Q1)")
        ax.set_xlabel("Earnings Call Year")
        ax.axhline(y=0, color="black", linewidth=0.5)
        avg_spread = spread.mean()
        ax.axhline(y=avg_spread, color="blue", linewidth=1, linestyle="--",
                    label=f"Avg spread: {avg_spread:+.1%}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/yearly_spread.png", dpi=150)
    plt.close()

    # Save full joined dataset
    df.to_csv(f"{OUTPUT_DIR}/scores_with_returns.csv", index=False)

    print(f"\nOutputs saved:")
    print(f"  {OUTPUT_DIR}/scores_with_returns.csv — full joined dataset ({len(df)} rows)")
    print(f"  {OUTPUT_DIR}/quintile_analysis.csv — quintile stats")
    print(f"  {OUTPUT_DIR}/yearly_quintile_returns.csv — year-by-year quintile returns")
    print(f"  {OUTPUT_DIR}/quintile_returns.png — quintile bar chart")
    print(f"  {OUTPUT_DIR}/correlation_analysis.png — scatter + correlation bars")
    print(f"  {OUTPUT_DIR}/yearly_spread.png — Q5-Q1 annual spread")


def main():
    scores_df = pd.read_csv(SCORES_FILE, parse_dates=["date"])
    prices_df = load_prices()

    joined = compute_forward_returns(scores_df, prices_df)
    analyze_and_plot(joined)
    print("\nDone!")


if __name__ == "__main__":
    main()
