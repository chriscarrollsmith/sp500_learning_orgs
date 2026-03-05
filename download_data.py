"""
Download required datasets if not already present.

Sources:
  - Earnings transcripts: huggingface.co/datasets/kurry/sp500_earnings_transcripts
  - Stock prices: huggingface.co/datasets/defeatbeta/yahoo-finance-data
"""

import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    "transcripts.parquet": (
        "https://huggingface.co/datasets/kurry/sp500_earnings_transcripts"
        "/resolve/main/parquet_files/part-0.parquet"
    ),
    "stock_prices.parquet": (
        "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data"
        "/resolve/main/data/stock_prices.parquet"
    ),
}


def ensure_data(filename):
    """Download a dataset file if it doesn't exist. Returns the local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        print(f"  {filename} already exists, skipping download.")
        return path

    url = DATASETS[filename]
    print(f"  Downloading {filename}...")
    subprocess.check_call(
        ["curl", "-L", "-o", path, url],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    print(f"  Saved to {path}")
    return path


def ensure_all():
    """Download all datasets."""
    print("Checking datasets...")
    for filename in DATASETS:
        ensure_data(filename)
    print("All datasets ready.")


if __name__ == "__main__":
    ensure_all()
