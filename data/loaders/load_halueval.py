"""
Loader for the HaluEval benchmark.

HaluEval isn't packaged as a clean HuggingFace `datasets` entry with a single
canonical name, so this script pulls the raw JSON files directly from the
official repo and loads them into pandas DataFrames.

Usage:
    python load_halueval.py --split qa
    python load_halueval.py --split all
"""

import argparse
import json
import os
import urllib.request

import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/{split}_data.json"
)

SPLITS = ["qa", "dialogue", "summarization", "general"]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def download_split(split: str) -> str:
    """Download the raw JSON for a given split if not already cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, f"{split}_data.json")

    if os.path.exists(local_path):
        print(f"[cache] Using cached file for '{split}'")
        return local_path

    url = BASE_URL.format(split=split)
    print(f"[download] Fetching {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{split}' split from {url}. "
            f"Check your internet connection or verify the file path in the "
            f"HaluEval repo hasn't changed. Original error: {e}"
        )
    return local_path


def load_split(split: str) -> pd.DataFrame:
    """Load a single HaluEval split into a DataFrame."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose from {SPLITS}.")

    path = download_split(split)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"[loaded] '{split}' split: {len(df)} records, columns: {list(df.columns)}")
    return df


def load_all() -> dict:
    """Load every split into a dict of DataFrames."""
    return {split: load_split(split) for split in SPLITS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HaluEval benchmark splits.")
    parser.add_argument(
        "--split",
        type=str,
        default="qa",
        choices=SPLITS + ["all"],
        help="Which split to load (default: qa). Use 'all' to load everything.",
    )
    args = parser.parse_args()

    if args.split == "all":
        data = load_all()
        for name, df in data.items():
            print(f"\n{name}: {df.shape}")
            print(df.head(2))
    else:
        df = load_split(args.split)
        print(df.head(2))
