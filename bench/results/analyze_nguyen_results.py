#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def nguyen_sort_key(path: Path):
    m = re.search(r"Nguyen-(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return 10**9


def format_float(x):
    return f"{x:.8f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Nguyen benchmark result files and report mean/median "
            "R2 on train and test for each dataset."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing Nguyen-*.csv files (default: current script folder).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the aggregated summary.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    files = sorted(results_dir.glob("Nguyen-*.csv"), key=nguyen_sort_key)
    if not files:
        raise FileNotFoundError(f"No Nguyen-*.csv files found in {results_dir}")

    dataframes = []
    for file in files:
        df = pd.read_csv(file, sep=";")
        required = {"problem", "r2_train", "r2_test"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file.name}: missing columns {sorted(missing)}")
        dataframes.append(df[["problem", "r2_train", "r2_test"]].copy())

    all_results = pd.concat(dataframes, ignore_index=True)
    summary_df = (
        all_results.groupby("problem", as_index=False)
        .agg(
            num_trials=("problem", "size"),
            r2_train_mean=("r2_train", "mean"),
            r2_train_median=("r2_train", "median"),
            r2_test_mean=("r2_test", "mean"),
            r2_test_median=("r2_test", "median"),
        )
        .rename(columns={"problem": "dataset"})
    )
    summary_df["dataset_id"] = (
        summary_df["dataset"].str.extract(r"Nguyen-(\d+)").astype(int)
    )
    summary_df = summary_df.sort_values("dataset_id").drop(columns=["dataset_id"])

    header = [
        "dataset",
        "num_trials",
        "r2_train_mean",
        "r2_train_median",
        "r2_test_mean",
        "r2_test_median",
    ]

    pretty_df = summary_df.copy()
    pretty_df["num_trials"] = pretty_df["num_trials"].astype(int)
    for col in [
        "r2_train_mean",
        "r2_train_median",
        "r2_test_mean",
        "r2_test_median",
    ]:
        pretty_df[col] = pretty_df[col].map(format_float)
    print(pretty_df.to_string(index=False))

    if args.output_csv is not None:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out, sep=";", index=False, columns=header)
        print(f"\nSaved summary to: {out}")


if __name__ == "__main__":
    main()
