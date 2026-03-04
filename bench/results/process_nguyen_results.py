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
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Nguyen benchmark result files and report a markdown table "
            "with average R2 test and success rate (R2 == 1)."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing Nguyen-*.csv files (default: current folder).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the aggregated summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional output Markdown path for the summary table.",
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
            total_runs=("problem", "size"),
            r2_test_mean=("r2_test", "mean"),
            r2_test_median=("r2_test", "median"),
            successful_runs=("r2_test", lambda s: (s == 1.0).sum()),
        )
        .rename(columns={"problem": "dataset"})
    )
    summary_df["success_rate"] = (
        summary_df["successful_runs"].astype(int).astype(str)
        + "/"
        + summary_df["total_runs"].astype(int).astype(str)
    )
    summary_df["dataset_id"] = (
        summary_df["dataset"].str.extract(r"Nguyen-(\d+)").astype(int)
    )
    summary_df = summary_df.sort_values("dataset_id").drop(columns=["dataset_id"])

    md_df = summary_df[
        ["dataset", "r2_test_mean", "r2_test_median", "success_rate"]
    ].copy()
    md_df["r2_test_mean"] = md_df["r2_test_mean"].map(format_float)
    md_df["r2_test_median"] = md_df["r2_test_median"].map(format_float)
    markdown_table = md_df.to_markdown(index=False, disable_numparse=True)
    print(markdown_table)

    if args.output_csv is not None:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(
            out,
            sep=";",
            index=False,
            columns=[
                "dataset",
                "total_runs",
                "r2_test_mean",
                "successful_runs",
                "success_rate",
            ],
        )
        print(f"\nSaved summary to: {out}")

    if args.output_md is not None:
        out_md = args.output_md.resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown_table + "\n")
        print(f"\nSaved markdown table to: {out_md}")


if __name__ == "__main__":
    main()
