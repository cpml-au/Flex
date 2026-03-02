#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


def _safe_mean(values):
    return mean(values) if values else float("nan")


def _safe_std(values):
    return pstdev(values) if len(values) > 1 else 0.0


def _entropy_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    return entropy


def _simpson_diversity(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    sq_sum = 0.0
    for c in counts.values():
        p = c / total
        sq_sum += p * p
    return 1.0 - sq_sum


def _token_stats(exprs):
    all_tokens = []
    for expr in exprs:
        all_tokens.extend(TOKEN_RE.findall(expr))
    total = len(all_tokens)
    unique = len(set(all_tokens))
    richness = (unique / total) if total > 0 else 0.0
    return unique, total, richness


def _group_metrics(rows):
    exprs = [r["expr"] for r in rows]
    lengths = [r["length"] for r in rows]
    fitnesses = [r["fitness"] for r in rows if math.isfinite(r["fitness"])]

    expr_counts = Counter(exprs)
    n = len(exprs)
    unique_exprs = len(expr_counts)

    unique_tokens, total_tokens, token_richness = _token_stats(exprs)

    return {
        "population_size": n,
        "unique_exprs": unique_exprs,
        "unique_expr_ratio": (unique_exprs / n) if n > 0 else 0.0,
        "expr_entropy": _entropy_from_counts(expr_counts),
        "simpson_diversity": _simpson_diversity(expr_counts),
        "length_mean": _safe_mean(lengths),
        "length_std": _safe_std(lengths),
        "unique_lengths": len(set(lengths)),
        "unique_length_ratio": (len(set(lengths)) / n) if n > 0 else 0.0,
        "fitness_mean": _safe_mean(fitnesses),
        "fitness_std": _safe_std(fitnesses),
        "fitness_min": min(fitnesses) if fitnesses else float("nan"),
        "fitness_max": max(fitnesses) if fitnesses else float("nan"),
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
        "token_richness": token_richness,
    }


def _read_log(path):
    per_gen_island = defaultdict(list)
    per_gen = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "generation",
            "island",
            "individual_idx",
            "length",
            "fitness",
            "expr",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns in detailed log: {sorted(missing)}"
            )

        for row in reader:
            generation = int(row["generation"])
            island = int(row["island"])
            length = int(row["length"])
            try:
                fitness = float(row["fitness"])
            except ValueError:
                fitness = float("nan")
            expr = row["expr"]

            parsed = {
                "generation": generation,
                "island": island,
                "length": length,
                "fitness": fitness,
                "expr": expr,
            }
            per_gen_island[(generation, island)].append(parsed)
            per_gen[generation].append(parsed)

    return per_gen_island, per_gen


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_generation_metrics(gen_rows, output_prefix):
    generations = [r["generation"] for r in gen_rows]
    metrics = [
        ("unique_expr_ratio", "Unique Expr Ratio"),
        ("expr_entropy", "Expression Entropy"),
        ("simpson_diversity", "Simpson Diversity"),
        ("unique_length_ratio", "Unique Length Ratio"),
        ("length_std", "Length Std"),
        ("fitness_std", "Fitness Std"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, (key, title) in zip(axes.flatten(), metrics):
        values = [r[key] for r in gen_rows]
        ax.plot(generations, values, linewidth=1.8)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Generation")
    fig.suptitle("Population Diversity Trends (All Islands Aggregated)")
    fig.tight_layout()
    out = Path(str(output_prefix) + "_plots_generation_metrics.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return [out]


def _plot_island_trajectories(island_rows, output_prefix):
    by_island = defaultdict(list)
    for row in island_rows:
        by_island[row["island"]].append(row)
    for island in by_island:
        by_island[island] = sorted(by_island[island], key=lambda r: r["generation"])

    metrics = [
        ("unique_expr_ratio", "Unique Expr Ratio"),
        ("expr_entropy", "Expression Entropy"),
        ("simpson_diversity", "Simpson Diversity"),
        ("fitness_std", "Fitness Std"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (key, title) in zip(axes.flatten(), metrics):
        for island, rows in sorted(by_island.items()):
            generations = [r["generation"] for r in rows]
            values = [r[key] for r in rows]
            ax.plot(generations, values, linewidth=1.4, label=f"island {island}")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Generation")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5))
    fig.suptitle("Per-Island Diversity Trajectories")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(str(output_prefix) + "_plots_islands_combined.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return [out]


def _plot_island_divergence(island_rows, output_prefix):
    by_gen_metric = defaultdict(list)
    for row in island_rows:
        by_gen_metric[row["generation"]].append(row["unique_expr_ratio"])

    generations = sorted(by_gen_metric.keys())
    spread = []
    for gen in generations:
        vals = by_gen_metric[gen]
        spread.append(max(vals) - min(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(generations, spread, linewidth=1.8)
    ax.set_title("Between-Island Divergence in Unique Expr Ratio")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Max - Min across islands")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(str(output_prefix) + "_plots_island_divergence.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Analyze population diversity from detailed GP logs."
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to population_detailed_log.csv generated by GPSymbolicRegressor.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Prefix for output CSVs. Defaults to '<logfile_without_suffix>_diversity'."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of PNG plots.",
    )
    args = parser.parse_args()

    logfile = args.logfile
    if not logfile.exists():
        raise FileNotFoundError(f"Detailed log file not found: {logfile}")

    if args.output_prefix is None:
        output_prefix = logfile.with_suffix("")
        output_prefix = Path(str(output_prefix) + "_diversity")
    else:
        output_prefix = args.output_prefix

    per_gen_island, per_gen = _read_log(logfile)

    island_rows = []
    for (generation, island), rows in sorted(per_gen_island.items()):
        metrics = _group_metrics(rows)
        island_rows.append({"generation": generation, "island": island, **metrics})

    gen_rows = []
    for generation, rows in sorted(per_gen.items()):
        metrics = _group_metrics(rows)
        metrics["num_islands_present"] = len({r["island"] for r in rows})
        gen_rows.append({"generation": generation, **metrics})

    island_out = Path(str(output_prefix) + "_per_island.csv")
    gen_out = Path(str(output_prefix) + "_per_generation.csv")
    _write_csv(island_out, island_rows)
    _write_csv(gen_out, gen_rows)

    print(f"Wrote per-island diversity metrics to: {island_out}")
    print(f"Wrote per-generation diversity metrics to: {gen_out}")

    if not args.no_plots and island_rows:
        # Only generate one combined figure where all islands are overlaid
        # with different colors (no separate plot files per metric).
        plots = _plot_island_trajectories(island_rows, output_prefix)
        for plot_file in plots:
            print(f"Wrote plot: {plot_file}")

    if gen_rows:
        last = gen_rows[-1]
        print(
            "Final generation summary: "
            f"gen={last['generation']}, "
            f"unique_expr_ratio={last['unique_expr_ratio']:.4f}, "
            f"expr_entropy={last['expr_entropy']:.4f}, "
            f"simpson_diversity={last['simpson_diversity']:.4f}"
        )


if __name__ == "__main__":
    main()
