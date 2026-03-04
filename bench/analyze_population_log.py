#!/usr/bin/env python3
"""Analyze GP population diversity from detailed evolution logs."""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

TOKEN_RE = re.compile(r"[A-Za-z_]\w*")
NAN = float("nan")


def _safe_mean(values):
    return mean(values) if values else NAN


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


def _token_stats(exprs):
    all_tokens = [token for expr in exprs for token in TOKEN_RE.findall(expr)]
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
        "length_mean": _safe_mean(lengths),
        "length_std": _safe_std(lengths),
        "unique_lengths": len(set(lengths)),
        "unique_length_ratio": (len(set(lengths)) / n) if n > 0 else 0.0,
        "fitness_mean": _safe_mean(fitnesses),
        "fitness_std": _safe_std(fitnesses),
        "fitness_min": min(fitnesses) if fitnesses else NAN,
        "fitness_max": max(fitnesses) if fitnesses else NAN,
        "unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
        "token_richness": token_richness,
    }


def _add_normalized_best_fitness(island_rows):
    by_island = defaultdict(list)
    for row in island_rows:
        by_island[row["island"]].append(row)

    for rows in by_island.values():
        best_vals = [r["fitness_min"] for r in rows if math.isfinite(r["fitness_min"])]
        if not best_vals:
            for r in rows:
                r["fitness_best_norm"] = NAN
            continue

        # Lower fitness is better; normalize so 1.0 is the best seen for that island.
        island_min = min(best_vals)
        island_max = max(best_vals)
        denom = island_max - island_min
        if denom <= 0.0:
            for r in rows:
                r["fitness_best_norm"] = (
                    1.0 if math.isfinite(r["fitness_min"]) else NAN
                )
            continue

        for r in rows:
            if math.isfinite(r["fitness_min"]):
                r["fitness_best_norm"] = (island_max - r["fitness_min"]) / denom
            else:
                r["fitness_best_norm"] = NAN


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
            except (TypeError, ValueError):
                fitness = NAN
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


def _plot_island_trajectories(island_rows, output_prefix):
    by_island = defaultdict(list)
    for row in island_rows:
        by_island[row["island"]].append(row)
    for island in by_island:
        by_island[island] = sorted(by_island[island], key=lambda r: r["generation"])

    metrics = [
        ("unique_expr_ratio", "Unique Expr Ratio"),
        ("expr_entropy", "Expression Entropy"),
        ("fitness_best_norm", "Normalized Best Fitness"),
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
    _add_normalized_best_fitness(island_rows)

    gen_rows = []
    for generation, rows in sorted(per_gen.items()):
        metrics = _group_metrics(rows)
        metrics["num_islands_present"] = len({r["island"] for r in rows})
        island_norm_vals = [
            r["fitness_best_norm"]
            for r in island_rows
            if r["generation"] == generation and math.isfinite(r["fitness_best_norm"])
        ]
        metrics["fitness_best_norm_mean"] = _safe_mean(island_norm_vals)
        metrics["fitness_best_norm_std"] = _safe_std(island_norm_vals)
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
        plot_file = _plot_island_trajectories(island_rows, output_prefix)
        print(f"Wrote plot: {plot_file}")

    if gen_rows:
        last = gen_rows[-1]
        print(
            "Final generation summary: "
            f"gen={last['generation']}, "
            f"unique_expr_ratio={last['unique_expr_ratio']:.4f}, "
            f"expr_entropy={last['expr_entropy']:.4f}, "
            f"fitness_best_norm_mean={last['fitness_best_norm_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
