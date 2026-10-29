"""Profile a serial (no Ray) flex run with cProfile and break the time down by phase.

Run from the ``bench`` directory:

    python profile_flex.py Nguyen_13_consts_pso.yaml Nguyen-13 --generations 20

With ``--generations`` the config is copied to ``results/`` with that number of
generations, and the copy is used for the run.
"""

import argparse
import cProfile
import pathlib
import pstats
import random

import numpy as np
import yaml

from bench import eval as bench_eval

# phase label -> (file name suffix, function name) of the entry whose cumulative time
# is reported. pygmo calls the Python fitness from C++, which breaks the cProfile call
# stack: only leaf-level entries are reliable, so no enclosing function is reported.
PHASES = [
    ("objective of the constant optimizer (fitness)", ("bench.py", "fitness")),
    ("gradient (vectorized finite differences)", ("bench.py", "gradient")),
    ("tree evaluation without constants (compute_MSE)", ("bench.py", "compute_MSE")),
    ("numba evaluation and constant fitting (lm)",
     ("bench.py", "eval_MSE_and_fit_constants_numba")),
    ("pygmo population init (evaluates all points)", ("__init__.py", "_population_init")),
    ("compile tree (compile_individual_with_consts)",
     ("util.py", "compile_individual_with_consts")),
    ("variation (varAnd)", ("algorithms.py", "varAnd")),
    ("variation (varOr)", ("algorithms.py", "varOr")),
    ("parent selection", ("selection.py", "selStochasticUniversalSampling")),
    ("population statistics", ("regressor.py", "__stats")),
]


def find_stats(stats, file_suffix, func_name):
    """Number of calls and cumulative time of the entries matching ``file_suffix``
    and ``func_name``."""
    entries = [
        (nc, ct)
        for (fname, _, name), (_, nc, _, ct, _) in stats.stats.items()
        if fname.endswith(file_suffix) and name == func_name
    ]
    return sum(e[0] for e in entries), sum(e[1] for e in entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfgfile", help="Path of the YAML config file.")
    parser.add_argument("problem", help="Name of the Nguyen or PMLB dataset.")
    parser.add_argument("--seed", type=int, default=29802)
    parser.add_argument("--generations", type=int, default=None)
    args = parser.parse_args()

    stem = pathlib.Path(args.cfgfile).stem
    cfgfile = args.cfgfile
    if args.generations is not None:
        with open(args.cfgfile) as f:
            cfg = yaml.safe_load(f)
        cfg["gp"]["generations"] = args.generations
        stem = f"{stem}_gen{args.generations}"
        cfgfile = pathlib.Path("results") / f"profile_{stem}.yaml"
        with open(cfgfile, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    random.seed(args.seed)
    np.random.seed(args.seed)

    prof = cProfile.Profile()
    prof.enable()
    bench_eval(problem=args.problem, cfgfile=cfgfile, seed=args.seed,
               remove_init_duplicates=False, multiprocessing=False)
    prof.disable()

    out = pathlib.Path("results") / f"profile_{stem}.prof"
    prof.dump_stats(out)

    stats = pstats.Stats(prof)
    total = stats.total_tt
    print("\n" + "=" * 78)
    print(f"Profile of {args.problem} ({cfgfile}), total {total:.2f} s")
    print(f"Raw stats saved to {out}")
    print("-" * 78)
    print(f"{'phase':<50}{'calls':>10}{'time (s)':>10}{'share':>8}")
    for label, (file_suffix, func_name) in PHASES:
        nc, t = find_stats(stats, file_suffix, func_name)
        if nc > 0:
            print(f"{label:<50}{nc:>10}{t:>10.2f}{100 * t / total:>7.1f}%")
    print("=" * 78 + "\n")

    stats.sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
