"""Where the time of a parallel (Ray) flex run goes: Ray timeline of the fitness tasks
run by the workers and, with ``--profile``, cProfile of the driver process (which
slows down the driver: fit time and worker utilization are measured without it).
Early stopping is disabled.

Run from the ``bench`` directory:

    env OMP_NUM_THREADS=1 taskset -c 0-7 python scaling_flex.py Nguyen-13 \\
        --cfg Nguyen_13_consts_lbfgs.yaml --cores 8 --generations 40 [--profile]
"""

import argparse
import cProfile
import json
import pathlib
import pstats
import random
import tempfile
import time

import numpy as np
import ray
import yaml

import bench
from bench_strategies import warmup_workers
from profile_flex import find_stats

# phase label -> (file name suffix, function name), cumulative time on the driver
DRIVER_PHASES = [
    ("waiting for fitness results (ray.get)", ("util.py", "mapper")),
    ("variation (varAnd)", ("algorithms.py", "varAnd")),
    ("variation (varOr)", ("algorithms.py", "varOr")),
    ("parent selection", ("selection.py", "selStochasticUniversalSampling")),
    ("parent selection (random.sample)", ("random.py", "sample")),
    ("assign fitness and constants (callback)", ("bench.py", "assign_attributes")),
    ("population statistics", ("regressor.py", "__stats")),
    ("survival selection (selBest)", ("selection.py", "selBest")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--seed", type=int, default=29802)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    tag = f"{args.problem}_{args.cores}c"
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["gp"]["generations"] = args.generations
    cfgfile = pathlib.Path("results") / f"scaling_{tag}.yaml"
    with open(cfgfile, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    ray.init(num_cpus=args.cores)
    warmup_workers()
    random.seed(args.seed)
    np.random.seed(args.seed)

    prof = cProfile.Profile()
    t_start = time.time()
    if args.profile:
        prof.enable()
    _, _, _, fit_time = bench.eval(
        problem=args.problem, cfgfile=cfgfile, seed=args.seed,
        remove_init_duplicates=False, early_stop_fitness_threshold=None,
    )
    prof.disable()
    t_end = time.time()

    # fitness tasks of the run from the Ray timeline (times in microseconds)
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        ray.timeline(filename=tmp.name)
        events = json.load(open(tmp.name))
    tasks = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat") == "task::compute_attributes"
        and t_start * 1e6 <= e["ts"] <= t_end * 1e6
    ]
    busy = sum(e["dur"] for e in tasks) / 1e6
    durations = np.array([e["dur"] for e in tasks]) / 1e3
    # time with at least one fitness task running (union of the task intervals)
    span, end = 0.0, -np.inf
    for start, stop in sorted((e["ts"], e["ts"] + e["dur"]) for e in tasks):
        span += max(0.0, stop - max(start, end))
        end = max(end, stop)
    span /= 1e6

    print("\n" + "=" * 78)
    print(f"{args.problem} ({args.cfg}), {args.cores} cores, {args.generations} "
          f"generations, batch_size {cfg['gp']['batch_size']}")
    print(f"fit time {fit_time:.2f} s")
    print(f"fitness tasks: {len(tasks)} ({len(tasks) / (args.generations + 1):.1f} per "
          f"generation), duration median {np.median(durations):.1f} ms, "
          f"max {durations.max():.1f} ms")
    print(f"worker busy time {busy:.2f} s = {100 * busy / (args.cores * fit_time):.1f}% "
          f"of {args.cores} cores x fit time")
    print(f"time with fitness tasks running {span:.2f} s (workers busy "
          f"{100 * busy / (args.cores * span):.1f}% of it); time with no fitness task "
          f"running {fit_time - span:.2f} s ({100 * (fit_time - span) / fit_time:.1f}%)")
    if args.profile:
        stats = pstats.Stats(prof)
        print("-" * 78)
        print(f"{'driver phase (under cProfile)':<50}{'time (s)':>10}{'share':>8}")
        for label, (file_suffix, func_name) in DRIVER_PHASES:
            _, t = find_stats(stats, file_suffix, func_name)
            if t > 0:
                print(f"{label:<50}{t:>10.2f}{100 * t / fit_time:>7.1f}%")
        prof.dump_stats(pathlib.Path("results") / f"scaling_{tag}.prof")
    print("=" * 78 + "\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
