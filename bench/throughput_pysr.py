"""Throughput (expression evaluations per second) of flex (L-BFGS constant tuning) and
PySR at matched cores and wall time, with early stopping disabled. PySR uses the
operators of the primitives of the flex config (``--flex-cfg``).

Evaluations follow the convention of SymbolicRegression.jl (PySR ``num_evals``): one
per candidate expression plus the objective calls of the constant optimizer, without
the calls made to estimate gradients by finite differences. For flex, the throughput
including the finite-difference calls (``raw``) is also reported.

Both throughputs are averages over the window from ``--timeout`` - 20 s to
``--timeout`` after the start of the search:

- PySR: stopped by ``timeout_in_seconds``; the throughput is the last "Expressions
  evaluated per second" printed by SymbolicRegression.jl (average over the last ~20 s).
- flex: evaluations counted in ``bench.assign_attributes`` and recorded with their time
  at the end of each generation; the counts at the window bounds are interpolated
  between generations. The run length is set by ``--flex-generations``, which must
  make the fit last at least ``--timeout``; the config is copied to ``results/`` with
  that number of generations.

Results are written to ``results/throughput_<problem>_<cores>c_<method>.csv``.

Run from the ``bench`` directory:

    taskset -c 0 python throughput_pysr.py Nguyen-13 --cores 1 --flex-generations 70
    taskset -c 0 python throughput_pysr.py 225_puma8NH --flex-cfg PMLB_lbfgs.yaml \
        --cores 1 --flex-generations 11 --turbo
"""

import argparse
import os
import pathlib
import random
import re
import tempfile
import time

import numpy as np
import sympy
import yaml

from compare_pysr import SEEDS, make_pysr

HEADER = ["method", "seed", "time_s", "evals_per_s", "raw_evals_per_s"]

# flex primitive -> (PySR operator, arity); aq is defined as a custom PySR operator
PYSR_OPERATORS = {
    "add": ("+", 2),
    "sub": ("-", 2),
    "mul": ("*", 2),
    "div": ("/", 2),
    "aq": ("aq(x, y) = x / sqrt(1 + y^2)", 2),
    "sin": ("sin", 1),
    "cos": ("cos", 1),
    "exp": ("exp", 1),
    "log": ("log", 1),
}
SYMPY_MAPPINGS = {"aq": lambda x, y: x / sympy.sqrt(1 + y**2)}


def pysr_operators(cfg):
    """Binary and unary PySR operators of the primitives used in a flex config."""
    ops = [PYSR_OPERATORS[p["name"]] for p in cfg["gp"]["primitives"]["used"]]
    return [op for op, a in ops if a == 2], [op for op, a in ops if a == 1]


def run_flex(problem, cfgfile, seed, cores, timeout):
    import bench

    random.seed(seed)
    np.random.seed(seed)
    bench.EVAL_COUNTS.update(evals=0, raw_evals=0)
    # evaluation counts at the end of each generation
    times, evals, raw_evals = [time.time()], [0], [0]

    def record(best_inds):
        times.append(time.time())
        evals.append(bench.EVAL_COUNTS["evals"])
        raw_evals.append(bench.EVAL_COUNTS["raw_evals"])

    _, _, _, fit_time = bench.eval(
        problem=problem, cfgfile=cfgfile, seed=seed, remove_init_duplicates=False,
        multiprocessing=cores > 1, early_stop_fitness_threshold=None,
        custom_logger=record,
    )
    t = np.array(times) - times[0]
    assert t[-1] >= timeout, f"flex ran {t[-1]:.1f} s < {timeout} s: more generations"
    window = [timeout - 20.0, timeout]
    e, r = np.interp(window, t, evals), np.interp(window, t, raw_evals)
    return fit_time, (e[1] - e[0]) / 20.0, (r[1] - r[0]) / 20.0


def run_pysr(problem, timeout, seed, binary_operators, unary_operators, turbo):
    from datasets import generate_dataset
    from juliacall import Main as jl

    X_train, y_train, *_ = generate_dataset(problem, random_state=seed)
    model = make_pysr(
        timeout, seed, early_stop=False, verbosity=1,
        binary_operators=binary_operators, unary_operators=unary_operators,
        extra_sympy_mappings=SYMPY_MAPPINGS, turbo=turbo,
    )

    # SymbolicRegression.jl prints the speed to the Julia stdout: redirect it to a file
    redirect = jl.seval(
        'path -> (io = open(path, "w"); old = stdout; redirect_stdout(io); (io, old))'
    )
    restore = jl.seval("(io, old) -> (flush(stdout); redirect_stdout(old); close(io))")
    fd, path = tempfile.mkstemp()
    os.close(fd)
    io, old = redirect(path)
    try:
        tic = time.time()
        model.fit(X_train, y_train.ravel())
        fit_time = time.time() - tic
    finally:
        restore(io, old)
    with open(path) as f:
        out = f.read()
    os.remove(path)

    speeds = re.findall(r"Expressions evaluated per second: (\S+)", out)
    return fit_time, float(speeds[-1]), float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="?", default="Nguyen-13")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--flex-cfg", default="Nguyen_13_consts_lbfgs.yaml")
    parser.add_argument("--flex-generations", type=int, help="Required for flex.")
    parser.add_argument("--turbo", action="store_true", help="PySR turbo mode.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["flex-lbfgs", "pysr"])
    args = parser.parse_args()

    # the number of Julia threads must be set before juliacall is imported
    os.environ["PYTHON_JULIACALL_THREADS"] = str(args.cores)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    seeds = SEEDS[: args.num_seeds]
    tag = f"{args.problem}_{args.cores}c"
    with open(args.flex_cfg) as f:
        cfg = yaml.safe_load(f)
    binary_operators, unary_operators = pysr_operators(cfg)

    if "flex-lbfgs" in args.methods:
        cfg["gp"]["generations"] = args.flex_generations
        flex_cfg = pathlib.Path("results") / f"throughput_flex-lbfgs_{tag}.yaml"
        with open(flex_cfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    if "flex-lbfgs" in args.methods and args.cores > 1:
        import ray
        from bench_strategies import warmup_workers

        ray.init(num_cpus=args.cores)
        warmup_workers()

    if "pysr" in args.methods:
        # absorb the Julia JIT compilation before the timed runs
        X = np.random.rand(20, 1)
        make_pysr(
            None, 0, niterations=2, binary_operators=binary_operators,
            unary_operators=unary_operators, extra_sympy_mappings=SYMPY_MAPPINGS,
            turbo=args.turbo,
        ).fit(X, X[:, 0])

    for method in args.methods:
        out = f"results/throughput_{tag}_{method}.csv"
        with open(out, "w") as f:
            f.write(";".join(HEADER) + "\n")
        for seed in seeds:
            print(f"\n### {method}  seed={seed}\n", flush=True)
            if method == "pysr":
                res = run_pysr(args.problem, args.timeout, seed, binary_operators,
                               unary_operators, args.turbo)
            else:
                res = run_flex(args.problem, flex_cfg, seed, args.cores, args.timeout)
            fit_time, evals_per_s, raw_evals_per_s = res
            row = [method, seed, f"{fit_time:.2f}", f"{evals_per_s:.4g}",
                   f"{raw_evals_per_s:.4g}"]
            with open(out, "a") as f:
                f.write(";".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
