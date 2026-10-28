"""Compare flex (PSO and L-BFGS constant tuning) with PySR at matched cores and
wall-time budget.

Flex has no wall-time limit: its budget is set through the ``generations`` of the
config files, calibrated so that a run takes about ``--timeout`` seconds. PySR is
stopped by ``timeout_in_seconds=--timeout``. Both stop early if the training MSE
drops below 1e-12.

Run from the ``bench`` directory:

    python compare_pysr.py Nguyen-13 --cores 8 --timeout 60 --num-seeds 5
"""

import argparse
import os
import random
import time

import numpy as np
import sympy as sp
from sklearn.metrics import r2_score

FLEX_CFGS = {
    "flex-pso": "Nguyen_13_consts_pso.yaml",
    "flex-lbfgs": "Nguyen_13_consts_lbfgs.yaml",
}
SEEDS = [29802, 22118, 860, 15795, 21575, 5390, 11964, 6265, 23654, 11284]
HEADER = ["method", "seed", "time_s", "r2_train", "r2_test", "size", "model"]


def model_size(model_str):
    """Number of nodes of the (simplified) sympy expression tree."""
    return len(list(sp.preorder_traversal(sp.sympify(model_str))))


def run_flex(problem, cfgfile, seed):
    from bench import eval as bench_eval

    random.seed(seed)
    np.random.seed(seed)
    r2_train, r2_test, model_str, fit_time = bench_eval(
        problem=problem, cfgfile=cfgfile, seed=seed, remove_init_duplicates=False
    )
    return fit_time, r2_train, r2_test, model_str


def make_pysr(
    timeout,
    seed,
    niterations=100000,
    early_stop=True,
    verbosity=0,
    binary_operators=("+", "-", "*", "/"),
    unary_operators=("sin", "cos", "exp", "log"),
    extra_sympy_mappings=None,
    turbo=False,
):
    from pysr import PySRRegressor

    return PySRRegressor(
        binary_operators=list(binary_operators),
        unary_operators=list(unary_operators),
        extra_sympy_mappings=extra_sympy_mappings,
        turbo=turbo,
        # evaluate on the full dataset, as flex does
        batching=False,
        maxsize=40,
        niterations=niterations,
        timeout_in_seconds=timeout,
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-12" if early_stop else None
        ),
        parallelism="multithreading",
        precision=64,
        random_state=seed,
        progress=False,
        verbosity=verbosity,
        temp_equation_file=True,
    )


def run_pysr(problem, timeout, seed):
    from datasets import generate_dataset

    X_train, y_train, X_test, y_test, *_ = generate_dataset(problem, random_state=seed)
    y_train, y_test = y_train.ravel(), y_test.ravel()

    model = make_pysr(timeout, seed)
    tic = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - tic

    model_str = str(sp.simplify(model.sympy()))
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    return fit_time, r2_train, r2_test, model_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", nargs="?", default="Nguyen-13")
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--methods", nargs="+",
                        default=["flex-pso", "flex-lbfgs", "pysr"])
    args = parser.parse_args()

    # the number of Julia threads must be set before juliacall is imported
    os.environ["PYTHON_JULIACALL_THREADS"] = str(args.cores)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    seeds = SEEDS[: args.num_seeds]
    out = f"results/compare_pysr_{args.problem}.csv"
    with open(out, "w") as f:
        f.write(";".join(HEADER) + "\n")

    if any(m.startswith("flex") for m in args.methods):
        import ray
        from bench_strategies import warmup_workers

        ray.init(num_cpus=args.cores)
        warmup_workers()

    if "pysr" in args.methods:
        # absorb the Julia JIT compilation before the timed runs
        X = np.random.rand(20, 1)
        make_pysr(None, 0, niterations=2).fit(X, X[:, 0])

    for method in args.methods:
        for seed in seeds:
            print(f"\n### {method}  seed={seed}\n", flush=True)
            if method == "pysr":
                res = run_pysr(args.problem, args.timeout, seed)
            else:
                res = run_flex(args.problem, FLEX_CFGS[method], seed)
            fit_time, r2_train, r2_test, model_str = res
            row = [method, seed, f"{fit_time:.2f}", f"{r2_train:.6f}",
                   f"{r2_test:.6f}", model_size(model_str), model_str]
            with open(out, "a") as f:
                f.write(";".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
