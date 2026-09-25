"""Benchmark: coarse-grained island parallelism vs. the original (fine-grained)
master-worker strategy, on the Nguyen-like polynomial problem from
``simple_sr_noyaml.py``.

Both runs use identical hyperparameters and the same RNG seed, so the only
difference is the parallelization strategy. We report wall-clock time and the best
training fitness reached by each.
"""

import time
import random
import warnings
import re

import numpy as np
import ray
from deap import gp

from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.util import detect_nested_trigonometric_functions, compile_individuals
from flex.gp.primitives import add_primitives_to_pset_from_dict


# Ground truth: x^4 + x^3 + x^2 + x.
# A larger sample makes each fitness evaluation more representative of real workloads
# (the trivial 20-point version is dominated by Ray/serialization overhead, which
# hides the effect of the parallelization strategy).
x = np.linspace(-1.0, 1.0, 2000).reshape(-1, 1)
y = (x**4 + x**3 + x**2 + x).ravel()


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    return detect_nested_trigonometric_functions(str(ind))


def get_features_batch(
    individuals_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    features_batch = [
        [fe(i) for i in individuals_batch] for fe in individ_feature_extractors
    ]
    return features_batch[0], features_batch[1], features_batch[2]


def eval_MSE_sol(individual, X, y):
    warnings.filterwarnings("ignore")
    y_pred = individual(X).ravel()
    MSE = np.mean(np.square(y_pred - y))
    if np.isnan(MSE):
        MSE = 1e5
    return MSE, y_pred


def predict(individuals_batch, toolbox, X, penalty):
    callables = compile_individuals(toolbox, individuals_batch)
    u = [None] * len(individuals_batch)
    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, X, None)
    return u


def score(individuals_batch, toolbox, X, y, penalty):
    callables = compile_individuals(toolbox, individuals_batch)
    MSE = [None] * len(individuals_batch)
    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, X, y)
    return MSE


def fitness(individuals_batch, toolbox, X, y, penalty):
    callables = compile_individuals(toolbox, individuals_batch)
    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_batch)
    fitnesses = [None] * len(individuals_batch)
    for i, ind in enumerate(callables):
        if individ_length[i] >= 50:
            fitnesses[i] = (1e8,)
        else:
            MSE, _ = eval_MSE_sol(ind, X, y)
            fitnesses[i] = (
                MSE
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )
    return fitnesses


def build_pset():
    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    pset.renameArguments(ARG0="x")
    primitives = {
        "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
        "used": [
            {"name": "add", "dimension": None, "rank": None},
            {"name": "sub", "dimension": None, "rank": None},
            {"name": "mul", "dimension": None, "rank": None},
            {"name": "div", "dimension": None, "rank": None},
            {"name": "sin", "dimension": None, "rank": None},
            {"name": "cos", "dimension": None, "rank": None},
            {"name": "exp", "dimension": None, "rank": None},
            {"name": "log", "dimension": None, "rank": None},
        ],
    }
    return add_primitives_to_pset_from_dict(pset, primitives)


def run(coarse_grained, seed):
    # reset RNGs so both strategies start from identical populations
    random.seed(seed)
    np.random.seed(seed)

    pset = build_pset()
    common_data = {"penalty": {"reg_param": 0.0}}

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        num_individuals=300,
        num_islands=10,
        generations=100,
        mig_freq=10,
        mut_prob=0.1,
        min_height=2,
        max_height=6,
        crossover_prob=0.9,
        overlapping_generation=True,
        print_log=False,
        batch_size=100,
        coarse_grained_islands=coarse_grained,
    )

    tic = time.perf_counter()
    gpsr.fit(x, y)
    toc = time.perf_counter()

    elapsed = toc - tic
    evals_per_sec = (
        (gpsr.get_last_gen() + 1) * gpsr.num_individuals * gpsr.num_islands / elapsed
    )
    best = gpsr.get_best_individuals(n_ind=1)[0]
    return elapsed, best.fitness.values[0], evals_per_sec, str(best)


def main():
    ray.init(ignore_reinit_error=True)

    seed = 42

    print("Running FINE-GRAINED (original master-worker)...")
    t_fine, fit_fine, eps_fine, best_fine = run(coarse_grained=False, seed=seed)

    print("Running COARSE-GRAINED (island-parallel)...")
    t_coarse, fit_coarse, eps_coarse, best_coarse = run(coarse_grained=True, seed=seed)

    print("\n" + "=" * 70)
    print(f"{'Strategy':<18}{'Time (s)':>12}{'Best fitness':>16}{'Evals/s':>14}")
    print("-" * 70)
    print(f"{'fine-grained':<18}{t_fine:>12.2f}{fit_fine:>16.3e}{eps_fine:>14.0f}")
    print(
        f"{'coarse-grained':<18}{t_coarse:>12.2f}{fit_coarse:>16.3e}{eps_coarse:>14.0f}"
    )
    print("-" * 70)
    speedup = t_fine / t_coarse if t_coarse > 0 else float("nan")
    print(f"Speed-up (fine / coarse): {speedup:.2f}x")
    print("=" * 70)
    print(f"best (fine)   = {best_fine}")
    print(f"best (coarse) = {best_coarse}")

    ray.shutdown()


if __name__ == "__main__":
    main()
