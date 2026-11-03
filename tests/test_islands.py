import random

from deap import gp
import numpy as np
import ray

from flex.gp.numba_eval import (
    assign_attributes,
    fitness_batch,
    predict_batch,
    score_batch,
)
from flex.gp.numpy_primitives import numpy_primitives
from flex.gp.regressor import GPSymbolicRegressor


def run_coarse_islands(seed):
    random.seed(seed)
    np.random.seed(seed)

    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    for name in ("add", "mul"):
        params = numpy_primitives[name]
        pset.addPrimitive(params.op, params.in_types, params.out_type, name=name)
    pset.addTerminal(object, float, "c")

    X = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    y = (X**3 + 2.0 * X).ravel()

    model = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness_batch,
        predict_func=predict_batch,
        score_func=score_batch,
        callback_func=assign_attributes,
        select_fun="random.sample",
        select_args="{'k': self.num_individuals}",
        # draws from NumPy's global generator
        mut_fun="gp.mixedMutate",
        mut_args="{'expr': toolbox.expr_mut, 'pset': pset, 'prob': [0.7, 0.2, 0.1]}",
        expr_mut_args="{'min_': 1, 'max_': 2}",
        crossover_prob=0.0,
        mut_prob=1.0,
        num_individuals=20,
        generations=4,
        num_islands=2,
        mig_freq=2,
        batch_size=20,
        overlapping_generation=True,
        coarse_grained_islands=True,
    )
    model.fit(X, y)
    return [str(ind) for ind in model.get_population_individuals()]


def test_coarse_islands_repeat_with_the_same_seed():
    ray.init()
    first = run_coarse_islands(1)
    second = run_coarse_islands(1)
    third = run_coarse_islands(2)
    ray.shutdown()

    assert first == second
    assert first != third
