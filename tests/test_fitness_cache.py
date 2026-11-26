import numpy as np
import pytest
from deap import gp

from flex.gp.primitives import add_primitives_to_pset_from_dict
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.util import (
    FitnessCache,
    cached_map,
    dummy_fitness,
    dummy_predict,
    dummy_score,
)

X = np.ones((4, 1))
y = np.ones(4)


def make_regressor(fitness=dummy_fitness, **kwargs):
    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    primitives = {
        "imports": {"flex.gp.numpy_primitives": ["numpy_primitives"]},
        "used": [
            {"name": "add", "dimension": None, "rank": None},
            {"name": "mul", "dimension": None, "rank": None},
        ],
    }
    pset = add_primitives_to_pset_from_dict(pset, primitives)
    pset.addTerminal(1.0, float, "one")
    return GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=dummy_score,
        predict_func=dummy_predict,
        common_data={},
        multiprocessing=False,
        **kwargs,
    )


def recording_function(calls):
    def f(individuals, toolbox):
        calls.extend(individuals)
        return [len(ind) for ind in individuals]

    return f


def test_cached_map_evaluates_each_distinct_individual_once():
    calls = []

    def map_func(f, individuals):
        calls.append(individuals)
        return [f(ind) for ind in individuals]

    cache = FitnessCache()
    assert cached_map(map_func, cache, len, ["a", "bb", "a"]) == [1, 2, 1]
    assert cache.hits == 1
    assert cached_map(map_func, cache, len, ["ccc", "bb", "a"]) == [3, 2, 1]
    assert cache.hits == 3
    assert calls == [["a", "bb"], ["ccc"]]


@pytest.mark.parametrize(
    "cache_fitness, expected", [(False, ["a", "a"]), (True, ["a"])]
)
def test_map_train_only_caches_when_enabled(cache_fitness, expected):
    toolbox = make_regressor(cache_fitness=cache_fitness)._prepare_fit(X, y, None, None)
    calls = []
    toolbox.map_train(recording_function(calls), ["a", "a"])
    assert calls == expected


def test_map_is_not_cached():
    toolbox = make_regressor(cache_fitness=True)._prepare_fit(X, y, None, None)
    calls = []
    toolbox.map(recording_function(calls), ["a", "a"])
    assert calls == ["a", "a"]


def test_each_fit_starts_with_an_empty_cache():
    model = make_regressor(cache_fitness=True)
    calls = []
    for _ in range(2):
        toolbox = model._prepare_fit(X, y, None, None)
        toolbox.map_train(recording_function(calls), ["a"])
    assert calls == ["a", "a"]


def test_fit_evaluates_each_tree_once_and_assigns_every_fitness():
    calls = []

    def fitness(individuals, toolbox, X, y):
        calls.extend(str(ind) for ind in individuals)
        return [(float(len(ind)),) for ind in individuals]

    model = make_regressor(
        fitness=fitness,
        cache_fitness=True,
        num_individuals=30,
        num_islands=2,
        generations=5,
        crossover_prob=1.0,
    )
    model.fit(X, y)

    assert len(calls) == len(set(calls))
    population = model.get_population_individuals()
    assert all(ind.fitness.valid for ind in population)
    assert all(ind.fitness.values[0] == len(ind) for ind in population)
    assert model.get_cache_hit_rate() > 0
