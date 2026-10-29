from deap import gp
from flex.gp.regressor import GPSymbolicRegressor
import numpy as np
import ray
from flex.gp.util import get_features_batch
from flex.gp.numba_eval import mse, predict_tree
from flex.gp.primitives import add_primitives_to_pset_from_dict


# Ground truth
x = np.array([x / 10.0 for x in range(-10, 10)]).reshape(-1, 1)
y = (x**4 + x**3 + x**2 + x).ravel()


def predict(individuals_batch, toolbox, X, penalty):
    pset = toolbox.compile.keywords["pset"]
    return [predict_tree(ind, pset, X) for ind in individuals_batch]


def score(individuals_batch, toolbox, X, y, penalty):
    pset = toolbox.compile.keywords["pset"]
    return [mse(ind, pset, X, y) for ind in individuals_batch]


def fitness(individuals_batch, toolbox, X, y, penalty):
    pset = toolbox.compile.keywords["pset"]

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_batch)

    fitnesses = [None] * len(individuals_batch)
    for i, ind in enumerate(individuals_batch):
        if individ_length[i] >= 50:
            fitnesses[i] = (1e8,)
        else:
            MSE = mse(ind, pset, X, y)
            if np.isnan(MSE):
                MSE = 1e5

            fitnesses[i] = (
                MSE
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )

    return fitnesses


def main():

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
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

    pset = add_primitives_to_pset_from_dict(pset, primitives)

    penalty = {"reg_param": 0.0}
    common_data = {"penalty": penalty}

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness,
        score_func=score,
        predict_func=predict,
        common_data=common_data,
        num_individuals=100,
        num_islands=10,
        generations=200,
        mut_prob=0.1,
        min_height=2,
        max_height=6,
        crossover_prob=0.9,
        overlapping_generation=True,
        print_log=True,
        batch_size=100,
    )

    gpsr.fit(x, y)

    ray.shutdown()


if __name__ == "__main__":
    main()
