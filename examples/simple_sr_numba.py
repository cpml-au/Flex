from deap import gp
from flex.gp.numba_eval import (
    assign_attributes,
    fitness_batch,
    predict_batch,
    score_batch,
)
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.primitives import add_primitives_to_pset_from_dict
import numpy as np
import ray


# Ground truth with constants
x = np.linspace(-1.0, 1.0, 50).reshape(-1, 1)
y = 2.5 * np.sin(1.5 * x[:, 0]) + 0.7 * x[:, 0] ** 2


def main():

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
            {"name": "exp", "dimension": None, "rank": None},
        ],
    }

    pset = add_primitives_to_pset_from_dict(pset, primitives)
    # terminal whose values are fitted by the fitness function
    pset.addTerminal(object, float, "c")

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=fitness_batch,
        score_func=score_batch,
        predict_func=predict_batch,
        callback_func=assign_attributes,
        common_data={"parsimony": 1e-4, "max_iter": 8},
        num_individuals=200,
        num_islands=4,
        generations=50,
        mut_prob=0.1,
        min_height=2,
        max_height=6,
        crossover_prob=0.9,
        overlapping_generation=True,
        print_log=True,
        batch_size=50,
    )

    gpsr.fit(x, y)

    best = gpsr.get_best_individuals()[0]
    print("Best individual:", best, "constants:", best.consts)
    print("MSE:", gpsr.score(x, y))

    ray.shutdown()


if __name__ == "__main__":
    main()
