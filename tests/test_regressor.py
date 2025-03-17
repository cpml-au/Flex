from sklearn.utils.estimator_checks import check_estimator
from alpine.gp.regressor import GPSymbolicRegressor
from alpine.gp import util
from deap import gp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from alpine.gp.util import dummy_fitness, dummy_score, dummy_predict


def test_regressor():
    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.renameArguments(ARG0="x")

    primitives = {
        "imports": {"alpine.gp.numpy_primitives": ["numpy_primitives"]},
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

    pset = util.add_primitives_to_pset_from_dict(pset, primitives)

    common_data = {}

    gpsr = GPSymbolicRegressor(
        pset=pset,
        fitness=dummy_fitness,
        error_metric=dummy_score,
        predict_func=dummy_predict,
        common_data=common_data,
        NINDIVIDUALS=100,
        num_islands=10,
        NGEN=200,
        MUTPB=0.1,
        min_height=2,
        max_height=6,
        crossover_prob=0.9,
        overlapping_generation=True,
        print_log=True,
        batch_size=100,
    )

    print(gpsr.get_params())
    check_estimator(gpsr)

    # # Generate synthetic data
    # X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # # Parameter grid
    # param_grid = {"NGEN": [10, 20]}

    # # Grid search
    # grid_search = GridSearchCV(
    #     estimator=gpsr,
    #     param_grid=param_grid,
    #     cv=3,
    #     scoring="r2",
    #     verbose=1,
    #     n_jobs=1,
    # )

    # # Fit the grid search
    # grid_search.fit(X_train, y_train)


if __name__ == "__main__":
    test_regressor()
