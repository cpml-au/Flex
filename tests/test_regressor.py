from sklearn.utils.estimator_checks import check_estimator
from flex.gp.regressor import GPSymbolicRegressor
from flex.gp.primitives import add_primitives_to_pset_from_dict
from deap import gp

# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split, GridSearchCV
from flex.gp.util import dummy_fitness, dummy_score, dummy_predict


def test_check_regressor():
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

    common_data = {}

    gpsr = GPSymbolicRegressor(
        pset_config=pset,
        fitness=dummy_fitness,
        score_func=dummy_score,
        predict_func=dummy_predict,
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

    check_estimator(
        gpsr,
        expected_failed_checks={
            "check_regressors_train": "dummy model",
            "check_complex_data": "check_array=False",
            "check_dtype_object": "check_array=False",
            "check_estimators_empty_data_messages": "check_array=False",
            "check_estimators_nan_inf": "check_array=False",
            "check_estimator_sparse_tag": "check_array=False",
            "check_estimator_sparse_container": "check_array=False",
            "check_estimator_sparse_array": "check_array=False",
            "check_estimator_sparse_matrix": "check_array=False",
            "check_regressor_data_not_an_array": "check_array=False",
            "check_supervised_y_2d": "check_array=False",
            "check_supervised_y_no_nan": "check_array=False",
            "check_fit1d": "check_array=False",
            "check_fit2d_predict1d": "check_array=False",
        },
    )

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
    test_check_regressor()
