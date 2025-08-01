import yaml
from itertools import chain
import ray
import numpy as np
from deap import gp


def load_config_data(filename):
    """Load problem settings from YAML file."""
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    regressor_params = dict()
    regressor_params["num_individuals"] = config_file_data["gp"]["num_individuals"]
    regressor_params["generations"] = config_file_data["gp"]["generations"]
    regressor_params["num_islands"] = config_file_data["gp"]["multi_island"][
        "num_islands"
    ]
    regressor_params["mig_freq"] = config_file_data["gp"]["multi_island"]["migration"][
        "freq"
    ]
    regressor_params["mig_frac"] = config_file_data["gp"]["multi_island"]["migration"][
        "frac"
    ]
    regressor_params["crossover_prob"] = config_file_data["gp"]["crossover_prob"]
    regressor_params["mut_prob"] = config_file_data["gp"]["mut_prob"]
    regressor_params["frac_elitist"] = config_file_data["gp"]["frac_elitist"]
    regressor_params["overlapping_generation"] = config_file_data["gp"][
        "overlapping_generation"
    ]

    regressor_params["select_fun"] = config_file_data["gp"]["select"]["fun"]
    regressor_params["select_args"] = config_file_data["gp"]["select"]["kargs"]
    regressor_params["mut_fun"] = config_file_data["gp"]["mutate"]["fun"]
    regressor_params["mut_args"] = config_file_data["gp"]["mutate"]["kargs"]
    regressor_params["expr_mut_fun"] = config_file_data["gp"]["mutate"]["expr_mut"]
    regressor_params["expr_mut_args"] = config_file_data["gp"]["mutate"][
        "expr_mut_kargs"
    ]
    regressor_params["crossover_fun"] = config_file_data["gp"]["crossover"]["fun"]
    regressor_params["crossover_args"] = config_file_data["gp"]["crossover"]["kargs"]

    regressor_params["validate"] = config_file_data["gp"]["validate"]

    return regressor_params, config_file_data


def detect_nested_trigonometric_functions(equation):
    # List of trigonometric functions
    trig_functions = ["sin", "cos"]
    nested = 0  # Flag to indicate if nested functions are found
    function_depth = 0  # Track depth within trigonometric function calls
    i = 0

    while i < len(equation) and not nested:
        # Look for trigonometric function
        trig_found = any(
            equation[i : i + len(trig)].lower() == trig for trig in trig_functions
        )
        if trig_found:
            # If a trig function is found, look for its opening parenthesis
            j = i
            while j < len(equation) and equation[j] not in ["(", " "]:
                j += 1
            if j < len(equation) and equation[j] == "(":
                if function_depth > 0:
                    # We are already inside a trig function, this is a nested trig
                    # function
                    nested = 1
                function_depth += 1
                i = j  # Move i to the position of '('
        elif equation[i] == "(" and function_depth > 0:
            # Increase depth if we're already in a trig function
            function_depth += 1
        elif equation[i] == ")":
            if function_depth > 0:
                # Leaving a trigonometric function or nested parentheses
                function_depth -= 1
        i += 1

    return nested


def mapper(f, individuals, toolbox_ref, batch_size):
    fitnesses = [] * len(individuals)
    for i in range(0, len(individuals), batch_size):
        individuals_batch = individuals[i : i + batch_size]
        fitnesses.append(f(individuals_batch, toolbox_ref))
    fitnesses = list(chain(*ray.get(fitnesses)))
    return fitnesses


def dummy_fitness(individuals_str, toolbox, X, y):
    fitnesses = [(0.0,)] * len(individuals_str)

    return fitnesses


def dummy_score(individuals_str, toolbox, X, y):

    MSE = [0.0] * len(individuals_str)

    return MSE


def dummy_predict(individuals_str, toolbox, X):
    pred = [np.zeros(len(X))] * len(individuals_str)
    return pred


def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]


def compile_individual_with_consts(tree, toolbox, special_term_name="c"):
    const_idx = 0
    tree_clone = toolbox.clone(tree)
    for i, node in enumerate(tree_clone):
        if isinstance(node, gp.Terminal) and not node.name.startswith("ARG"):
            if node.name == special_term_name:
                tree_clone[i] = gp.Terminal(
                    f"{special_term_name}[{const_idx}]", True, float
                )
                const_idx += 1

    individual = toolbox.compile(expr=tree_clone, extra_args=[special_term_name])
    return individual, const_idx


def fitness_value(ind):
    return ind.fitness.values


def avg_func(values):
    return np.around(np.mean(values), 4)


def std_func(values):
    return np.around(np.std(values), 4)


def min_func(values):
    return np.around(np.min(values), 4)


def max_func(values):
    return np.around(np.max(values), 4)
