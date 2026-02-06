[![Documentation Status](https://readthedocs.org/projects/flexgp/badge/?version=latest)](https://flexgp.readthedocs.io/en/latest/?badge=latest)

# Flex

_Flex_ is a Python library for **symbolic regression** via _Genetic Programming_.
It provides a high-level interface to the [`DEAP`](https://github.com/cpml-au/deap)
library, including distributed computing functionalities.

Besides solving classical symbolic regression problems involving _algebraic equations_
(see, for example, the benchmark problems contained in the
[SRBench](https://github.com/cavalab/srbench) repository), _Flex_ is specifically
designed to help identifying _symbolic_ models of _physical systems_ governed by **field equations**.
To this aim, it allows to exploit the **discrete calculus** framework defined and implemented in the library
[`dctkit`](https://github.com/cpml-au/dctkit) as a natural and effective language to express physical models
(i.e., conservation laws).

_Flex_ has been introduced in the paper [_Discovering interpretable physical models
with symbolic regression and discrete exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2),
along with several benchmark problems (heat transfer, linear elasticity, Euler's
elastica). Scripts to reproduce these benchmarks can be found [here](https://github.com/cpml-au/SR-DEC_Examples).

**Features**:

- distributed computing provided by the [`ray`](https://www.ray.io) library;
- scikit-learn compatible interface;
- hyperparameter configuration via YAML files and optimization via `optuna`;
- support for custom operators (with/without strong-typing);
- benchmark suite (Nguyen and SRBench).

## Installation

Dependencies should be installed within a `conda` environment. We recommend using
[`mamba`](https://github.com/mamba-org/mamba) since it is much faster than `conda` at
solving the environment and downloading the dependencies. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ mamba env create -f environment.yaml
```

Otherwise, you can update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
$ pip install -e .
```

to install a development version of the library.

Running the tests:

```bash
$ tox
```

Generating the docs:

```bash
$ tox -e docs
```

## Benchmarks

To run the benchmarks, cd into the `bench` folder and execute:

```bash
$ ./bench.sh
```

Then, process the results using the `process_results` notebook.

## Usage

Setting up a symbolic regression problem in _Flex_ involves several key steps:

1. Define the functions that return the **prediction** and the **fitness**
   associated to an individual. These functions **must** have at least the following
   arguments in the first three positions:
   - the list of trees to be evaluated by the current worker;
   - the `toolbox` object used to compile the individual trees into callable functions;
   - the dataset features needed for the evaluation of the individuals. The name of the
     argument **must** be `X`.

Additionally, the fourth argument of the **fitness** function **must** be the dataset
labels, called `y`. For unsupervised problems, `None` must be passed as a `y` argument
to the fit method of the regressor, while the `y` argument of the fitness function must
be dropped. Any additional arguments can be set using
the `common_data` argument of the `GPSymbolicRegressor` object (see below).

```python
def predict(individuals_batch, toolbox, X):

    predictions = [None] * len(individuals_batch)

    for i, tree in enumerate(individuals_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        # predictions[i] = ...
    # ...
    return predictions

def fitness(individuals_batch, toolbox, X, y):

    fitnesses = [None]*len(trees)

    for i, tree in enumerate(individuals_batch):

        callable, _ = compile_individual_with_consts(tree, toolbox)
        # MSE = ...

        # each fitness MUST be a tuple (required by DEAP)
        fitnesses[i] = (MSE,)

    return fitnesses
```

_Optionally_, you can define a `score` function with the same arguments as those of the
fitness function, to compute the score (e.g. $R^2$) of the best model found during the
SR run on a given dataset.

2. Set up and solve the symbolic regression problem. The configuration of the
   `GPSymbolicRegressor` object can be specified via the arguments of its constructor
   (see the API docs), or loaded from a YAML file.

```python
yamlfile = "simple_sr.yaml"
filename = os.path.join(os.path.dirname(__file__), yamlfile)

regressor_params, config_file_data = load_config_data(filename)

# pset = gp.PrimitiveSetTyped(...)
# pset.renameArguments(ARG0="x")

pset = add_primitives_to_pset_from_dict(
    pset, config_file_data["gp"]["primitives"]
)

# define extra common arguments of fitness and predict functions
penalty = config_file_data["gp"]["penalty"]
common_data = {"penalty": penalty}

# ...
# ...

# load datasets...

gpsr = GPSymbolicRegressor(
    pset_config=pset,
    fitness=fitness,
    score_func=score,
    predict_func=predict,
    common_data=common_data,
    print_log=True,
    batch_size=100,
    **regressor_params
)

# run the symbolic regression
gpsr.fit(X_train, y_train)

# compute the prediction on the test dataset given by the best model found during the SR
pred_test = gpsr.predict(X_test)

ray.shutdown()
```

Complete example scripts can be found in the `examples` directory. In particular, the
`simple_sr.py` script may be used as a template/skeleton for defining your symbolic
regression problem.

## Citing

```
@article{Manti_2024,
    doi = {10.1088/2632-2153/ad1af2},
    url = {https://dx.doi.org/10.1088/2632-2153/ad1af2},
    year = {2024},
    publisher = {IOP Publishing},
    volume = {5},
    number = {1},
    pages = {015005},
    author = {Simone Manti and Alessandro Lucantonio},
    title = {Discovering interpretable physical models using symbolic regression and discrete exterior calculus},
    journal = {Machine Learning: Science and Technology}
}
```

## FAQ

1. **How do you optimize the constants if the fitness is non-differentiable?**

   In this case, we can use gradient-free optimizer, for example the evolutionary strategies. Using `pygmo` one may use, for example, the `pso` or `sea` algorithm.

2. **What are the most impactful hyperparameters to tune?**

   According to our experience, usually it is very important to tune `num_individuals`, `generations`, `num_islands` and `reg_param`.

3. **Is it possible to get the best individuals in a sympy format?**

   Yes, but only for standard algebraic symbolic regression, by using the class method `get_best_individual_sympy`

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC), ALPS, 101039481). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
