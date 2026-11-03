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
mamba env create -f environment.yaml
```

Otherwise, you can update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
pip install -e .
```

to install a development version of the library.

Running the tests:

```bash
tox
```

Generating the docs:

```bash
tox -e docs
```

## Benchmarks

To run the benchmarks on the Nguyen and PMLB datasets, cd into the `bench` folder and
execute:

```bash
bash bench_Nguyen.sh
bash bench_PMLB.sh
```

Each line of these scripts runs `python bench.py <config file> <dataset>` (for example,
`python bench.py PMLB.yaml 1028_SWD`), which solves the problem with 10 random seeds
and writes the training and test $R^2$ of each run to `results/<dataset>.csv`. Then,
process the results using the `results/process_results.ipynb` notebook.

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
SR run on a given dataset. When a validation dataset is used (`validate=True`), the
score is also computed on it at each generation and the maximum is reported as the best
validation score, so a higher score must mean a better model (e.g. $R^2$, or the
negative of an error).

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
regression problem. It evaluates the trees with the numba-compiled interpreter of
`flex.gp.numba_eval` (see below), which supports the primitives of `numpy_primitives`;
with other primitives, compile the trees into Python functions with
`compile_individuals` or `compile_individual_with_consts` of `flex.gp.util`, as in the
fitness function shown above.

### Fast evaluation and constant fitting with numba

The module `flex.gp.numba_eval` evaluates trees built from the primitives of
`numpy_primitives` with a numba-compiled interpreter, which avoids compiling each tree
into a Python function. It also fits the constants of a tree (the terminals named `c`)
by minimizing the mean squared error (MSE) with the Levenberg-Marquardt method, using
the exact derivatives of the tree with respect to its constants, which are computed in
forward mode in the module `flex.gp.numba_grad`. Numerical terminals
(fixed numbers added with `pset.addTerminal`, or ephemeral constants) are evaluated
with their values and are not fitted.

Per-tree functions, to be called inside your own fitness, score and predict functions
(`pset` is the primitive set; its i-th argument reads the i-th column of `X`):

- `fit_mse(tree, pset, X, y, bounds=(-5.0, 5.0), max_iter=8)` fits the constants,
  keeping them within `bounds`, and returns the MSE and the fitted constants. The fit
  starts from random values, uniformly distributed within `bounds`, whose generator is
  seeded with a hash of the tree: a given tree always starts from the same values, in
  serial and in parallel runs. If a derivative is not finite at some data point, the fit
  stops and returns the constants reached so far;
- `mse(tree, pset, X, y, consts)` returns the MSE of the tree with the given constants;
- `predict_tree(tree, pset, X, consts)` returns the values of the tree with the given
  constants.

Ready-made batch functions, to be passed directly to `GPSymbolicRegressor`:

- `fitness_batch` (`fitness`): the fitness of an individual is its MSE after fitting its
  constants with `fit_mse`, plus `parsimony` times its size (number of nodes plus
  number of constants); it is 1e8 if the MSE is not finite;
- `assign_attributes` (`callback_func`): stores the fitness and the fitted constants on
  each individual (`ind.consts`);
- `score_batch` (`score_func`) and `predict_batch` (`predict_func`): MSE and
  predictions of an individual with its stored constants. Since lower MSE means a
  better model, `score_batch` is suited to the `score` method but not to validation
  (`validate=True`), which requires a score where higher is better.

The options `bounds`, `max_iter` and `parsimony` of `fitness_batch` are set through
`common_data`:

```python
from flex.gp.numba_eval import (
    assign_attributes, fitness_batch, predict_batch, score_batch
)

pset.addTerminal(object, float, "c")  # constants fitted by fitness_batch

gpsr = GPSymbolicRegressor(
    pset_config=pset,
    fitness=fitness_batch,
    score_func=score_batch,
    predict_func=predict_batch,
    callback_func=assign_attributes,
    common_data={"parsimony": 1e-4, "max_iter": 8},
    # ...
)
```

The derivatives at points where the differentiation rules are not defined follow these
conventions:

- In the chain rule, a product with a zero factor is zero, also if the other factor is
  infinite or undefined. This gives the true derivative in cases such as
  `sqrt(mul(c, x))` at `x = 0`, where the tree is 0 for every `c`, and `exp(div(c, x))`
  at `x = 0` with `c < 0`, where the tree underflows to 0. The convention is not always
  exact:
  - where the true derivative does not exist but the inner derivative is zero, as for
    `sqrt(square(sub(x, c)))`, which is `|x - c|`, at `x = c`, the derivative is 0 and the
    fit continues instead of stopping;
  - where the limit of the product is not zero, the derivative is wrong. For example,
    `div(x, div(x, sub(c, c)))` is `c1 - c2`, but with two equal constants its derivative
    is computed as 0 instead of 1. This requires a denominator that depends on the
    constants to be exactly zero.
- At a zero base, `pow(a, b)` with `b >= 1` uses the derivative from the right,
  `b * 0^(b - 1)`, also when `b` is not an integer and `a^b` is not a real number for
  negative `a`. For example, `pow(sub(x, c), 1.5)` at `x = c` has a finite derivative
  and the fit continues, even if the tree is not finite for larger `c`.

A step of the fit is accepted only if it reduces the error, which is computed from the
values of the tree: an inaccurate derivative can make the fit less effective, but it does
not affect the reported error, and steps to constants at which the tree is not finite on
the data are rejected.

A tree that is not finite on the data at the start values is not fitted, also if a limit
exists at the point, as for `div(sin(mul(c, x)), x)` at `x = 0`; since the start values
depend only on the tree, this happens at each evaluation of that tree. Finally, `sin` and
`cos` of very large values, such as `sin(exp(mul(c, x)))` at `c * x = 40`, are not
accurate in double precision: the value and the derivative of the tree are finite but
unreliable.

The complete script is `examples/simple_sr_numba.py`.

### Parallelization strategies

_Flex_ uses Ray to evaluate the islands in parallel with one of two strategies, selected
by the `coarse_grained_islands` argument of the `GPSymbolicRegressor` constructor (the
YAML configuration does not set this argument):

```python
gpsr = GPSymbolicRegressor(
    # ...
    num_islands=16,
    coarse_grained_islands=True,  # False (the default) selects the fine-grained strategy
)
```

**Fine-grained strategy (`coarse_grained_islands=False`, the default).** The main process
performs selection and variation for all the islands at each generation. It then splits
the new individuals of all the islands into batches of `batch_size` individuals and
sends them to the Ray workers for the evaluation of the fitness, so any free core can
evaluate an individual of any island. Statistics, validation and the early stopping
check are computed at each generation.

Use this strategy when the number of islands is smaller than the number of cores, because
all the cores can still evaluate the fitness; when individuals of different islands
differ strongly in evaluation cost, because the shared queue of batches balances the
load; when statistics, validation or early stopping are needed at each generation; and
when using `preprocess_args`, which this strategy alone supports.

**Coarse-grained strategy (`coarse_grained_islands=True`).** Each island runs as one Ray
task, which performs selection, variation and fitness evaluation for `mig_freq`
generations. The main process then collects the populations, performs the migration, and
starts the next block of generations. As a result:

- the main process and Ray communicate once per block instead of at each generation;
- each block ends when the slowest island finishes;
- statistics, validation, logging and the early stopping check are computed at the end of
  each block, so a run can continue for up to `mig_freq` generations after reaching the
  early stopping threshold;
- `batch_size` applies only to the evaluation of the initial population.

Use this strategy when selection and variation take a significant share of the run, or
when the fitness of each individual is cheap to evaluate, because these are the cases
where the communication at each generation limits the fine-grained strategy. Each island
uses one core at a time (with the default `num_cpus=1`), so set `num_islands` at least
equal to the number of cores. With more islands than cores, each island waiting in the
queue starts as soon as a core becomes free, which keeps the cores busy when some islands
are slower than others. The argument `num_individuals` is the size of each island, so
reduce it when increasing `num_islands` to keep the same total population. The benchmarks
in the `bench` folder use this strategy.

With both strategies, a run is repeatable when `random` and `numpy.random` are seeded in
the main process before calling `fit`, provided that Ray is started before the seeding,
because starting Ray can draw from `random`. With `multiprocessing=False`, the islands
run in the main process and `coarse_grained_islands` has no effect; the same holds for a
single island, which always follows the fine-grained strategy.

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

   Yes, but only for standard algebraic symbolic regression, by using the class method `get_best_individual_sympy`. You need to define an appropriate `conversion_rules` dictionary, that converts `flex` primitives to the `sympy` format.
   The default value for `conversion_rules` is defined in `numpy_primitives.py`. We suggest to have a look on the API documentation and on the `simple_sr` tutorial for more details.

4. **Why isn't there a pre-defined fitness/predict/score function?**

   To maintain maximum _flexibility_, this library avoids "black-box" functions. While this requires a slightly more verbose setup, it ensures you have total control over the evolution logic. To get started quickly, we recommend using one of our tutorials as a foundation for your specific implementation.

## Repositories using Flex

The following projects leverage Flex for their evolution pipelines. If you have a project you would like to showcase, please submit a pull request with your repository details to be included in this list.

- [SR-DEC_Examples](https://github.com/cpml-au/SR-DEC_Examples): benchmarks for symbolic regression paired with discrete exterior calculus.
- [SR-Traffic](https://github.com/cpml-au/SR-Traffic): symbolic regression of fundamental diagrams of first-order traffic flow models.
- [SR-CLF](https://github.com/cpml-au/SR-CLF): symbolic regression of Control Lyapunov Functions.

## Acknowledgements

This work is supported by the European Union (European Research Council (ERC), ALPS, 101039481). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.
