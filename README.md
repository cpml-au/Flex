[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library for **symbolic regression** via _Genetic Programming_.
It provides a high-level interface to the [`DEAP`](https://github.com/alucantonio/DEAP)
library, including distributed computing functionalities.

Besides solving classical symbolic regression problems involving _algebraic equations_
(see, for example, the benchmark problems contained in the
[SRBench](https://github.com/cavalab/srbench) repository), _AlpineGP_ is specifically
designed to help identifying _symbolic_ models of _physical systems_ governed by **field equations**.
To this aim, it allows to exploit the **discrete calculus** framework defined and implemented in the library
[`dctkit`](https://github.com/alucantonio/dctkit) as a natural and effective language to express physical models
(i.e., conservation laws).

_AlpineGP_ has been introduced in the paper [_Discovering interpretable physical models
with symbolic regression and discrete exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2),
along with several benchmark problems (heat transfer, linear elasticity, Euler's
elastica). Scripts to reproduce these benchmarks can be found [here](https://github.com/alucantonio/MLST2024_SRDEC).

**Features**:
- distributed computing provided by the [`ray`](https://www.ray.io) library;
- scikit-learn compatible interface;
- hyperparameter configuration via YAML files;
- support for custom operators (with/without strong-typing);
- benchmark suite (Nguyen and SRBench) 

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
Then process the results using the `process_results` notebook.

## Usage

Setting up a symbolic regression problem in _AlpineGP_ involves several key steps:

1. Define the function that computes the prediction associated to an _individual_
(model expression tree). Its arguments may be a _function_ obtained by parsing the
individual tree and possibly other parameters, such as the features (`X`) needed to evaluate
the model. It returns both the error between the predictions and the labels (`y`) and
the predictions themselves. 
```python
def eval_MSE_sol(individual, X, y):

    # ...
    return MSE, prediction
```

2. Define the functions that return the **prediction** and the **fitness** 
   associated to an individual. These functions **must** have at least the following
   arguments in the first three positions:
   - the list of trees to be evaluated by the current worker;
   - the `toolbox` object used to compile the individual trees into callable functions;
   - the dataset features needed for the evaluation of the individuals. The name of the
     argument **must** be `X`.

Additionally, the fourth argument of the **fitness** function **must** be the dataset
labels, called `y`. For unsupervised problems, `None` can be passed for the labels to
the `fit` method of the regressor. Both functions **must** be decorated with `ray.remote` to support
distributed evaluation (multiprocessing). Any additional arguments can be set using
the `common_data` argument of the `GPSymbolicRegressor` object (see below). 
```python
@ray.remote
def predict(trees, toolbox, X):

    callables = compile_individuals(toolbox, trees)

    preds = [None]*len(trees)

    for i, ind in enumerate(callables):
        _, preds[i] = eval_MSE_sol(ind, X, None)

    return preds

@ray.remote
def fitness(trees, toolbox, X, y):
    callables = compile_individuals(toolbox, trees)

    fitnesses = [None]*len(trees)

    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, X, y)
        
        # each fitness MUST be a tuple (required by DEAP)
        fitnesses[i] = (MSE,)

    return fitnesses
```

3. Set up and solve the symbolic regression problem. The configuration of the
   `GPSymbolicRegressor` object can be specified via the arguments of its constructor
   (see the API docs), or loaded from a YAML file.
```python
# read config parameters from YAML file
yamlfile = "ex1.yaml"
filename = os.path.join(os.path.dirname(__file__), yamlfile)

regressor_params, config_file_data = util.load_config_data(filename)

# ...
# ...

# load datasets...

# define the primitive set (input/output types)
pset = gp.PrimitiveSetTyped(...)

# rename arguments of the tree function
pset.renameArguments(ARG0="u")

# define extra common arguments of fitness and predict functions
common_params = {'penalty': penalty}

# create the Symbolic Regression Problem object
gpsr = gps.GPSymbolicRegressor(pset=pset, fitness=fitness.remote,
                               predict_func=predict.remote, common_data=common_params,
                               print_log=True, 
                               **regressor_params)

# solve the symbolic regression problem
gpsr.fit(X_train, y_train)

# compute the prediction on the test dataset given by the best model found during the SR
pred_test = gpsr.predict(X_test)
```

A complete example notebook can be found in the `examples` directory. Also check the
`simple_sr.py` script for an introductory example (use it as a template/skeleton for
defining your symbolic regression problem).

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