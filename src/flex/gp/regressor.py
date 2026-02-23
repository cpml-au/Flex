from deap import algorithms, tools, gp, base, creator
from deap.tools import migRing
import numpy as np
import operator
from typing import List, Dict, Callable
from os.path import join
import os
import ray
import random
from flex.gp.util import mapper, max_func, min_func, avg_func, std_func, fitness_value
from flex.gp.sympy import stringify_for_sympy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sympy.parsing.sympy_parser import parse_expr
from functools import partial
from itertools import chain
import numpy.typing as npt
from jax import Array


# reducing the number of threads launched by fitness evaluations
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)


class GPSymbolicRegressor(RegressorMixin, BaseEstimator):
    """Symbolic regression via Genetic Programming (GP).

    This regressor evolves symbolic expressions represented as GP trees in order
    to minimize a user-defined fitness function. It is built on top of DEAP and
    follows the scikit-learn estimator interface.

    The regressor supports:
    - Arbitrary user-defined fitness, prediction, and scoring functions
    - Multi-island evolution with periodic migration
    - Elitism and overlapping or non-overlapping generations
    - Parallel fitness evaluation using Ray
    - Validation-set monitoring
    - Conversion of the best individual to a SymPy expression

    Args:
        pset_config: set of primitives and terminals (loosely or strongly typed).
        fitness: fitness evaluation function. It must return a tuple containing a
            single scalar fitness value, e.g. `(fitness_value,)`.
        predict_func: function that returns a prediction given an individual and
            a test dataset as inputs.
        score_func: score metric used for validation and for the `score` method.
        select_fun: string representing the selection operator to use.
        select_args: stringified dictionary of keyword arguments passed to the
            selection operator. The string is evaluated at runtime.
        mut_fun: mutation operator.
        mut_args: arguments for the mutation operator.
        expr_mut_fun: expression generator used during mutation.
        expr_mut_args: arguments for the mutation expression generator.
        crossover_fun: crossover operator.
        crossover_args: arguments for the crossover operator.
        min_height: minimum height of GP trees at initialization.
        max_height: maximum height of GP trees at initialization.
        max_length: maximum number of nodes allowed in a GP tree.
        num_individuals: population size per island.
        generations: number of generations.
        num_islands: number of islands (for a multi-island model).
        remove_init_duplicates: whether to remove duplicate individuals from
            the initial populations.
        crossover_prob: cross-over probability.
        mig_freq: migration frequency (in generations).
        mig_frac: fraction of individuals exchanged during migration.
        crossover_prob: probability of applying crossover.
        mut_prob: probability of applying mutation.
        frac_elitist: fraction of elite individuals preserved each generation.
        overlapping_generation: True if the offspring competes with the parents
            for survival.
        common_data: dictionary of arguments shared between fitness, prediction,
          and scoring functions.
        validate: whether to use a validation dataset.
        preprocess_args: configuration for a function applied to individuals prior
          to fitness evaluation. It must contain three keys: `func`, the callable to
          execute. It must accept an individual and the toolbox as its first two
          arguments; `func_args`: a dictionary of additional arguments for
          func; `callback`: a function used to assign the resulting preprocessed
          values back to each individual.
        callback_func: function called after fitness evaluation to perform custom
            processing.
        seed_str: list of GP expressions used to seed the initial population.
        print_log: whether to print the log containing the population statistics
            during the run.
        num_best_inds_str: number of best individuals printed at each generation.
        save_best_individual: whether to save the string representation of the best
            individual.
        save_train_fit_history: whether to save the training fitness history.
        output_path: directory where outputs are saved.
        batch_size : batch size used for Ray-based fitness evaluation.
        num_cpus: number of CPUs allocated to each Ray task.
        max_calls: maximum number of tasks a Ray worker can execute before restart.
            The default is `0`, which means infinite number of tasks.
        custom_logger: user-defined logging function called with the best individuals.
        special_term_name: name used for ephemeral constants during SymPy conversion.
        sympy_conversion_rules: mapping from GP primitives (DEAP) to SymPy primitives.
            The default is `None`, which means that sympy conversion is disabled.
        multiprocessing: whether to use Ray for parallel fitness evaluation.
    """

    def __init__(
        self,
        pset_config: gp.PrimitiveSet | gp.PrimitiveSetTyped,
        fitness: Callable,
        predict_func: Callable,
        score_func: Callable | None = None,
        select_fun: str = "tools.selection.tournament_with_elitism",
        select_args: str = "{'num_elitist': self.n_elitist, 'tournsize': 3, 'stochastic_tourn': { 'enabled': False, 'prob': [0.8, 0.2] }}",  # noqa: E501
        mut_fun: str = "gp.mutUniform",
        mut_args: str = "{'expr': toolbox.expr_mut, 'pset': pset}",
        expr_mut_fun: str = "gp.genHalfAndHalf",
        expr_mut_args: str = "{'min_': 1, 'max_': 3}",
        crossover_fun: str = "gp.cxOnePoint",
        crossover_args: str = "{}",
        min_height: int = 1,
        max_height: int = 3,
        max_length: int = 100,
        num_individuals: int = 10,
        generations: int = 1,
        num_islands: int = 1,
        remove_init_duplicates: bool = False,
        mig_freq: int = 10,
        mig_frac: float = 0.05,
        crossover_prob: float = 0.5,
        mut_prob: float = 0.2,
        frac_elitist: float = 0.0,
        overlapping_generation: bool = False,
        common_data: Dict | None = None,
        validate: bool = False,
        preprocess_args: Dict | None = None,
        callback_func: Callable | None = None,
        seed_str: List[str] | None = None,
        print_log: bool = False,
        num_best_inds_str: int = 1,
        save_best_individual: bool = False,
        save_train_fit_history: bool = False,
        output_path: str | None = None,
        batch_size: int = 1,
        num_cpus: int = 1,
        max_calls: int = 0,
        custom_logger: Callable = None,
        special_term_name: str = "c",
        sympy_conversion_rules: Dict = None,
        multiprocessing: bool = True,
    ):
        super().__init__()
        self.pset_config = pset_config

        self.fitness = fitness
        self.score_func = score_func
        self.predict_func = predict_func

        self.print_log = print_log
        self.num_best_inds_str = num_best_inds_str
        self.preprocess_args = preprocess_args
        self.callback_func = callback_func
        self.save_best_individual = save_best_individual
        self.save_train_fit_history = save_train_fit_history
        self.output_path = output_path
        self.batch_size = batch_size

        self.common_data = common_data

        self.num_individuals = num_individuals
        self.generations = generations
        self.num_islands = num_islands
        self.crossover_prob = crossover_prob
        self.mut_prob = mut_prob
        self.select_fun = select_fun
        self.select_args = select_args
        self.mut_fun = mut_fun
        self.mut_args = mut_args
        self.expr_mut_fun = expr_mut_fun
        self.expr_mut_args = expr_mut_args
        self.crossover_fun = crossover_fun
        self.crossover_args = crossover_args
        self.min_height = min_height
        self.max_height = max_height
        self.max_length = max_length
        self.mig_freq = mig_freq
        self.mig_frac = mig_frac

        self.overlapping_generation = overlapping_generation
        self.validate = validate

        self.frac_elitist = frac_elitist

        self.seed_str = seed_str
        self.num_cpus = num_cpus
        self.remove_init_duplicates = remove_init_duplicates
        self.max_calls = max_calls
        self.custom_logger = custom_logger
        self.special_term_name = special_term_name
        self.sympy_conversion_rules = sympy_conversion_rules
        self.multiprocessing = multiprocessing

    def __sklearn_tags__(self):
        # since we are allowing cases in which y=None
        # we need to modify the tag requires_y to False
        # (check sklearn docs)
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags

    @property
    def n_elitist(self):
        """Compute the number of elitists in the population"""
        return int(self.frac_elitist * self.num_individuals)

    def get_params(self, deep: bool = True):
        return self.__dict__

    def __creator_toolbox_pset_config(self):
        """Initialize toolbox and individual creator based on config file.

        Returns:
            a tuple containing the initialized toolbox and the primitive set.

        """
        pset = self.pset_config
        toolbox = base.Toolbox()

        # SELECTION
        toolbox.register("select", eval(self.select_fun), **eval(self.select_args))

        # MUTATION
        toolbox.register(
            "expr_mut", eval(self.expr_mut_fun), **eval(self.expr_mut_args)
        )

        toolbox.register("mutate", eval(self.mut_fun), **eval(self.mut_args))

        # CROSSOVER
        toolbox.register("mate", eval(self.crossover_fun), **eval(self.crossover_args))
        toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

        # INDIVIDUAL GENERATOR/CREATOR
        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=pset,
            min_=self.min_height,
            max_=self.max_height,
            max_length=self.max_length,
        )
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        createIndividual = creator.Individual
        toolbox.register(
            "individual", tools.initIterate, createIndividual, toolbox.expr
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        if self.seed_str is not None:
            self.seed_ind = [
                createIndividual.from_string(i, pset) for i in self.seed_str
            ]
        return toolbox, pset

    def __init_data_store(self):
        """Initialize the store data dict with the common parameters."""
        self.__data_store = dict()

        if self.common_data is not None:
            # FIXME: does everything work when the functions do not have common args?
            self.__store_fit_score_common_args(self.common_data)

    def __store_fit_score_common_args(self, data: Dict):
        """Store names and values of the arguments that are in common between
        the fitness and the error metric functions in the common object space.

        Args:
            data: dictionary containing arguments names and values.
        """
        self.__store_shared_objects("common", data)

    def __store_datasets(self, datasets: Dict[str, npt.NDArray | Array]):
        """Store datasets with the corresponding label ("train", "val" or "test")
        in the common object space. The datasets are passed as parameters to
        the fitness, and possibly to the error metric and the prediction functions.

        Args:
            datasets: the keys are 'train', 'val' and 'test' denoting the training,
                the validation and the test datasets, respectively. The associated
                values are numpy or jax arrays.
        """
        for dataset_label, dataset_data in datasets.items():
            self.__store_shared_objects(dataset_label, dataset_data)

    def __store_shared_objects(self, label: str, data: Dict):
        """Store a dictionary of data in the internal data store, optionally
        converting values to Ray object references for shared-memory access.

        Args:
            label: key under which the data dictionary will be stored internally.
            data: dictionary of objects to store.
        """
        for key, value in data.items():
            # replace each item of the dataset with its obj ref
            if not isinstance(value, ray.ObjectRef) and self.multiprocessing:
                data[key] = ray.put(value)
        self.__data_store[label] = data

    def __fetch_shared_objects(self, stored_data: Dict):
        """Retrieve objects from the Ray object store and reconstruct
        a local dictionary of concrete values.

        Args:
            stored_data: dictionary potentially containing ``ray.ObjectRef`` values.

        Returns:
            a new dictionary where all Ray object references have been
            dereferenced into concrete Python objects.
        """
        fetched_data = dict()
        for key, value in stored_data.items():
            if isinstance(value, ray.ObjectRef):
                fetched_data[key] = ray.get(value)
            else:
                fetched_data[key] = value

        return fetched_data

    def __print(self, message: str):
        """Helper to handle conditional printing.

        Args:
            message: message to print.
        """
        if self.print_log:
            print(message, flush=True)

    def __init_stats_log(self):
        """Initialize logbook to collect statistics."""
        self.__logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if self.validate:
            self.__logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.__logbook.chapters["valid"].header = ("valid_score",)
        else:
            self.__logbook.header = "gen", "evals", "fitness", "size"
        self.__logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.__logbook.chapters["size"].header = "min", "avg", "max", "std"

        # Initialize variables for statistics
        self.__stats_fit = tools.Statistics(fitness_value)
        self.__stats_size = tools.Statistics(len)
        self.__mstats = tools.MultiStatistics(
            fitness=self.__stats_fit, size=self.__stats_size
        )
        self.__mstats.register("avg", avg_func)
        self.__mstats.register("std", std_func)
        self.__mstats.register("min", min_func)
        self.__mstats.register("max", max_func)

        self.__train_fit_history = []

    def __compute_valid_stats(self, pop: List, toolbox: base.Toolbox):
        """Compute the validation score of the best individual.

        Args:
            pop: a given population.
            toolbox: the toolbox for the evolution.

        Returns:
            the validation score.
        """
        best = tools.selBest(pop, k=1)
        # FIXME: ugly way of handling lists/tuples; assume evaluate_val_score returns a
        # single-valued tuple as eval_val_fit
        valid_score = toolbox.map(toolbox.evaluate_val_score, best)[0]
        return valid_score

    def get_pop_stats(self):
        """Get population stats."""
        pop = self.__flatten_list(self.__pop)
        return self.__mstats.compile(pop)

    def __stats(self, pop: List, gen: int, evals: int, toolbox: base.Toolbox):
        """Compute and print statistics of a population.

        Args:
            pop: a given population.
            gen: the generation number.
            evals: the number of the evaluations in the current generation.
            toolbox: the toolbox for the evolution.
        """

        # LINE_UP = '\033[1A'
        # LINE_CLEAR = '\x1b[2K'
        # Compile statistics for the current population
        record = self.get_pop_stats()

        # record the statistics in the logbook
        if self.validate:
            # compute satistics related to the validation set
            valid_score = self.__compute_valid_stats(pop, toolbox)
            record["valid"] = {"valid_score": valid_score}

        self.__logbook.record(gen=gen, evals=evals, **record)

        if self.print_log:
            # Print statistics for the current population
            # print(LINE_UP, end=LINE_CLEAR, flush=True)
            print(self.__logbook.stream, flush=True)

    def __get_remote(self, f: Callable):
        """Wraps a function for parallel execution if multiprocessing is enabled.

        Args:
            f: The function to be executed, typically a task or objective function.

        Returns:
            The Ray remote handle if multiprocessing is active, otherwise the
            original function.
        """
        if self.multiprocessing:
            return ray.remote(num_cpus=self.num_cpus, max_calls=self.max_calls)(
                f
            ).remote
        else:
            return f

    def __register_fitness_func(self, toolbox: base.Toolbox):
        """Register fitness function in the toolbox.

        Args:
            toolbox: the toolbox for the evolution.
        """
        store = self.__data_store
        args_train = store["common"] | store["train"]
        toolbox.register(
            "evaluate_train", self.__get_remote(self.fitness), **args_train
        )

    def __register_val_funcs(self, toolbox: base.Toolbox):
        """Register the functions needed for validation, i.e. the score metric and the
        fitness function. Must be called after storing the datasets in the common
        obj space.

        Args:
            toolbox: the toolbox for the evolution.
        """
        store = self.__data_store
        args_val = store["common"] | store["val"]
        toolbox.register(
            "evaluate_val_fit", self.__get_remote(self.fitness), **args_val
        )
        toolbox.register(
            "evaluate_val_score", self.__get_remote(self.score_func), **args_val
        )

    def __register_map(self, toolbox: base.Toolbox):
        """Register mapper in the toolbox.

        Args:
            toolbox: the toolbox for the evolution.
        """
        if self.multiprocessing:
            toolbox_ref = ray.put(toolbox)
            toolbox.register(
                "map", mapper, toolbox_ref=toolbox_ref, batch_size=self.batch_size
            )
        else:

            def base_mapper(f, individuals, toolbox):
                individuals_batch = [[ind] for ind in individuals]
                fitnesses = map(partial(f, toolbox=toolbox), individuals_batch)
                return list(chain.from_iterable(fitnesses))

            toolbox.register("map", base_mapper, toolbox=toolbox)

    def _prepare_fit(
        self,
        X: npt.NDArray | Array,
        y: npt.NDArray | Array,
        X_val: npt.NDArray | Array,
        y_val: npt.NDArray | Array,
    ):
        """Prepare datasets, internal state, and the DEAP toolbox for evolution.

        Args:
            X: training input features.
            y: training target values. Can be None for unsupervised tasks.
            X_val: validation input features.
            y_val: validation target values. Can be None for unsupervised tasks.

        Returns:
            a configured DEAP toolbox containing registered evaluation and
            preprocessing functions.
        """
        validated_data = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            skip_check_array=True,
            # ensure_2d=False,
            # allow_nd=True,
            # multi_output=True,
        )
        if y is None:
            X = validated_data
            train_data = {"X": X}
        else:
            X, y = validated_data
            train_data = {"X": X, "y": y}

        if self.validate and X_val is not None:
            if y_val is None:
                val_data = {"X": X_val}
            else:
                val_data = {"X": X_val, "y": y_val}
            datasets = {"train": train_data, "val": val_data}
        else:
            datasets = {"train": train_data}

        # config individual creator and toolbox
        toolbox, _ = self.__creator_toolbox_pset_config()

        self.__init_data_store()

        self.__store_datasets(datasets)

        self.__init_stats_log()

        # register functions for fitness evaluation (train/val)
        self.__register_map(toolbox)
        self.__register_fitness_func(toolbox)
        if self.validate and self.score_func is not None:
            self.__register_val_funcs(toolbox)

        if self.preprocess_args is not None:
            toolbox.register(
                "preprocess_func",
                self.__get_remote(self.preprocess_args["func"]),
                **self.preprocess_args["func_args"],
            )

        return toolbox

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: npt.NDArray | Array,
        y: npt.NDArray | Array = None,
        X_val: npt.NDArray | Array = None,
        y_val: npt.NDArray | Array = None,
    ):
        """Fits the training data using GP-based symbolic regression.

        This method initializes the populations, evaluates the fitness of the
        individuals, and evolves the populations for the specified number of
        generations.

        Args:
            X: training input data.
            y: training targets. If None, the fitness function must not require
                targets.
            X_val: validation input data.
            y_val: validation targets.
        """
        toolbox = self._prepare_fit(X, y, X_val, y_val)
        self.__run(toolbox)
        self.is_fitted_ = True
        return self

    def predict(self, X: npt.NDArray | Array):
        """Predict outputs using the best evolved individual.

        Args:
            X: Input data.

        Returns:
            predictions computed by the best individual.
        """
        check_is_fitted(self)
        toolbox, pset = self.__creator_toolbox_pset_config()
        X = validate_data(
            self, X, accept_sparse=False, reset=False, skip_check_array=True
        )
        test_data = {"X": X}
        store = self.__data_store
        args_predict_func = self.__fetch_shared_objects(store["common"]) | test_data
        u_best = self.predict_func((self._best,), toolbox=toolbox, **args_predict_func)[
            0
        ]
        return u_best

    def score(self, X: npt.NDArray | Array, y: npt.NDArray | Array = None):
        """Compute the score of the best evolved individual.
        This method evaluates the user-provided `score_func` on the given dataset.

        Args:
            X: input data.
            y: target values.

        Returns:
            score value returned by `score_func`.
        """
        check_is_fitted(self)
        toolbox, pset = self.__creator_toolbox_pset_config()
        validated_data = validate_data(
            self, X, y, accept_sparse=False, reset=False, skip_check_array=True
        )
        if y is None:
            X = validated_data
            test_data = {"X": X}
        else:
            X, y = validated_data
            test_data = {"X": X, "y": y}
        store = self.__data_store
        args_score_func = self.__fetch_shared_objects(store["common"]) | test_data
        score = self.score_func((self._best,), toolbox=toolbox, **args_score_func)[0]
        return score

    def __flatten_list(self, nested_lst: List):
        """Convert a list of lists into a single flat list.

        Args:
            nested_lst: a list containing multiple sublists.

        Returns:
            a single list containing all elements of the sublists in order.
        """
        flat_list = []
        for lst in nested_lst:
            flat_list += lst
        return flat_list

    def __unflatten_list(self, flat_lst: List, lengths: List):
        """Restore a flat list into a list of sublists based on provided lengths.

        Args:
            flat_lst: the single-dimensional list to be partitioned.
            lengths: a list of integers specifying the size of each original sublist.

        Returns:
            a list of lists reconstructed to match the original structure.
        """
        result = []
        start = 0  # Starting index of the current sublist
        for length in lengths:
            # Slice the list from the current start index to start+length
            end = start + length
            result.append(flat_lst[start:end])
            start = end  # Update the start index for the next sublist
        return result

    def __evolve_islands(self, cgen: int, toolbox: base.Toolbox):
        """Performs a single iteration of the evolution pipeline with the
        multi-islands strategy.

        Args:
            cgen: current generation index.
            toolbox: the toolbox for the evolution.

        Returns:
            the total number of evaluations.
        """
        num_evals = 0

        invalid_inds = [None] * self.num_islands
        offsprings = [None] * self.num_islands
        elite_inds = [None] * self.num_islands

        for i in range(self.num_islands):
            # Select the parents for the offspring
            offsprings[i] = list(map(toolbox.clone, toolbox.select(self.__pop[i])))

            # Apply crossover and mutation to the offspring with elitism
            elite_inds[i] = tools.selBest(offsprings[i], self.n_elitist)
            offsprings[i] = elite_inds[i] + algorithms.varOr(
                offsprings[i],
                toolbox,
                self.num_individuals - self.n_elitist,
                self.crossover_prob,
                self.mut_prob,
            )

            # add individuals subject to cross-over and mutation to the list of invalids
            invalid_inds[i] = [ind for ind in offsprings[i] if not ind.fitness.valid]

            num_evals += len(invalid_inds[i])

            if self.preprocess_args is not None:
                preprocess_values = toolbox.map(
                    toolbox.preprocess_func, invalid_inds[i]
                )
                self.preprocess_args["callback"](invalid_inds[i], preprocess_values)

        fitnesses = toolbox.map(
            toolbox.evaluate_train, self.__flatten_list(invalid_inds)
        )
        fitnesses = self.__unflatten_list(fitnesses, [len(i) for i in invalid_inds])

        for i in range(self.num_islands):
            if self.callback_func is not None:
                self.callback_func(invalid_inds[i], fitnesses[i])
            else:
                for ind, fit in zip(invalid_inds[i], fitnesses[i]):
                    ind.fitness.values = fit

            # survival selection
            if not self.overlapping_generation:
                # The population is entirely replaced by the offspring
                self.__pop[i][:] = offsprings[i]
            else:
                # parents and offspring compete for survival (truncation selection)
                self.__pop[i] = tools.selBest(
                    self.__pop[i] + offsprings[i], self.num_individuals
                )

        # migrations among islands
        if cgen % self.mig_frac == 0 and self.num_islands > 1:
            migRing(
                self.__pop,
                int(self.mig_frac * self.num_individuals),
                selection=random.sample,
            )

        return num_evals

    def __remove_duplicates(self, toolbox: base.Toolbox):
        """Remove duplicates in the population.

        Args:
            toolbox: the toolbox for the evolution.
        """
        for i in range(self.num_islands):
            while True:
                fitnesses = toolbox.map(toolbox.evaluate_train, self.__pop[i])
                if self.callback_func is not None:
                    self.callback_func(self.__pop[i], fitnesses)
                else:
                    for ind, fit in zip(self.__pop[i], fitnesses):
                        ind.fitness.values = fit
                fitness_array = np.array(
                    [ind.fitness.values[0] for ind in self.__pop[i]]
                )
                # Identify unique fitness indices
                _, idx_unique = np.unique(fitness_array, return_index=True)
                # Identify duplicate indices
                dup_indices = np.setdiff1d(np.arange(len(fitnesses)), idx_unique)
                # Identify indices with fitness above threshold
                threshold_indices = np.where(fitness_array > 1e5)[0]
                # Combine both types of bad indices
                bad_indices = np.unique(
                    np.concatenate([dup_indices, threshold_indices])
                )
                if len(bad_indices) == 0:
                    break
                for idx in bad_indices:
                    self.__pop[i][idx] = toolbox.individual()

    def get_best_individuals(self, n_ind: int = 1):
        """Returns the best individuals across all islands.

        Args:
            n_ind : number of top individuals to return.

        Returns:
            List of the best GP individuals.
        """
        best_inds = tools.selBest(self.__flatten_list(self.__pop), k=n_ind)
        return best_inds[:n_ind]

    def _step(self, toolbox: base.Toolbox, cgen: int):
        """Performs a single step of the evolution pipeline.

        Args:
            toolbox: the toolbox for the evolution.
            cgen: current generation index.
        """
        num_evals = self.__evolve_islands(cgen, toolbox)

        # select the best individuals in the current population
        # (including all islands)
        best_inds = self.get_best_individuals(self.num_best_inds_str)

        # compute and print population statistics (including all islands)
        self.__stats(self.__flatten_list(self.__pop), cgen, num_evals, toolbox)

        if self.print_log:
            print("Best individuals of this generation:", flush=True)
            for i in range(self.num_best_inds_str):
                print(str(best_inds[i]), flush=True)
            if self.custom_logger is not None:
                self.custom_logger(best_inds)

        # Update history of best fitness and best validation score
        self.__train_fit_history = self.__logbook.chapters["fitness"].select("min")
        if self.validate:
            self.__val_score_history = self.__logbook.chapters["valid"].select(
                "valid_score"
            )
            self.max_val_score = max(self.__val_score_history)

        self._best = best_inds[0]

    def _restart(self, toolbox: base.Toolbox, save_best_inds: bool = True):
        """Re-initializes the population while optionally preserving the best
        individuals.

        Args:
            toolbox: the toolbox for the evolution.
            save_best_inds: whether to keep the best individual from each island
                in the new population. Defaults to True.
        """
        best_inds = [None] * self.num_islands
        for i in range(self.num_islands):
            best_inds[i] = tools.selBest(self.__pop[i], k=1)[0]
        self._generate_init_pop(toolbox)
        for i in range(self.num_islands):
            self.__pop[i][0] = best_inds[i]

    def _generate_init_pop(self, toolbox: base.Toolbox):
        """Generates the initial population.

        Args:
            toolbox: the toolbox for the evolution.
        """
        self.__pop = [None] * self.num_islands
        for i in range(self.num_islands):
            self.__pop[i] = toolbox.population(n=self.num_individuals)

        # Seeds the first island with individuals
        if self.seed_str is not None:
            self.__print(" Seeding population with individuals...")
            self.__pop[0][: len(self.seed_ind)] = self.seed_ind

        if self.remove_init_duplicates:
            self.__print(" Removing duplicates from initial population(s)...")
            self.__remove_duplicates(toolbox)
            self.__print(" DONE.")

        if self.preprocess_args is not None:
            for i in range(self.num_islands):
                preprocess_values = toolbox.map(toolbox.preprocess_func, self.__pop[i])
                self.preprocess_args["callback"](self.__pop[i], preprocess_values)

    def _evaluate_init_pop(self, toolbox: base.Toolbox):
        """Evaluates the initial population.

        Args:
            toolbox: the toolbox for the evolution.
        """
        for i in range(self.num_islands):
            fitnesses = toolbox.map(toolbox.evaluate_train, self.__pop[i])

            if self.callback_func is not None:
                self.callback_func(self.__pop[i], fitnesses)
            else:
                for ind, fit in zip(self.__pop[i], fitnesses):
                    ind.fitness.values = fit

    def __run(self, toolbox: base.Toolbox):
        """Performs the evolution pipeline.

        Args:
            toolbox: the toolbox for the evolution.
        """

        self.__print("Generating initial population(s)...")
        self._generate_init_pop(toolbox)
        self.__print("DONE.")

        # Evaluate the fitness of the entire population on the training set
        self.__print("Evaluating initial population(s)...")
        self._evaluate_init_pop(toolbox)
        self.__print("DONE.")

        if self.validate:
            self.__print("Using validation dataset.")

        self.__print(" -= START OF EVOLUTION =- ")

        for gen in range(self.generations):
            self.__cgen = gen + 1

            self._step(toolbox, self.__cgen)

            if self._best.fitness.values[0] <= 1e-15:
                self.__print("Fitness threshold reached - STOPPING.")
                break

        self.__print(" -= END OF EVOLUTION =- ")

        self.__last_gen = self.__cgen

        # define sympy representation of the best individual
        if self.sympy_conversion_rules is not None:
            self.__best_sympy = parse_expr(
                stringify_for_sympy(
                    self._best, self.sympy_conversion_rules, self.special_term_name
                )
            )
            best_str = self.__best_sympy
        else:
            best_str = self._best

        self.__print(f"The best individual is {best_str}")

        self.__print(
            f"The best fitness on the training set is {self.__train_fit_history[-1]}"
        )

        if self.validate:
            self.__print(
                f"The best score on the validation set is {self.max_val_score}"
            )

        if self.save_best_individual and self.output_path is not None:
            self.__save_best_individual(self.output_path)
            self.__print("String of the best individual saved to disk.")

        if self.save_train_fit_history and self.output_path is not None:
            self.__save_train_fit_history(self.output_path)
            self.__print("Training fitness history saved to disk.")

        # NOTE: ray.shutdown should be manually called by the user

    def __save_best_individual(self, output_path: str):
        """Saves the string of the best individual of the population in a .txt file.

        Args:
            output_path: path where the history should be saved.
        """
        file = open(join(output_path, "best_ind.txt"), "w")
        file.write(str(self._best))
        file.close()

    def __save_train_fit_history(self, output_path: str):
        """Saves the training (and validation) history in a .npy file.

        Args:
            output_path: path where the history should be saved.
        """
        np.save(join(output_path, "train_fit_history.npy"), self.__train_fit_history)
        if self.validate:
            np.save(
                join(output_path, "val_score_history.npy"), self.__val_score_history
            )

    def get_best_individual_sympy(self):
        """Returns the SymPy expression of the best individual.

        Returns:
            sympy representation of the best individual if conversion is enabled.
        """

        if self.sympy_conversion_rules is not None:
            return self.__best_sympy

    def get_train_fit_history(self):
        """Returns the training score history.

        Returns:
            list containing the validation scores at each generation.
        """
        return self.__train_fit_history

    def get_val_score_history(self):
        """Returns the validation score history.

        Returns:
            list containing the validation scores at each generation.
        """
        return self.__val_score_history

    def get_last_gen(self):
        """Returns the last generation index.

        Returns:
            the last generation.
        """
        return self.__last_gen

    def save_best_test_sols(self, X_test: npt.NDArray | Array, output_path: str):
        """Compute and save the predictions corresponding to the best individual
        at the end of the evolution, evaluated over the test dataset.

        Args:
            X_test: test input data.
            output_path: path where the predictions should be saved (one .npy file for
                each sample in the test dataset).
        """
        best_test_sols = self.predict(X_test)

        for i, sol in enumerate(best_test_sols):
            np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)

        print("Best individual solution evaluated over the test set saved to disk.")
