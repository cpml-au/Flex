from deap import algorithms, tools, gp, base, creator
from deap.tools import migRing
import matplotlib.pyplot as plt
import numpy as np
import operator
from typing import List, Dict, Callable
from os.path import join
import networkx as nx
from alpine.data import Dataset
import os
import ray
import random
from alpine.gp.util import mapper, add_primitives_to_pset_from_dict
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils.validation import check_is_fitted


# reducing the number of threads launched by fitness evaluations
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)


class GPSymbolicRegressor(RegressorMixin, BaseEstimator):
    """Symbolic regression problem via Genetic Programming.

    Args:
        pset: set of primitives and terminals (loosely or strongly typed).
        predict_func: function that returns a prediction given an individual and
            a test `Dataset` as inputs.
        NINDIVIDUALS: number of individuals in the parent population.
        NGEN: number of generations.
        num_islands: number of islands (for a multi-island model).
        crossover_prob: cross-over probability.
        MUTPB: mutation probability.
        frac_elitist: best individuals to keep expressed as a percentage of the
            population (ex. 0.1 = keep top 10% individuals)
        overlapping_generation: True if the offspring competes with the parents
            for survival.
        plot_history: whether to plot fitness vs generation number.
        print_log: whether to print the log containing the population statistics
            during the run.
        print_best_inds_str: number of best individuals' strings to print after
            each generation.
        plot_best: whether to show the plot of the solution corresponding to the
            best individual every plot_freq generations.
        plot_freq: frequency (number of generations) of the plot of the best
            individual.
        seed: list of individual strings to seed in the initial population.
        preprocess_func: function to call before evaluating the fitness of the
            individuals of each generation.
        callback_func: function to call after evaluating the fitness of the
            individuals of each generation. It takes the population/batch of
            individuals and the list containing all the values of the attributes
            returned by the fitness evaluation function.
    """

    def __init__(
        self,
        pset_config: gp.PrimitiveSet | gp.PrimitiveSetTyped,
        fitness: Callable,
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
        NINDIVIDUALS: int = 10,
        NGEN: int = 1,
        num_islands: int = 1,
        mig_freq: int = 10,
        mig_frac: float = 0.05,
        crossover_prob: float = 0.5,
        MUTPB: float = 0.2,
        frac_elitist: float = 0.0,
        overlapping_generation: bool = False,
        immigration_enabled: bool = False,
        immigration_freq: int = 0,
        immigration_frac: float = 0.0,
        error_metric: Callable | None = None,
        predict_func: Callable | None = None,
        common_data: Dict | None = None,
        validate: bool = False,
        preprocess_func: Callable | None = None,
        callback_func: Callable | None = None,
        seed: List[str] | None = None,
        plot_history: bool = False,
        print_log: bool = False,
        num_best_inds_str: int = 1,
        plot_best: bool = False,
        plot_freq: int = 5,
        plot_best_genealogy: bool = False,
        plot_best_individual_tree: bool = False,
        save_best_individual: bool = False,
        save_train_fit_history: bool = False,
        output_path: str | None = None,
        batch_size=1,
    ):
        super().__init__()
        self.pset_config = pset_config

        self.fitness = fitness
        self.error_metric = error_metric
        self.predict_func = predict_func

        self.plot_best = plot_best

        self.plot_best_genealogy = plot_best_genealogy

        self.plot_history = plot_history
        self.print_log = print_log
        self.num_best_inds_str = num_best_inds_str
        self.plot_freq = plot_freq
        self.preprocess_func = preprocess_func
        self.callback_func = callback_func
        self.plot_best_individual_tree = plot_best_individual_tree
        self.save_best_individual = save_best_individual
        self.save_train_fit_history = save_train_fit_history
        self.output_path = output_path
        self.batch_size = batch_size

        self.common_data = common_data

        self.NINDIVIDUALS = NINDIVIDUALS
        self.NGEN = NGEN
        self.num_islands = num_islands
        self.crossover_prob = crossover_prob
        self.MUTPB = MUTPB
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
        self.mig_freq = mig_freq
        self.mig_frac = mig_frac
        self.immigration_enabled = immigration_enabled
        self.immigration_frac = immigration_frac
        self.immigration_freq = immigration_freq

        self.overlapping_generation = overlapping_generation
        self.validate = validate

        self.frac_elitist = frac_elitist

        self.seed = seed

    @property
    def n_elitist(self):
        return int(self.frac_elitist * self.NINDIVIDUALS)

    def get_params(self, deep=True):
        return self.__dict__

    def __pset_config(self):
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

        pset = add_primitives_to_pset_from_dict(pset, primitives)
        return pset

    def __creator_toolbox_pset_config(self):
        """Initialize toolbox and individual creator based on config file."""
        pset = self.__pset_config()
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
        )
        toolbox.register(
            "expr_pop",
            gp.genHalfAndHalf,
            pset=pset,
            min_=self.min_height,
            max_=self.max_height,
            is_pop=True,
        )
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        createIndividual = creator.Individual
        toolbox.register(
            "individual", tools.initIterate, createIndividual, toolbox.expr
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        self.__createIndividual = createIndividual

        if self.seed is not None:
            self.seed = [
                self.__createIndividual.from_string(i, pset) for i in self.seed
            ]
        return toolbox, pset

    def __store_fit_error_common_args(self, data: Dict):
        """Store names and values of the arguments that are in common between
        the fitness and the error metric functions in the common object space.

        Args:
            data: dictionary containing arguments names and values.
        """
        self.__store_shared_objects("common", data)

    def __store_datasets(self, datasets: Dict[str, Dataset]):
        """Store datasets with the corresponding label ("train", "val" or "test")
        in the common object space. The datasets are passed as parameters to
        the fitness, and possibly to the error metric and the prediction functions.

        Args:
            datasets: the keys are 'train', 'val' and 'test' denoting the training,
                the validation and the test datasets, respectively. The associated
                values are `Dataset` objects.
        """
        for dataset_label, dataset_data in datasets.items():
            self.__store_shared_objects(dataset_label, dataset_data)

    def __store_shared_objects(self, label: str, data: Dict):
        for key, value in data.items():
            # replace each item of the dataset with its obj ref
            if not isinstance(value, ray.ObjectRef):
                data[key] = ray.put(value)
        self.__data_store[label] = data

    def __init_logbook(self):
        # Initialize logbook to collect statistics
        self.__logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if self.validate:
            self.__logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.__logbook.chapters["valid"].header = "valid_fit", "valid_err"
        else:
            self.__logbook.header = "gen", "evals", "fitness", "size"
        self.__logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.__logbook.chapters["size"].header = "min", "avg", "max", "std"

    def __compute_valid_stats(self, pop, toolbox):
        best = tools.selBest(pop, k=1)
        # FIXME: ugly way of handling lists/tuples; assume eval_val_MSE returns a
        # single-valued tuple as eval_val_fit
        valid_fit = toolbox.map(toolbox.evaluate_val_fit, best)[0][0]
        valid_err = toolbox.map(toolbox.evaluate_val_MSE, best)[0]

        return valid_fit, valid_err

    def __stats(self, pop, gen, evals):
        """Compute and print statistics of a population."""

        # LINE_UP = '\033[1A'
        # LINE_CLEAR = '\x1b[2K'
        # Compile statistics for the current population
        record = self.__mstats.compile(pop)

        # record the statistics in the logbook
        if self.validate:
            # compute satistics related to the validation set
            valid_fit, valid_err = self.__compute_valid_stats(pop)
            record["valid"] = {"valid_fit": valid_fit, "valid_err": valid_err}

        self.__logbook.record(gen=gen, evals=evals, **record)

        if self.print_log:
            # Print statistics for the current population
            # print(LINE_UP, end=LINE_CLEAR, flush=True)
            print(self.__logbook.stream, flush=True)

    def __plot_history(self):
        """Plots the fitness of the best individual vs generation number."""
        if not self.__plot_initialized:
            self.__plot_initialized = True
            # new figure number when starting with new evolution
            self.__fig_id = self.__fig_id + 1
            plt.figure(self.__fig_id).show()
            plt.pause(0.01)

        plt.figure(self.__fig_id)
        fig = plt.gcf()

        # Array of generations starts from 1
        x = range(1, len(self.__train_fit_history) + 1)
        plt.plot(x, self.__train_fit_history, "b", label="Training Fitness")
        if self.validate:
            plt.plot(x, self.val_fit_history, "r", label="Validation Fitness")
            fig.legend(loc="upper right")

        plt.xlabel("Generation #")
        plt.ylabel("Best Fitness")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

    def __plot_genealogy(self, best):
        # Get genealogy of best individual
        import networkx

        gen_best = self.__history.getGenealogy(best)
        graph = networkx.DiGraph(gen_best)
        graph = graph.reverse()
        pos = networkx.nx_agraph.graphviz_layout(
            graph, prog="dot", args="-Gsplines=True"
        )
        # Retrieve individual strings for graph node labels
        labels = gen_best.copy()
        for key in labels.keys():
            labels[key] = str(self.__history.genealogy_history[key])
        plt.figure()
        networkx.draw_networkx(graph, pos=pos)
        label_options = {"ec": "k", "fc": "lightblue", "alpha": 1.0}
        networkx.draw_networkx_labels(
            graph, pos=pos, labels=labels, font_size=10, bbox=label_options
        )

        # Save genealogy to file
        # networkx.nx_agraph.write_dot(graph, "genealogy.dot")

    def __get_remote(self, f):
        return (ray.remote(f)).remote

    def __register_fitness_func(self, toolbox):
        store = self.__data_store
        args_train = store["common"] | store["train"]
        toolbox.register(
            "evaluate_train", self.__get_remote(self.fitness), **args_train
        )

    def __register_val_funcs(self, toolbox):
        """Register the functions needed for validation, i.e. the error metric and the
        fitness function. Must be called after storing the datasets in the common
        obj space.
        """
        store = self.__data_store
        args_val = store["common"] | store["val"]
        toolbox.register(
            "evaluate_val_fit", self.__get_remote(self.fitness), **args_val
        )
        toolbox.register(
            "evaluate_val_MSE", self.__get_remote(self.error_metric), **args_val
        )

    def __register_score_func(self, toolbox):
        store = self.__data_store
        args_score_func = store["common"] | store["test"]
        toolbox.register(
            "evaluate_test_score",
            self.__get_remote(self.error_metric),
            **args_score_func,
        )

    def __register_predict_func(self, toolbox):
        store = self.__data_store
        args_predict_func = store["common"] | store["test"]
        toolbox.register(
            "evaluate_test_sols",
            self.__get_remote(self.predict_func),
            **args_predict_func,
        )

    def __register_map(self, toolbox):
        toolbox_ref = ray.put(toolbox)
        toolbox.register(
            "map", mapper, toolbox_ref=toolbox_ref, batch_size=self.batch_size
        )

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, X_val=None, y_val=None):
        """Fits the training data using GP-based symbolic regression."""
        X, y = self._validate_data(X, y, accept_sparse=False)

        # config individual creator and toolbox
        toolbox, pset = self.__creator_toolbox_pset_config()

        self.__data_store = dict()

        if self.common_data is not None:
            # FIXME: does everything work when the functions do not have common args?
            self.__store_fit_error_common_args(self.common_data)

        # Initialize variables for statistics
        self.__stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        self.__stats_size = tools.Statistics(len)
        self.__mstats = tools.MultiStatistics(
            fitness=self.__stats_fit, size=self.__stats_size
        )
        self.__mstats.register("avg", lambda ind: np.around(np.mean(ind), 4))
        self.__mstats.register("std", lambda ind: np.around(np.std(ind), 4))
        self.__mstats.register("min", lambda ind: np.around(np.min(ind), 4))
        self.__mstats.register("max", lambda ind: np.around(np.max(ind), 4))

        self.__init_logbook()

        self.__train_fit_history = []

        # Create history object to build the genealogy tree
        self.__history = tools.History()

        if self.plot_best_genealogy:
            # Decorators for history
            toolbox.decorate("mate", self.__history.decorator)
            toolbox.decorate("mutate", self.__history.decorator)

        self.__register_map(toolbox)

        self.__plot_initialized = False
        self.__fig_id = 0

        train_data = {"X": X, "y": y}
        if self.validate and X_val is not None:
            val_data = {"X": X_val, "y": y_val}
            datasets = {"train": train_data, "val": val_data}
        else:
            datasets = {"train": train_data}
        self.__store_datasets(datasets)
        self.__register_fitness_func(toolbox)
        if self.validate and self.error_metric is not None:
            self.__register_val_funcs(toolbox)
        self.__run(toolbox)
        self._is_fitted = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        toolbox, pset = self.__creator_toolbox_pset_config()
        X = self._validate_data(X, accept_sparse=False, reset=False)
        test_data = {"X": X}
        datasets = {"test": test_data}
        self.__store_datasets(datasets)
        if not hasattr(self, "_predict_func_registered"):
            self.__register_predict_func(toolbox)
            self._predict_func_registered = True
        u_best = toolbox.map(toolbox.evaluate_test_sols, (self.__best,))[0]
        # u_best = toolbox.map(toolbox.evaluate_test_sols, (self.__best,))
        return u_best

    def score(self, X, y):
        """Computes the error metric (passed to the `GPSymbolicRegressor` constructor)
        on a given dataset.
        """
        check_is_fitted(self)
        toolbox, pset = self.__creator_toolbox_pset_config()
        X, y = self._validate_data(X, y, accept_sparse=False, reset=False)
        test_data = {"X": X, "y": y}
        datasets = {"test": test_data}
        self.__store_datasets(datasets)
        self.__register_score_func(toolbox)
        score = toolbox.map(toolbox.evaluate_test_score, (self.__best,))[0]
        # score = toolbox.map(toolbox.evaluate_test_score, (self.__best,))
        return score

    def __immigration(self, pop, num_immigrants: int, toolbox):
        immigrants = toolbox.population(n=num_immigrants)
        for i in range(num_immigrants):
            idx_individual_to_replace = random.randint(0, self.NINDIVIDUALS - 1)
            pop[idx_individual_to_replace] = immigrants[i]

    def __flatten_list(self, nested_lst):
        flat_list = []
        for lst in nested_lst:
            flat_list += lst
        return flat_list

    def __unflatten_list(self, flat_lst, lengths):
        result = []
        start = 0  # Starting index of the current sublist
        for length in lengths:
            # Slice the list from the current start index to start+length
            end = start + length
            result.append(flat_lst[start:end])
            start = end  # Update the start index for the next sublist
        return result

    # def __local_search(
    #     self, n_iter: int = 1, n_mutations: int = 500, n_inds_to_refine: int = 10
    # ):

    #     for i in range(self.num_islands):
    #         # select N best individuals for refinement
    #         sel_individuals = tools.selBest(self.__pop[i], k=n_inds_to_refine)

    #         # store indices of best individuals in the population
    #         idx_ind = [
    #             self.__pop[i].index(sel_individuals[j]) for j in range(n_inds_to_refine)
    #         ]

    #         # initialize best-so-far individuals and fitnesses with the
    #         # current individuals
    #         best_so_far_fits = [
    #             sel_individuals[j].fitness.values[0] for j in range(n_inds_to_refine)
    #         ]
    #         best_so_far_inds = self.__toolbox.clone(sel_individuals)

    #         for _ in range(n_iter):
    #             mutants = self.__toolbox.clone(best_so_far_inds)
    #             # generate mutations for each of the best individuals
    #             mut_ind = [
    #                 [
    #                     gp.mixedMutate(
    #                         mutants[j],
    #                         self.__toolbox.expr_mut,
    #                         self.__pset,
    #                         [0.4, 0.3, 0.3],
    #                     )[0]
    #                     for _ in range(n_mutations)
    #                 ]
    #                 for j in range(n_inds_to_refine)
    #             ]
    #             for j in range(n_inds_to_refine):
    #                 # evaluate fitnesses of mutated individuals
    #                 fitness_mutated_inds = self.__toolbox.map(
    #                     self.__toolbox.evaluate_train, mut_ind[j]
    #                 )

    #                 # assign fitnesses to mutated individuals
    #                 for ind, fit in zip(mut_ind[j], fitness_mutated_inds):
    #                     ind.fitness.values = fit

    #                 # select best mutation
    #                 best_mutation = tools.selBest(mut_ind[j], k=1)[0]

    #                 if best_mutation.fitness.values[0] < best_so_far_fits[j]:
    #                     print("Found better individual in tabu search")
    #                     best_so_far_inds[j] = best_mutation
    #                     best_so_far_fits[j] = best_mutation.fitness.values[0]

    #         # replace individuals with refined ones (if improved)
    #         for j in range(n_inds_to_refine):
    #             self.__pop[i][idx_ind[j]] = best_so_far_inds[j]

    def __evolve_islands(self, cgen: int, toolbox):
        num_evals = 0

        invalid_inds = [None] * self.num_islands
        offsprings = [None] * self.num_islands
        elite_inds = [None] * self.num_islands

        for i in range(self.num_islands):
            if self.immigration_enabled:
                if cgen % self.immigration_freq == 0:
                    self.__immigration(
                        self.__pop[i], int(self.immigration_frac * self.NINDIVIDUALS)
                    )

            # Select the parents for the offspring
            offsprings[i] = list(map(toolbox.clone, toolbox.select(self.__pop[i])))

            # Apply crossover and mutation to the offspring with elitism
            elite_inds[i] = tools.selBest(offsprings[i], self.n_elitist)
            offsprings[i] = elite_inds[i] + algorithms.varOr(
                offsprings[i],
                toolbox,
                self.NINDIVIDUALS - self.n_elitist,
                self.crossover_prob,
                self.MUTPB,
            )

            # add individuals subject to cross-over and mutation to the list of invalids
            invalid_inds[i] = [ind for ind in offsprings[i] if not ind.fitness.valid]

            num_evals += len(invalid_inds[i])

            if self.preprocess_func is not None:
                self.preprocess_func(invalid_inds[i])

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
                    self.__pop[i] + offsprings[i], self.NINDIVIDUALS
                )

        # migrations among islands
        if cgen % self.mig_frac == 0 and self.num_islands > 1:
            migRing(
                self.__pop,
                int(self.mig_frac * self.NINDIVIDUALS),
                selection=random.sample,
            )

        # self.__local_search()

        return num_evals

    def __run(self, toolbox):
        """Runs symbolic regression."""

        # Generate initial population
        print("Generating initial population(s)...", flush=True)
        self.__pop = [None] * self.num_islands
        for i in range(self.num_islands):
            self.__pop[i] = toolbox.population(n=self.NINDIVIDUALS)

        print("DONE.", flush=True)

        if self.plot_best_genealogy:
            # Populate the history and the Hall Of Fame of the first island
            self.__history.update(self.__pop[0])

        # Seeds the first island with individuals
        if self.seed is not None:
            print("Seeding population with individuals...", flush=True)
            self.__pop[0][: len(self.seed)] = self.seed

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the fitness of the entire population on the training set
        print("Evaluating initial population(s)...", flush=True)

        if self.preprocess_func is not None:
            self.preprocess_func(self.__pop)

        for i in range(self.num_islands):
            fitnesses = toolbox.map(toolbox.evaluate_train, self.__pop[i])

            if self.callback_func is not None:
                self.callback_func(self.__pop[i], fitnesses)
            else:
                for ind, fit in zip(self.__pop[i], fitnesses):
                    ind.fitness.values = fit

        if self.validate:
            print("Using validation dataset.")

        print("DONE.", flush=True)

        for gen in range(self.NGEN):
            self.__cgen = gen + 1

            num_evals = self.__evolve_islands(self.__cgen, toolbox)

            # select the best individuals in the current population
            # (including all islands)
            best_inds = tools.selBest(
                self.__flatten_list(self.__pop), k=self.num_best_inds_str
            )

            # compute and print population statistics (including all islands)
            self.__stats(self.__flatten_list(self.__pop), self.__cgen, num_evals)

            if self.print_log:
                print("Best individuals of this generation:", flush=True)
                for i in range(self.num_best_inds_str):
                    print(str(best_inds[i]))

            # Update history of best fitness and best validation error
            self.__train_fit_history = self.__logbook.chapters["fitness"].select("min")
            if self.validate:
                self.val_fit_history = self.__logbook.chapters["valid"].select(
                    "valid_fit"
                )
                self.val_fit_history = self.__logbook.chapters["valid"].select(
                    "valid_fit"
                )
                self.min_valerr = min(self.val_fit_history)

            if self.plot_history and (
                self.__cgen % self.plot_freq == 0 or self.__cgen == 1
            ):
                self.__plot_history()

            if (
                self.plot_best
                and (self.__toolbox.plot_best_func is not None)
                and (
                    self.__cgen % self.plot_freq == 0
                    or self.__cgen == 1
                    or self.__cgen == self.NGEN
                )
            ):
                toolbox.plot_best_func(best_inds[0])

            self.__best = best_inds[0]
            if self.__best.fitness.values[0] <= 1e-15:
                print("EARLY STOPPING.")
                break

        self.__plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        print(f"The best individual is {self.__best}", flush=True)
        print(f"The best fitness on the training set is {self.__train_fit_history[-1]}")

        if self.validate:
            print(f"The best fitness on the validation set is {self.min_valerr}")

        if self.plot_best_genealogy:
            self.__plot_genealogy(self.__best)

        if self.plot_best_individual_tree:
            self.__plot_best_individual_tree()

        if self.save_best_individual and self.output_path is not None:
            self.__save_best_individual(self.output_path)
            print("String of the best individual saved to disk.")

        if self.save_train_fit_history and self.output_path is not None:
            self.__save_train_fit_history(self.output_path)
            print("Training fitness history saved to disk.")

        # NOTE: ray.shutdown should be manually called by the user

    def __plot_best_individual_tree(self):
        """Plots the tree of the best individual at the end of the evolution."""
        nodes, edges, labels = gp.graph(self.__best)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        plt.figure(figsize=(7, 7))
        nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, labels)
        plt.axis("off")
        plt.show()

    def __save_best_individual(self, output_path: str):
        """Saves the string of the best individual of the population in a .txt file."""
        file = open(join(output_path, "best_ind.txt"), "w")
        file.write(str(self.__best))
        file.close()

    def __save_train_fit_history(self, output_path: str):
        np.save(join(output_path, "train_fit_history.npy"), self.__train_fit_history)
        if self.validate:
            np.save(join(output_path, "val_fit_history.npy"), self.val_fit_history)

    def save_best_test_sols(self, X_test, output_path: str):
        """Compute and save the predictions corresponding to the best individual
        at the end of the evolution, evaluated over the test dataset.

        Args:
            test_data: test dataset.
            output_path: path where the predictions should be saved (one .npy file for
                each sample in the test dataset).
        """
        best_test_sols = self.predict(X_test)

        for i, sol in enumerate(best_test_sols):
            np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)

        print("Best individual solution evaluated over the test set saved to disk.")
