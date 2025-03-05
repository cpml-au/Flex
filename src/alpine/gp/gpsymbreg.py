from deap import algorithms, tools, gp, base, creator
from deap.tools import migRing
import matplotlib.pyplot as plt
import numpy as np
import operator
from typing import List, Dict, Callable
from os.path import join
import networkx as nx
from .primitives import add_primitives_to_pset
from alpine.data import Dataset
import os
import ray
import random
from itertools import chain
from importlib import import_module

# reducing the number of threads launched by fitness evaluations
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
)


class GPSymbolicRegressor:
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
        pset: gp.PrimitiveSet | gp.PrimitiveSetTyped,
        fitness: Callable,
        select_fun: str = "tools.selection.tournament_with_elitism",
        select_args: str = "{'num_elitist': self.n_elitist, 'tournsize': 3, 'stochastic_tourn': { 'enabled': False, 'prob': [0.8, 0.2] }}",
        mut_fun: str = "gp.mutUniform",
        mut_args: str = "{'expr': self.toolbox.expr_mut, 'pset': self.pset}",
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

        self.pset = pset

        self.fitness = fitness
        self.error_metric = error_metric
        self.predict_func = predict_func

        self.data_store = dict()

        self.plot_best = plot_best

        self.plot_best_genealogy = plot_best_genealogy

        self.plot_history = plot_history
        self.print_log = print_log
        self.num_best_inds_str = num_best_inds_str
        self.plot_freq = plot_freq
        self.preprocess_func = preprocess_func
        self.callback_fun = callback_func
        self.is_plot_best_individual_tree = plot_best_individual_tree
        self.is_save_best_individual = save_best_individual
        self.is_save_train_fit_history = save_train_fit_history
        self.output_path = output_path
        self.batch_size = batch_size

        if common_data is not None:
            # FIXME: does everything work when the functions do not have common args?
            self.store_fit_error_common_args(common_data)

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

        # Elitism settings
        self.n_elitist = int(frac_elitist * self.NINDIVIDUALS)

        # config individual creator and toolbox
        self.__creator_toolbox_config()
        # self.createIndividual = individualCreator

        # self.toolbox = toolbox

        self.seed = seed

        if self.seed is not None:
            self.seed = [self.createIndividual.from_string(i, pset) for i in seed]

        # Initialize variables for statistics
        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(
            fitness=self.stats_fit, size=self.stats_size
        )
        self.mstats.register("avg", lambda ind: np.around(np.mean(ind), 4))
        self.mstats.register("std", lambda ind: np.around(np.std(ind), 4))
        self.mstats.register("min", lambda ind: np.around(np.min(ind), 4))
        self.mstats.register("max", lambda ind: np.around(np.max(ind), 4))

        self.__init_logbook()

        self.train_fit_history = []

        # Create history object to build the genealogy tree
        self.history = tools.History()

        if self.plot_best_genealogy:
            # Decorators for history
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)

        self.__register_map()

        self.plot_initialized = False
        self.fig_id = 0

    def __creator_toolbox_config(self):
        """Initialize toolbox and individual creator based on config file."""
        self.toolbox = base.Toolbox()

        # SELECTION
        self.toolbox.register("select", eval(self.select_fun), **eval(self.select_args))

        # MUTATION
        self.toolbox.register(
            "expr_mut", eval(self.expr_mut_fun), **eval(self.expr_mut_args)
        )

        self.toolbox.register("mutate", eval(self.mut_fun), **eval(self.mut_args))

        # CROSSOVER
        self.toolbox.register(
            "mate", eval(self.crossover_fun), **eval(self.crossover_args)
        )
        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )

        # INDIVIDUAL GENERATOR/CREATOR
        self.toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.min_height,
            max_=self.max_height,
        )
        self.toolbox.register(
            "expr_pop",
            gp.genHalfAndHalf,
            pset=self.pset,
            min_=self.min_height,
            max_=self.max_height,
            is_pop=True,
        )
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        createIndividual = creator.Individual
        self.toolbox.register(
            "individual", tools.initIterate, createIndividual, self.toolbox.expr
        )

        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.createIndividual = createIndividual

    def store_fit_error_common_args(self, data: Dict):
        """Store names and values of the arguments that are in common between
        the fitness and the error metric functions in the common object space.

        Args:
            data: dictionary containing arguments names and values.
        """
        self.__store_shared_objects("common", data)

    def store_datasets(self, datasets: Dict[str, Dataset]):
        """Store datasets with the corresponding label ("train", "val" or "test")
        in the common object space. The datasets are passed as parameters to
        the fitness, and possibly to the error metric and the prediction functions.

        Args:
            datasets: the keys are 'train', 'val' and 'test' denoting the training,
                the validation and the test datasets, respectively. The associated
                values are `Dataset` objects.
        """
        for dataset_label in datasets.keys():
            dataset_name_data = {datasets[dataset_label].name: datasets[dataset_label]}
            self.__store_shared_objects(dataset_label, dataset_name_data)

    def __store_shared_objects(self, label: str, data: Dict):
        for key, value in data.items():
            data[key] = ray.put(value)
        self.data_store[label] = data

    def __init_logbook(self):
        # Initialize logbook to collect statistics
        self.logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if self.validate:
            self.logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.logbook.chapters["valid"].header = "valid_fit", "valid_err"
        else:
            self.logbook.header = "gen", "evals", "fitness", "size"
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

    def __compute_valid_stats(self, pop):
        best = tools.selBest(pop, k=1)
        # FIXME: ugly way of handling lists/tuples; assume eval_val_MSE returns a
        # single-valued tuple as eval_val_fit
        valid_fit = self.toolbox.map(self.toolbox.evaluate_val_fit, best)[0][0]
        valid_err = self.toolbox.map(self.toolbox.evaluate_val_MSE, best)[0]

        return valid_fit, valid_err

    def __stats(self, pop, gen, evals):
        """Compute and print statistics of a population."""

        # LINE_UP = '\033[1A'
        # LINE_CLEAR = '\x1b[2K'
        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # record the statistics in the logbook
        if self.validate:
            # compute satistics related to the validation set
            valid_fit, valid_err = self.__compute_valid_stats(pop)
            record["valid"] = {"valid_fit": valid_fit, "valid_err": valid_err}

        self.logbook.record(gen=gen, evals=evals, **record)

        if self.print_log:
            # Print statistics for the current population
            # print(LINE_UP, end=LINE_CLEAR, flush=True)
            print(self.logbook.stream, flush=True)

    def __plot_history(self):
        """Plots the fitness of the best individual vs generation number."""
        if not self.plot_initialized:
            self.plot_initialized = True
            # new figure number when starting with new evolution
            self.fig_id = self.fig_id + 1
            plt.figure(self.fig_id).show()
            plt.pause(0.01)

        plt.figure(self.fig_id)
        fig = plt.gcf()

        # Array of generations starts from 1
        x = range(1, len(self.train_fit_history) + 1)
        plt.plot(x, self.train_fit_history, "b", label="Training Fitness")
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

        gen_best = self.history.getGenealogy(best)
        graph = networkx.DiGraph(gen_best)
        graph = graph.reverse()
        pos = networkx.nx_agraph.graphviz_layout(
            graph, prog="dot", args="-Gsplines=True"
        )
        # Retrieve individual strings for graph node labels
        labels = gen_best.copy()
        for key in labels.keys():
            labels[key] = str(self.history.genealogy_history[key])
        plt.figure()
        networkx.draw_networkx(graph, pos=pos)
        label_options = {"ec": "k", "fc": "lightblue", "alpha": 1.0}
        networkx.draw_networkx_labels(
            graph, pos=pos, labels=labels, font_size=10, bbox=label_options
        )

        # Save genealogy to file
        # networkx.nx_agraph.write_dot(graph, "genealogy.dot")

    def __register_fitness_func(self):
        store = self.data_store
        args_train = store["common"] | store["train"]
        self.toolbox.register("evaluate_train", self.fitness, **args_train)

    def __register_val_funcs(self):
        """Register the functions needed for validation, i.e. the error metric and the
        fitness function. Must be called after storing the datasets in the common
        obj space.
        """
        store = self.data_store
        args_val = store["common"] | store["val"]
        self.toolbox.register("evaluate_val_fit", self.fitness, **args_val)
        self.toolbox.register("evaluate_val_MSE", self.error_metric, **args_val)

    def __register_score_func(self):
        store = self.data_store
        args_score_func = store["common"] | store["test"]
        self.toolbox.register(
            "evaluate_test_score", self.error_metric, **args_score_func
        )

    def __register_predict_func(self):
        store = self.data_store
        args_predict_func = store["common"] | store["test"]
        self.toolbox.register(
            "evaluate_test_sols", self.predict_func, **args_predict_func
        )

    def __register_map(self):
        def mapper(f, individuals, toolbox_ref):
            fitnesses = [] * len(individuals)
            for i in range(0, len(individuals), self.batch_size):
                individuals_batch = individuals[i : i + self.batch_size]
                fitnesses.append(f(individuals_batch, toolbox_ref))
            fitnesses = list(chain(*ray.get(fitnesses)))
            return fitnesses

        toolbox_ref = ray.put(self.toolbox)
        self.toolbox.register("map", mapper, toolbox_ref=toolbox_ref)

    def fit(self, train_data: Dataset, val_data: Dataset | None = None):
        """Fits the training data using GP-based symbolic regression."""
        if self.validate and val_data is not None:
            datasets = {"train": train_data, "val": val_data}
        else:
            datasets = {"train": train_data}
        self.store_datasets(datasets)
        self.__register_fitness_func()
        if self.validate and self.error_metric is not None:
            self.__register_val_funcs()
        self.__run()

    def predict(self, test_data: Dataset):
        datasets = {"test": test_data}
        self.store_datasets(datasets)
        self.__register_predict_func()
        u_best = self.toolbox.map(self.toolbox.evaluate_test_sols, (self.best,))[0]
        return u_best

    def score(self, test_data: Dataset):
        """Computes the error metric (passed to the `GPSymbolicRegressor` constructor)
        on a given dataset.
        """
        datasets = {"test": test_data}
        self.store_datasets(datasets)
        self.__register_score_func()
        score = self.toolbox.map(self.toolbox.evaluate_test_score, (self.best,))[0]
        return score

    def immigration(self, pop, num_immigrants: int):
        immigrants = self.toolbox.population(n=num_immigrants)
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

    def __local_search(
        self, n_iter: int = 1, n_mutations: int = 500, n_inds_to_refine: int = 10
    ):

        for i in range(self.num_islands):
            # select N best individuals for refinement
            sel_individuals = tools.selBest(self.pop[i], k=n_inds_to_refine)

            # store indices of best individuals in the population
            idx_ind = [
                self.pop[i].index(sel_individuals[j]) for j in range(n_inds_to_refine)
            ]

            # initialize best-so-far individuals and fitnesses with the
            # current individuals
            best_so_far_fits = [
                sel_individuals[j].fitness.values[0] for j in range(n_inds_to_refine)
            ]
            best_so_far_inds = self.toolbox.clone(sel_individuals)

            for _ in range(n_iter):
                mutants = self.toolbox.clone(best_so_far_inds)
                # generate mutations for each of the best individuals
                mut_ind = [
                    [
                        gp.mixedMutate(
                            mutants[j],
                            self.toolbox.expr_mut,
                            self.pset,
                            [0.4, 0.3, 0.3],
                        )[0]
                        for _ in range(n_mutations)
                    ]
                    for j in range(n_inds_to_refine)
                ]
                for j in range(n_inds_to_refine):
                    # evaluate fitnesses of mutated individuals
                    fitness_mutated_inds = self.toolbox.map(
                        self.toolbox.evaluate_train, mut_ind[j]
                    )

                    # assign fitnesses to mutated individuals
                    for ind, fit in zip(mut_ind[j], fitness_mutated_inds):
                        ind.fitness.values = fit

                    # select best mutation
                    best_mutation = tools.selBest(mut_ind[j], k=1)[0]

                    if best_mutation.fitness.values[0] < best_so_far_fits[j]:
                        print("Found better individual in tabu search")
                        best_so_far_inds[j] = best_mutation
                        best_so_far_fits[j] = best_mutation.fitness.values[0]

            # replace individuals with refined ones (if improved)
            for j in range(n_inds_to_refine):
                self.pop[i][idx_ind[j]] = best_so_far_inds[j]

    def __evolve_islands(self, cgen: int):
        num_evals = 0

        invalid_inds = [None] * self.num_islands
        offsprings = [None] * self.num_islands
        elite_inds = [None] * self.num_islands

        for i in range(self.num_islands):
            if self.immigration_enabled:
                if cgen % self.immigration_freq == 0:
                    self.immigration(
                        self.pop[i], int(self.immigration_frac * self.NINDIVIDUALS)
                    )

            # Select the parents for the offspring
            offsprings[i] = list(
                map(self.toolbox.clone, self.toolbox.select(self.pop[i]))
            )

            # Apply crossover and mutation to the offspring with elitism
            elite_inds[i] = tools.selBest(offsprings[i], self.n_elitist)
            offsprings[i] = elite_inds[i] + algorithms.varOr(
                offsprings[i],
                self.toolbox,
                self.NINDIVIDUALS - self.n_elitist,
                self.crossover_prob,
                self.MUTPB,
            )

            # add individuals subject to cross-over and mutation to the list of invalids
            invalid_inds[i] = [ind for ind in offsprings[i] if not ind.fitness.valid]

            num_evals += len(invalid_inds[i])

            if self.preprocess_func is not None:
                self.preprocess_func(invalid_inds[i])

        fitnesses = self.toolbox.map(
            self.toolbox.evaluate_train, self.__flatten_list(invalid_inds)
        )
        fitnesses = self.__unflatten_list(fitnesses, [len(i) for i in invalid_inds])

        for i in range(self.num_islands):
            if self.callback_fun is not None:
                self.callback_fun(invalid_inds[i], fitnesses[i])
            else:
                for ind, fit in zip(invalid_inds[i], fitnesses[i]):
                    ind.fitness.values = fit

            # survival selection
            if not self.overlapping_generation:
                # The population is entirely replaced by the offspring
                self.pop[i][:] = offsprings[i]
            else:
                # parents and offspring compete for survival (truncation selection)
                self.pop[i] = tools.selBest(
                    self.pop[i] + offsprings[i], self.NINDIVIDUALS
                )

        # migrations among islands
        if cgen % self.mig_frac == 0 and self.num_islands > 1:
            migRing(
                self.pop,
                int(self.mig_frac * self.NINDIVIDUALS),
                selection=random.sample,
            )

        # self.__local_search()

        return num_evals

    def __run(self):
        """Runs symbolic regression."""

        # Generate initial population
        print("Generating initial population(s)...", flush=True)
        self.pop = [None] * self.num_islands
        for i in range(self.num_islands):
            self.pop[i] = self.toolbox.population(n=self.NINDIVIDUALS)

        print("DONE.", flush=True)

        if self.plot_best_genealogy:
            # Populate the history and the Hall Of Fame of the first island
            self.history.update(self.pop[0])

        # Seeds the first island with individuals
        if self.seed is not None:
            print("Seeding population with individuals...", flush=True)
            self.pop[0][: len(self.seed)] = self.seed

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the fitness of the entire population on the training set
        print("Evaluating initial population(s)...", flush=True)

        if self.preprocess_func is not None:
            self.preprocess_func(self.pop)

        for i in range(self.num_islands):
            fitnesses = self.toolbox.map(self.toolbox.evaluate_train, self.pop[i])

            if self.callback_fun is not None:
                self.callback_fun(self.pop[i], fitnesses)
            else:
                for ind, fit in zip(self.pop[i], fitnesses):
                    ind.fitness.values = fit

        if self.validate:
            print("Using validation dataset.")

        print("DONE.", flush=True)

        for gen in range(self.NGEN):
            self.cgen = gen + 1

            num_evals = self.__evolve_islands(self.cgen)

            # select the best individuals in the current population
            # (including all islands)
            best_inds = tools.selBest(
                self.__flatten_list(self.pop), k=self.num_best_inds_str
            )

            # compute and print population statistics (including all islands)
            self.__stats(self.__flatten_list(self.pop), self.cgen, num_evals)

            if self.print_log:
                print("Best individuals of this generation:", flush=True)
                for i in range(self.num_best_inds_str):
                    print(str(best_inds[i]))

            # Update history of best fitness and best validation error
            self.train_fit_history = self.logbook.chapters["fitness"].select("min")
            if self.validate:
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit"
                )
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit"
                )
                self.min_valerr = min(self.val_fit_history)

            if self.plot_history and (
                self.cgen % self.plot_freq == 0 or self.cgen == 1
            ):
                self.__plot_history()

            if (
                self.plot_best
                and (self.toolbox.plot_best_func is not None)
                and (
                    self.cgen % self.plot_freq == 0
                    or self.cgen == 1
                    or self.cgen == self.NGEN
                )
            ):
                self.toolbox.plot_best_func(best_inds[0])

            self.best = best_inds[0]
            if self.best.fitness.values[0] <= 1e-15:
                print("EARLY STOPPING.")
                break

        self.plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        print(f"The best individual is {self.best}", flush=True)
        print(f"The best fitness on the training set is {self.train_fit_history[-1]}")

        if self.validate:
            print(f"The best fitness on the validation set is {self.min_valerr}")

        if self.plot_best_genealogy:
            self.__plot_genealogy(self.best)

        if self.is_plot_best_individual_tree:
            self.plot_best_individual_tree()

        if self.is_save_best_individual and self.output_path is not None:
            self.save_best_individual(self.output_path)
            print("String of the best individual saved to disk.")

        if self.is_save_train_fit_history and self.output_path is not None:
            self.save_train_fit_history(self.output_path)
            print("Training fitness history saved to disk.")

        # NOTE: ray.shutdown should be manually called by the user

    def plot_best_individual_tree(self):
        """Plots the tree of the best individual at the end of the evolution."""
        nodes, edges, labels = gp.graph(self.best)
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

    def save_best_individual(self, output_path: str):
        """Saves the string of the best individual of the population in a .txt file."""
        file = open(join(output_path, "best_ind.txt"), "w")
        file.write(str(self.best))
        file.close()

    def save_train_fit_history(self, output_path: str):
        np.save(join(output_path, "train_fit_history.npy"), self.train_fit_history)
        if self.validate:
            np.save(join(output_path, "val_fit_history.npy"), self.val_fit_history)

    def save_best_test_sols(self, test_data: Dataset, output_path: str):
        """Compute and save the predictions corresponding to the best individual
        at the end of the evolution, evaluated over the test dataset.

        Args:
            test_data: test dataset.
            output_path: path where the predictions should be saved (one .npy file for
                each sample in the test dataset).
        """
        best_test_sols = self.predict(test_data)

        for i, sol in enumerate(best_test_sols):
            np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)

        print("Best individual solution evaluated over the test set saved to disk.")
