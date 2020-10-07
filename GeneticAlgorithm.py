from helpers.conway_rules import conway_steps
import numpy as np
import multiprocessing as mp
from initialization_strategies.strategies import InitStrategy, RandomWithWarmup, Random, BoundedSpace
from selection_strategies.strategies import Selection
from crossover_strategies.strategies import Crossover
from functools import partial


class GeneticAlgorithm:
    def __init__(self, population_size, max_gen, initialization_strategy: InitStrategy,
                 selection_strategy: Selection, crossover_strategy: Crossover,
                 parents_ratio=0.8, mutation_probability=0.1, elitism_ratio=0.01,
                 fitness_parallel=False, random_state=-1):

        self.population_size = population_size
        self.max_gen = max_gen
        self.initialization_strategy = initialization_strategy
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.parents_ratio = parents_ratio
        self.mutation_probability = mutation_probability
        self.elitism_ratio = elitism_ratio
        self.fitness_parallel = fitness_parallel
        if fitness_parallel:
            self.pool = mp.Pool(mp.cpu_count())
        else:
            self.pool = None

        self._population = None
        self._retain_len = int(population_size * self.parents_ratio)
        self._num_elites = int(np.ceil(elitism_ratio*population_size))

        if random_state != -1:
            np.random.seed(random_state)

    def run(self, stop_board, delta):
        # Initialize population
        self._population = self.initialization_strategy.generate(self.population_size)
        fitness_values = np.zeros(len(self._population))

        for generation in range(self.max_gen):
            self._population, scores = self.evolve(stop_board, delta)
            if generation % 100 == 0:
                print("Best scores:", scores[:self._retain_len])

        return self._population[0], fitness_values[0]

    def evolve(self, stop_board, delta):
        fitness_values = self._fitness_population(stop_board, delta)
        sorted_indices = np.argsort(fitness_values)[::-1]
        self._population = [self._population[idx] for idx in sorted_indices]

        best_fitnesses = fitness_values[sorted_indices][:self._retain_len]

        # SELECTION
        parents = self.selection_strategy.select(self._population, self._retain_len, fitness_values)

        # MUTATION
        mutated_parents = []
        for gene in parents[self._num_elites:]:
            if np.random.rand() < self.mutation_probability:
                mutated_parents.append(self.mutate(gene))
            else:
                mutated_parents.append(gene)
        mutated_parents.extend(parents[:self._num_elites])

        # CROSSOVER

        places_left = self.population_size - self._retain_len
        children = []
        while len(children) < places_left:
            mom_idx, dad_idx = np.random.randint(0, self._retain_len - 1, 2)
            if mom_idx != dad_idx:
                child1, child2 = self.crossover_strategy.crossover(parents[mom_idx], parents[dad_idx])
                children.append(child1)
                if len(children) < places_left:
                    children.append(child2)
        mutated_parents.extend(children)
        return mutated_parents, best_fitnesses

    @staticmethod
    def _fitness(start_board, stop_board, delta):
        candidate = conway_steps(start_board, delta)
        return np.sum(candidate == stop_board) / 625.

    def _fitness_population(self, stop_board, delta):
        if self.fitness_parallel:
            return np.array(self.pool.map(partial(self._fitness, stop_board=stop_board, delta=delta), self._population))
        else:
            return np.array([self._fitness(x, stop_board, delta) for x in self._population])

    def mutate(self, gene):
        mask = np.random.binomial(1, self.mutation_probability, size=(25, 25)).astype('bool')
        gene[mask] += 1
        gene[mask] %= 2
        return gene




