from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from helpers.conway_rules import conway_steps


class InitStrategy(ABC):
    @abstractmethod
    def generate(self, population_size):
        pass


class Random(InitStrategy):
    def generate(self, population_size):
        return np.random.binomial(1, 0.5, (population_size, 25, 25))


class RandomWithWarmup(InitStrategy):

    def __init__(self, delta):
        self.delta = delta

    def generate(self, population_size):
        start_population = np.random.binomial(1, 0.5, (population_size, 25, 25))
        population = list()
        for board in start_population:
            end_board = conway_steps(board, self.delta)
            population.append(end_board)
        return population


class BoundedSpace(InitStrategy):

    def __init__(self, seed_board, mutation_prob=0.01):
        self.seed_board = seed_board
        self.mutation_prob = mutation_prob

    def generate(self, population_size):
        population = list()
        for i in range(population_size):
            seed_board = self.seed_board.copy()
            mask = np.random.binomial(1, self.mutation_prob, size=(25, 25)).astype('bool')
            seed_board[mask] += 1
            seed_board[mask] %= 2
            population.append(seed_board)
        return population
