from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Selection(ABC):
    @abstractmethod
    def select(self, population, retain_len, fitness_values=None):
        pass


class BestFitnessSelection(Selection):

    def __init__(self, leftovers_probability):
        self.leftovers_probability = leftovers_probability

    def select(self, population, retain_len, fitness_values=None):
        parents = population[:retain_len]
        leftovers = population[retain_len:]

        for gene in leftovers:
            if np.random.rand() < self.leftovers_probability:
                parents.append(gene)
        return parents


class ProportionateSelection(Selection):

    def select(self, population, retain_len, fitness_values=None):
        parents = []
        cum_fitness = np.cumsum(fitness_values / sum(fitness_values))
        for i in range(retain_len):
            random_try = np.random.rand()
            for index, j in enumerate(cum_fitness):
                if random_try < j:
                    parents.append(population[index])
                    break
        return parents
