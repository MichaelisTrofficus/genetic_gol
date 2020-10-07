from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Crossover(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        pass


class UniformCrossover(Crossover):

    def crossover(self, parent1, parent2):
        select_mask = np.random.binomial(1, 0.5, size=(25, 25)).astype('bool')
        child1, child2 = np.copy(parent1), np.copy(parent2)
        child1[select_mask] = parent2[select_mask]
        child2[select_mask] = parent1[select_mask]
        return child1, child2
