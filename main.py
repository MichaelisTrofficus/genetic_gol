from crossover_strategies.strategies import UniformCrossover
from initialization_strategies.strategies import Random, RandomWithWarmup, BoundedSpace
from selection_strategies.strategies import ProportionateSelection, BestFitnessSelection
from GeneticAlgorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from helpers.conway_rules import conway_steps
from tqdm import tqdm


data_path = "./data/"
delta1 = pd.read_csv(data_path + "delta1.csv")
result = delta1.copy()

experiments = delta1.values

for index, experiment in enumerate(tqdm(experiments)):
    delta = experiment[1]
    stop_board = np.reshape(experiment[2: 2+625], (25, 25))
    init_strategy = BoundedSpace(seed_board=conway_steps(stop_board, delta))
    # init_strategy = RandomWithWarmup(delta=5)
    selection_strategy = BestFitnessSelection(leftovers_probability=0.05)
    crossover_strategy = UniformCrossover()

    ga = GeneticAlgorithm(population_size=20,
                      max_gen=300,
                      initialization_strategy=init_strategy,
                      selection_strategy=selection_strategy,
                      crossover_strategy=crossover_strategy,
                      parents_ratio=0.8,
                      mutation_probability=0.1,
                      elitism_ratio=0.01)

    gen_start_board, fitness_value = ga.run(stop_board, delta)
    print("Fitness Value of Solution:", fitness_value)
    result.loc[index, 2:] = gen_start_board.reshape((-1, 625))
    result.to_csv("./data/result_delta1.csv")

