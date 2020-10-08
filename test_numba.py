from numba import jit, njit
import numpy as np
from helpers.plots import visualize_boards


def generate(population_size):
    return np.random.binomial(1, 0.5, (population_size, 25, 25))


@jit(nopython=True)
def fitness(start_board, stop_board, delta):
    candidate = conway_steps(start_board, delta)
    return np.sum(candidate == stop_board) / 625.


def fitness_population(population, stop_board, delta):
    return np.array([fitness(x, stop_board, delta) for x in population])


@jit(nopython=True)
def fitness_population_numba(population, stop_board, delta):
    return np.array([fitness(x, stop_board, delta) for x in population])


@njit
def fitness_numba(population, stop_board, delta):

    gen_stop_boards = []
    for start_board in population:
        gen_stop_board = start_board.copy()
        for i in range(delta):
            for idx, row in enumerate(gen_stop_board):
                for idy, column in enumerate(row):

                    idx_plus = (idx + 1 if idx != 24 else 0)
                    idx_minus = (idx - 1 if idx != 0 else 24)
                    idy_plus = (idy + 1 if idy != 24 else 0)
                    idy_minus = (idy - 1 if idy != 0 else 24)

                    neighbours_indices = [[idx, idy_minus], [idx, idy_plus], [idx_minus, idy], [idx_plus, idy],
                                          [idx_minus, idy_minus], [idx_minus, idy_plus], [idx_plus, idy_minus],
                                          [idx_plus, idy_plus]]
                    neighbours = np.array([gen_stop_board[x[0], x[1]] for x in neighbours_indices])
                    live_neighbours = np.sum(neighbours)

                    if column:
                        # Cell is alive
                        if live_neighbours < 2:
                            new_value = 0
                        elif live_neighbours > 3:
                            new_value = 0
                        else:
                            new_value = 1

                    else:
                        # Cell is dead
                        if live_neighbours == 3:
                            new_value = 1
                        else:
                            new_value = 0

                    gen_stop_board[idx, idy] = new_value
            gen_stop_board = gen_stop_board.copy()
        gen_stop_boards.append(gen_stop_board)

    return np.array([np.sum(x == stop_board) / 625. for x in gen_stop_boards])


stop_board = generate(1)[0]
population = generate(100)
delta = 1

fitness_values = fitness_numba(population, stop_board, delta)
fitness_values = fitness_numba(population, stop_board, delta)
fitness_values = fitness_numba(population, stop_board, delta)
