from GeneticAlgorithm import GeneticAlgorithm
import multiprocessing as mp
import random


def work(solver, target, delta):
    # this is required for every worker to have different initial seed. Otherwise they inherit it from this thread
    random.seed()
    return solver.run(target, delta)


class ParalellGeneticAlgorithm:
    def __init__(self, n_proc='auto', *args, **kwargs):
        """
        Multi-process version of Genetic Solver with different initial conditions
        :param n_proc: number of processes to create
        :param args: GeneticSolver arguments (see its documentation for more)
        :param kwargs: GeneticSolver key-value arguments
        """
        if n_proc == 'auto':
            n_proc = mp.cpu_count()
        self.n_proc = n_proc
        self.pool = mp.Pool(mp.cpu_count() if n_proc == 'auto' else n_proc)
        self.args = args
        self.kwargs = kwargs
        self._solvers = None
        if 'fitness_parallel' in self.args or ('fitness_parallel' in self.kwargs and self.kwargs['fitness_parallel']):
            raise ValueError("Fitness function cannot be parallelized in MPGeneticSolver")

    def run(self, stop_board, delta, return_all=True):
        """
        Solve RGoL problem
        :param stop_board: 20x20 array that represents field in stopping condition
        :param delta: number of steps to revert
        :param return_all: if True, returns all of the results from different runners, as well as their scores.
                           If False only solution associated with the best score is returned
        :return: either list of (solution, score) pairs or the best solution (see `return_all`)
        """
        self._solvers = [GeneticAlgorithm(*self.args, **self.kwargs) for _ in range(self.n_proc)]
        tasks = [(solver, stop_board, delta) for solver in self._solvers]
        results = self.pool.starmap(work, tasks)
        return results if return_all else self.select_best(results)

    @classmethod
    def select_best(cls, solutions):
        """
        Using output of solve method, select the best solution
        :param solutions: list of (solution, score) pairs
        :return: 20x20 array that represents the solution (starting board condition)
        """
        return sorted(solutions, key=lambda x: x[1], reverse=True)[0]


if __name__ == '__main__':
    print("Registered number of cores: ", mp.cpu_count())
