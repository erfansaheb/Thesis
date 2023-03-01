from app.utils import (
    cost_function,
    random_init_sol,
)
from app.load import load_problem
from time import time
from app.ALNS import ALNS
from app.Operators import one_week_swap, multi_week_swap, one_game_flip, multi_game_flip
import numpy as np
import pandas as pd

from app.named_tuples import Solution

if __name__ == "__main__":
    problem = load_problem("Instances//ITC2021_Early_2.xml")
    solution = Solution(
        problem=problem,
        representative=random_init_sol(
            problem, rng=np.random.default_rng(3)
        ),  # pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy(),
    )
    # initial_sol = pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy()
    init_cost = solution.total_cost
    operators = [one_week_swap, multi_week_swap, one_game_flip, multi_game_flip]
    operators_len = len(operators)
    probabilities = [[1 / operators_len for _ in operators]]
    repeat = 1
    repeat_range = range(repeat)
    for prb in probabilities:
        start = time()
        best_sol, last_improvement, weights, feas_sols = (
            [solution.copy() for _ in repeat_range],
            [0 for _ in repeat_range],
            [[] for _ in repeat_range],
            [[] for _ in repeat_range],
        )
        for i in repeat_range:
            rng = np.random.default_rng(31 + i)
            (best_sol[i], last_improvement[i], weights[i], feas_sols[i],) = ALNS(
                solution,
                prb,
                operators,
                [0, 1, 2, 3],
                problem,
                rng,
            )
        running_time = (time() - start) / repeat
        best_cost = [sol.total_cost for sol in best_sol]
        minidx = np.argmin(best_cost)
        print(
            prb,
            "\t",
            np.mean(best_cost),
            "\t",
            best_cost[minidx],
            "\t",
            100 * ((init_cost - best_cost[minidx]) / init_cost),
            "\t",
            running_time,
        )
        print("Solution: ", best_sol[minidx])
