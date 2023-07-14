from app.utils import cost_function, random_init_sol, dummy_init_sol
from app.load import load_problem
from time import time
from app.ALNS import ALNS
from app.Operators import (
    one_week_swap,
    multi_week_swap,
    one_game_flip,
    multi_game_flip,
    set_week_for_game,
)
import numpy as np
import pandas as pd
from app.model import create_model
from itertools import combinations, product
from app.named_tuples import Solution
import os

if __name__ == "__main__":
    # import required module

    # assign directory
    directory = "Instances"

    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        problem = load_problem(f)
        model = create_model(problem)
        os.makedirs("lp_models", exist_ok=True)
        model.write(f"lp_models\\{filename[:-4]}.lp")
    # solution = Solution(
    #     problem=problem,
    #     representative=dummy_init_sol(problem)
    # random_init_sol(
    #   problem, rng=np.random.default_rng(3)
    # ),  # pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy(),
    # )
    # initial_sol = pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy()
    # init_cost = solution.total_cost
    # operators = [
    #     set_week_for_game
    # ]  # [one_week_swap, multi_week_swap, one_game_flip, multi_game_flip]
    # operators_len = len(operators)
    # probabilities = [[1 / operators_len for _ in operators]]
    # repeat = 1
    # repeat_range = range(repeat)
    # for prb in probabilities:
    #     start = time()
    #     best_sol, last_improvement, weights, feas_sols = (
    #         [solution.copy() for _ in repeat_range],
    #         [0 for _ in repeat_range],
    #         [[] for _ in repeat_range],
    #         [[] for _ in repeat_range],
    #     )
    #     for i in repeat_range:
    #         rng = np.random.default_rng(31 + i)
    #         (
    #             best_sol[i],
    #             last_improvement[i],
    #             weights[i],
    #             feas_sols[i],
    #         ) = ALNS(
    #             solution,
    #             prb,
    #             operators,
    #             [0, 1, 2, 3],
    #             problem,
    #             rng,
    #         )
    #     running_time = (time() - start) / repeat
    #     best_cost = [sol.total_cost for sol in best_sol]
    #     minidx = np.argmin(best_cost)
    #     print(
    #         prb,
    #         "\t",
    #         np.mean(best_cost),
    #         "\t",
    #         best_cost[minidx],
    #         "\t",
    #         100 * ((init_cost - best_cost[minidx]) / init_cost),
    #         "\t",
    #         running_time,
    #     )
    #     print("Solution: ", best_sol[minidx].representative)
