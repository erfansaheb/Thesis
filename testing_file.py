from itertools import permutations
from app.load import (
    load_problem,
    load_solution,
)
from app.utils import (
    feasibility_check,
    cost_function,
    # compatibility_check,
    random_init_sol,
    cost_function_games,
)

# from app.utils_2 import cost_function_dummy
from time import time

# from itertools import product
# from time import time
# from ALNS import ALNS
# from Operators import one_week_swap, multi_week_swap, one_game_flip, multi_game_flip
import numpy as np

import pandas as pd

from app.named_tuples import Solution

problem = load_problem("Instances//ITC2021_Early_2.xml")
solution = Solution(
    problem=problem,
    representative=pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy(),
    total_cost=0,
    hard_cost=0,
    soft_cost=0,
    obj_fun=0,
)
# compatibility_check(initial_sol)
# solution.representative, total_cost = load_solution(
#     "Final_Solutions//E2.xml",
#     solution.representative.copy(),
# )

# solution.representative = random_init_sol(problem, rng=np.random.default_rng(31))
start = time()
print(feasibility_check(solution, problem, "slots", 0), time() - start)
start = time()
print(feasibility_check(solution, problem), time() - start)

cost = cost_function(solution, problem)
print(cost)
# best_sol, best_objective_value = load_solution(
#     "best_sol.xml", solution.representative.copy()
# )
# pd.DataFrame(solution).to_csv("their_sol.csv")
# for week in range(30):
#     home_w1 = np.where(initial_sol == week)[0]
#     home_w1_best = np.where(best_sol == week)[0]
#     print(home_w1, home_w1_best, np.intersect1d(home_w1, home_w1_best))
