from app.load import (
    load_problem,
    load_solution,
)
from app.utils import (
    # feasibility_check,
    cost_function,
    # compatibility_check,
)
from app.utils_2 import cost_function_dummy

# from itertools import product
# from time import time
# from ALNS import ALNS
# from Operators import one_week_swap, multi_week_swap, one_game_flip, multi_game_flip
import numpy as np

# import pandas as pd

from app.named_tuples import Solution

problem = load_problem("Instances//EarlyInstances_V3//ITC2021_Early_12.xml")
solution = Solution(
    representative=np.ones((problem["n_teams"], problem["n_teams"]), dtype=int) * (-1),
    total_cost=0,
    games_cost=np.zeros((problem["n_teams"], problem["n_teams"])),
    teams_cost=np.zeros(problem["n_teams"]),
    slots_cost=np.zeros(problem["n_slots"]),
)

# initial_sol = pd.read_csv("Erfan_init_sol.csv", header=None).to_numpy()
# compatibility_check(initial_sol)
# sol = np.ones((problem["n_teams"], problem["n_teams"]), dtype=int) * (-1)
solution.representative, solution.total_cost = load_solution(
    "..//Appendix_Files//Final_Solutions//Early_instances//E12.xml",
    solution.representative.copy(),
)
print(cost_function(solution.representative, problem))
print(cost_function_dummy(solution.representative, problem))
best_sol, best_objective_value = load_solution(
    "best_sol.xml", solution.representative.copy()
)
# pd.DataFrame(solution).to_csv("their_sol.csv")
# for week in range(30):
#     home_w1 = np.where(initial_sol == week)[0]
#     home_w1_best = np.where(best_sol == week)[0]
#     print(home_w1, home_w1_best, np.intersect1d(home_w1, home_w1_best))
