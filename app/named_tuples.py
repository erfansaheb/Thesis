# from collections import namedtuple

import numpy as np


class Solution:
    representative: np.array
    total_cost: int = 0
    teams_cost: np.array
    games_cost: np.array
    slots_cost: np.array

    def __init__(self, representative, total_cost, teams_cost, games_cost, slots_cost):
        self.representative = representative
        self.total_cost = total_cost
        self.teams_cost = teams_cost
        self.games_cost = games_cost
        self.slots_cost = slots_cost


# Solution = namedtuple(
#     "Solution",
#     ["representative", "total_cost", "teams_cost", "games_cost", "slots_cost"],
# )
# Problem = namedtuple("Problem", ["n_teams", "n_slots", "gameMode"])
# Constraints = namedtuple("Constraints"])
# CA1 = namedtuple("CA1", ["min", "max", "teams", "mode", "slots", "penalty", "type"])
# CA2 = namedtuple(
#     "CA2",
#     ["min", "max", "teams1", "teams2", "mode1", "mode2", "slots", "penalty", "type"],
# )
# CA3 = namedtuple(
#     "CA3",
#     [
#         "min",
#         "max",
#         "teams1",
#         "teams2",
#         "mode1",
#         "mode2",
#         "slots",
#         "intp",
#         "penalty",
#         "type",
#     ],
# )
# CA4 = namedtuple(
#     "CA4",
#     [
#         "min",
#         "max",
#         "teams1",
#         "teams2",
#         "mode1",
#         "mode2",
#         "slots",
#         "intp",
#         "penalty",
#         "type",
#     ],
# )
