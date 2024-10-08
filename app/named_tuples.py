# from collections import namedtuple

from itertools import product
import numpy as np
from copy import deepcopy
from app.utils import cost_function


class Solution:
    representative: np.array
    total_cost: int = 0
    teams_cost: np.array
    games_cost: np.array
    slots_cost: np.array
    hard_cost: int = 0
    soft_cost: int = 0
    obj_fun: int = 0
    problem: dict
    week_availability: np.array
    game_availability: np.array

    def __init__(
        self,
        problem,
        representative=None,
        total_cost=0,
        hard_cost=0,
        soft_cost=0,
        obj_fun=None,
        week_availability=None,
    ):
        self.problem = problem
        self.representative = representative
        self.total_cost = total_cost
        self.teams_cost = np.zeros(problem["n_teams"])
        self.games_cost = np.zeros((problem["n_teams"], problem["n_teams"]))
        self.slots_cost = np.zeros(problem["n_slots"] + 1)
        self.hard_cost = hard_cost
        self.soft_cost = soft_cost
        self.obj_fun = obj_fun or cost_function(self, problem)
        self.week_availability = (
            week_availability
            if week_availability is not None
            else np.ones((problem["n_teams"], problem["n_slots"]), dtype=int)
        )
        self.game_availability = np.ones(
            (problem["n_teams"], problem["n_teams"]), dtype=int
        )

    def copy(self):
        return deepcopy(self)


class Constraint:
    all: list
    teams: dict[int, list]
    slots: dict[int, list]
    games: dict[tuple[int, int], list]
    types: dict = {"ca": 4, "ba": 2, "ga": 1, "fa": 1, "sa": 1}

    def __init__(self, n_teams, n_slots):
        self.all = {
            type: [[] for _ in range(n_types)] for type, n_types in self.types.items()
        }
        self.teams = {
            i: {
                type: [[] for _ in range(n_types)]
                for type, n_types in self.types.items()
            }
            for i in range(n_teams)
        }

        self.slots = {
            i: {
                type: [[] for _ in range(n_types)]
                for type, n_types in self.types.items()
            }
            for i in range(n_slots)
        }

        self.games = {
            (i, j): {
                type: [[] for _ in range(n_types)]
                for type, n_types in self.types.items()
            }
            for i, j in product(range(n_teams), range(n_teams))
            if i != j
        }


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
