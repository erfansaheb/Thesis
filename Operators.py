from copy import deepcopy
import numpy as np


def one_week_swap(
    solution: np.array, rng: np.random.Generator, problem: dict
) -> np.array:
    """This function takes a solution, randomly selects two weeks and
    swaps all the games in those two weeks

    Args:
        solution (np.array): solution to be changed
        rng (np.random.Generator): random generator (seeded!)
        problem (dict): dictionary related to problem

    Returns:
        np.array: solution with changed weeks
    """
    sol = deepcopy(solution)
    weeks = rng.choice(np.arange(0, problem["n_slots"]), size=2, replace=False)
    sol[solution == weeks[0]] = weeks[1]
    sol[solution == weeks[1]] = weeks[0]
    return sol


def multi_week_swap(
    solution: np.array, rng: np.random.Generator, problem: dict
) -> np.array:
    """This function takes a solution
    - randomly selects the number of weeks to be swapped: r ,
    - randomly selects 2*r weeks
    - swaps the games of each couple of weeks with eachother

    Args:
        solution (np.array): solution to be changed
        rng (np.random.Generator): random generator (seeded!)
        problem (dict):  dictionary related to problem

    Returns:
        np.array: solution with changed weeks
    """
    sol = deepcopy(solution)
    rm_size = rng.integers(2, 4)
    weeks = rng.choice(
        np.arange(0, problem["n_slots"]), size=2 * rm_size, replace=False
    )
    for r in range(rm_size):
        sol[solution == weeks[2 * r]] = weeks[2 * r + 1]
        sol[solution == weeks[2 * r + 1]] = weeks[2 * r]
    return sol


# home away flip
# one
def one_game_flip(
    solution: np.array, rng: np.random.Generator, problem: dict
) -> np.array:
    """This function takes a solution, randomly selects two teams and flips the home-away games week

    Args:
        solution (np.array): solution to be changed
        rng (np.random.Generator): random generator (seeded!)
        problem (dict):  dictionary related to problem

    Returns:
        np.array: solution with flipped weeks
    """
    sol = deepcopy(solution)
    teams = rng.choice(np.arange(0, problem["n_teams"]), size=2, replace=False)
    sol[tuple(teams)] = solution[teams[1], teams[0]]
    sol[teams[1], teams[0]] = solution[tuple(teams)]
    return sol


# multi
def multi_game_flip(
    solution: np.array, rng: np.random.Generator, problem: dict
) -> np.array:
    """This function takes a solution
    - randomly selects the number of games to be flipped: r ,
    - randomly selects 2*r teams
    - flips the home-away games of each couple of teams

    Args:
        solution (np.array): solution to be changed
        rng (np.random.Generator): random generator (seeded!)
        problem (dict):  dictionary related to problem

    Returns:
        np.array: solution with flipped weeks
    """
    sol = deepcopy(solution)
    rm_size = rng.integers(2, 4)
    teams = rng.choice(
        np.arange(0, problem["n_teams"]), size=2 * rm_size, replace=False
    )
    for r in range(rm_size):
        sol[tuple(teams[2 * r : 2 * (r + 1)])] = solution[
            teams[2 * r + 1], teams[2 * r]
        ]
        sol[teams[2 * r + 1], teams[2 * r]] = solution[
            tuple(teams[2 * r : 2 * (r + 1)])
        ]
    return sol
