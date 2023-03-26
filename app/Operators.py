from copy import deepcopy
import numpy as np

from app.named_tuples import Solution
from app.utils import (
    cost_function_games,
    feasibility_check,
    compatibility_check,
    update_week_availability,
)


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
    rep = solution.representative
    sol = deepcopy(rep)
    weeks = rng.choice(np.arange(0, problem["n_slots"]), size=2, replace=False)
    sol[rep == weeks[0]] = weeks[1]
    sol[rep == weeks[1]] = weeks[0]
    return new_sol(rep=sol, prob=problem)


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
    rep = solution.representative
    sol = deepcopy(rep)
    rm_size = rng.integers(2, 4)
    weeks = rng.choice(
        np.arange(0, problem["n_slots"]), size=2 * rm_size, replace=False
    )
    for r in range(rm_size):
        sol[rep == weeks[2 * r]] = weeks[2 * r + 1]
        sol[rep == weeks[2 * r + 1]] = weeks[2 * r]
    return new_sol(rep=sol, prob=problem)


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
    rep = solution.representative
    sol = deepcopy(rep)
    teams = rng.choice(np.arange(0, problem["n_teams"]), size=2, replace=False)
    sol[tuple(teams)] = rep[teams[1], teams[0]]
    sol[teams[1], teams[0]] = rep[tuple(teams)]
    return new_sol(rep=sol, prob=problem)


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
    rep = solution.representative
    sol = deepcopy(rep)
    rm_size = rng.integers(2, 4)
    teams = rng.choice(
        np.arange(0, problem["n_teams"]), size=2 * rm_size, replace=False
    )
    for r in range(rm_size):
        sol[tuple(teams[2 * r : 2 * (r + 1)])] = rep[teams[2 * r + 1], teams[2 * r]]
        sol[teams[2 * r + 1], teams[2 * r]] = rep[tuple(teams[2 * r : 2 * (r + 1)])]
    return new_sol(rep=sol, prob=problem)


def set_week_for_game(
    solution: Solution,
    rng: np.random.Generator,
    problem: dict,
):
    # select a game from dummy week -> the most expensive one?
    # find a week for it -> use week_availability
    # if found:
    #   put the game on that week
    # if not: -> no available week for both teams
    #   go for the next game in dummy
    #       or
    #   find the most expensive games for teams and put it in dummy until an option for the first game is available
    dummies = np.where(solution.representative == problem["n_slots"])
    choice = rng.choice(len(dummies[0]))
    team1, team2 = (dummies[0][choice], dummies[1][choice])
    avail1 = np.where(solution.week_availability[team1] == 1)
    avail2 = np.where(solution.week_availability[team2] == 1)
    options = np.intersect1d(avail1, avail2)
    if options.size:
        best_cost = np.inf
        for option in options:
            rep = solution.representative.copy()
            rep[team1, team2] = option
            _, feasibility = feasibility_check(
                rep, problem, const_level="teams", index=team1
            )
            if not feasibility:
                continue
            _, feasibility = feasibility_check(
                rep, problem, const_level="teams", index=team2
            )
            if not feasibility:
                continue
            game_cost = cost_function_games(rep, problem, (team1, team2))
            new_cost = (
                solution.total_cost + game_cost - problem["dummy_costs"][team1, team2]
            )
            if new_cost < best_cost:
                best_cost = new_cost
                best_rep = rep.copy()
        if best_cost < np.inf:
            selected_week = best_rep[team1, team2]
            update_week_availability(
                solution,
                week_num=selected_week,
                team1=team1,
                team2=team2,
                method="remove",
            )
            return new_sol(
                rep=best_rep, prob=problem, week_availability=solution.week_availability
            )
    return solution


def set_week_for_all_game(
    solution: Solution,
    rng: np.random.Generator,
    problem: dict,
):
    # select a game from dummy week -> the most expensive one?
    # find a week for it -> use week_availability
    # if found:
    #   put the game on that week
    # if not: -> no available week for both teams
    #   go for the next game in dummy
    #       or
    #   find the most expensive games for teams and put it in dummy until an option for the first game is available
    dummies = np.where(solution.representative == problem["n_slots"])
    for team1, team2 in zip(*dummies):
        # team1, team2 = (dummies[0][choice], dummies[1][choice])
        avail1 = np.where(solution.week_availability[team1] == 1)
        avail2 = np.where(solution.week_availability[team2] == 1)
        options = np.intersect1d(avail1, avail2)
        if options.size:
            best_cost = np.inf
            for option in options:
                rep = solution.representative.copy()
                rep[team1, team2] = option
                _, feasibility = feasibility_check(
                    rep, problem, const_level="teams", index=team1
                )
                if not feasibility:
                    continue
                _, feasibility = feasibility_check(
                    rep, problem, const_level="teams", index=team2
                )
                if not feasibility:
                    continue
                game_cost = cost_function_games(rep, problem, (team1, team2))
                new_cost = (
                    solution.total_cost
                    + game_cost
                    - problem["dummy_costs"][team1, team2]
                )
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_rep = rep.copy()
            if best_cost < np.inf:
                selected_week = best_rep[team1, team2]
                update_week_availability(
                    solution,
                    week_num=selected_week,
                    team1=team1,
                    team2=team2,
                    method="remove",
                )
                return new_sol(
                    rep=best_rep,
                    prob=problem,
                    week_availability=solution.week_availability,
                )
    return solution


def new_sol(rep, prob, week_availability=None):
    # update rep
    # update costs: total, games, slots, teams, hard, soft
    # update availabilites: game, week
    return Solution(
        problem=prob, representative=rep, week_availability=week_availability
    )
