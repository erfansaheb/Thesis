# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:14:45 2022

@author: erfan
"""

import numpy as np


def circle_method(num_teams, rng):
    # num_teams = 20  # input('\nEnter number of teams: - ')
    # teams_list = []
    teams_list = list(rng.permutation(np.arange(0, num_teams)))
    # for i in range(int(num_teams)):
    #     teams_list.append(i+1)
    if (len(teams_list) % 2) != 0:
        teams_list.append(0)
        # team_list = np.append(teams_list,0)
    x = teams_list[: len(teams_list) // 2]
    y = teams_list[len(teams_list) // 2 :]
    Vars = []
    for i in range(len(teams_list) - 1):
        # rounds = {}
        if i != 0:
            x.insert(1, y.pop(0))
            y.append(x.pop())
        # matches.append(rounds)
        Vars.extend({"VarName": f"x[{x[j]},{y[j]},{i}]", "X": 1} for j in range(len(x)))
        Vars.extend(
            {"VarName": f"x[{y[j]},{x[j]},{i+len(teams_list) - 1}]", "X": 1}
            for j in range(len(x))
        )
        # rounds[x[j]] = y[j]
    # for i in range(len(matches)):
    #     print("\n Day", i + 1, " : ", matches[i])
    return Vars


# print(circle_method(5))
