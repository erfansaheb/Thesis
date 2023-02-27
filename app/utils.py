from itertools import combinations
import numpy as np


def cost_function(Solution, problem):
    ca, ga, ba, fa, sa = problem["obj_constr"].values()
    obj = 0
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        for c in cc:
            if i == 0:  # CA1 constraints
                p = np.zeros(len(c["teams"]), dtype=int)
                for slot in c["slots"]:
                    p += (
                        np.sum(Solution[c["teams"], :] == slot, axis=1)
                        if c["mode"] == "H"
                        else np.sum(Solution[:, c["teams"]] == slot, axis=0)
                    )
                obj += max([(p - c["max"]).sum(), 0]) * c["penalty"]
            elif i == 1:  # CA2 constraints
                for team1 in c["teams1"]:
                    if c["mode1"] == "HA":
                        p = np.sum(
                            np.isin(Solution[team1, c["teams2"]], c["slots"])
                        ) + np.sum(np.isin(Solution[c["teams2"], team1], c["slots"]))
                    elif c["mode1"] == "H":
                        p = np.sum(np.isin(Solution[team1, c["teams2"]], c["slots"]))
                    else:
                        p = np.sum(np.isin(Solution[c["teams2"], team1], c["slots"]))
                    obj += compute_penalty(c, p)
            elif i == 2:  # CA3 constraints
                for team in c["teams1"]:
                    slots_H = Solution[team, c["teams2"]]
                    slots_A = Solution[c["teams2"], team]
                    slots = np.concatenate([slots_A, slots_H])
                    if c["mode1"] == "HA":
                        obj = check_games_in_slots(problem, obj, c, slots)
                    elif c["mode1"] == "H":
                        obj = check_games_in_slots(problem, obj, c, slots_H)
                    else:
                        obj = check_games_in_slots(problem, obj, c, slots_A)
            else:  # CA4 constraints
                slots_H = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                slots_A = Solution[np.ix_(c["teams2"], c["teams1"])].flatten()
                slots = np.concatenate([slots_A, slots_H])
                if c["mode2"] == "GLOBAL":
                    if c["mode1"] == "HA":
                        p = np.sum(np.isin(slots, c["slots"]))
                    elif c["mode1"] == "H":
                        p = np.sum(np.isin(slots_H, c["slots"]))
                    else:
                        p = np.sum(np.isin(slots_A, c["slots"]))
                    obj += compute_penalty(c, p)
                else:
                    for slot in c["slots"]:
                        if c["mode1"] == "HA":
                            p = np.sum(slots == slot)
                        elif c["mode1"] == "H":
                            p = np.sum(slots_H == slot)
                        else:
                            p = np.sum(slots_A == slot)
                        obj += compute_penalty(c, p)
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:  # GA1 constraints
            p = sum(
                np.sum(Solution[meeting] == c["slots"]) for meeting in c["meetings"]
            )
            obj += compute_penalty(c, p, "min")
            obj += compute_penalty(c, p, "max")
    for i, bc in enumerate(ba):
        if len(bc) == 0:
            continue
        for c in bc:
            if i == 0:
                check_zero = 1 if c["slots"][0] == 0 else 0
                for team in c["teams"]:
                    p = sum(
                        np.isin(np.array(c["slots"][check_zero:]), Solution[team, :])
                        == np.isin(
                            np.array(c["slots"][check_zero:]) - 1, Solution[team, :]
                        )
                    )
                    obj += compute_penalty(c, p, "intp")
            elif i == 1:
                check_zero = 1 if c["slots"][0] == 0 else 0
                p = sum(
                    sum(
                        np.isin(
                            np.array(c["slots"][check_zero:]),
                            Solution[team, :],
                        )
                        == np.isin(
                            np.array(c["slots"][check_zero:]) - 1,
                            Solution[team, :],
                        )
                    )
                    for team in c["teams"]
                )
                obj += compute_penalty(c, p, "intp")
    for i, fc in enumerate(fa):
        if len(fc) == 0:
            continue
        for c in fc:  # FA1 constraints
            diff = np.zeros([len(c["teams"]), len(c["teams"])], dtype=int)
            for s in c["slots"]:
                home_count = (Solution[c["teams"], :] <= s).sum(
                    axis=1
                ) - 1  # excluding the column = team
                for i, j in combinations(c["teams"], 2):
                    diff[i, j] = max(abs(home_count[i] - home_count[j]), diff[i, j])
            diff -= c["intp"]
            diff[diff < 0] = 0
            obj += np.sum(diff) * c["penalty"]
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:  # SE1 constraints
            for team1, team2 in c["teams"]:
                first = Solution[team1, team2]
                second = Solution[team2, team1]
                diff = abs(second - first) - 1
                obj += compute_penalty(c, diff, "min")
    return obj


def feasibility_check(Solution, problem):
    ca, ga, ba, fa, sa = problem["feas_constr"].values()
    status, feasibility = compatibility_check(Solution)
    if not feasibility:
        return (status, feasibility)
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        if i == 0:  # CA1 constraints
            for c in cc:
                for team in c["teams"]:
                    p = 0
                    if c["mode"] == "H":
                        p += np.sum(
                            np.transpose(Solution[team : team + 1, :]) == c["slots"]
                        )
                    else:
                        p += np.sum(Solution[:, team : team + 1] == c["slots"])
                    if p > c["max"]:
                        feasibility = False
                        status = "Team {} has {} more {} games than max= {} during time slots {}:\t {}".format(
                            team, p - c["max"], c["mode"], c["max"], c["slots"], p
                        )
                        return status, feasibility
        elif i == 1:  # CA2 constraints
            for c in cc:
                if c["mode1"] == "HA":
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team1, team2] == c["slots"])
                            p += np.sum(Solution[team2, team1] == c["slots"])
                        if p > c["max"]:
                            feasibility = False
                            status = "Team {} has {} more {} games than max= {} during time slots {} against teams {}:\t {}".format(
                                team1,
                                p - c["max"],
                                c["mode1"],
                                c["max"],
                                c["slots"],
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
                elif c["mode1"] == "H":
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team1, team2] == c["slots"])
                        if p > c["max"]:
                            feasibility = False
                            status = "Team {} has {} more {} games than max= {} during time slots {} against teams {}:\t {}".format(
                                team1,
                                p - c["max"],
                                c["mode1"],
                                c["max"],
                                c["slots"],
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
                else:
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team2, team1] == c["slots"])
                        if p > c["max"]:
                            feasibility = False
                            status = "Team {} has {} more {} games than max= {} during time slots {} against teams {}:\t {}".format(
                                team1,
                                p - c["max"],
                                c["mode1"],
                                c["max"],
                                c["slots"],
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
        elif i == 2:  # CA3 constraints
            for c in cc:
                if c["mode1"] == "HA":
                    for team1 in c["teams1"]:
                        slots_H = Solution[team, c["teams2"]]
                        slots_A = Solution[c["teams2"], team]
                        slots = np.concatenate([slots_A, slots_H])
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and((slots < s), (slots >= s - c["intp"]))
                            )
                            if p > c["max"]:
                                feasibility = False
                                status = "Team {} has {} more games than max= {} during time slots {} against teams {}:\t {}".format(
                                    team1,
                                    p - c["max"],
                                    c["max"],
                                    list(range(c["intp"], problem["n_slots"] + 1)),
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
                elif c["mode1"] == "H":
                    for team1 in c["teams1"]:
                        slots_H = Solution[team, c["teams2"]]
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and(
                                    (slots_H < s), (slots_H >= s - c["intp"])
                                )
                            )
                            if p > c["max"]:
                                feasibility = False
                                status = "Team {} has {} more home games than max= {} during time slots {} against teams {}:\t {}".format(
                                    team1,
                                    p - c["max"],
                                    c["max"],
                                    list(range(c["intp"], problem["n_slots"] + 1)),
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
                else:
                    for team1 in c["teams1"]:
                        slots_A = Solution[c["teams2"], team]
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and(
                                    (slots_A < s), (slots_A >= s - c["intp"])
                                )
                            )
                            if p > c["max"]:
                                feasibility = False
                                status = "Team {} has {} more away games than max= {} during time slots {} against teams {}:\t {}".format(
                                    team1,
                                    p - c["max"],
                                    c["max"],
                                    list(range(c["intp"], problem["n_slots"] + 1)),
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
        else:  # CA4 constraints
            for c in cc:
                if c["mode1"] == "HA":
                    if c["mode2"] == "GLOBAL":
                        p = 0
                        slots_H = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        slots_A = Solution[np.ix_(c["teams2"], c["teams1"])].flatten()
                        slots = np.concatenate([slots_A, slots_H])
                        for slot in c["slots"]:
                            p += np.sum(slots == slot)
                        if p > c["max"]:
                            feasibility = False
                            status = "Teams {} has {} more games than max= {} during time slots {} against teams {}:\t {}".format(
                                c["teams1"],
                                p - c["max"],
                                c["max"],
                                c["slots"],
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                feasibility = False
                                status = "Teams {} has {} more games than max= {} at time slot {} against teams {}:\t {}".format(
                                    c["teams1"],
                                    p - c["max"],
                                    c["max"],
                                    c["slots"],
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
                elif c["mode1"] == "H":
                    if c["mode2"] == "GLOBAL":
                        p = 0
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p += np.sum(slots == slot)
                        if p > c["max"]:
                            feasibility = False
                            status = "Teams {} has {} more home games than max= {} during time slots {} against teams {}:\t {}".format(
                                c["teams1"],
                                p - c["max"],
                                c["max"],
                                slot,
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                feasibility = False
                                status = "Teams {} has {} more home games than max= {} at time slot {} against teams {}:\t {}".format(
                                    c["teams1"],
                                    p - c["max"],
                                    c["max"],
                                    c["slots"],
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
                else:
                    if c["mode2"] == "GLOBAL":
                        p = 0
                        slots = Solution[np.ix_(c["teams2"], c["teams1"])].flatten()
                        for slot in c["slots"]:
                            p += np.sum(slots == slot)
                        if p > c["max"]:
                            feasibility = False
                            status = "Teams {} has {} more away games than max= {} during time slots {} against teams {}:\t {}".format(
                                c["teams1"],
                                p - c["max"],
                                c["max"],
                                c["slots"],
                                c["teams2"],
                                p,
                            )
                            return status, feasibility
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                feasibility = False
                                status = "Teams {} has {} more away games than max= {} at time slot {} against teams {}:\t {}".format(
                                    c["teams1"],
                                    p - c["max"],
                                    c["max"],
                                    c["slots"],
                                    c["teams2"],
                                    p,
                                )
                                return status, feasibility
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:  # GA1 constraints
            p = 0
            for meeting in c["meetings"]:
                p += np.sum(Solution[tuple(meeting)] == c["slots"])
            if p < c["min"]:
                feasibility = False
                status = f"Less than min {c['min']} games from {c['meetings']} took place during time slots{c['slots']}:\t {p}"
                return status, feasibility
            elif p > c["max"]:
                feasibility = False
                status = "More/less than max/min {c['max']} games from {c['meetings']} took place during time slots{c['slots']}:\t {p}"
                return status, feasibility
    for i, bc in enumerate(ba):
        if len(bc) == 0:
            continue
        if i == 0:  # BR1 constraints
            for c in bc:
                for team in c["teams"]:
                    p = 0
                    for slot in c["slots"]:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot - 1).any()
                        if c["mode2"] == "HA":
                            p += cur == prev
                        elif c["mode2"] == "H":
                            p += cur == prev and cur
                        else:
                            p += cur == prev and not cur
                    if p > c["intp"]:
                        feasibility = False
                        status = "Team {} has {} more {} breaks than max= {} during time slots {} :\t {}".format(
                            team, p - c["intp"], c["mode2"], c["intp"], c["slots"], p
                        )
                        return status, feasibility
        elif i == 1:  # BR2 constraints
            for c in bc:
                p = 0
                for team in c["teams"]:
                    for slot in c["slots"]:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot - 1).any()
                        p += cur == prev
                if p > c["intp"]:
                    feasibility = False
                    status = "Teams {} has {} more breaks than max= {} during time slots {} :\t {}".format(
                        c["teams"], p - c["intp"], c["intp"], c["slots"], p
                    )
                    return status, feasibility
    for i, fc in enumerate(fa):
        if len(fc) == 0:
            continue
        for c in fc:  # FA1 constraints
            diff = np.zeros([len(c["teams"]), len(c["teams"])], dtype=int)
            for s in c["slots"]:
                p = 0
                home_count = np.zeros_like(c["teams"])
                for team in c["teams"]:
                    home_count[team] = (
                        np.sum(Solution[team, :] <= s) - 1
                    )  # excluding the column = team
                for i, j in combinations(c["teams"], 2):
                    diff[i, j] = max(abs(home_count[i] - home_count[j]), diff[i, j])
                    # if diff[i,j] > c['intp']:
                    #     p += (diff - c['intp'])
            diff -= c["intp"]
            diff[diff < 0] = 0
            if np.sum(diff) > 0:
                feasibility = False
                status = "The difference in home games played between {} is larger than {} during time slots {} :\t {}".format(
                    c["teams"], c["intp"], c["slots"], p
                )
                return status, feasibility
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:  # SE1 constraints
            for team1, team2 in c["teams"]:
                first = Solution[team1, team2]
                second = Solution[team2, team1]
                diff = abs(second - first) - 1
                if diff < c["min"]:
                    feasibility = False
                    status = "Team {} and team {} has {} less time slots between their mutual games than min= {}:\t {}".format(
                        team1, team2, c["min"] - diff, c["min"], diff
                    )
                    return status, feasibility
    return status, feasibility


def compatibility_check(Solution: np.array) -> bool:
    for i in range(len(Solution)):
        weeks, counts = np.unique(
            np.concatenate([Solution[i, :], Solution[:, i]]), return_counts=True
        )
        dummy_week = True if weeks[-1] == (len(Solution) - 1) * 2 else False
        if dummy_week:
            if (counts[1:-1] > 1).any():
                return (
                    "Incompatible solution, at least one team has more than one game on the same time slot",
                    False,
                )
        elif (counts[1:] > 1).any():
            return (
                "Incompatible solution, at least one team has more than one game on the same time slot",
                False,
            )
    return "Compatible", True


def random_init_sol(sol, problem, rng):
    num_teams = problem["n_teams"]
    teams_list = list(rng.permutation(np.arange(num_teams)))
    if (len(teams_list) % 2) != 0:
        teams_list.append(0)
    x = teams_list[0 : int(len(teams_list) / 2)]
    y = teams_list[int(len(teams_list) / 2) : len(teams_list)]
    matches = []
    for i in range(len(teams_list) - 1):
        rounds = {}
        if i != 0:
            x.insert(1, y.pop(0))
            y.append(x.pop())
        matches.append(rounds)
        for j in range(len(x)):
            sol[x[j], y[j]] = i
            rounds[x[j]] = y[j]
    teams_list = list(rng.permutation(np.arange(num_teams)))
    if (len(teams_list) % 2) != 0:
        teams_list.append(0)
    x = teams_list[0 : int(len(teams_list) / 2)]
    y = teams_list[int(len(teams_list) / 2) : len(teams_list)]
    # matches = []
    for i in range(len(teams_list) - 1):
        if i != 0:
            x.insert(1, y.pop(0))
            y.append(x.pop())
        # matches.append(rounds)
        for j in range(len(x)):
            if sol[x[j], y[j]] == -1:
                sol[x[j], y[j]] = i + len(teams_list) - 1
            else:
                sol[y[j], x[j]] = i + len(teams_list) - 1
    return sol


def compute_penalty(c, p, extremum="max"):
    if extremum in ["max", "intp"]:
        return max([p - c[extremum], 0]) * c["penalty"]
    elif extremum == "min":
        return max([c[extremum] - p, 0]) * c["penalty"]


def check_games_in_slots(problem, obj, c, slots):
    for s in range(c["intp"], problem["n_slots"] + 1):
        p = np.sum(np.logical_and((slots < s), (slots >= s - c["intp"])))
        obj += compute_penalty(c, p)
    return obj
