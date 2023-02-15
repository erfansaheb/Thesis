from itertools import combinations
import numpy as np


def cost_function_dummy(Solution, problem):
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
                check_zero = 1
                for team in c["teams"]:
                    # p = sum(
                    #     np.isin(np.array(c["slots"][check_zero:]), Solution[team, :])
                    #     == np.isin(
                    #         np.array(c["slots"][check_zero:]) - 1, Solution[team, :]
                    #     )
                    # )
                    p = 0
                    obj += compute_penalty(c, p, "intp")
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
                        obj += (p - c["intp"]) * c["penalty"]
            elif i == 1:
                p = 0
                for team in c["teams"]:
                    for slot in c["slots"]:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot - 1).any()
                        p += cur == prev
                if p > c["intp"]:
                    obj += (p - c["intp"]) * c["penalty"]
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
            obj += np.sum(diff) * c["penalty"]
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:  # SE1 constraints
            for team1, team2 in c["teams"]:
                first = Solution[team1, team2]
                second = Solution[team2, team1]
                diff = abs(second - first) - 1
                if diff < c["min"]:
                    obj += (c["min"] - diff) * c["penalty"]
    return obj


def check_games_in_slots(problem, obj, c, slots):
    for s in range(c["intp"], problem["n_slots"] + 1):
        p = np.sum(np.logical_and((slots < s), (slots >= s - c["intp"])))
        obj += compute_penalty(c, p)
    return obj


def compute_penalty(c, p, extremum="max"):
    if extremum in ["max", "intp"]:
        return max([p - c[extremum], 0]) * c["penalty"]
    elif extremum == "min":
        return max([c[extremum] - p, 0]) * c["penalty"]
