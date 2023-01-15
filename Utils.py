import xmltodict
from itertools import combinations
import numpy as np


def load_problem(file):
    # load file
    data_dict = load_xml_file(file)
    n_teams = len(data_dict["Instance"]["Resources"]["Teams"]["team"])
    n_slots = (n_teams - 1) * 2
    gameMode = data_dict["Instance"]["Structure"]["Format"]["gameMode"]
    # team_feas_const = {str(i):[] for i in range(n_teams)}
    ca_hard, ca_soft = [[] for _ in range(4)], [[] for _ in range(4)]
    if ca := data_dict["Instance"]["Constraints"]["CapacityConstraints"]:
        cas = ["CA1", "CA2", "CA3", "CA4"]
        for i, c in enumerate(cas):
            if c not in ca.keys():
                continue
            elif type(ca[c]) == list:
                cc = ca[c]
            else:
                cc = [ca[c]]
            for num in cc:
                const = {
                    "max": int(num["@max"]),
                    "min": int(num["@min"]),
                    "penalty": int(num["@penalty"]),
                }
                if "@teams" in num.keys():
                    const["teams"] = [int(x) for x in num["@teams"].split(";")]
                else:
                    const["teams1"] = [int(x) for x in num["@teams1"].split(";")]
                    const["teams2"] = [int(x) for x in num["@teams2"].split(";")]
                if "@mode" in num.keys():
                    const["mode"] = num["@mode"]
                else:
                    const["mode1"] = num["@mode1"]
                    const["mode2"] = num["@mode2"]
                if "@slots" in num.keys():
                    const["slots"] = [int(x) for x in num["@slots"].split(";")]
                if "@intp" in num.keys():
                    const["intp"] = int(num["@intp"])
                if num["@type"] == "HARD":
                    ca_hard[i].append(const)
                else:
                    ca_soft[i].append(const)

    ga_hard, ga_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    if ga := data_dict["Instance"]["Constraints"]["GameConstraints"]:
        gas = ["GA1"]
        for i, g in enumerate(gas):
            if g not in ga.keys():
                continue
            elif type(ga[g]) == list:
                ga1 = ga[g]
            else:
                ga1 = [ga[g]]
            for num in ga1:
                const = {
                    "meetings": [
                        [int(y), int(z)]
                        for y, z in (
                            x.split(",") for x in num["@meetings"].split(";") if x
                        )
                    ],
                    "slots": [int(x) for x in num["@slots"].split(";")],
                    "max": int(num["@max"]),
                    "min": int(num["@min"]),
                    "penalty": int(num["@penalty"]),
                }
                if num["@type"] == "HARD":
                    ga_hard[i].append(const)
                else:
                    ga_soft[i].append(const)
    ba_hard, ba_soft = [[] for _ in range(2)], [[] for _ in range(2)]
    if ba := data_dict["Instance"]["Constraints"]["BreakConstraints"]:
        bas = ["BR1", "BR2"]
        for i, b in enumerate(bas):
            if b not in ba.keys():
                continue
            elif type(ba[b]) == list:
                bc = ba[b]
            else:
                bc = [ba[b]]
            for num in bc:
                const = {
                    "teams": [int(x) for x in num["@teams"].split(";")],
                    "slots": [int(x) for x in num["@slots"].split(";")],
                    "intp": int(num["@intp"]),
                    "mode2": num["@mode2"],
                    "penalty": int(num["@penalty"]),
                }
                if "@mode1" in num.keys():
                    const["mode1"] = num["@mode1"]
                elif "@homeMode" in num.keys():
                    const["homeMode"] = num["@homeMode"]
                if num["@type"] == "HARD":
                    ba_hard[i].append(const)
                else:
                    ba_soft[i].append(const)
    fa_hard, fa_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    if fa := data_dict["Instance"]["Constraints"]["FairnessConstraints"]:
        fas = ["FA2"]
        for i, f in enumerate(fas):
            if f not in fa.keys():
                continue
            elif type(fa[f]) == list:
                fc = fa[f]
            else:
                fc = [fa[f]]
            for num in fc:
                const = {
                    "teams": [int(x) for x in num["@teams"].split(";")],
                    "slots": [int(x) for x in num["@slots"].split(";")],
                    "intp": int(num["@intp"]),
                    "mode": num["@mode"],
                    "penalty": int(num["@penalty"]),
                }
                if num["@type"] == "HARD":
                    fa_hard[i].append(const)
                else:
                    fa_soft[i].append(const)
    sa_hard, sa_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    if sa := data_dict["Instance"]["Constraints"]["SeparationConstraints"]:
        sas = ["SE1"]
        for i, s in enumerate(sas):
            if s not in sa.keys():
                continue
            elif type(sa[s]) == list:
                sc = sa[s]
            else:
                sc = [sa[s]]
            for num in sc:
                const = {
                    "teams": list(
                        combinations([int(x) for x in num["@teams"].split(";")], 2)
                    ),
                    "min": int(num["@min"]),
                    "mode1": num["@mode1"],
                    "penalty": int(num["@penalty"]),
                }
                if num["@type"] == "HARD":
                    sa_hard[i].append(const)
                else:
                    sa_soft[i].append(const)
    return {
        "feas_constr": {
            "ca": ca_hard,
            "ga": ga_hard,
            "ba": ba_hard,
            "fa": fa_hard,
            "sa": sa_hard,
        },
        "obj_constr": {
            "ca": ca_soft,
            "ga": ga_soft,
            "ba": ba_soft,
            "fa": fa_soft,
            "sa": sa_soft,
        },
        "n_teams": n_teams,
        "n_slots": n_slots,
        "gameMode": gameMode,
    }


def load_xml_file(file: str) -> dict:
    """Loads the xml file for initializing the problem

    Args:
        file (str): address to the file

    Returns:
        dict: output of xml to dict
    """
    with open(file, "r") as f:
        data_dict = xmltodict.parse(f.read())
        f.close()
    return data_dict


def cost_function(Solution, problem):
    ca, ga, ba, fa, sa = problem["obj_constr"].values()
    obj = 0
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
                        obj += (p - c["max"]) * c["penalty"]
        elif i == 1:  # CA2 constraints
            for c in cc:
                if c["mode1"] == "HA":
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team1, team2] == c["slots"])
                            p += np.sum(Solution[team2, team1] == c["slots"])
                        if p > c["max"]:
                            obj += (p - c["max"]) * c["penalty"]
                elif c["mode1"] == "H":
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team1, team2] == c["slots"])
                        if p > c["max"]:
                            obj += (p - c["max"]) * c["penalty"]
                else:
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(Solution[team2, team1] == c["slots"])
                        if p > c["max"]:
                            obj += (p - c["max"]) * c["penalty"]
        elif i == 2:  # CA3 constraints
            for c in cc:
                if c["mode1"] == "HA":
                    for team in c["teams1"]:
                        slots_H = Solution[team, c["teams2"]]
                        slots_A = Solution[c["teams2"], team]
                        slots = np.concatenate([slots_A, slots_H])
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and((slots < s), (slots >= s - c["intp"]))
                            )
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
                elif c["mode1"] == "H":
                    for team in c["teams1"]:
                        slots_H = Solution[team, c["teams2"]]
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and(
                                    (slots_H < s), (slots_H >= s - c["intp"])
                                )
                            )
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
                else:
                    for team in c["teams1"]:
                        slots_A = Solution[c["teams2"], team]
                        for s in range(c["intp"], problem["n_slots"] + 1):
                            p = np.sum(
                                np.logical_and(
                                    (slots_A < s), (slots_A >= s - c["intp"])
                                )
                            )
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
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
                            obj += (p - c["max"]) * c["penalty"]
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
                elif c["mode1"] == "H":
                    if c["mode2"] == "GLOBAL":
                        p = 0
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p += np.sum(slots == slot)
                        if p > c["max"]:
                            obj += (p - c["max"]) * c["penalty"]
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
                else:
                    if c["mode2"] == "GLOBAL":
                        p = 0
                        slots = Solution[np.ix_(c["teams2"], c["teams1"])].flatten()
                        for slot in c["slots"]:
                            p += np.sum(slots == slot)
                        if p > c["max"]:
                            obj += (p - c["max"]) * c["penalty"]
                    else:
                        slots = Solution[np.ix_(c["teams1"], c["teams2"])].flatten()
                        for slot in c["slots"]:
                            p = np.sum(slots == slot)
                            if p > c["max"]:
                                obj += (p - c["max"]) * c["penalty"]
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:  # GA1 constraints
            p = 0
            for meeting in c["meetings"]:
                p += np.sum(Solution[tuple(meeting)] == c["slots"])
            if p < c["min"]:
                obj += (c["min"] - p) * c["penalty"]
            elif p > c["max"]:
                obj += (p - c["max"]) * c["penalty"]
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
                        obj += (p - c["intp"]) * c["penalty"]
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


def feasibility_check(Solution, problem):
    ca, ga, ba, fa, sa = problem["feas_constr"].values()
    status, feasibility = "Feasible", True
    for i in range(len(Solution)):
        if (
            len(np.unique(np.concatenate([Solution[i, :], Solution[:, i]])))
            != len(Solution) * 2 - 1
        ):
            feasibility = False
            print(i)
            status = "Incompatible solution, at least one team has more than one game on the same time slot"
            return status, feasibility
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
                        # return status, feasibility
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
                            # return status, feasibility
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
                            # return status, feasibility
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
                            # return status, feasibility
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
                                # return status, feasibility
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
                                # return status, feasibility
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
                                # return status, feasibility
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
                            # return status, feasibility
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
                                # return status, feasibility
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
                            # return status, feasibility
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
                                # return status, feasibility
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
                            # return status, feasibility
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
                                # return status, feasibility
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
                # return status, feasibility
            elif p > c["max"]:
                feasibility = False
                status = "More/less than max/min {c['max']} games from {c['meetings']} took place during time slots{c['slots']}:\t {p}"
                # return status, feasibility
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
                        # return status, feasibility
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
                    # return status, feasibility
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
                # return status, feasibility
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
                    # return status, feasibility
    return status, feasibility


def load_solution(file, sol):
    # load file
    with open(file, "r") as f:
        data_dict = xmltodict.parse(f.read())
        f.close()
    Games = data_dict["Solution"]["Games"]["ScheduledMatch"]
    objective_value = int(
        data_dict["Solution"]["MetaData"]["ObjectiveValue"]["@objective"]
    )
    for game in Games:
        sol[int(game["@home"]), int(game["@away"])] = int(game["@slot"])
    return sol, objective_value


def random_init_sol(sol, problem, rng):
    num_teams = problem["n_teams"]
    teams_list = list(rng.permutation(np.arange(num_teams)))
    if (len(teams_list) % 2) != 0:
        teams_list.append(0)
    x = teams_list[0 : int(len(teams_list) / 2)]
    y = teams_list[int(len(teams_list) / 2) : len(teams_list)]
    matches = []
    for i in range(len(teams_list)):
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
    for i in range(len(teams_list)):
        if i != 0:
            x.insert(1, y.pop(0))
            y.append(x.pop())
        # matches.append(rounds)
        for j in range(len(x)):
            if sol[x[j], y[j]] == -1:
                sol[x[j], y[j]] = i + len(teams_list)
            else:
                sol[y[j], x[j]] = i + len(teams_list)
    return sol
