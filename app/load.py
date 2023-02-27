from collections import OrderedDict
from itertools import combinations
import xmltodict


def load_problem(file: str) -> dict[str, dict[str, list]]:
    data_dict = load_xml_file(file)
    n_teams = len(data_dict["Instance"]["Resources"]["Teams"]["team"])
    n_slots = (n_teams - 1) * 2
    gameMode = data_dict["Instance"]["Structure"]["Format"]["gameMode"]

    if capacity_constraints := data_dict["Instance"]["Constraints"][
        "CapacityConstraints"
    ]:
        ca_hard, ca_soft = load_capacity_constraints(capacity_constraints)
    else:
        ca_hard, ca_soft = [[] for _ in range(4)], [[] for _ in range(4)]

    if game_constraints := data_dict["Instance"]["Constraints"]["GameConstraints"]:
        ga_hard, ga_soft = load_game_constraints(game_constraints)
    else:
        ga_hard, ga_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    if break_constraints := data_dict["Instance"]["Constraints"]["BreakConstraints"]:
        ba_hard, ba_soft = load_break_constraints(break_constraints)
    else:
        ba_hard, ba_soft = [[] for _ in range(2)], [[] for _ in range(2)]
    if fairness_constraints := data_dict["Instance"]["Constraints"][
        "FairnessConstraints"
    ]:
        fa_hard, fa_soft = load_fairness_constraints(fairness_constraints)
    else:
        fa_hard, fa_soft = [[] for _ in range(1)], [[] for _ in range(1)]

    if separation_constraints := data_dict["Instance"]["Constraints"][
        "SeparationConstraints"
    ]:
        sa_hard, sa_soft = load_separation_constraints(separation_constraints)
    else:
        sa_hard, sa_soft = [[] for _ in range(1)], [[] for _ in range(1)]
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


def load_separation_constraints(
    separation_constraints: OrderedDict,
) -> tuple[list, list]:
    """This function extracts the separation constraints

    Args:
        separation_constraints (OrderedDict): dictionary containing each type of separation constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard separation constraints
    """
    sa_hard, sa_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    sas = ["SE1"]
    for i, s in enumerate(sas):
        if s not in separation_constraints.keys():
            continue
        elif type(separation_constraints[s]) == list:
            sc = separation_constraints[s]
        else:
            sc = [separation_constraints[s]]
        for num in sc:
            const = {
                "teams": list(
                    combinations([int(x) for x in num["@teams"].split(";")], 2)
                ),
                "min": int(num["@min"]),
                "mode1": num["@mode1"],
                "penalty": int(num["@penalty"]),
                "type": num["@type"],
            }
            if num["@type"] == "HARD":
                const["penalty"] *= 10
                sa_hard[i].append(const)
            else:
                sa_soft[i].append(const)
    return sa_hard, sa_soft


def load_fairness_constraints(
    fairness_constraints: OrderedDict,
) -> tuple[list, list]:
    """This function extracts the fairness constraints

    Args:
        fairness_constraints (OrderedDict): dictionary containing each type of fairness constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard fairness constraints
    """
    fa_hard, fa_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    fas = ["FA2"]
    for i, f in enumerate(fas):
        if f not in fairness_constraints.keys():
            continue
        elif type(fairness_constraints[f]) == list:
            fc = fairness_constraints[f]
        else:
            fc = [fairness_constraints[f]]
        for num in fc:
            const = {
                "teams": sorted([int(x) for x in num["@teams"].split(";")]),
                "slots": sorted([int(x) for x in num["@slots"].split(";")]),
                "intp": int(num["@intp"]),
                "mode": num["@mode"],
                "penalty": int(num["@penalty"]),
                "type": num["@type"],
            }
            if num["@type"] == "HARD":
                const["penalty"] *= 10
                fa_hard[i].append(const)
            else:
                fa_soft[i].append(const)
    return fa_hard, fa_soft


def load_break_constraints(
    break_constraints: OrderedDict,
) -> tuple[list, list]:
    """This function extracts the break constraints

    Args:
        break_constraints (OrderedDict): dictionary containing each type of break constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard break constraints
    """
    ba_hard, ba_soft = [[] for _ in range(2)], [[] for _ in range(2)]
    bas = ["BR1", "BR2"]
    for i, b in enumerate(bas):
        if b not in break_constraints.keys():
            continue
        elif type(break_constraints[b]) == list:
            bc = break_constraints[b]
        else:
            bc = [break_constraints[b]]
        for num in bc:
            const = {
                "teams": [int(x) for x in num["@teams"].split(";")],
                "slots": sorted([int(x) for x in num["@slots"].split(";")]),
                "intp": int(num["@intp"]),
                "mode2": num["@mode2"],
                "penalty": int(num["@penalty"]),
                "type": num["@type"],
            }
            if "@mode1" in num.keys():
                const["mode1"] = num["@mode1"]
            elif "@homeMode" in num.keys():
                const["homeMode"] = num["@homeMode"]
            if num["@type"] == "HARD":
                const["penalty"] *= 10
                ba_hard[i].append(const)
            else:
                ba_soft[i].append(const)
    return ba_hard, ba_soft


def load_game_constraints(game_constraints: OrderedDict) -> tuple[list, list]:
    """This function extracts the game constraints

    Args:
        game_constraints (OrderedDict): dictionary containing each type of game constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard game constraints
    """
    ga_hard, ga_soft = [[] for _ in range(1)], [[] for _ in range(1)]
    gas = ["GA1"]
    for i, g in enumerate(gas):
        if g not in game_constraints.keys():
            continue
        elif type(game_constraints[g]) == list:
            ga1 = game_constraints[g]
        else:
            ga1 = [game_constraints[g]]
        for num in ga1:
            const = {
                "meetings": [
                    (int(y), int(z))
                    for y, z in (x.split(",") for x in num["@meetings"].split(";") if x)
                ],
                "slots": [int(x) for x in num["@slots"].split(";")],
                "max": int(num["@max"]),
                "min": int(num["@min"]),
                "penalty": int(num["@penalty"]),
                "type": num["@type"],
            }
            if num["@type"] == "HARD":
                const["penalty"] *= 10
                ga_hard[i].append(const)
            else:
                ga_soft[i].append(const)
    return ga_hard, ga_soft


def load_capacity_constraints(capacity_constraints: OrderedDict) -> tuple[list, list]:
    """This function extracts the capacity constraints

    Args:
        capacity_constraints (OrderedDict): dictionary containing each type of capacity constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard capacity constraints
    """
    ca_hard, ca_soft = [[] for _ in range(4)], [[] for _ in range(4)]
    cas = ["CA1", "CA2", "CA3", "CA4"]
    for i, c in enumerate(cas):
        if c not in capacity_constraints.keys():
            continue
        elif type(capacity_constraints[c]) == list:
            cc = capacity_constraints[c]
        else:
            cc = [capacity_constraints[c]]
        for num in cc:
            const = {
                "max": int(num["@max"]),
                "min": int(num["@min"]),
                "penalty": int(num["@penalty"]),
                "type": num["@type"],
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
                const["penalty"] *= 10
                ca_hard[i].append(const)
            else:
                ca_soft[i].append(const)
    return ca_hard, ca_soft


def load_xml_file(file: str) -> OrderedDict:
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
