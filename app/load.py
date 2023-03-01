from collections import OrderedDict
from itertools import combinations, product
import xmltodict
from app.named_tuples import Constraint


def load_problem(file: str) -> dict[str, dict[str, list]]:
    data_dict = load_xml_file(file)
    n_teams = len(data_dict["Instance"]["Resources"]["Teams"]["team"])
    n_slots = (n_teams - 1) * 2
    gameMode = data_dict["Instance"]["Structure"]["Format"]["gameMode"]
    constraints = data_dict["Instance"]["Constraints"]
    feas_constr, obj_constr = Constraint(n_teams=n_teams, n_slots=n_slots), Constraint(
        n_teams=n_teams, n_slots=n_slots
    )
    load_capacity_constraints(
        constraints.get("CapacityConstraints"),
        feas_constr,
        obj_constr,
    )
    load_game_constraints(
        constraints.get("GameConstraints"),
        feas_constr,
        obj_constr,
    )
    load_break_constraints(
        constraints.get("BreakConstraints"),
        feas_constr,
        obj_constr,
    )
    load_fairness_constraints(
        constraints.get("FairnessConstraints"),
        feas_constr,
        obj_constr,
    )
    load_separation_constraints(
        constraints.get("SeparationConstraints"),
        feas_constr,
        obj_constr,
    )
    return {
        "feas_constr": feas_constr,
        "obj_constr": obj_constr,
        "n_teams": n_teams,
        "n_slots": n_slots,
        "gameMode": gameMode,
    }


def load_separation_constraints(
    separation_constraints: OrderedDict,
    feas_constr: Constraint,
    obj_constr: Constraint,
) -> tuple[list, list]:
    """This function extracts the separation constraints

    Args:
        separation_constraints (OrderedDict): dictionary containing each type of separation constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard separation constraints
    """
    if separation_constraints:
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
                    feas_constr.all["sa"][i].append(const)
                    for team in const["teams"]:
                        feas_constr.teams[team]["sa"][i].append(const)
                    for game in combinations(const["teams"], 2):
                        feas_constr.games[game]["sa"][i].append(const)
                        feas_constr.games[(game[1], game[0])]["sa"][i].append(const)
                else:
                    obj_constr.all["sa"][i].append(const)
                    for team in const["teams"]:
                        obj_constr.teams[team]["sa"][i].append(const)
                    for game in combinations(const["teams"], 2):
                        obj_constr.games[game]["sa"][i].append(const)
                        obj_constr.games[(game[1], game[0])]["sa"][i].append(const)
    return


def load_fairness_constraints(
    fairness_constraints: OrderedDict,
    feas_constr: Constraint,
    obj_constr: Constraint,
) -> tuple[list, list]:
    """This function extracts the fairness constraints

    Args:
        fairness_constraints (OrderedDict): dictionary containing each type of fairness constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard fairness constraints
    """
    if fairness_constraints:
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
                    feas_constr.all["fa"][i].append(const)
                    for team in const["teams"]:
                        feas_constr.teams[team]["fa"][i].append(const)
                    for slot in const["slots"]:
                        feas_constr.slots[slot]["fa"][i].append(const)
                else:
                    obj_constr.all["fa"][i].append(const)
                    for team in const["teams"]:
                        obj_constr.teams[team]["fa"][i].append(const)
                    for slot in const["slots"]:
                        obj_constr.slots[slot]["fa"][i].append(const)
    return


def load_break_constraints(
    break_constraints: OrderedDict,
    feas_constr: Constraint,
    obj_constr: Constraint,
) -> tuple[list, list]:
    """This function extracts the break constraints

    Args:
        break_constraints (OrderedDict): dictionary containing each type of break constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard break constraints
    """
    if break_constraints:
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
                    "teams": sorted([int(x) for x in num["@teams"].split(";")]),
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
                    feas_constr.all["ba"][i].append(const)
                    for team in const["teams"]:
                        feas_constr.teams[team]["ba"][i].append(const)
                    for slot in const["slots"]:
                        feas_constr.slots[slot]["ba"][i].append(const)
                else:
                    obj_constr.all["ba"][i].append(const)
                    for team in const["teams"]:
                        obj_constr.teams[team]["ba"][i].append(const)
                    for slot in const["slots"]:
                        obj_constr.slots[slot]["ba"][i].append(const)
    return


def load_game_constraints(
    game_constraints: OrderedDict,
    feas_constr: Constraint,
    obj_constr: Constraint,
) -> tuple[list, list]:
    """This function extracts the game constraints

    Args:
        game_constraints (OrderedDict): dictionary containing each type of game constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard game constraints
    """
    if game_constraints:
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
                    "meetings": sorted(
                        [
                            (int(y), int(z))
                            for y, z in (
                                x.split(",") for x in num["@meetings"].split(";") if x
                            )
                        ]
                    ),
                    "slots": [int(x) for x in num["@slots"].split(";")],
                    "max": int(num["@max"]),
                    "min": int(num["@min"]),
                    "penalty": int(num["@penalty"]),
                    "type": num["@type"],
                }
                teams = set()
                if num["@type"] == "HARD":
                    feas_constr.all["ga"][i].append(const)
                    for slot in const["slots"]:
                        feas_constr.slots[slot]["ga"][i].append(const)
                    for game in const["meetings"]:
                        feas_constr.games[game]["ga"][i].append(const)
                        teams.update(game)
                    for team in teams:
                        feas_constr.teams[team]["ga"][i].append(const)
                else:
                    obj_constr.all["ga"][i].append(const)
                    for slot in const["slots"]:
                        obj_constr.slots[slot]["ga"][i].append(const)
                    for game in const["meetings"]:
                        obj_constr.games[game]["ga"][i].append(const)
                        teams.update(game)
                    for team in teams:
                        obj_constr.teams[team]["ga"][i].append(const)
    return


def load_capacity_constraints(
    capacity_constraints: OrderedDict,
    feas_constr: Constraint,
    obj_constr: Constraint,
) -> tuple[list, list]:
    """This function extracts the capacity constraints

    Args:
        capacity_constraints (OrderedDict): dictionary containing each type of capacity constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard capacity constraints
    """
    if capacity_constraints:
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
                    const["teams"] = sorted([int(x) for x in num["@teams"].split(";")])
                else:
                    const["teams1"] = sorted(
                        [int(x) for x in num["@teams1"].split(";")]
                    )
                    const["teams2"] = sorted(
                        [int(x) for x in num["@teams2"].split(";")]
                    )
                if "@mode" in num.keys():
                    const["mode"] = num["@mode"]
                else:
                    const["mode1"] = num["@mode1"]
                    const["mode2"] = num["@mode2"]
                if "@slots" in num.keys():
                    const["slots"] = sorted([int(x) for x in num["@slots"].split(";")])
                if "@intp" in num.keys():
                    const["intp"] = int(num["@intp"])
                teams = (
                    const["teams"]
                    if const.get("teams")
                    else const["teams1"] + const["teams2"]
                )
                if num["@type"] == "HARD":
                    feas_constr.all["ca"][i].append(const)
                    for team in teams:
                        feas_constr.teams[team]["ca"][i].append(const)
                    if const.get("mode2"):
                        for slot in feas_constr.slots.keys():
                            feas_constr.slots[slot]["ca"][i].append(const)
                    else:
                        for slot in const["slots"]:
                            feas_constr.slots[slot]["ca"][i].append(const)
                    if "teams2" in const:
                        for game in product(const["teams1"], const["teams2"]):
                            if game[0] != game[1]:
                                feas_constr.games[game]["ca"][i].append(const)
                else:
                    obj_constr.all["ca"][i].append(const)
                    for team in teams:
                        obj_constr.teams[team]["ca"][i].append(const)
                    if const.get("mode2"):
                        for slot in obj_constr.slots.keys():
                            obj_constr.slots[slot]["ca"][i].append(const)
                    else:
                        for slot in const["slots"]:
                            obj_constr.slots[slot]["ca"][i].append(const)
                    if "teams2" in const:
                        for game in product(const["teams1"], const["teams2"]):
                            if game[0] != game[1]:
                                obj_constr.games[game]["ca"][i].append(const)
    return


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


# def get_team_consts(constraints, team):
# for const in constraints:
