from itertools import combinations, product
import numpy as np
import xml.etree.ElementTree as ET


def write_sol_xml(model, filename):
    root = ET.Element("Solution")

    # Create an element to store the objective value
    metadata_element = ET.SubElement(root, "MetaData")
    objective_element = ET.SubElement(metadata_element, "ObjectiveValue")
    objective_element.set("objective", str(model.objVal))

    # Create an element to store variable values
    games_element = ET.SubElement(root, "Games")
    for variable in model.getVars():
        if variable.x > 0 and variable.varName[0] == "x":
            variable_element = ET.SubElement(games_element, "ScheduledMatch")
            home, away, slot = variable.varName.split("[")[1].split("]")[0].split(",")
            variable_element.set("home", home)
            variable_element.set("away", away)
            variable_element.set("slot", slot)

    # Create the XML tree
    tree = ET.ElementTree(root)

    # Write the XML tree to a file
    tree.write(filename)

    print("Optimization results saved to results.xml.")


def cost_function(Solution, problem: dict) -> int:
    """Compute the cost of the solution based on the constraints.
    Args:
        Solution (Solution): the solution
        problem (dict): the problem instance
    Returns:
        int: the cost of the solution
    """
    representative = Solution.representative
    ca, ba, ga, fa, sa = problem["obj_constr"].all.values()
    obj = 0
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        for c in cc:
            if i == 0:  # CA1 constraints
                p = np.zeros(len(c["teams"]), dtype=int)
                for slot in c["slots"]:
                    p += (
                        np.sum(representative[c["teams"], :] == slot, axis=1)
                        if c["mode"] == "H"
                        else np.sum(representative[:, c["teams"]] == slot, axis=0)
                    )
                penalty = max([(p - c["max"]).sum(), 0]) * c["penalty"]
                update_costs(Solution, penalty, const_type=c["type"])
                Solution.teams_cost[c["teams"]] += (
                    np.maximum((p[i] - c["max"]), 0) * c["penalty"]
                )
                Solution.slots_cost[c["slots"]] += penalty
            elif i == 1:  # CA2 constraints
                for team1 in c["teams1"]:
                    if c["mode1"] == "HA":
                        p = np.sum(
                            np.isin(representative[team1, c["teams2"]], c["slots"])
                        ) + np.sum(
                            np.isin(representative[c["teams2"], team1], c["slots"])
                        )
                        meetings = list(product([team1], c["teams2"])) + list(
                            product(c["teams2"], [team1])
                        )
                    elif c["mode1"] == "H":
                        p = np.sum(
                            np.isin(representative[team1, c["teams2"]], c["slots"])
                        )
                        meetings = list(product([team1], c["teams2"]))
                    else:
                        p = np.sum(
                            np.isin(representative[c["teams2"], team1], c["slots"])
                        )
                        meetings = list(product(c["teams2"], [team1]))
                    if penalty := compute_penalty(c, p):
                        Solution.teams_cost[c["teams2"] + [team1]] += penalty
                        for meeting in meetings:
                            Solution.games_cost[meeting] += penalty
                        Solution.slots_cost[c["slots"]] += penalty
                        update_costs(Solution, penalty, const_type=c["type"])
            elif i == 2:  # CA3 constraints
                for team in c["teams1"]:
                    slots_H = representative[team, c["teams2"]]
                    slots_A = representative[c["teams2"], team]
                    slots = np.concatenate([slots_A, slots_H])
                    obj_bef = obj
                    if c["mode1"] == "HA":
                        obj = check_games_in_slots(problem, obj, c, slots)
                        slots_check = slots
                        meetings = list(product([team], c["teams2"])) + list(
                            product(c["teams2"], [team])
                        )
                    elif c["mode1"] == "H":
                        obj = check_games_in_slots(problem, obj, c, slots_H)
                        slots_check = slots_H
                        meetings = list(product([team], c["teams2"]))
                    else:
                        obj = check_games_in_slots(problem, obj, c, slots_A)
                        slots_check = slots_A
                        meetings = list(product(c["teams2"], [team]))
                    if penalty := obj - obj_bef:
                        update_costs(Solution, penalty, const_type=c["type"])
                        Solution.teams_cost[c["teams2"] + [team]] += penalty
                        for meeting in meetings:
                            Solution.games_cost[meeting] += penalty
                        Solution.slots_cost[slots_check] += penalty
            else:  # CA4 constraints
                slots_H = representative[np.ix_(c["teams1"], c["teams2"])].flatten()
                slots_A = representative[np.ix_(c["teams2"], c["teams1"])].flatten()
                slots = np.concatenate([slots_A, slots_H])
                if c["mode2"] == "GLOBAL":
                    if c["mode1"] == "HA":
                        p = np.sum(np.isin(slots, c["slots"]))
                        slots_check = slots
                        meetings = list(product(c["teams1"], c["teams2"])) + list(
                            product(c["teams2"], c["teams1"])
                        )
                    elif c["mode1"] == "H":
                        p = np.sum(np.isin(slots_H, c["slots"]))
                        slots_check = slots_H
                        meetings = list(product(c["teams1"], c["teams2"]))
                    else:
                        p = np.sum(np.isin(slots_A, c["slots"]))
                        slots_check = slots_A
                        meetings = list(product(c["teams2"], c["teams1"]))

                    if penalty := compute_penalty(c, p):
                        update_costs(Solution, penalty, const_type=c["type"])
                        Solution.teams_cost[c["teams2"] + c["teams1"]] += penalty
                        for meeting in meetings:
                            Solution.games_cost[meeting] += penalty
                        Solution.slots_cost[slots_check] += penalty
                else:
                    for slot in c["slots"]:
                        if c["mode1"] == "HA":
                            p = np.sum(slots == slot)
                            slots_check = slots
                            meetings = list(product(c["teams1"], c["teams2"])) + list(
                                product(c["teams2"], c["teams1"])
                            )
                        elif c["mode1"] == "H":
                            p = np.sum(slots_H == slot)
                            slots_check = slots_H
                            meetings = list(product(c["teams1"], c["teams2"]))
                        else:
                            p = np.sum(slots_A == slot)
                            slots_check = slots_A
                            meetings = list(product(c["teams2"], c["teams1"]))
                        if penalty := compute_penalty(c, p) > 0:
                            update_costs(Solution, penalty, const_type=c["type"])
                            Solution.teams_cost[c["teams2"] + c["teams1"]] += penalty
                            for meeting in meetings:
                                Solution.games_cost[meeting] += penalty
                            Solution.slots_cost[slots_check] += penalty
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:  # GA1 constraints
            p = sum(
                np.sum(representative[meeting] == c["slots"])
                for meeting in c["meetings"]
            )
            obj_bef = obj
            obj += compute_penalty(c, p, "min")
            obj += compute_penalty(c, p, "max")
            if penalty := obj - obj_bef:
                update_costs(Solution, penalty, const_type=c["type"])
                for meeting in c["meetings"]:
                    Solution.teams_cost[[*meeting]] += penalty
                    Solution.games_cost[meeting] += penalty
                Solution.slots_cost[c["slots"]] += penalty
    for i, bc in enumerate(ba):
        if len(bc) == 0:
            continue
        for c in bc:
            if i == 0:
                check_zero = 1 if c["slots"][0] == 0 else 0
                for team in c["teams"]:
                    p = sum(
                        np.isin(
                            np.array(c["slots"][check_zero:]), representative[team, :]
                        )
                        == np.isin(
                            np.array(c["slots"][check_zero:]) - 1,
                            representative[team, :],
                        )
                    )
                    if penalty := compute_penalty(c, p, "intp"):
                        update_costs(Solution, penalty, const_type=c["type"])
                        Solution.teams_cost[team] += penalty
                        Solution.slots_cost[c["slots"]] += penalty
            elif i == 1:
                check_zero = 1 if c["slots"][0] == 0 else 0
                p = sum(
                    sum(
                        np.isin(
                            np.array(c["slots"][check_zero:]),
                            representative[team, :],
                        )
                        == np.isin(
                            np.array(c["slots"][check_zero:]) - 1,
                            representative[team, :],
                        )
                    )
                    for team in c["teams"]
                )
                if penalty := compute_penalty(c, p, "intp"):
                    update_costs(Solution, penalty, const_type=c["type"])
                    Solution.teams_cost[c["teams"]] += penalty / len(c["teams"])
                    Solution.slots_cost[c["slots"]] += penalty / len(c["slots"])
    for i, fc in enumerate(fa):
        if len(fc) == 0:
            continue
        for c in fc:  # FA1 constraints
            diff = np.zeros([len(c["teams"]), len(c["teams"])], dtype=int)
            for s in c["slots"]:
                home_count = (representative[c["teams"], :] <= s).sum(
                    axis=1
                ) - 1  # excluding the column = team
                for i, j in combinations(c["teams"], 2):
                    difference = abs(home_count[i] - home_count[j])
                    diff[i, j] = max(difference, diff[i, j])
                    if difference > c["intp"]:
                        Solution.teams_cost[[i, j]] += c["penalty"]
                        Solution.slots_cost[s] += c["penalty"]
            diff -= c["intp"]
            diff[diff < 0] = 0
            if penalty := np.sum(diff) * c["penalty"] > 0:
                update_costs(Solution, penalty, const_type=c["type"])
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:  # SE1 constraints
            for team1, team2 in c["teams"]:
                first = representative[team1, team2]
                second = representative[team2, team1]
                diff = abs(second - first) - 1
                if penalty := compute_penalty(c, diff, "min"):
                    update_costs(Solution, penalty, const_type=c["type"])
                    Solution.teams_cost[[team1, team2]] += penalty
                    Solution.slots_cost[[first, second]] += penalty
                    Solution.games_cost[team1, team2] += penalty
                    Solution.games_cost[team2, team1] += penalty

    return Solution.total_cost


def cost_function_games(
    representative: np.array, problem: dict, game: tuple[int, int]
) -> int:
    """Compute the cost of a game based on the constraints.
    Args:
        representative (np.array): the representative of the solution
        problem (dict): the problem instance
        game (tuple[int, int]): the game to compute the cost for
    Returns:
        int: the cost of the game
    """
    ca, _, ga, _, sa = problem["obj_constr"].games[game].values()
    soft_cost, hard_cost, total_cost, obj = 0, 0, 0, 0
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        for c in cc:
            if i == 0:  # CA1 constraints
                p = np.zeros(len(c["teams"]), dtype=int)
                for slot in c["slots"]:
                    p += (
                        np.sum(representative[c["teams"], :] == slot, axis=1)
                        if c["mode"] == "H"
                        else np.sum(representative[:, c["teams"]] == slot, axis=0)
                    )
                penalty = max([(p - c["max"]).sum(), 0]) * c["penalty"]
                soft_cost, hard_cost, total_cost = update_costs_games(
                    soft_cost, hard_cost, total_cost, penalty, c["type"]
                )
            elif i == 1:  # CA2 constraints
                for team1 in c["teams1"]:
                    if c["mode1"] == "HA":
                        p = np.sum(
                            np.isin(representative[team1, c["teams2"]], c["slots"])
                        ) + np.sum(
                            np.isin(representative[c["teams2"], team1], c["slots"])
                        )
                    elif c["mode1"] == "H":
                        p = np.sum(
                            np.isin(representative[team1, c["teams2"]], c["slots"])
                        )
                    else:
                        p = np.sum(
                            np.isin(representative[c["teams2"], team1], c["slots"])
                        )
                    if penalty := compute_penalty(c, p):
                        soft_cost, hard_cost, total_cost = update_costs_games(
                            soft_cost, hard_cost, total_cost, penalty, c["type"]
                        )
            elif i == 2:  # CA3 constraints
                for team in c["teams1"]:
                    slots_H = representative[team, c["teams2"]]
                    slots_A = representative[c["teams2"], team]
                    slots = np.concatenate([slots_A, slots_H])
                    obj_bef = obj
                    if c["mode1"] == "HA":
                        obj = check_games_in_slots(problem, obj, c, slots)
                    elif c["mode1"] == "H":
                        obj = check_games_in_slots(problem, obj, c, slots_H)
                    else:
                        obj = check_games_in_slots(problem, obj, c, slots_A)
                    if penalty := obj - obj_bef:
                        soft_cost, hard_cost, total_cost = update_costs_games(
                            soft_cost, hard_cost, total_cost, penalty, c["type"]
                        )
            else:  # CA4 constraints
                slots_H = representative[np.ix_(c["teams1"], c["teams2"])].flatten()
                slots_A = representative[np.ix_(c["teams2"], c["teams1"])].flatten()
                slots = np.concatenate([slots_A, slots_H])
                if c["mode2"] == "GLOBAL":
                    if c["mode1"] == "HA":
                        p = np.sum(np.isin(slots, c["slots"]))
                    elif c["mode1"] == "H":
                        p = np.sum(np.isin(slots_H, c["slots"]))
                    else:
                        p = np.sum(np.isin(slots_A, c["slots"]))

                    if penalty := compute_penalty(c, p):
                        soft_cost, hard_cost, total_cost = update_costs_games(
                            soft_cost, hard_cost, total_cost, penalty, c["type"]
                        )
                else:
                    for slot in c["slots"]:
                        if c["mode1"] == "HA":
                            p = np.sum(slots == slot)
                            slots_check = slots
                        elif c["mode1"] == "H":
                            p = np.sum(slots_H == slot)
                        else:
                            p = np.sum(slots_A == slot)
                        if penalty := compute_penalty(c, p) > 0:
                            soft_cost, hard_cost, total_cost = update_costs_games(
                                soft_cost, hard_cost, total_cost, penalty, c["type"]
                            )
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:  # GA1 constraints
            p = sum(
                np.sum(representative[meeting] == c["slots"])
                for meeting in c["meetings"]
            )
            obj_bef = obj
            obj += compute_penalty(c, p, "min")
            obj += compute_penalty(c, p, "max")
            if penalty := obj - obj_bef:
                soft_cost, hard_cost, total_cost = update_costs_games(
                    soft_cost, hard_cost, total_cost, penalty, c["type"]
                )
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:  # SE1 constraints
            for team1, team2 in c["teams"]:
                first = representative[team1, team2]
                second = representative[team2, team1]
                diff = abs(second - first) - 1
                if penalty := compute_penalty(c, diff, "min"):
                    soft_cost, hard_cost, total_cost = update_costs_games(
                        soft_cost, hard_cost, total_cost, penalty, c["type"]
                    )

    return total_cost


def feasibility_check(
    representative: np.array, problem: dict, const_level: str = "all", index: int = None
) -> tuple[str, bool]:
    """Check the feasibility of the solution based on the constraints either at the team, slot, game, or all levels.
    Args:
        representative (np.array): the representative of the solution
        problem (dict): the problem instance
        const_level (str, optional): the level of constraints to check. Defaults to "all".
        index (int, optional): the index of the constraint to check. Defaults to None.
    Returns:
        tuple[str, bool]: the status of the solution and its feasibility (True/False)
    """
    if const_level != "all":
        ca, ba, ga, fa, sa = getattr(problem["feas_constr"], const_level)[
            index
        ].values()
    else:
        ca, ba, ga, fa, sa = getattr(problem["feas_constr"], const_level).values()
    status, feasibility = compatibility_check(representative)
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
                            np.transpose(representative[team : team + 1, :])
                            == c["slots"]
                        )
                    else:
                        p += np.sum(representative[:, team : team + 1] == c["slots"])
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
                            p += np.sum(representative[team1, team2] == c["slots"])
                            p += np.sum(representative[team2, team1] == c["slots"])
                        if p > c["max"]:
                            feasibility = False
                            status = f"Team {team1} has {p - c['max']} more {c['mode1']} games than max={c['max']}\
                            during time slots {c['slots']} against teams {c['teams2']}: {p}"
                            return status, feasibility
                elif c["mode1"] == "H":
                    for team1 in c["teams1"]:
                        p = 0
                        for team2 in c["teams2"]:
                            p += np.sum(representative[team1, team2] == c["slots"])
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
                            p += np.sum(representative[team2, team1] == c["slots"])
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
                        slots_H = representative[team1, c["teams2"]]
                        slots_A = representative[c["teams2"], team1]
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
                        slots_H = representative[team1, c["teams2"]]
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
                        slots_A = representative[c["teams2"], team1]
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
                        slots_H = representative[
                            np.ix_(c["teams1"], c["teams2"])
                        ].flatten()
                        slots_A = representative[
                            np.ix_(c["teams2"], c["teams1"])
                        ].flatten()
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
                        slots = representative[
                            np.ix_(c["teams1"], c["teams2"])
                        ].flatten()
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
                        slots = representative[
                            np.ix_(c["teams1"], c["teams2"])
                        ].flatten()
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
                        slots = representative[
                            np.ix_(c["teams1"], c["teams2"])
                        ].flatten()
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
                        slots = representative[
                            np.ix_(c["teams2"], c["teams1"])
                        ].flatten()
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
                        slots = representative[
                            np.ix_(c["teams1"], c["teams2"])
                        ].flatten()
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
                p += np.sum(representative[tuple(meeting)] == c["slots"])
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
                        home_games = representative[team, :]
                        games = np.concatenate([home_games, representative[:, team]])
                        if slot == 0 or slot not in games:
                            continue
                        cur = (home_games == slot).any()
                        prev = (home_games == slot - 1).any()
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
                        home_games = representative[team, :]
                        games = np.concatenate([home_games, representative[:, team]])
                        if slot == 0 or slot not in games:
                            continue
                        cur = (representative[team, :] == slot).any()
                        prev = (representative[team, :] == slot - 1).any()
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
                        np.sum(representative[team, :] <= s) - 1
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
                first = representative[team1, team2]
                second = representative[team2, team1]
                diff = abs(second - first) - 1
                if diff < c["min"]:
                    feasibility = False
                    status = "Team {} and team {} has {} less time slots between their mutual games than min= {}:\t {}".format(
                        team1, team2, c["min"] - diff, c["min"], diff
                    )
                    return status, feasibility
    return status, feasibility


def compatibility_check(Solution: np.array) -> tuple[str, bool]:
    """Check if the solution is compatible, i.e. if each team has exactly one game per time slot
    Args:
        Solution (np.array): The solution to be checked (solution representation)

    Returns:
        tuple[str, bool]: The status of the check and the feasibility of the solution
    """
    for i in range(len(Solution)):
        weeks, counts = np.unique(
            np.concatenate([Solution[i, :], Solution[:, i]]), return_counts=True
        )
        dummy_week = weeks[-1] == (len(Solution) - 1) * 2
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


def random_init_sol(problem: dict, rng: np.random.Generator) -> np.array:
    """Create a random initial solution using the circle method
    Args:
        problem (dict): The problem dictionary
        rng (np.random.Generator): The random number generator

    Returns:
        np.array: The random initial solution
    """
    sol = np.ones((problem["n_teams"], problem["n_teams"]), dtype=int) * (-1)
    num_teams = problem["n_teams"]
    for round in range(2):
        teams_list = list(rng.permutation(np.arange(num_teams)))
        if (len(teams_list) % 2) != 0:
            teams_list.append(0)
        x = teams_list[0 : int(len(teams_list) / 2)]
        y = teams_list[int(len(teams_list) / 2) : len(teams_list)]
        for i in range(len(teams_list) - 1):
            if i != 0:
                x.insert(1, y.pop(0))
                y.append(x.pop())
            for j in range(len(x)):
                if sol[x[j], y[j]] == -1:
                    sol[x[j], y[j]] = i + len(teams_list) * round - 1
                else:
                    sol[y[j], x[j]] = i + len(teams_list) * round - 1
    return sol


def dummy_init_sol(problem: dict) -> np.array:
    """Create a dummy initial solution with all games assigned to the dummy week which is the last week of the season + 1
    and main diagonal set to -1
    Args:
        problem (dict): The problem dictionary

    Returns:
        np.array: The dummy initial solution
    """
    sol = (
        np.ones((problem["n_teams"], problem["n_teams"]), dtype=int)
        * problem["n_slots"]
    )
    for i in range(len(sol)):
        sol[i, i] = -1
    return sol


def compute_penalty(c: dict, p: int, extremum: str = "max") -> int:
    """Compute the penalty for a constraint

    Args:
        c (dict): The constraint dictionary
        p (int): The number of violations
        extremum (str, optional): The extremum to be used, either "max", "min" or "intp". Defaults to "max".

    Returns:
        int: The penalty
    """
    if extremum in ["max", "intp"]:
        return max([p - c[extremum], 0]) * c["penalty"]
    elif extremum == "min":
        return max([c[extremum] - p, 0]) * c["penalty"]


def check_games_in_slots(problem: dict, obj: int, c: dict, slots: np.array) -> int:
    """Check if the number of games in the slots is in the interval [s - c["intp"], s)
    If not, the penalty is computed and added to the objective function

    Args:
        problem (dict): The problem dictionary
        obj (int): The objective value before the check
        c (dict): The constraint dictionary
        slots (np.array): The slots to be checked

    Returns:
        int: The objective value after the check
    """
    for s in range(c["intp"], problem["n_slots"] + 1):
        p = np.sum(np.logical_and((slots < s), (slots >= s - c["intp"])))
        obj += compute_penalty(c, p)
    return obj


def update_costs(
    solution, penalty: int, const_type: str, hard_const_degree: int = 10
) -> None:
    """Update the costs of the solution with the penalty
    If the constraint is soft, the soft cost, the total cost and objective function are updated
    If the constraint is hard, the hard cost, the total cost and objective function are updated
    If the constraint is dummy, the dummy cost and the total cost are updated

    Args:
        solution (Solution): The solution object
        penalty (int): The penalty to be added
        const_type (str): The type of the constraint, either "SOFT", "HARD" or "DUMMY"
        hard_const_degree (int, optional): The degree of the hard constraint. Defaults to 10.

    Returns:
        None
    """
    if const_type == "SOFT":
        solution.obj_fun += penalty
        solution.soft_cost += penalty
        solution.total_cost += penalty
    elif const_type == "HARD":
        solution.hard_cost += penalty
        solution.total_cost += penalty * hard_const_degree
    elif const_type == "DUMMY":
        solution.dummy_cost += penalty
        solution.total_cost += penalty
    return


def update_costs_games(
    soft_cost: int,
    hard_cost: int,
    total_cost: int,
    penalty: int,
    const_type: str,
    hard_const_degree: int = 10,
) -> tuple[int, int, int]:
    """Update the costs of the game with the penalty
    If the constraint is soft, the soft cost and the total cost are updated
    If the constraint is hard, the hard cost and the total cost are updated

    Args:
        soft_cost (int): The soft cost of the game
        hard_cost (int): The hard cost of the game
        total_cost (int): The total cost of the game
        penalty (int): The penalty to be added
        const_type (str): The type of the constraint, either "SOFT" or "HARD"
        hard_const_degree (int, optional): The degree of the hard constraint. Defaults to 10.

    Returns:
        tuple[int, int, int]: The updated soft cost, hard cost and total cost
    """
    if const_type == "SOFT":
        soft_cost += penalty
        total_cost += penalty
    elif const_type == "HARD":
        hard_cost += penalty
        total_cost += penalty * hard_const_degree
    return soft_cost, hard_cost, total_cost


def update_week_availability(
    solution, week_num: int, team1: int, team2: int, method: str
) -> None:
    """Update the week availability matrix of the solution
    If the game between team1 and team2 has been assigned to week_num,
    the availability of each team is set to 0 -> not available in that week
    If the game between team1 and team2 has been removed from week_num,
    the availability of each team is set to 1 -> available

    Args:
        solution (Solution): The solution object
        week_num (int): The week to be updated
        team1 (int): The first team to be updated
        team2 (int): The second team to be updated
        method (str): The method to be used, either "add" or "remove"

    Returns:
        None
    """
    if method == "add":
        solution.week_availability[team1, week_num] = 1
        solution.week_availability[team2, week_num] = 1
    elif method == "remove":
        solution.week_availability[team1, week_num] = 0
        solution.week_availability[team2, week_num] = 0


def update_game_availability(solution, game: tuple[int, int], method: str) -> None:
    """Update the game availability matrix of the solution
    If the game has been added, the availability is set to 1
    If the game has been removed, the availability is set to 0

    Args:
        solution (Solution): The solution object
        game (tuple[int, int]): The game to be updated
        method (str): The method to be used, either "add" or "remove"

    Returns:
        None
    """
    if method == "add":
        solution.game_availability[game] = 1
    elif method == "remove":
        solution.game_availability[game] = 0
