from collections import OrderedDict, defaultdict
from itertools import combinations, product
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from app.named_tuples import Constraint


def create_model(problem):
    model = gp.Model(problem["instance_name"])
    n_teams = problem["n_teams"]
    n_slots = problem["n_slots"]
    gameMode = problem["gameMode"]
    x, h, a = add_variables(model, n_teams, n_slots)
    model.addConstrs(
        (x.sum(i, i, "*") == 0 for i in range(n_teams)), name="no_self_play"
    )
    # games_happen_only_once
    model.addConstrs(
        (
            x.sum(i, j, "*") == 1
            for i, j in product(range(n_teams), range(n_teams))
            if i != j
        ),
        name="games_happen_only_once",
    )
    # one_game_per_slot
    model.addConstrs(
        (
            x.sum(i, "*", k) + x.sum("*", i, k) == 1
            for i, k in product(range(n_teams), range(n_slots))
        ),
        name="one_game_per_slot",
    )
    if gameMode == "P":
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] + x[j, i, k] for k in range(n_teams - 1)) == 1
                for i, j in product(range(n_teams), range(n_teams))
                if i != j
            ),
            name="at_least_one_game_first_half",
        )
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] + x[j, i, k] for k in range(n_teams, n_slots))
                == 1
                for i, j in product(range(n_teams), range(n_teams))
                if i != j
            ),
            name="at_least_one_game_second_half",
        )
    # home break
    model.addConstrs(
        (
            x.sum(i, "*", k) + x.sum(i, "*", k - 1) <= 1 + h[i, k]
            for i, k in product(range(n_teams), range(1, n_slots))
        ),
        name="home_break",
    )
    # away break
    model.addConstrs(
        (
            x.sum("*", i, k) + x.sum("*", i, k - 1) <= 1 + a[i, k]
            for i, k in product(range(n_teams), range(1, n_slots))
        ),
        name="away_break",
    )
    ca_soft, br_soft, ga_soft, fa_soft, sa_soft = problem["obj_constr"].all.values()
    ca_hard, br_hard, ga_hard, fa_hard, sa_hard = problem["feas_constr"].all.values()

    # capacity constraints
    dca1, dca2, dca3, dca4 = add_capacity_constraints(
        ca_hard, ca_soft, model, x, n_teams, n_slots
    )
    dbr1, dbr2 = add_break_constraints(br_hard, br_soft, model, h, a, n_teams)
    dga1 = add_game_constraints(
        ga_hard,
        ga_soft,
        model,
        x,
    )
    dfa2 = add_fairness_constraints(fa_hard, fa_soft, model, x, n_teams)
    dse1 = add_separation_constraints(sa_hard, sa_soft, model, x, n_teams, n_slots)
    model.update()
    return model


def add_separation_constraints(sa_hard, sa_soft, model, x, n_teams, n_slots):
    dse1 = gp.tuplelist()
    for sa_type, sc in enumerate(sa_soft):
        if len(sc) == 0:
            continue
        if sa_type == 0:  # SA1 constraints
            dse1 = model.addVars(
                n_teams, n_teams, len(sc), name="dse1", vtype=GRB.INTEGER, lb=0.0
            )
            _add_separation_constraints1(model, x, sc, n_slots, dse1)
    for sa_type, sc in enumerate(sa_hard):
        if len(sc) == 0:
            continue
        if sa_type == 0:  # SA1 constraints
            _add_separation_constraints1(model, x, sc, n_slots, None)
    return dse1


def _add_separation_constraints1(model, x, sc, n_slots, dse1: None):
    for l, se1 in enumerate(sc):
        model.addConstrs(
            (
                gp.quicksum(k * (x[i, j, k] - x[j, i, k]) for k in range(n_slots))
                - (dse1[i, j, l] if dse1 is not None else 0)
                >= se1["min"]
                for i, j in se1["teams"]
            ),
            name=f"se1_{'soft' if dse1 is not None else 'hard'}_1",
        )
        model.addConstrs(
            (
                gp.quicksum(k * (x[j, i, k] - x[i, j, k]) for k in range(n_slots))
                + (dse1[i, j, l] if dse1 is not None else 0)
                <= -se1["min"]
                for i, j in se1["teams"]
            ),
            name=f"se1_{'soft' if dse1 is not None else 'hard'}_2",
        )


def add_fairness_constraints(
    fa_hard: list[list[dict]],
    fa_soft: list[list[dict]],
    model: gp.Model,
    x: gp.tupledict,
    n_teams: int,
) -> np.array:
    """This function extracts the fairness constraints

    Args:
        fairness_constraints (OrderedDict): dictionary containing each type of fairness constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard fairness constraints
    """
    dfa2 = gp.tuplelist()
    for fa_type, fc in enumerate(fa_soft):
        if len(fc) == 0:
            continue
        if fa_type == 0:  # FA2 constraints
            dfa2 = model.addVars(
                n_teams, n_teams, len(fc), name="dfa2", vtype=GRB.INTEGER, lb=0.0
            )
            _add_fairness_constraints2(model, x, fc, dfa2)
    for fa_type, fc in enumerate(fa_hard):
        if len(fc) == 0:
            continue
        if fa_type == 0:  # FA2 constraints
            _add_fairness_constraints2(model, x, fc, None)
    return dfa2


def _add_fairness_constraints2(model, x, fc, dfa2: None):
    for l, fc2 in enumerate(fc):
        for i, j in combinations(fc2["teams"], 2):
            model.addConstrs(
                (
                    gp.quicksum(
                        x.sum(i, "*", k) - x.sum(j, "*", k) for k in range(slot)
                    )
                    - (dfa2[i, j, l] if dfa2 is not None else 0)
                    <= fc2["intp"]
                    for slot in fc2["slots"]
                ),
                name=f"fa2_{'soft' if dfa2 is not None else 'hard'}_1[{i},{j}]",
            )
            model.addConstrs(
                (
                    gp.quicksum(
                        x.sum(j, "*", k) - x.sum(i, "*", k) for k in range(slot)
                    )
                    - (dfa2[i, j, l] if dfa2 is not None else 0)
                    <= fc2["intp"]
                    for slot in fc2["slots"]
                ),
                name=f"fa2_{'soft' if dfa2 is not None else 'hard'}_2[{i},{j}]",
            )


def add_game_constraints(
    ga_hard: list[list[dict]],
    ga_soft: list[list[dict]],
    model: gp.Model,
    x: gp.tupledict,
) -> np.array:
    """This function extracts the game constraints

    Args:
        game_constraints (OrderedDict): dictionary containing each type of game constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard game constraints
    """
    dga1 = gp.tuplelist()
    for ga_type, gc in enumerate(ga_soft):
        if len(gc) == 0:
            continue
        if ga_type == 0:  # GA1 constraints
            dga1 = model.addVars(len(gc), 2, name="dga1", vtype=GRB.INTEGER, lb=0.0)
            _add_game_constraints1(model, x, gc, dga1)
    for ga_type, gc in enumerate(ga_hard):
        if len(gc) == 0:
            continue
        if ga_type == 0:  # GA1 constraints
            _add_game_constraints1(model, x, gc, None)
    return dga1


def _add_game_constraints1(model, x, gc, dga1: None):
    for l, ga1 in enumerate(gc):
        rhs = gp.quicksum(
            x.sum(i, j, k) for i, j in ga1["meetings"] for k in ga1["slots"]
        )
        model.addConstr(
            rhs - (dga1[l, 0] if dga1 is not None else 0) <= ga1["max"],
            name=f"ga1_{'soft' if dga1 is not None else 'hard'}_max[{l}]",
        )
        model.addConstr(
            rhs - (dga1[l, 1] if dga1 is not None else 0) >= ga1["min"],
            name=f"ga1_{'soft' if dga1 is not None else 'hard'}_min[{l}]",
        )


def add_break_constraints(
    br_hard: list[list[dict]],
    br_soft: list[list[dict]],
    model: gp.Model,
    h: gp.tupledict,
    a: gp.tupledict,
    n_teams: int,
) -> np.array:
    """This function extracts the break constraints

    Args:
        break_constraints (OrderedDict): dictionary containing each type of break constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard break constraints
    """
    dbr1, dba2 = gp.tuplelist(), gp.tuplelist()
    for br_type, bc in enumerate(br_soft):
        if len(bc) == 0:
            continue
        if br_type == 0:  # BA1 constraints
            dbr1 = model.addVars(
                n_teams, len(bc), name="dba1", vtype=GRB.INTEGER, lb=0.0
            )
            _add_break_constraints1(model, h, a, bc, dbr1)
        else:  # BA2 constraints
            dba2 = model.addVars(len(bc), name="dba2", vtype=GRB.INTEGER, lb=0.0)
            _add_break_constraints2(model, h, a, bc, dba2)
    for br_type, bc in enumerate(br_hard):
        if len(bc) == 0:
            continue
        if br_type == 0:  # BA1 constraints
            _add_break_constraints1(model, h, a, bc, None)
        else:  # BA2 constraints
            _add_break_constraints2(model, h, a, bc, None)
    return dbr1, dba2


def _add_break_constraints1(model, h, a, bc, dbr1: None):
    for j, br1 in enumerate(bc):
        if br1["mode2"] == "H":
            for i in br1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(h[i, k] for k in br1["slots"])
                        - (dbr1[i, j] if dbr1 is not None else 0)
                        <= br1["intp"]
                    ),
                    name=f"br1_{'soft' if dbr1 is not None else 'hard'}_home[{j}]",
                )
        elif br1["mode2"] == "A":
            for i in br1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(a[i, k] for k in br1["slots"])
                        - (dbr1[i, j] if dbr1 is not None else 0)
                        <= br1["intp"]
                    ),
                    name=f"br1_{'soft' if dbr1 is not None else 'hard'}_away[{j}]",
                )
        else:
            for i in br1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(h[i, k] + a[i, k] for k in br1["slots"])
                        - (dbr1[i, j] if dbr1 is not None else 0)
                        <= br1["intp"]
                    ),
                    name=f"br1_{'soft' if dbr1 is not None else 'hard'}_both[{j}]",
                )


def _add_break_constraints2(model, h, a, bc, dbr2: None):
    for j, br2 in enumerate(bc):
        model.addConstr(
            (
                gp.quicksum(
                    a[i, k] + h[i, k] for i in br2["teams"] for k in br2["slots"]
                )
                - (dbr2[j] if dbr2 is not None else 0)
                <= br2["intp"]
            ),
            name=f"br2_{'soft' if dbr2 is not None else 'hard'}_both[{j}]",
        )


def add_capacity_constraints(
    ca_hard: list[list[dict]],
    ca_soft: list[list[dict]],
    model: gp.Model,
    x: gp.tupledict,
    n_teams: int,
    n_slots: int,
) -> np.array:
    """This function extracts the capacity constraints

    Args:
        capacity_constraints (OrderedDict): dictionary containing each type of capacity constraints

    Returns:
        Tuple[list, list]: separated lists of soft and hard capacity constraints
    """
    dca1, dca2, dca3, dca4 = (
        gp.tuplelist(),
        gp.tuplelist(),
        gp.tuplelist(),
        gp.tuplelist(),
    )
    for ca_type, cc in enumerate(ca_soft):
        if len(cc) == 0:
            continue
        if ca_type == 0:  # CA1 constraints
            dca1 = model.addVars(
                n_teams, len(cc), name="dca1", vtype=GRB.INTEGER, lb=0.0
            )
            _add_capacity_constraints1(model, x, cc, dca1)
        elif ca_type == 1:  # CA2 constraints
            dca2 = model.addVars(
                n_teams, len(cc), name="dca2", vtype=GRB.INTEGER, lb=0.0
            )
            _add_capacity_constraints2(model, x, cc, dca2)
        elif ca_type == 2:  # CA3 constraints
            dca3 = model.addVars(
                n_teams, n_slots, len(cc), name="dca3", vtype=GRB.INTEGER, lb=0.0
            )
            _add_capacity_constraints3(model, x, cc, n_slots, dca3)
        else:
            dca4 = model.addVars(
                n_slots, len(cc), name="dca4", vtype=GRB.INTEGER, lb=0.0
            )
            _add_capacity_constraints4(model, x, cc, n_slots, dca4)
    for ca_type, cc in enumerate(ca_hard):
        if len(cc) == 0:
            continue
        if ca_type == 0:  # CA1 constraints
            _add_capacity_constraints1(model, x, cc, None)
        elif ca_type == 1:  # CA2 constraints
            _add_capacity_constraints2(model, x, cc, None)
        elif ca_type == 2:  # CA3 constraints
            _add_capacity_constraints3(model, x, cc, n_slots, None)
        else:
            _add_capacity_constraints4(model, x, cc, n_slots, None)
    return dca1, dca2, dca3, dca4


def _add_capacity_constraints1(model, x, cc, dca1: None):
    for j, ca1 in enumerate(cc):
        if ca1["mode"] == "H":
            for i in ca1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(x.sum(i, "*", k) for k in ca1["slots"])
                        - (dca1[i, j] if dca1 is not None else 0)
                        <= ca1["max"]
                    ),
                    name=f"ca1_{'soft' if dca1 is not None else 'hard'}_home[{j}]",
                )
        elif ca1["mode"] == "A":
            for i in ca1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(x.sum("*", i, k) for k in ca1["slots"])
                        - (dca1[i, j] if dca1 is not None else 0)
                        <= ca1["max"]
                    ),
                    name=f"ca1_{'soft' if dca1 is not None else 'hard'}_away[{j}]",
                )
        else:
            for i in ca1["teams"]:
                model.addConstr(
                    (
                        gp.quicksum(
                            x.sum(i, "*", k) + x.sum("*", i, k) for k in ca1["slots"]
                        )
                        - (dca1[i, j] if dca1 is not None else 0)
                        <= ca1["max"]
                    ),
                    name=f"ca1_{'soft' if dca1 is not None else 'hard'}_both[{j}]",
                )


def _add_capacity_constraints2(model, x, cc, dca2: None):
    for l, ca2 in enumerate(cc):
        if ca2["mode1"] == "H":
            for i in ca2["teams1"]:
                model.addConstr(
                    (
                        gp.quicksum(
                            x.sum(i, j, k) for k in ca2["slots"] for j in ca2["teams2"]
                        )
                        - (dca2[i, l] if dca2 is not None else 0)
                        <= ca2["max"]
                    ),
                    name=f"ca2_{'soft' if dca2 is not None else 'hard'}_home[{l}]",
                )
        elif ca2["mode1"] == "A":
            for i in ca2["teams1"]:
                model.addConstr(
                    (
                        gp.quicksum(
                            x.sum(j, i, k) for k in ca2["slots"] for j in ca2["teams2"]
                        )
                        - (dca2[i, l] if dca2 is not None else 0)
                        <= ca2["max"]
                    ),
                    name=f"ca2_{'soft' if dca2 is not None else 'hard'}_away[{l}]",
                )
        else:
            for i in ca2["teams1"]:
                model.addConstr(
                    (
                        gp.quicksum(
                            x.sum(i, j, k) + x.sum(j, i, k)
                            for k in ca2["slots"]
                            for j in ca2["teams2"]
                        )
                        - (dca2[i, l] if dca2 is not None else 0)
                        <= ca2["max"]
                    ),
                    name=f"ca2_{'soft' if dca2 is not None else 'hard'}_both[{l}]",
                )


def _add_capacity_constraints3(model, x, cc, n_slots, dca3: None):
    for l, ca3 in enumerate(cc):
        if ca3["mode1"] == "H":
            for i in ca3["teams1"]:
                for s in range(ca3["intp"], n_slots + 1):
                    model.addConstr(
                        gp.quicksum(
                            x.sum(i, j, k)
                            for k in range(s - ca3["intp"], s)
                            for j in ca3["teams2"]
                        )
                        - (dca3[i, s - ca3["intp"], l] if dca3 is not None else 0)
                        <= ca3["max"],
                        name=f"ca3_{'soft' if dca3 is not None else 'hard'}_home[{l}]",
                    )
        elif ca3["mode1"] == "A":
            for i in ca3["teams1"]:
                for s in range(ca3["intp"], n_slots + 1):
                    model.addConstr(
                        gp.quicksum(
                            x.sum(j, i, k)
                            for k in range(s - ca3["intp"], s)
                            for j in ca3["teams2"]
                        )
                        - (dca3[i, s - ca3["intp"], l] if dca3 is not None else 0)
                        <= ca3["max"],
                        name=f"ca3_{'soft' if dca3 is not None else 'hard'}_away[{l}]",
                    )
        else:
            for i in ca3["teams1"]:
                for s in range(ca3["intp"], n_slots + 1):
                    model.addConstr(
                        gp.quicksum(
                            x.sum(i, j, k) + x.sum(j, i, k)
                            for k in range(s - ca3["intp"], s)
                            for j in ca3["teams2"]
                        )
                        - (dca3[i, s - ca3["intp"], l] if dca3 is not None else 0)
                        <= ca3["max"],
                        name=f"ca3_{'soft' if dca3 is not None else 'hard'}_both[{l}]",
                    )


def _add_capacity_constraints4(model, x, cc, n_slots, dca4: None):
    for l, ca4 in enumerate(cc):
        if ca4["mode1"] == "H":
            if ca4["mode2"] == "GLOBAL":
                model.addConstr(
                    gp.quicksum(
                        x.sum(i, j, k)
                        for i, j, k in product(
                            ca4["teams1"], ca4["teams2"], ca4["slots"]
                        )
                    )
                    - (dca4[0, l] if dca4 is not None else 0)
                    <= ca4["max"],
                    name=f"ca4_{'soft' if dca4 is not None else 'hard'}_home_g[{l}]",
                )
            else:
                for k in ca4["slots"]:
                    model.addConstr(
                        gp.quicksum(
                            x.sum(i, j, k)
                            for i, j in product(ca4["teams1"], ca4["teams2"])
                        )
                        - (dca4[k, l] if dca4 is not None else 0)
                        <= ca4["max"],
                        name=f"ca4_{'soft' if dca4 is not None else 'hard'}_home_e[{l}]",
                    )
        elif ca4["mode1"] == "A":
            if ca4["mode2"] == "GLOBAL":
                model.addConstr(
                    gp.quicksum(
                        x.sum(j, i, k)
                        for i, j, k in product(
                            ca4["teams1"], ca4["teams2"], ca4["slots"]
                        )
                    )
                    - (dca4[0, l] if dca4 is not None else 0)
                    <= ca4["max"],
                    name=f"ca4_{'soft' if dca4 is not None else 'hard'}_away_g[{l}]",
                )
            else:
                for k in ca4["slots"]:
                    model.addConstr(
                        gp.quicksum(
                            x.sum(j, i, k)
                            for i, j in product(ca4["teams1"], ca4["teams2"])
                        )
                        - (dca4[k, l] if dca4 is not None else 0)
                        <= ca4["max"],
                        name=f"ca4_{'soft' if dca4 is not None else 'hard'}_away_e[{l}]",
                    )
        elif ca4["mode2"] == "GLOBAL":
            model.addConstr(
                gp.quicksum(
                    x.sum(i, j, k) + x.sum(j, i, k)
                    for i, j, k in product(ca4["teams1"], ca4["teams2"], ca4["slots"])
                )
                - (dca4[0, l] if dca4 is not None else 0)
                <= ca4["max"],
                name=f"ca4_{'soft' if dca4 is not None else 'hard'}_both_g[{l}]",
            )
        else:
            for k in ca4["slots"]:
                model.addConstr(
                    gp.quicksum(
                        x.sum(i, j, k) + x.sum(j, i, k)
                        for i, j in product(ca4["teams1"], ca4["teams2"])
                    )
                    - (dca4[k, l] if dca4 is not None else 0)
                    <= ca4["max"],
                    name=f"ca4_{'soft' if dca4 is not None else 'hard'}_both_e[{l}]",
                )


def add_variables(
    model: gp.Model, n_teams: int, n_slots: int
) -> tuple[gp.tuplelist, gp.tuplelist, gp.tuplelist]:
    """Adds the variables to the model

    Args:
        model (gp.Model): Gurobi model
        n_teams (int): number of teams
        n_slots (int): number of slots
    Returns:
        tuple[gp.tuplelist, gp.tuplelist]: x, h and a variables
    """
    x = model.addVars(
        n_teams,
        n_teams,
        n_slots,
        vtype=GRB.BINARY,
        name="x",
    )
    h = model.addVars(
        n_teams,
        n_slots,
        vtype=GRB.BINARY,
        name="h",
    )
    a = model.addVars(
        n_teams,
        n_slots,
        vtype=GRB.BINARY,
        name="a",
    )
    return x, h, a
