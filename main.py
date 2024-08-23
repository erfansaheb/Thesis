from contextlib import redirect_stdout
import time

from app.utils import (
    set_initial_solution,
    set_initial_solution_json,
)
from app.load import (
    load_init_sol_json,
    load_problem,
    load_sol_from_ampl_output,
)
from app.utils import (
    write_sol_xml,
    fix_less_teams_and_optimize_random,
    fix_less_weeks_and_optimize_random,
    fix_more_teams_and_optimize_random,
    fix_more_weeks_and_optimize_random,
    fix_more_teams_and_optimize_costliest,
    fix_more_weeks_and_optimize_costliest,
    fix_less_teams_and_optimize_least_costliest,
    fix_less_weeks_and_optimize_least_costliest,
)
import numpy as np
from app.model import create_model


import os
import gurobipy as gp


def neighborhood_size(model, n_free_slots, counter):
    if model.status == gp.GRB.OPTIMAL and model.Runtime < 60:
        counter += 1
        if counter == 10:
            n_free_slots += 1
            counter = 0
    elif model.status == gp.GRB.TIME_LIMIT:
        n_free_slots -= 1
        counter = 0
    else:
        counter = 0
    return n_free_slots, counter


if __name__ == "__main__":
    # import required module

    # assign directory
    directory = "Instances"
    settings = {
        # 1: {"mip_focus": 3, "timelimit": 600},
        # 2: {"mip_focus": 0, "timelimit": 600},
        3: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "random",
            "num_const": "weeks_fix",
        },
        4: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "random",
            "num_const": "teams_fix",
        },
        5: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "costliest",
            "num_const": "weeks_fix",
            "func": "all",
        },
        6: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "costliest",
            "num_const": "teams_fix",
            "func": "all",
        },
        7: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "costliest",
            "num_const": "weeks_fix",
            "func": "ones",
        },
        8: {
            "mip_focus": 3,
            "timelimit": 600,
            "type": "costliest",
            "num_const": "teams_fix",
            "func": "ones",
        },
    }
    for xml_file in os.listdir(directory):
        for key, setting in settings.items():
            rng = np.random.default_rng(12345)
            filename = xml_file[:-4]
            f = os.path.join(directory, f"{filename}.xml")
            problem = load_problem(f)
            model, variables = create_model(problem)
            folder = f"results/dynamic_MIP{setting['mip_focus']}_TIME{setting['timelimit']}_{setting['type']}_{setting.get('num_const','')}_{setting.get('func','')}/{filename}/"
            if not os.path.exists(f"{folder}"):
                os.makedirs(f"{folder}")
            if not os.path.exists(f"{folder}sols/"):
                os.makedirs(f"{folder}sols/")
            if os.path.exists(f"{folder}/logs.txt"):
                print(f"Already solved {filename}")
                continue
            start = time.time()
            with open(
                f"{folder}/logs.txt",
                "w",
            ) as log_file:
                with redirect_stdout(log_file):
                    original_objective = model.getObjective()
                    if os.path.exists(f"Initial_sols/{filename}.json"):
                        variables_init = load_init_sol_json(
                            f"Initial_sols/{filename}.json"
                        )
                        set_initial_solution_json(model, variables_init)
                        model.setParam("MIPFocus", 1)
                        # model.setObjective(0.0)
                        model.setParam("TimeLimit", 10)
                        model.optimize()
                        print("-----------------------------------")
                    elif os.path.exists(
                        f"Feasible_sols_main_model/solution_{filename}.txt"
                    ):
                        variables_init = load_sol_from_ampl_output(
                            f"Feasible_sols_main_model/solution_{filename}.txt"
                        )
                        model.update()
                        set_initial_solution(model, variables_init)
                        model.setParam("MIPFocus", 1)
                        model.setParam("TimeLimit", 60)
                        model.optimize()
                        print("-----------------------------------")
                    else:
                        continue
                        # model.setParam("MIPFocus", 1)
                        # model.setParam("TimeLimit", 60)
                        # model.setObjective(0.0)
                        # model.optimize()
                        # print("-----------------------------------")

                        # if model.status == gp.GRB.OPTIMAL:
                        #     print("Initial optimal solution found.")
                        #     write_sol_xml(model, f"results/{filename}/initial_sol.xml")
                        #     model.write(
                        #         f"results/MIP{setting['mip_focus']}_TIME{setting['time_limit']}/{filename}/initial_sol.json"
                        #     )
                    # model.setObjective(original_objective)
                    # model.update()
                    model.setParam("MIPFocus", setting["mip_focus"])
                    # model.setParam("TimeLimit", setting["timelimit"])
                    if key in [1, 2]:
                        fix_less_weeks_and_optimize_random(
                            model,
                            variables,
                            problem["n_slots"],
                            folder,
                        )
                        fix_less_teams_and_optimize_random(
                            model,
                            variables,
                            problem["n_teams"],
                            folder,
                        )
                    else:
                        index = 1
                        time_limit = setting["timelimit"]
                        model_vars = model.getVars()
                        obj_variables = [var for var in model_vars if var.Obj > 0]
                        n_free_slots = problem["n_slots"] // 4
                        n_free_teams = problem["n_teams"] // 3
                        counter = 0
                        while time_limit > 0:
                            start_here = time.time()
                            if key == 3:
                                fix_more_weeks_and_optimize_random(
                                    model,
                                    variables,
                                    problem["n_slots"],
                                    folder,
                                    time_limit=max(min(300, time_limit), 5),
                                    rng=rng,
                                    index=index,
                                    n_free_slots=n_free_slots,
                                )
                                n_free_slots, counter = neighborhood_size(
                                    model, n_free_slots, counter
                                )
                            elif key == 4:
                                fix_more_teams_and_optimize_random(
                                    model,
                                    variables,
                                    problem["n_teams"],
                                    folder,
                                    time_limit=max(min(300, time_limit), 5),
                                    n_free_teams=n_free_teams,
                                    rng=rng,
                                    index=index,
                                )
                                n_free_teams, counter = neighborhood_size(
                                    model, n_free_teams, counter
                                )
                            elif key in [5, 7]:
                                fix_more_weeks_and_optimize_costliest(
                                    model,
                                    variables,
                                    problem["n_slots"],
                                    folder,
                                    n_free_slots=n_free_slots,
                                    time_limit=max(min(300, time_limit), 5),
                                    rng=rng,
                                    index=index,
                                    selection_criteria=setting["func"],
                                    obj_variables=obj_variables,
                                )
                                n_free_slots, counter = neighborhood_size(
                                    model, n_free_slots, counter
                                )
                            elif key in [6, 8]:
                                fix_more_teams_and_optimize_costliest(
                                    model,
                                    variables,
                                    problem["n_teams"],
                                    folder,
                                    n_free_teams=n_free_teams,
                                    time_limit=max(min(300, time_limit), 5),
                                    rng=rng,
                                    index=index,
                                    selection_criteria=setting["func"],
                                    obj_variables=obj_variables,
                                )
                                n_free_teams, counter = neighborhood_size(
                                    model, n_free_teams, counter
                                )
                            time_limit -= time.time() - start_here
                            index += 1

                    print("-----------------------------------")
                    print(f"it took {time.time() - start} seconds")
