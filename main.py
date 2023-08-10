from contextlib import redirect_stdout
import time

from app.utils import (
    cost_function,
    random_init_sol,
    dummy_init_sol,
    set_initial_solution,
    set_initial_solution_json,
)
from app.load import (
    load_init_sol_json,
    load_problem,
    load_sol_from_ampl_output,
    load_solution,
)
from app.utils import (
    write_sol_xml,
    fix_less_teams_and_optimize_random,
    fix_less_weeks_and_optimize_random,
    fix_more_teams_and_optimize_random,
    fix_more_weeks_and_optimize_random,
)

# from time import time
# from app.ALNS import ALNS
# from app.Operators import (
#     one_week_swap,
#     multi_week_swap,
#     one_game_flip,
#     multi_game_flip,
#     set_week_for_game,
# )

# import numpy as np
# import pandas as pd
from app.model import create_model

# from itertools import combinations, product
from app.named_tuples import Solution
import os
import gurobipy as gp

if __name__ == "__main__":
    # import required module

    # assign directory
    directory = "Instances"
    settings = {
        1: {"mip_focus": 3, "timelimit": 600},
        2: {"mip_focus": 0, "timelimit": 600},
        3: {"mip_focus": 3, "timelimit": 300},
        4: {"mip_focus": 0, "timelimit": 300},
    }
    # iterate over files in
    # that directory
    for key, setting in settings.items():
        for xml_file in os.listdir(directory):
            filename = xml_file[:-4]
            # if not os.path.exists("lp_models"):
            f = os.path.join(directory, f"{filename}.xml")
            problem = load_problem(f)
            model, variables = create_model(problem)
            # os.makedirs("lp_models", exist_ok=True)
            # model.write(f"lp_models\\{filename}.lp")
            # else:
            # model = gp.read(f"./lp_models/{filename}.lp")
            if not os.path.exists(
                f"results/MIP{setting['mip_focus']}_TIME{setting['timelimit']}/{filename}"
            ):
                os.makedirs(
                    f"results/MIP{setting['mip_focus']}_TIME{setting['timelimit']}/{filename}"
                )
            # if not os.path.exists(f"results/{filename}"):
            #     os.makedirs(f"results/{filename}")
            if os.path.exists(
                f"results/MIP{setting['mip_focus']}_TIME{setting['timelimit']}/{filename}/logs.txt"
            ):
                print(f"Already solved {filename}")
                continue
            start = time.time()
            with open(
                f"results/MIP{setting['mip_focus']}_TIME{setting['timelimit']}/{filename}/logs.txt",
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
                        model.setObjective(0.0)
                        model.setParam("TimeLimit", 60)
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
                        model.setParam("MIPFocus", 1)
                        model.setParam("TimeLimit", 60)
                        model.setObjective(0.0)
                        model.optimize()
                        print("-----------------------------------")

                        if model.status == gp.GRB.OPTIMAL:
                            print("Initial optimal solution found.")
                            write_sol_xml(model, f"results/{filename}/initial_sol.xml")
                            model.write(
                                f"results/MIP{setting['mip_focus']}_TIME{setting['time_limit']}/{filename}/initial_sol.json"
                            )
                    model.setObjective(original_objective)
                    if key in [1, 2]:
                        fix_less_weeks_and_optimize_random(
                            model,
                            variables,
                            problem["n_slots"],
                            filename,
                            time_limit=setting["time_limit"],
                            mip_focus=setting["mip_focus"],
                        )
                        fix_less_teams_and_optimize_random(
                            model,
                            variables,
                            problem["n_teams"],
                            filename,
                            time_limit=setting["time_limit"],
                            mip_focus=setting["mip_focus"],
                        )
                    elif key in [4, 3]:
                        fix_more_weeks_and_optimize_random(
                            model,
                            variables,
                            problem["n_slots"],
                            filename,
                            time_limit=setting["time_limit"],
                            mip_focus=setting["mip_focus"],
                        )
                        fix_more_teams_and_optimize_random(
                            model,
                            variables,
                            problem["n_teams"],
                            filename,
                            time_limit=setting["time_limit"],
                            mip_focus=setting["mip_focus"],
                        )
                    print("-----------------------------------")
                    print(f"it took {time.time() - start} seconds")
                # else:
                #     print(
                #         "Optimization did not converge to an initial optimal solution."
                #     )
