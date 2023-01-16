import numpy as np
from Utils import cost_function, feasibility_check


def update_weights(weights, thetas, scores, r=0.2):
    new_weights = []
    for w, weight in enumerate(weights):
        if thetas[w] > 0:
            new_weights += [(weight * (1 - r) + r * (scores[w] / thetas[w]))]
        else:
            new_weights += [weight]
    print(new_weights)
    return new_weights


def normalize_weights(weights):
    normalized = weights / np.sum(weights)
    less = normalized < 0.05
    normalized[less] = 0.05
    normalized[~less] = (
        (1 - (0.05 * sum(less))) * normalized[~less] / sum(normalized[~less])
    )
    return normalized


def ALNS(
    init_sol,
    init_cost,
    probability,
    operators,
    prob,
    rng,
    T_f=0.1,
    warm_up=100,
):
    feas_sols = []
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost
    r = 0.2
    operators_len_range = range(len(operators))
    scores = [0 for i in operators_len_range]
    thetas = [0 for i in operators_len_range]
    last_improvement = 0
    best_cost = cost_incumb
    delta = [0]
    non_imp_count = 0
    weights = probability
    ws = np.array([])
    for itr in range(10000):
        if itr == warm_up and np.mean(delta) == 0:
            print(weights)
            warm_up += 100
        if itr < warm_up:
            op_id = rng.choice(operators_len_range, replace=True, p=probability)
            operator = operators[op_id]
            thetas[op_id] += 1
            new_sol = operator(
                incumbent,
                rng,
                prob,
            )
            new_cost = cost_function(new_sol, prob)
            delta_E = new_cost - cost_incumb
            if delta_E >= 0:
                non_imp_count += 1
            c, feasiblity = feasibility_check(new_sol, prob)
            if feasiblity and delta_E < 0:
                scores[op_id] += 1
                incumbent = new_sol
                cost_incumb = new_cost
                if cost_incumb < best_cost:
                    scores[op_id] += 2
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr
                    non_imp_count = 0
            elif feasiblity:
                feas_sols.append(new_sol)
                if rng.uniform() < 0.8:
                    incumbent = new_sol
                    cost_incumb = new_cost
                if delta_E > 0:
                    delta += [delta_E]
        else:
            if itr == warm_up:
                delta_avg = np.mean(delta[1:])
                T_0 = -delta_avg / np.log(0.8)
                alpha = 0.9995  # np.power((T_f/T_0), (1/(10000-warm_up)))
                T = T_0
                # Ts = [T]
                # Ps = [np.exp(-delta_avg / T)]
            # if non_imp_count > 300:
            #     ops = [multi_ins_rand_worst_remove]
            #     ops_len = len(ops)
            #     for i in range(1, 31):
            #         rm_size = rng.choice(np.arange(2, 5))
            #         op_id = rng.choice(ops_len)
            #         escape = ops[op_id]  # operators[-1]
            #         incumbent, costs, features, call_costs = escape(
            #             incumbent,
            #             copy_costs(costs),
            #             copy_features(features),
            #             copy_call_costs(call_costs),
            #             rng,
            #             prob,
            #             rm_size,
            #         )
            #         new_cost = sum(costs)
            #         if new_cost < best_cost:
            #             scores[op_id + 6] += 4
            #             best_call_costs = call_costs
            #             best_sol = incumbent
            #             best_cost = new_cost
            #             last_improvement = itr
            #             break
            #     non_imp_count = 0
            #     thetas[op_id + 6] += 1
            #     continue

            if (itr - warm_up) % 100 == 0:
                if itr - warm_up == 0:
                    ws = np.append(ws, [weights])
                weights = update_weights(weights, thetas, scores, r)

                scores = [0 for i in operators_len_range]
                thetas = [0 for i in operators_len_range]
                weights = normalize_weights(weights)
                ws = np.append(ws, [weights])

            op_id = rng.choice(operators_len_range, replace=True, p=weights)
            operator = operators[op_id]
            thetas[op_id] += 1
            new_sol = operator(
                incumbent,
                rng,
                prob,
            )

            new_cost = cost_function(new_sol, prob)
            delta_E = new_cost - cost_incumb
            if delta_E >= 0:
                non_imp_count += 1
            c, feasiblity = feasibility_check(new_sol, prob)

            if feasiblity and delta_E < 0:
                scores[op_id] += 1
                incumbent = new_sol
                cost_incumb = new_cost
                if cost_incumb < best_cost:
                    scores[op_id] += 2
                    best_sol = incumbent
                    best_cost = cost_incumb
                    last_improvement = itr

            elif feasiblity:
                feas_sols.append(new_sol)
                prbb = np.exp(-delta_E / T)
                if rng.uniform() < prbb:
                    incumbent = new_sol
                    cost_incumb = new_cost

            T *= alpha
    return best_sol, best_cost, last_improvement, ws, feas_sols
