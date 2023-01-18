import numpy as np
from Utils import cost_function, feasibility_check


def update_weights(
    weights: list[float], thetas: list[int], scores: list[int], r: float = 0.2
) -> list[float]:
    """This function updates the weights using the adaptive weight adjustment method

    Args:
        weights (list[float]): list of weights
        thetas (list[int]): a list containing number of times each operator is selected
        scores (list[int]): a list containing the performance of each operator
        r (float, optional): fixed constant. Defaults to 0.2.

    Returns:
        list[float]: list of updated weights
    """
    new_weights = []
    for w, weight in enumerate(weights):
        if thetas[w] > 0:
            new_weights += [(weight * (1 - r) + r * (scores[w] / thetas[w]))]
        else:
            new_weights += [weight]
    return new_weights


def normalize_weights(weights: np.array(float), threshold: float = 0.05) -> np.array:
    """This function normalizes the weights to prevent any of them from getting zero

    Args:
        weights (np.array): updated weights

    Returns:
        np.array: normalized weights
    """
    normalized = weights / np.sum(weights)
    less = normalized < threshold
    normalized[less] = threshold
    normalized[~less] = (
        (1 - (threshold * sum(less))) * normalized[~less] / sum(normalized[~less])
    )
    return normalized


def ALNS(
    init_sol: np.array,
    init_cost: int,
    probability: list[float],
    operators: list,
    prob: dict,
    rng: np.random.Generator,
    T_f: float = 0.1,
    warm_up: int = 100,
) -> tuple[np.array, int, int, list, list]:
    """This is an implementation of ALNS algorithm for my problem

    Args:
        init_sol (np.array): initial solution, 2d numpy array
        init_cost (int): cost of initial solution
        probability (list[float]): probabilities of choosing each operator
        operators (list): list of operators
        prob (dict): dictionary related to problem
        rng (np.random.Generator): random generator (seeded!)
        T_f (float, optional): final temperature. Defaults to 0.1.
        warm_up (int, optional): number of iteration for warm-up phase. Defaults to 100.

    Returns:
        tuple[np.array, int, int, list, list]: best solution, best cost, last iteration of improvement, list of weights, list of feasible solutions found
    """
    feas_sols = []
    incumbent = init_sol
    best_sol = init_sol
    cost_incumb = init_cost
    r = 0.2
    operators_len_range = range(len(operators))
    scores = [0 for _ in operators_len_range]
    thetas = [0 for _ in operators_len_range]
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
            op_id, operator = choose_operator(
                probability, operators, rng, operators_len_range, thetas
            )
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

                scores = [0 for _ in operators_len_range]
                thetas = [0 for _ in operators_len_range]
                weights = normalize_weights(weights)
                ws = np.append(ws, [weights])

            op_id, operator = choose_operator(
                weights, operators, rng, operators_len_range, thetas
            )
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


def choose_operator(
    weights: list[float],
    operators: list,
    rng: np.random.Generator,
    operators_len_range: int,
    thetas: list[int],
) -> tuple:
    """This function takes the list of operators and their weights and returns the selected operator

    Args:
        weights (list[float]): list of operator selection weights
        operators (list): list of operators
        rng (np.random.Generator): random generator(seeded)
        operators_len_range (int): number of operators
        thetas (list[int]): list of number of times each operator has been selected

    Returns:
        tuple: selected operator and its index
    """
    op_id = rng.choice(operators_len_range, replace=True, p=weights)
    operator = operators[op_id]
    thetas[op_id] += 1
    return op_id, operator
