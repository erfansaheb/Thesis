import numpy as np
from app.utils import cost_function, feasibility_check
from typing import Callable

from app.named_tuples import Solution


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
    init_sol: Solution,
    probability: list[float],
    operators: list[Callable],
    escape_op_ids: list[int],
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
        escape_op_ids (list): list of escape operator ids
        prob (dict): dictionary related to problem
        rng (np.random.Generator): random generator (seeded!)
        T_f (float, optional): final temperature. Defaults to 0.1.
        warm_up (int, optional): number of iteration for warm-up phase. Defaults to 100.

    Returns:
        tuple[np.array, int, int, list, list]: best solution, best cost, last iteration of improvement, list of weights, list of feasible solutions found
    """
    (
        feas_sols,
        incumbent,
        r,
        operators_len_range,
        scores,
        thetas,
        best_sol,
        delta,
        last_improvement,
        non_imp_count,
        weights,
        ws,
    ) = initiate_ALNS(init_sol, probability, operators)
    for itr in range(1000):
        if itr == warm_up and np.mean(delta) == 0:
            warm_up = _update_warm_up_number(warm_up)
        if itr < warm_up:
            best_sol, scores, thetas, last_improvement, delta = do_iteration(
                operators,
                prob,
                rng,
                feas_sols,
                incumbent,
                operators_len_range,
                scores,
                thetas,
                best_sol,
                non_imp_count,
                last_improvement,
                weights,
                itr,
                None,
                delta,
                phase="warm_up",
            )
        else:
            if itr == warm_up:
                alpha, T = calc_alpha(delta, warm_up, T_f)
            if non_imp_count > 300:
                (
                    incumbent,
                    scores,
                    best_sol,
                    last_improvement,
                    non_imp_count,
                ) = apply_escape(
                    operators,
                    escape_op_ids,
                    prob,
                    rng,
                    incumbent,
                    thetas,
                    scores,
                    itr,
                )

            if (itr - warm_up) % 100 == 0:
                scores, thetas, weights, ws = do_update_weights(
                    warm_up, r, operators_len_range, scores, thetas, weights, ws, itr
                )

            best_sol, scores, thetas, last_improvement, _ = do_iteration(
                operators,
                prob,
                rng,
                feas_sols,
                incumbent,
                operators_len_range,
                scores,
                thetas,
                best_sol,
                non_imp_count,
                last_improvement,
                weights,
                itr,
                T,
            )

            T *= alpha
    return best_sol, last_improvement, ws, feas_sols


def do_iteration(
    operators,
    prob,
    rng,
    feas_sols,
    incumbent,
    operators_len_range,
    scores,
    thetas,
    best_sol,
    non_imp_count,
    last_improvement,
    weights,
    itr,
    T,
    delta=None,
    phase=None,
):
    op_id, operator, thetas = choose_operator(
        weights, operators, rng, operators_len_range, thetas
    )
    new_sol, delta_E = apply_operator(prob, rng, incumbent, operator)
    if delta_E >= 0:
        non_imp_count += 1
    c, feasiblity = "feasible", True  # feasibility_check(new_sol, prob)
    if feasiblity and delta_E < 0:
        scores[op_id] += 1
        incumbent = new_sol.copy()
        if incumbent.total_cost < best_sol.total_cost:
            (
                scores,
                best_sol,
                last_improvement,
                non_imp_count,
            ) = update_bests(incumbent, scores, itr, op_id)
    elif feasiblity:
        feas_sols.append(new_sol)
        prbb = calc_acc_prb(delta_E, T, phase)
        if rng.uniform() < prbb:
            incumbent = new_sol.copy()
        if phase == "warm_up":
            delta += [delta_E]

    return best_sol, scores, thetas, last_improvement, delta


def calc_acc_prb(delta_E: int, T: float, phase: str = None) -> float:
    return 0.8 if phase == "warm_up" else np.exp(-delta_E / T)


def do_update_weights(
    warm_up, r, operators_len_range, scores, thetas, weights, ws, itr
):
    if itr - warm_up == 0:
        ws = np.append(ws, [weights])
    weights = update_weights(weights, thetas, scores, r)

    scores = [0 for _ in operators_len_range]
    thetas = [0 for _ in operators_len_range]
    weights = normalize_weights(weights)
    ws = np.append(ws, [weights])
    return scores, thetas, weights, ws


def apply_escape(
    operators,
    escape_op_ids,
    prob,
    rng,
    incumbent,
    thetas,
    scores,
    itr,
):
    ops_len = len(escape_op_ids)
    for _ in range(1, 31):
        op_id = rng.choice(ops_len)
        escape = operators[escape_op_ids[op_id]]
        new_sol, _ = apply_operator(prob, rng, incumbent, escape)
        if new_sol.total_cost < incumbent.total_cost:
            (
                scores,
                best_sol,
                last_improvement,
                non_imp_count,
            ) = update_bests(incumbent, scores, itr, escape_op_ids[op_id])
            break
    non_imp_count = 0
    thetas[escape_op_ids[op_id]] += 1
    return incumbent, scores, best_sol, last_improvement, non_imp_count


def calc_alpha(delta, warm_up, T_f):
    delta_avg = np.mean(delta[1:])
    T_0 = -delta_avg / np.log(0.8)
    alpha = 0.9995  # np.power((T_f/T_0), (1/(10000-warm_up)))
    T = T_0
    return alpha, T


def _update_warm_up_number(warm_up):
    warm_up += 100
    return warm_up


def update_bests(incumbent, scores, itr, op_id):
    scores[op_id] += 2
    best_sol = incumbent.copy()
    last_improvement = itr
    non_imp_count = 0
    return scores, best_sol, last_improvement, non_imp_count


def update_sol_cost(new_sol, new_cost):
    return new_sol, new_cost


def initiate_ALNS(
    init_sol: Solution,
    probability: list[float],
    operators: list[Callable],
) -> tuple:
    incumbent = init_sol.copy()
    best_sol = init_sol.copy()
    operators_len_range = range(len(operators))
    scores = [0 for _ in operators_len_range]
    thetas = [0 for _ in operators_len_range]
    last_improvement = 0
    delta = [0]
    non_imp_count = 0
    weights = probability
    ws = np.array([])
    return (
        [],
        incumbent,
        0.2,
        operators_len_range,
        scores,
        thetas,
        best_sol,
        delta,
        last_improvement,
        non_imp_count,
        weights,
        ws,
    )


def apply_operator(
    prob: dict,
    rng: np.random.Generator,
    solution: Solution,
    operator: Callable,
) -> tuple[np.array, int, int]:
    """This function applies the operator on solution and returns new solution, its cost, and difference of costs

    Args:
        prob (dict): dictionary related to problem
        rng (np.random.Generator): random generator (seeded)
        incumbent (np.array): current solution 2d np array
        cost_incumb (int): cost of current solution
        operator (Callable): selected operator

    Returns:
        tuple[np.array, int, int]: new solution, new cost, new_cost - current cost
    """
    new_sol = Solution(
        problem=prob,
        representative=operator(
            solution.representative,
            rng,
            prob,
        ),
    )
    # new_cost = cost_function(new_sol, prob)
    delta_E = new_sol.total_cost - solution.total_cost
    return new_sol, delta_E


def choose_operator(
    weights: list[float],
    operators: list[Callable],
    rng: np.random.Generator,
    operators_len_range: int,
    thetas: list[int],
) -> tuple[int, Callable]:
    """This function takes the list of operators and their weights and returns the selected operator

    Args:
        weights (list[float]): list of operator selection weights
        operators (list): list of operators
        rng (np.random.Generator): random generator(seeded)
        operators_len_range (int): number of operators
        thetas (list[int]): list of number of times each operator has been selected

    Returns:
        tuple[int, Callable]: selected operator and its index
    """
    op_id = rng.choice(operators_len_range, replace=True, p=weights)
    operator = operators[op_id]
    thetas[op_id] += 1
    return op_id, operator, thetas
