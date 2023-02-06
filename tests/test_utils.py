import numpy as np
from app.utils import compatibility_check, random_init_sol
import pytest


@pytest.fixture
def valid_input():
    problem = {"n_teams": 6}
    sol = np.ones((problem["n_teams"], problem["n_teams"]), dtype=int) * (-1)
    rng = np.random.default_rng(1)
    return random_init_sol(sol, problem, rng)


def test_that_compatibility_check_works_correctly(valid_input):
    assert compatibility_check(valid_input) == ("Compatible", True)


def test_that_compatibility_check_works_correctly_with_dummy(valid_input):
    solution = valid_input
    solution[1, 2:5] = 10
    assert compatibility_check(solution) == ("Compatible", True)
