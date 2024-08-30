#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Installed libs
import pyomo.environ as pyo
import pytest

# User-defined libs
from primo.utils.opt_utils import (
    in_bounds,
    is_binary_value,
    is_integer_value,
    is_pyomo_model_feasible,
)


@pytest.mark.parametrize(
    "value,tol,expected",
    [
        (0.0, 1e-5, True),
        (1.0, 1e-5, True),
        (0.00001, 1e-5, True),
        (1.00001, 1e-5, True),
        (0.5, 1e-5, False),
    ],
)
def test_is_binary_value(value, tol, expected):
    assert is_binary_value(value, tol) == expected


@pytest.mark.parametrize(
    "value,tol,expected",
    [
        (1.0, 1e-5, True),
        (2.0, 1e-5, True),
        (2.99999, 1e-4, True),
        (3.00001, 1e-4, True),
        (3.5, 1e-5, False),
    ],
)
def test_is_integer_value(value, tol, expected):
    assert is_integer_value(value, tol) == expected


@pytest.mark.parametrize(
    "value,lower_bound,upper_bound,tol,expected",
    [
        (5.0, 0.0, 10.0, 1e-5, True),
        (5.0, 6.0, 10.0, 1e-5, False),
        (5.0, 0.0, 4.0, 1e-5, False),
        (5.0, 5.0, 5.0, 1e-5, True),
        (5.00001, 5.0, 5.0, 1e-5, True),
    ],
)
def test_in_bounds(value, lower_bound, upper_bound, tol, expected):
    assert in_bounds(value, lower_bound, upper_bound, tol) == expected


@pytest.fixture
def create_test_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 10))
    model.y = pyo.Var(within=pyo.Binary)
    model.z = pyo.Var(within=pyo.Integers, bounds=(0, 5))
    model.obj = pyo.Objective(expr=model.x + model.y + model.z)
    model.con = pyo.Constraint(expr=model.x + model.y <= 10)
    return model


def test_is_pyomo_model_feasible(create_test_model):
    solver = pyo.SolverFactory("gurobi")
    solver.solve(create_test_model)

    assert is_pyomo_model_feasible(create_test_model, 1e-5)
