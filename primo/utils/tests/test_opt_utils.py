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
# pylint: disable=missing-function-docstring

# Installed libs
import pyomo.environ as pyo
import pytest
from pyomo.opt import SolverStatus, TerminationCondition

# User-defined libs
from primo.utils.opt_utils import (
    OptimizationException,
    in_bounds,
    is_binary_value,
    is_integer_value,
    is_pyomo_model_feasible,
    optimization_results_handler,
)
from primo.utils.solvers import get_solver


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


@pytest.fixture(name="create_test_model")
def my_create_test_model():
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


@pytest.fixture(name="infeasible_model")
def my_infeasible_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.y = pyo.Var(bounds=(0, 1))
    model.obj = pyo.Objective(expr=model.x + model.y)
    model.con = pyo.Constraint(expr=model.x + model.y >= 4)
    return model


@pytest.fixture(name="unbounded_model")
def my_unbounded_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))
    model.y = pyo.Var(bounds=(0, 1), initialize=1)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)
    model.con = pyo.Constraint(expr=model.x + model.y >= 4)
    return model


# this function ensures that the results object is correctly excepted by the function
def test_optimization_results_handler(
    infeasible_model, create_test_model, unbounded_model
):
    solver = get_solver("gurobi")
    results = solver.solve(infeasible_model)
    try:
        optimization_results_handler(results, infeasible_model)
    except OptimizationException as e:
        assert e.args[0] == "Optimization did not terminate feasibly or optimally"
        assert e.args[1].solver.termination_condition == TerminationCondition.infeasible
        assert e.args[1].solver.status == SolverStatus.warning

    # check the successful branch
    results = solver.solve(create_test_model)
    assert optimization_results_handler(results, create_test_model) == 1

    # test out highs...
    solver_highs = get_solver("highs")
    with pytest.raises(RuntimeError):
        # highs will throw an error when trying to load a solution if the model is infeasible
        # we'll just catch the error at the web API level and deal with it
        results = solver_highs.solve(infeasible_model)
        optimization_results_handler(results, infeasible_model)

    results = solver_highs.solve(create_test_model)
    assert optimization_results_handler(results, create_test_model) == 1

    # couldn't find an example of an actual optimization problem where this happens
    results.solver.termination_condition = TerminationCondition.maxTimeLimit
    assert optimization_results_handler(results, create_test_model) == 2

    with pytest.raises(RuntimeError):
        results = solver_highs.solve(unbounded_model)
        optimization_results_handler(results, unbounded_model)
