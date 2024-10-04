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

# Standard libs
import logging
from typing import Dict, Tuple, Union

# Installed libs
import numpy as np
import pyomo.environ as pyo

LOGGER = logging.getLogger(__name__)


def is_binary_value(value: float, tol: float) -> bool:
    """
    Returns True if the value is 0 or 1, subject to numerical tolerances.

    Parameters
    ----------
    value : float
        The value to be checked for binary status
    tol : float
        The absolute tolerance to be used to check for binary status

    Returns
    -------
    bool
        True if the value is 0 or 1

    """
    return np.isclose(value, 0.0, atol=tol) or np.isclose(value, 1.0, atol=tol)


def is_integer_value(value: float, tol: float) -> bool:
    """
    Returns True if the value is integral, subject to numerical tolerances.

    Parameters
    ----------
    value : float
        The value to be checked for integrality
    tol : float
        The absolute tolerance to be used to check for integrality

    Returns
    -------
    bool
        True if the value is integral

    """
    return np.isclose(np.round(value) - value, 0.0, atol=tol)


def in_bounds(
    value: float,
    lower_bound: Union[float, None],
    upper_bound: Union[float, None],
    tol: float,
) -> bool:
    """
    Returns True if the value is within lower and upper bound, subject to
    numerical tolerances.

    Parameters
    ----------
    value : float
        The value to be checked
    lower_bound : float
        The lower bound to be checked
    upper_bound : float
        The upper bound to be checked
    tol : float
        The absolute tolerance to be used

    Returns
    -------
    bool
        True if the value is within bounds

    """
    if lower_bound is not None:
        if value < lower_bound - tol:
            return False

    if upper_bound is not None:
        if value > upper_bound + tol:
            return False
    return True


def is_pyomo_model_feasible(model: pyo.ConcreteModel, tol: float) -> bool:
    """
    Checks whether a Pyomo model solution is feasible subject to a tolerance.
    The user must ensure that the model is solved or has an initial solution.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to be checked
    tol : float
        The absolute tolerance to be used for verifying variable bounds
        and constraints.

    Returns
    -------
    bool
        True if the solution is feasible for the model; False otherwise

    """

    for var in model.component_data_objects(ctype=pyo.Var, descend_into=True):
        val = var.value
        if val is None:
            val = 0
        lower_bound = None
        upper_bound = None
        if var.has_lb():
            lower_bound = pyo.value(var.lower, exception=False)
        if var.has_ub():
            upper_bound = pyo.value(var.upper, exception=False)
        if not in_bounds(val, lower_bound, upper_bound, tol):
            LOGGER.info(
                f"Variable {var} with value: {val} violates bounds"
                f" lower: {lower_bound}, upper: {upper_bound}"
            )
            return False

        if var.is_binary():
            if not is_binary_value(val, tol):
                LOGGER.info(f"Variable: {var} took a non-binary value: {val}")
                return False

        if var.is_integer():
            if not is_integer_value(val, tol):
                LOGGER.info(f"Variable: {var} took a non-integer value: {val}")
                return False

    for con in model.component_data_objects(ctype=pyo.Constraint, descend_into=True):
        val = pyo.value(con.body, exception=False)
        if val is None:
            val = 0
        lower_bound = None
        upper_bound = None

        if con.has_lb():
            lower_bound = pyo.value(con.lower, exception=False)
        if con.has_ub():
            upper_bound = pyo.value(con.upper, exception=False)
        if not in_bounds(val, lower_bound, upper_bound, tol):
            LOGGER.info(
                f"Constraint: {con} with value: {val} violated bounds "
                f"lower: {lower_bound}, upper: {upper_bound}"
            )
            return False

    return True


# TODO: delete this function after the demo notebook has been fully updated
def budget_slack_variable_scaling(
    model_inputs: object, objective_weights: Dict[str, float]
) -> Tuple:
    """
    Check whether the budget is sufficient to plug all wells and
    calculate the scaling factor for the budget slack variable based on the
    corresponding scenario.

    Parameters
    ----------
    model_inputs : OptInputs
        The full inputs to the optimization model.
    objective_weights : Dict[str,float]
        A dictionary mapping well IDs to their corresponding priority scores

    Returns
    -------
    Tuple
        A tuple containing: the scaling factor for the budget slack variable;
        a boolean indicating whether the budget is sufficient to plug all wells (True) or not (False).

    """
    # estimate the maximum number of wells can be plugged with the budget.
    unit_cost = max(model_inputs.mobilization_cost.values()) / max(
        model_inputs.mobilization_cost, key=model_inputs.mobilization_cost.get
    )
    max_well_num = model_inputs.budget / unit_cost

    # calculate the scaling factor for the budget slack variable if the
    # budget is not sufficient to plug all wells
    if max_well_num < len(objective_weights):
        scaling_budget_slack = (
            max_well_num * max(objective_weights.values())
        ) / model_inputs.budget
        budget_sufficient = False
    # calculate the scaling factor for the budget slack variable if the
    # budget is sufficient to plug all wells
    else:
        scaling_budget_slack = (
            len(objective_weights) * max(objective_weights.values())
        ) / model_inputs.budget
        budget_sufficient = True
    return scaling_budget_slack, budget_sufficient
