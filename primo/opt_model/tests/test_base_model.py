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

# Installed libs
import pyomo.environ as pyo
import pytest

# User defined libs
from primo.opt_model.base_model import BaseModel

LOGGER = logging.getLogger(__name__)


# Basic linear model with known solution using the BaseModel class as an inheritance
class LinearOptModel(BaseModel):
    """
    A simple linear model class for a two-variable two-constraint model
    """

    def _add_sets(self):
        """
        Define sets associated with the optimization model
        """
        LOGGER.info("Initializing set of wells!")
        model = self.model

        model.s_x = pyo.Set(
            doc="Set of x variables",
            initialize=[1, 2],
        )

    def _add_parameters(self):
        """
        Define parameters associated with the optimization model
        """
        model = self.model

        model.p_x = pyo.Param(
            doc="Parameter on x for constraints",
            initialize=2,
            mutable=False,
            within=pyo.NonNegativeReals,
        )

    def _add_variables(self):
        """
        Define variables associated with the optimization model
        """

        model = self.model
        model.v_x = pyo.Var(model.s_x, within=pyo.NonNegativeReals)

    def _add_constraints(self):
        model = self.model

        model.con_1 = pyo.Constraint(
            expr=model.v_x[1] + model.p_x * model.v_x[2] <= 1, doc="constraint 1 on x"
        )
        model.con_2 = pyo.Constraint(
            expr=model.p_x * model.v_x[1] + model.v_x[2] <= 1, doc="constraint 2 on x"
        )

    def _set_objective(self):
        """
        Define the objective function
        """
        model = self.model

        model.obj = pyo.Objective(expr=model.v_x[1] + model.v_x[2], sense=pyo.maximize)

    def build_model(self) -> None:
        """
        Initializes a model to represent the optimization problem at hand
        """
        self._add_sets()
        self._add_parameters()
        self._add_variables()
        self._add_constraints()
        self._set_objective()


# Integration testing for building a model using BaseModel class
def test_base_model_build():

    model = LinearOptModel("linear_model")
    model.build_model()

    # Checking that each of the model components got built.
    assert isinstance(model.model.s_x, pyo.Set)
    assert isinstance(model.model.v_x, pyo.Var)
    assert isinstance(model.model.p_x, pyo.Param)
    assert isinstance(model.model.con_1, pyo.Constraint)
    assert isinstance(model.model.con_2, pyo.Constraint)
    assert isinstance(model.model.obj, pyo.Objective)


# TODO: Need to add integration tests for scip and glpk
# Integration testing for solving a model using BaseModel class
def test_solve_linear_model_highs():

    model = LinearOptModel("linear_model")
    model.build_model()

    solver_params = {"solver": "highs"}
    status = model.solve_model(solver_params)
    # Checking to see if the model was solved
    assert status
    # Checking that the objective found is what we expect
    assert pyo.value(model.model.obj) == pytest.approx((2 / 3), 0.01)
