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
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, Union

# Installed libs
import pyomo
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis

# User-defined libs
from primo.utils import get_solver

LOGGER = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Base Optimization Model class. To be inherited and used as needed
    """

    def __init__(self, model_name: str):
        """
        Initializes the base model class.

        Parameters
        ----------
        model_name : str
            A string identifier to be assigned to the optimization model

        Returns
        -------
        BaseModel
            A fully instantiated BaseModel object
        """
        self.model_name = model_name
        self.model = pyo.ConcreteModel()

    @abstractmethod
    def _add_variables(self) -> None:
        pass

    @abstractmethod
    def _add_constraints(self) -> None:
        pass

    def _set_objective(self) -> None:
        """
        Override method as needed. By default, assumes that we have a
        feasibility problem
        """
        self.model.obj = pyo.Objective(expr=(0), sense=pyo.minimize)

    def build_model(self) -> None:
        """
        Initializes a model to represent the optimization problem at hand
        """
        self._add_variables()
        self._add_constraints()
        self._set_objective()

    def solve_model(
        self, solver_params: Dict = {}
    ) -> pyomo.opt.results.results_.SolverResults:
        """
        Solve the optimization model that has previously been set up using
        the `build_model()` method.

        Parameters
        ----------
        solver_params : Dict, optional
            A dictionary specifying solver parameters and values for them to be used.
            Valid options include "solver," "stream_output," "mip_gap," "time_limit,"
            and "solver_options"

        Returns
        -------
        results
            The Pyomo results object
        """
        solver = get_solver(**solver_params)
        results = solver.solve(self.model)

        return results

    def get_model_objective(self) -> float:
        """
        Return the objective value for a solved model.

        Note:
            You are responsible for checking if the model was solved successfully

        Returns
        -------
        float
            The objective value for the solved model. Maybe feasible, optimal,
            or meaningless
        """
        obj_list = []
        for v in self.model.component_data_objects(pyo.Objective):
            obj_list.append(v)

        if len(obj_list) == 0:
            LOGGER.warning("No objectives found in the model.")
            return

        elif len(obj_list) == 1:
            return pyo.value(obj_list[0])

        elif len(obj_list) > 1:
            LOGGER.warning(
                "Found multiple objectives in the model. Returning all of them."
            )
            return {obj.name: pyo.value(obj) for obj in obj_list}

    def write_model(self, file_name: str) -> None:
        """
        Writes the optimization model to a file that has previously been set up using
        :py:meth:`base_model.BaseModel.build_model()` method.
        The format of the file depends on the filename provided.
        If the filename ends with .mps, MPS format is used. If the filename
        ends with .lp, LP format is used. If the filename ends with .rew, a file in
        MPS format is written with all symbol information stripped.

        Parameters
        ----------
        file_name : str
            The path where the model is to be written.
            Must end with .mps or .lp extension
        """
        ext = pathlib.Path(file_name).suffix

        if ext in (".lp", ".mps", ".rew"):
            LOGGER.info(f"The model will be written in {ext[1:].upper()} format")

        else:
            LOGGER.warning(
                f"Unknown format {ext} encountered for optimization " "model. "
            )
            LOGGER.warning("Skipping write_model")
            return

        if os.path.exists(file_name):
            LOGGER.warning(f"File: {file_name} exists and will be overwritten")

        if ext in (".lp", ".mps"):
            # Ask Pyomo to explicitly retain all symbols
            self.model.write(file_name, io_options={"symbolic_solver_labels": True})
        else:
            temp_file_name = file_name[:-4] + ".mps"
            self.model.write(temp_file_name)
            os.rename(temp_file_name, file_name)

    def write_iis(self, iis_name: str, solver: Union[str, None] = None) -> None:
        """
        For an infeasible model, computes and writes an IIS to the file specified
        by iis_name. This assumes that a model has been built and solved by calling
        methods :py:meth:`base_model.BaseModel.build_model()` and
        :py:meth:`base_model.BaseModel.solved_model()` respectively and has been found
        to be infeasible.

        Parameters
        ----------
        iis_name : str
            The path where a valid IIS is to be written

        solver : str, optional
            The solver to be utilized for IIS calculation.
            Used by Pyomo, ignored if method_api is Gurobipy
        """

        if os.path.exists(iis_name):
            LOGGER.warning(f"File: {iis_name} exists and will be overwritten")

        write_iis(self.model, iis_name, solver, LOGGER)
