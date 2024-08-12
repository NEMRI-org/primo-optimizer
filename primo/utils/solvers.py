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

# Standard lib
import logging

# Installed lib
from pyomo.environ import SolverFactory
from pyomo.contrib.appsi.solvers import Highs
from pyomo.environ import check_optimal_termination as pyo_opt_term
from pyomo.contrib.appsi.base import TerminationCondition


LOGGER = logging.getLogger(__name__)


def get_solver(
    solver="highs",
    stream_output=True,
    mip_gap=0.01,
    time_limit=10000,
    solver_options={},
    **kwargs,
):
    """
    Returns a solver object. A few open-source solvers can be installed using conda.
    HiGHS and Gurobi python dependencies have already been installed. To install
    SCIP - Run `conda install -c conda-forge scip` in the command prompt
    GLPK - Run `conda install -c conda-forge glpk` in the command prompt

    Parameters
    ----------
    solver : str, default = "highs"
        Choice of solver

    stream_output : bool, default = True
        Display log output from the solver

    mip_gap : float, default = 0.01
        Duality gap for convergence/termination of MIP solve

    time_limit : float, default = 10000
        Time limit [in s] for termination of MIP solve

    solver_options : dict, default = {}
        Additional solver options

    **kwargs : dict
        Extra keyword arguments that are ignored

    Returns
    -------
    sol_obj :
        Pyomo solver object

    Raises
    ------
    ValueError
        If an unrecognized solver is provided as an input.
        Supported solvers include highs, gurobi, scip, and glpk
    """

    if not kwargs:
        LOGGER.warning(
            f"get_solver method received unknown arguments {kwargs}. "
            "These will be ignored"
        )

    if solver == "highs":
        sol_obj = Highs()
        sol_obj.config.stream_solver = stream_output
        sol_obj.config.mip_gap = mip_gap
        sol_obj.config.time_limit = time_limit

        for k, v in solver_options:
            if hasattr(sol_obj.config, k):
                setattr(sol_obj.config, k, v)

        return sol_obj

    elif solver == "gurobi":
        sol_obj = SolverFactory("gurobi", solver_io="python")
        sol_obj.options["MIPGap"] = mip_gap
        sol_obj.options["TimeLimit"] = time_limit
        sol_obj.options["OutputFlag"] = int(stream_output)
        sol_obj.options.update(solver_options)

        return sol_obj

    # For SCIP and GLPK, it is not clear if there is an option to
    # control the stream output with an option. The user needs to do it
    # with `tee` argument when they call the `solve` method.
    elif solver == "scip":
        sol_obj = SolverFactory("scip")
        sol_obj.options["limits/gap"] = mip_gap
        sol_obj.options["limits/time"] = time_limit
        sol_obj.options.update(solver_options)

        return sol_obj

    elif solver == "glpk":
        sol_obj = SolverFactory("glpk")
        sol_obj.options["tmlim"] = time_limit
        sol_obj.options["mipgap"] = mip_gap
        sol_obj.options.update(solver_options)

        return sol_obj

    else:
        raise ValueError(f"Solver {solver} is not recognized!")


def check_optimal_termination(results, solver):
    """
    Checks if the solver found the optimal solution or not.

    Parameters
    ----------
    results : Pyomo results object

    solver : str
        Solver used to solve the MIP

    Returns
    -------
    bool :
        True, if optimal solution is found; False, otherwise.

    Raises
    ------
    ValueError
        If an unrecognized solver is provided as an input.
        Supported solvers include highs, gurobi, scip, and glpk
    """

    if solver in ["gurobi", "scip", "glpk"]:
        # This works for Gurobi, SCIP, and GLPK, but not for HiGHS
        return pyo_opt_term(results)

    elif solver == "highs":
        # This part works for HiGHS
        if results.termination_condition == TerminationCondition.optimal:
            return True

        return False

    else:
        raise ValueError(f"Solver {solver} is not recognized")
