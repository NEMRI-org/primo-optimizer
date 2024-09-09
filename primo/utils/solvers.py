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
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs
from pyomo.environ import SolverFactory
from pyomo.environ import check_optimal_termination as pyo_opt_term

LOGGER = logging.getLogger(__name__)


def get_solver(
    solver="scip",
    stream_output=True,
    mip_gap=0.01,
    time_limit=10000,
    solver_options={},
):
    """
    Returns a solver object. A few open-source solvers can be installed using conda.
    HiGHS and Gurobi python dependencies have already been installed. To install
    SCIP - Run `conda install -c conda-forge scip` in the command prompt
    GLPK - Run `conda install -c conda-forge glpk` in the command prompt

    Parameters
    ----------
    solver : str, default = "scip"
        Choice of solver

    stream_output : bool, default = True
        Display log output from the solver. This is necessary for HiGHS
        which does not accept the pyomo tee keyword

    mip_gap : float, default = 0.01
        Duality gap for convergence/termination of MIP solve

    time_limit : float, default = 10000
        Time limit [in s] for termination of MIP solve

    solver_options : dict, default = {}
        Additional solver options

    Returns
    -------
    sol_obj :
        Pyomo solver object

    Raises
    ------
    ValueError
        If an unrecognized solver is provided as an input.
        Supported solvers include glpk, gurobi, gurobi_persistent, highs, and scip
    """
    if solver == "highs":
        sol_obj = Highs()
        sol_obj.config.stream_solver = stream_output
        sol_obj.config.mip_gap = mip_gap
        sol_obj.config.time_limit = time_limit

        for k, v in solver_options.items():
            if hasattr(sol_obj.config, k):
                setattr(sol_obj.config, k, v)

        return sol_obj

    if solver in ("gurobi", "gurobi_persistent"):
        sol_obj = SolverFactory(solver, solver_io="python")
        sol_obj.options["MIPGap"] = mip_gap
        sol_obj.options["TimeLimit"] = time_limit
        sol_obj.options.update(solver_options)

        return sol_obj

    if solver == "scip":
        sol_obj = SolverFactory("scip")
        sol_obj.options["limits/gap"] = mip_gap
        sol_obj.options["limits/time"] = time_limit
        sol_obj.options.update(solver_options)

        return sol_obj

    if solver == "glpk":
        sol_obj = SolverFactory("glpk")
        sol_obj.options["tmlim"] = time_limit
        sol_obj.options["mipgap"] = mip_gap
        sol_obj.options.update(solver_options)

        return sol_obj

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

    if solver in ["glpk", "gurobi", "gurobi_persistent", "scip"]:
        # This works for Gurobi, SCIP, and GLPK, but not for HiGHS
        return pyo_opt_term(results)

    if solver == "highs":
        # This part works for HiGHS
        if results.termination_condition == TerminationCondition.optimal:
            return True

        return False

    raise ValueError(f"Solver {solver} is not recognized")
