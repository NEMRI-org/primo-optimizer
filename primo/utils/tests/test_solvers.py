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
import pytest

# User-defined libs
from primo.utils.solvers import get_solver


@pytest.mark.parametrize(
    "solver", ["highs", "scip", "glpk", "gurobi", "gurobi_persistent", "unknown_solver"]
)
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.parametrize("mip_gap", [0.01, 0.1, 1])
@pytest.mark.parametrize("time_limit", [5, 10, 1000])
@pytest.mark.parametrize("solver_options", [{"keepfiles": True}])
def test_get_solver(solver, stream_output, mip_gap, time_limit, solver_options):
    """
    Test the get solver method
    """
    if solver == "unknown_solver":
        with pytest.raises(ValueError):
            get_solver(
                solver,
                stream_output,
                mip_gap,
                time_limit,
                solver_options,
            )

    else:
        get_solver(solver, stream_output, mip_gap, time_limit, solver_options)
        assert True, "Solver object successfully created"


def test_unspecified_get_solver(caplog):
    """Test auto-detection of the solver"""
    _ = get_solver()

    assert "Optimization solver is not specified. " in caplog.text


def test_highs_solver_options():
    """Tests setting additional solver options for highs"""
    _ = get_solver(solver="appsi_highs", solver_options={"mip_gap": 0.0001})
