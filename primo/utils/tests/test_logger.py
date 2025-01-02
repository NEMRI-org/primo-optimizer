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
from primo.utils import setup_logger


def test_logger(tmp_path):
    """Checks if if the setup_logger function works or not"""
    # Catch the invalid log-level error
    with pytest.raises(ValueError):
        setup_logger(10, False)

    d = tmp_path / "mylog.log"
    setup_logger(2, True, d)

    # Catch the log file exists error
    with pytest.raises(ValueError):
        setup_logger(2, True, d)


def test_debug_logger():
    """
    Runs the if-condition corresponding to debug mode for
    code coverage.
    """
    setup_logger(3, True)
