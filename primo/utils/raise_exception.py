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

LOGGER = logging.getLogger(__name__)


def raise_exception(msg: str, exception_type: Exception) -> None:
    """
    Raises an exception with a message and logs the details.

    Parameters
    ----------
    msg : str
        Helpful informative message to be passed with the exception
    exception_type : Exception
        The exception type to be raised

    Raises
    ------
    exception_type
        The specified exception with the provided message
    """
    try:
        raise exception_type(msg)
    except:
        # Try-except block ensures that the stacktrace is captured in a log file
        # if one is configured
        LOGGER.exception(msg)
        raise
