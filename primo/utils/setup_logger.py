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
import sys

# User-defined libs
from primo.utils.raise_exception import raise_exception

LOGGER_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
LOGGER_DATE = "%d-%b-%y %H:%M:%S"


def setup_logger(
    log_level: int = 2,
    log_to_console: bool = True,
    log_file: pathlib.Path = pathlib.Path(os.devnull),
):
    """
    Set up logging objects based on user input.

    Parameters
    ----------
    log_level : int, default = 2
        The level of logging messages---0: off; 1: warning; 2: info; 3: debug;

    log_to_console : bool, default = True
        If True, log messages are displayed on the screen in addition
        to the log file (if configured)

    log_file : pathlib.Path, default = pathlib.Path(os.devnull)
        The path on the disk where log files are written

    Returns
    -------
    logging.Logger
        A logger object set up as required

    Raises
    ------
    ValueError
        If the log_file specified already exists or if an invalid value for
        log_level is provided
    """
    supported_log_levels = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }
    if log_level not in supported_log_levels:
        raise_exception(
            f"Invalid value for log_level: {log_level}. Acceptable values are: [0, 1, 2, 3]",
            ValueError,
        )

    handlers = []
    if log_to_console:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)

    if log_file != pathlib.Path(os.devnull):
        if os.path.exists(log_file):
            raise_exception(
                f"Log file: {str(log_file)} already exists. Please specify new log file.",
                ValueError,
            )
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=supported_log_levels[log_level],
        format=LOGGER_FORMAT,
        datefmt=LOGGER_DATE,
        handlers=handlers,
    )
