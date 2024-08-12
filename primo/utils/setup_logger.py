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

# Installed libs

# User defined libs
from primo.utils.raise_exception import raise_exception
from primo.utils import LOGGER_FORMAT, LOGGER_DATE


def setup_logger(log_file: pathlib.Path, log_level: int, log_to_console: bool):
    """
    Set up logging objects based on user input.

    Parameters
    ----------
    log_file : pathlib.Path
        The path on the disk where log files are written; defaults to None
    log_level : int
        The level of logging messages---0: off; 1: warning; 2: info; 3: debug;
    log_to_console : bool
        If True, log messages are displayed on the screen in addition
        to the log file (if configured); defaults to False

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
    logging_level = logging.DEBUG

    if log_level == 0:
        # Instead of fully turning off, we only retain critical messages
        logging_level = logging.CRITICAL
    elif log_level == 1:
        logging_level = logging.WARNING
    elif log_level == 2:
        logging_level = logging.INFO
    elif log_level == 3:
        logging_level = logging.DEBUG
    else:
        raise_exception(
            f"Invalid value for log_level: {log_level}. "
            "Acceptable values are: {0,1,2,3}",
            ValueError,
        )

    handlers = []
    if log_to_console:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)

    if log_file != pathlib.Path(os.devnull):
        if os.path.exists(log_file):
            raise_exception(
                f"Log file: {log_file} already exists. " "Please specify new log file",
                ValueError,
            )
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging_level,
        format=LOGGER_FORMAT,
        datefmt=LOGGER_DATE,
        handlers=handlers,
    )
