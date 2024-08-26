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
import datetime
import logging

# User-defined libs
from primo.data_parser.default_data import (
    BING_MAPS_BASE_URL,
    CENSUS_YEAR,
    CONVERSION_FACTOR,
    EARTH_RADIUS,
    START_COORDINATES as Start_coordinates,
)
from primo.utils.solvers import get_solver, check_optimal_termination
from primo.utils.setup_logger import setup_logger, LogLevel

LOGGER = logging.getLogger(__name__)

# pylint: disable = logging-fstring-interpolation
# Ensure that census year is as recent as possible
if datetime.date.today().year - CENSUS_YEAR > 10:
    LOGGER.warning(f"Package is using {CENSUS_YEAR} CENSUS Data by default")
    LOGGER.warning("Consider upgrading to newer version")
