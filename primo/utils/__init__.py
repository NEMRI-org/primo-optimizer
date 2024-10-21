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
)
from primo.data_parser.default_data import START_COORDINATES as Start_coordinates
from primo.utils.setup_logger import setup_logger
from primo.utils.solvers import check_optimal_termination, get_solver

LOGGER = logging.getLogger(__name__)

# Ensure that census year is as recent as possible
if datetime.date.today().year - CENSUS_YEAR > 10:
    LOGGER.warning(f"Package is using {CENSUS_YEAR} CENSUS Data by default")
    LOGGER.warning("Consider upgrading to newer version")
