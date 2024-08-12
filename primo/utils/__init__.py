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

import datetime
import logging
from .solvers import get_solver, check_optimal_termination

LOGGER_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
LOGGER_DATE = "%d-%b-%y %H:%M:%S"

BING_MAPS_BASE_URL = "http://dev.virtualearth.net/REST/V1/Routes/Driving"


# TODO: Find a way for a user to pass this
CENSUS_YEAR = 2020

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS = 3959.0  # Earth's radius in Miles

Start_coordinates = (
    40.44,
    -79.94,
)  # fixed start coordinates in elevation util to get nearest road point


CONVERSION_FACTOR = 5.614583  # convert bbl/year to mcf/year

# Ensure that census year is as recent as possible
if datetime.date.today().year - CENSUS_YEAR > 10:
    LOGGER.warning(f"Package is using {CENSUS_YEAR} CENSUS Data by default")
    LOGGER.warning("Consider upgrading to newer version")
