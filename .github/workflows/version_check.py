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

# User-defined libs
import primo
from primo.utils.raise_exception import raise_exception
from primo.utils.setup_logger import setup_logger

setup_logger(log_level=3)

LOGGER = logging.getLogger(__name__)

RELEASE_VERSION = os.environ.get("RELEASE_VERSION")
if RELEASE_VERSION is None:
    raise_exception("PRIMO version not found", ValueError)

LOGGER.info(f"GitHub tag indicates RELEASE_VERSION is: {RELEASE_VERSION}")
LOGGER.info(f"PRIMO source indicates RELEASE_VERSION is: {primo.RELEASE}")
if RELEASE_VERSION != primo.RELEASE:
    raise_exception("GitHub tag does not match RELEASE version", ValueError)

LOGGER.info("PRIMO Version matches GitHub Tag!")
