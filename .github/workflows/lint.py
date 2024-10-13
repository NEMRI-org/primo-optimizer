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
import argparse
import logging

# Installed libs
from pylint.lint import Run

# User-defined libs
from primo.utils.raise_exception import raise_exception
from primo.utils.setup_logger import setup_logger

setup_logger(log_level=3)

LOGGER = logging.getLogger(__name__)
PARSER = argparse.ArgumentParser(prog="Linter")
PARSER.add_argument(
    "-t", "--threshold", help="Threshold to allow linter to pass", type=float
)

PARSER.add_argument("-p", "--path", help="Path to run linter on", type=str)

ARGS = PARSER.parse_args()
PATH = str(ARGS.path)
THRESHOLD = float(ARGS.threshold)

LOGGER.info(f"Beginning linting for: {PATH} with threshold: {THRESHOLD}")
RESULTS = Run(
    [
        PATH,
        "--rc-file",
        ".pylintrc",
        "--reports",
        "n",
        "--exit-zero",
        "--output-format",
        "primo_reporters.DisplayProgress,primo_reporters.DisplayOutput:pylint.txt",
    ],
    exit=False,
)


SCORE = RESULTS.linter.stats.global_note

if SCORE < THRESHOLD:
    msg = f"Pylint score was: {SCORE}. Linter failed"
    raise_exception(msg, ValueError)
else:
    msg = f"Pylint score was: {SCORE}. Linter passed!"
    LOGGER.info(msg)
