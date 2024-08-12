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
import os
import pathlib
from typing import List, Union

# Installed libs


def parse_args(args_list: Union[None, List[str]]) -> argparse.Namespace:
    """
    Set up and parse command-line arguments.

    Parameters
    ----------
    args_list : Union[None, List[str]], optional
        The arguments to be parsed; defaults to None

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """

    parser = argparse.ArgumentParser(
        description="Small-scale demo for PRIMO",
        epilog=(
            "For questions or comments, " "please reach out to  " "primo@netl.doe.gov"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = parser.add_argument_group("required_arguments")

    required.add_argument(
        "-f",
        "--input-file",
        action="store",
        type=pathlib.Path,
        required=True,
        help="Specify location of the csv file with input data",
    )

    required.add_argument(
        "-b",
        "--total-budget",
        action="store",
        type=float,
        required=True,
        help="The total budget available",
    )

    parser.add_argument(
        "-d",
        "--dac-weight",
        action="store",
        type=float,
        required=False,
        help="The weight assigned to disadvantaged community score",
    )

    parser.add_argument(
        "-df",
        "--dac-weight-percent",
        action="store",
        type=float,
        required=False,
        help="The percentage score that must be assigned to a well to consider it disadvantaged",
    )

    parser.add_argument(
        "-wd",
        "--maximum-well-distance",
        action="store",
        type=float,
        required=False,
        default=10,
        help="The maximum distance between two wells in a project in miles",
    )

    parser.add_argument(
        "-owc",
        "--max-wells-per-owner",
        action="store",
        type=float,
        required=False,
        default=-1,
        help=(
            "The maximum number of wells belonging to a single owner that can be plugged. "
            "The default value of -1 is a special value, indicating that this value and its"
            " related constraints should be ignored."
        ),
    )

    parser.add_argument(
        "-p",
        "--lifelong-production",
        action="store",
        type=float,
        required=False,
        default=1000,
        help="The maximum lifelong production of a well",
    )

    parser.add_argument(
        "-l",
        "--log-file",
        action="store",
        type=pathlib.Path,
        required=False,
        help="Specify location for a log-file if desired",
        default=os.devnull,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store",
        type=int,
        choices=[0, 1, 2, 3],
        required=False,
        default=1,
        help="Specify level of output",
    )

    parser.add_argument(
        "-o",
        "--output-to-console",
        action="store_true",
        required=False,
        help="Should logging output be displayed on console",
    )
    return parser.parse_args(args_list)
