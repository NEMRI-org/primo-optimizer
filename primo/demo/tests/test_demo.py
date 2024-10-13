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
import os

# Installed libs
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


def test_example():
    notebook_path = os.path.join("primo", "demo", "PRIMO - Example_1.ipynb")
    with open(notebook_path) as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    ep = ExecutePreprocessor()
    try:
        ep.preprocess(notebook, {"metadata": {"path": os.path.dirname(notebook_path)}})
        assert True, "Running demo notebook succeeded"
    except:
        assert False, "Demo notebook did not run successfully"
