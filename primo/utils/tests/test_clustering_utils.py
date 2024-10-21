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
# Standard lib
# Standard libs
import os

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.well_data import WellData
from primo.utils.clustering_utils import distance_matrix, perform_clustering


# Sample data for testing
@pytest.mark.parametrize(
    "well_data, weight, result, status",
    [  # Case1: Passed case
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                    "Op Name": "Owner 2",
                },
            ],
            {"age": 0.5, "depth": 0.5},  # Weight
            [[0.0, 505.0], [505.0, 0.0]],  # Result
            True,  # Status
        ),
        # Case 3: Summation of feature weights is not 1
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 42.0,
                    "Longitude": -70,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 43.0,
                    "Longitude": -74,
                    "Age [Years]": 30,
                    "Depth [ft]": 5000,
                    "Op Name": "Owner 2",
                },
            ],
            {"distance": 0.3, "age": 0.8, "depth": 0.5},  # Weight
            "Feature weights do not add up to 1",  # Result
            False,  # Status
        ),
        # Case 6: Spurious feature provided
        (  # Well data
            [
                {
                    "Well API": "W1",
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                    "Op Name": "Owner 1",
                },
                {
                    "Well API": "W2",
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                    "Op Name": "Owner 2",
                },
            ],
            {"distance": 0.5, "depth": 0.2, "ages": 0.3, "age": 0.3},  # Weight
            (
                "Received feature(s) [ages] that are not "
                "supported in the clustering step."
            ),  # Result
            False,  # Status
        ),
        # Add more test cases as needed
    ],
)
def test_distance_matrix(well_data, weight, result, status):
    """
    Tests for distance_matrix method
    """
    well_df = pd.DataFrame(well_data)
    well_cn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Op Name",
    )
    wd = WellData(well_df, well_cn)
    result_arr = np.array(result)
    if status:
        assert np.allclose(
            distance_matrix(wd, weight), result_arr, rtol=1e-5, atol=1e-8
        )
    else:
        with pytest.raises(ValueError):
            _ = distance_matrix(wd, weight) == result


def test_perform_clustering(caplog):
    """
    Tests for perform_clustering method
    """
    filename = os.path.dirname(os.path.realpath(__file__))[:-12]  # Primo folder
    filename += "//data_parser//tests//random_well_data.csv"

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
    )

    wd = WellData(data=filename, column_names=col_names)
    assert "Clusters" not in wd
    assert not hasattr(col_names, "cluster")

    num_clusters = perform_clustering(wd)
    assert "Clusters" in wd
    assert hasattr(col_names, "cluster")
    assert num_clusters == 16
    assert num_clusters == len(set(wd.data["Clusters"]))

    assert (
        "Found cluster attribute in the WellDataColumnNames object."
        "Assuming that the data is already clustered. If the corresponding "
        "column does not correspond to clustering information, please use a "
        "different name for the attribute cluster while instantiating the "
        "WellDataColumnNames object."
    ) not in caplog.text

    # Capture the warning if the data has already been clustered
    num_clusters = perform_clustering(wd)
    assert num_clusters == 16

    assert (
        "Found cluster attribute in the WellDataColumnNames object."
        "Assuming that the data is already clustered. If the corresponding "
        "column does not correspond to clustering information, please use a "
        "different name for the attribute cluster while instantiating the "
        "WellDataColumnNames object."
    ) in caplog.text
