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
from types import SimpleNamespace

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import WellData, WellDataColumnNames
from primo.utils.clustering_utils import (
    distance_matrix,
    get_admissible_well_pairs,
    get_pairwise_metrics,
    get_wells_in_dac,
    perform_clustering,
)


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
    """Tests the distance_matrix function"""
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


@pytest.fixture(name="get_well_data", scope="function")
def get_well_data_fixture():
    """Constructs a dummy well data"""
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

    return WellData(data=filename, column_names=col_names)


def test_perform_clustering(caplog, get_well_data):
    """Tests the perform clustering function"""
    wd = get_well_data
    col_names = wd.column_names
    assert "Clusters" not in wd
    assert not hasattr(col_names, "cluster")

    well_clusters = perform_clustering(wd, distance_threshold=10.0)
    assert "Clusters" in wd
    assert hasattr(col_names, "cluster")
    assert isinstance(well_clusters, dict)
    assert len(well_clusters) == 16
    assert len(well_clusters) == len(set(wd.data["Clusters"]))

    assert "Input well data is already clustered." not in caplog.text

    # Capture the warning if the data has already been clustered
    well_clusters = perform_clustering(wd, distance_threshold=10.0)
    assert isinstance(well_clusters, dict)
    assert len(well_clusters) == 16

    assert "Input well data is already clustered." in caplog.text


def test_get_admissible_well_pairs(get_well_data):
    """
    Tests the get_pairwise_metrics method and the
    get_admissible_well_pairs functions.
    """

    wd = get_well_data
    well_clusters = perform_clustering(wd, distance_threshold=10.0)
    pairwise_metrics = get_pairwise_metrics(wd, well_clusters)

    assert isinstance(pairwise_metrics, SimpleNamespace)
    assert isinstance(pairwise_metrics.distance, dict)
    assert isinstance(pairwise_metrics.age, dict)
    assert isinstance(pairwise_metrics.depth, dict)
    assert len(pairwise_metrics.distance) == 16
    assert len(pairwise_metrics.age) == 16
    assert len(pairwise_metrics.depth) == 16

    well_pairs = get_admissible_well_pairs(pairwise_metrics, max_distance=10.0)
    well_pairs_keep = well_pairs[0]
    well_pairs_remove = well_pairs[1]
    assert len(well_pairs_keep) == 16
    assert len(well_pairs_remove) == 16
    for i in range(16):
        # No well pairs must be removed in this case
        assert len(well_pairs_remove[i]) == 0

    well_pairs = get_admissible_well_pairs(
        pairwise_metrics,
        max_distance=10.0,
        max_age_range=10,
        max_depth_range=1000,
    )
    well_pairs_keep = well_pairs[0]
    well_pairs_remove = well_pairs[1]
    assert len(well_pairs_keep) == 16
    assert len(well_pairs_remove) == 16
    for i in range(16):
        # Well pairs will be removed in this case
        if i != 13:
            assert len(well_pairs_remove[i]) > 0


def test_get_wells_in_dac(get_well_data):
    """Tests the get_wells_in_dac function"""
    wd = get_well_data
    well_clusters = perform_clustering(wd, distance_threshold=10)
    wells_dac = get_wells_in_dac(wd, well_clusters)

    # Test the output when the DAC information is not available
    assert isinstance(wells_dac, dict)
    assert len(wells_dac) == 16
    for i in range(16):
        assert isinstance(wells_dac[i], list)
        assert len(wells_dac[i]) == 0

    # Test the output when the DAC information is available
    wd.column_names.is_disadvantaged = "is_disadvantaged"
    wd.data["is_disadvantaged"] = 0
    wd.data.loc[[27, 28, 29, 30], "is_disadvantaged"] = 1
    wells_dac = get_wells_in_dac(wd, well_clusters)

    assert len(wells_dac) == 16
    print(wells_dac)
    assert len(wells_dac[6]) == 1
    assert len(wells_dac[3]) == 3
    for i in range(16):
        if i not in [3, 6]:
            assert len(wells_dac[i]) == 0
