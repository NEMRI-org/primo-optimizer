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

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User defined libs
from primo.utils.clustering_utils import distance_matrix


# Sample data for testing
@pytest.mark.parametrize(
    "well_data, weight, result, status",
    [  # Case1: Passed case
        (  # Well data
            [
                {
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                },
                {
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                },
            ],
            {"distance": 0, "age": 0.5, "depth": 0.5},  # Weight
            [[0.0, 505.0], [505.0, 0.0]],  # Result
            True,  # Status
        ),
        # Case 2: Missing information
        ({}, {}, {}, False),
        # Case 3: Summation of feature weights is not 1
        (  # Well data
            [
                {
                    "Latitude": 42.0,
                    "Longitude": -70,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                },
                {
                    "Latitude": 43.0,
                    "Longitude": -74,
                    "Age [Years]": 550,
                    "Depth [ft]": 5000,
                },
            ],
            {"distance": 0.3, "age": 0.8, "depth": 0.5},  # Weight
            "Feature weights do not add up to 1",  # Result
            False,  # Status
        ),
        # Case 4: Missing information in the input well DataFrame
        (  # Well data
            [
                {
                    "Latitude": 40.0,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                },
                {
                    "Latitude": 41.0,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                },
            ],
            {"distance": 0, "age": 0.5, "depth": 0.5},  # Weight
            "The latitude or longitude information of well is not provided",  # Result
            False,  # Status
        ),
        # Case 5: Missing weights
        (  # Well data
            [
                {
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                },
                {
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                },
            ],
            {"distance": 0.5, "depth": 0.5},  # Weight
            "The weight for feature age is not provided",  # Result
            False,  # Status
        ),
        # Case 6: Spurious feature provided
        (  # Well data
            [
                {
                    "Latitude": 40.0,
                    "Longitude": -71,
                    "Age [Years]": 20,
                    "Depth [ft]": 1000,
                },
                {
                    "Latitude": 41.0,
                    "Longitude": -72,
                    "Age [Years]": 30,
                    "Depth [ft]": 2000,
                },
            ],
            {"distance": 0.5, "depth": 0.2, "ages": 0.3, "age": 0.3},  # Weight
            "Currently, the feature ages is not supported in the clustering step",  # Result
            False,  # Status
        ),
        # Add more test cases as needed
    ],
)
def test_distance_matrix(well_data, weight, result, status):
    well_df = pd.DataFrame(well_data)
    result_arr = np.array(result)
    if status:
        assert np.allclose(
            distance_matrix(well_df, weight), result_arr, rtol=1e-5, atol=1e-8
        )
    else:
        with pytest.raises(ValueError):
            _ = distance_matrix(well_df, weight) == result


# Run tests
if __name__ == "__main__":
    pytest.main()
