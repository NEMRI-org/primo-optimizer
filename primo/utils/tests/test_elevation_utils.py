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

# Installed libs
import numpy as np
import pytest

# User defined libs
from primo.utils.elevation_utils import haversine_distance


@pytest.mark.parametrize(
    "lat1, lon1, lat2, lon2, return_value, status",
    [  # Case 1: pass case
        (
            40.589335,
            -79.92741,
            40.642804,
            -79.715295,
            11.723084879535591,
            True,
        ),
        # Case 2: missing data
        (
            np.nan,
            -79.92741,
            40.642804,
            -79.715295,
            np.nan,
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_haversine_distance(lat1, lon1, lat2, lon2, return_value, status):
    if status:
        assert np.isclose(
            haversine_distance(lat1, lon1, lat2, lon2),
            return_value,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        assert np.isnan(haversine_distance(lat1, lon1, lat2, lon2))


# TODO: Need a small test raster file for testing the get_elevation
# and get_elevation_delta methods.
