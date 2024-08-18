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
from primo.utils.elevation_utils import (
    accessibility,
    get_bing_maps_api_key,
    get_nearest_road_point,
    get_route,
    haversine_distance,
)


@pytest.mark.secrets
def test_get_bing_maps_api_key():
    key = get_bing_maps_api_key()
    assert len(key) == 64


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


@pytest.mark.parametrize(
    "start_coord, end_coord, status",
    [  # Case 1: pass case
        (
            [40.589335, -79.92741],
            [40.642804, -79.715295],
            True,
        ),
        # Case 2: missing data
        (
            [np.nan, -79.92741],
            [40.642804, -79.715295],
            False,
        ),
        # Add more test cases as needed
    ],
)
@pytest.mark.secrets
def test_get_route(start_coord, end_coord, status):
    key = get_bing_maps_api_key()
    if status:
        assert isinstance(get_route(key, start_coord, end_coord), dict)
        assert "routePath" in get_route(key, start_coord, end_coord)
    else:
        assert get_route(key, start_coord, end_coord) is None


@pytest.mark.parametrize(
    "lat, lon, return_tuple, status",
    [  # Case 1: pass case
        (
            40.589335,
            -79.92741,
            (40.589202, -79.927406),
            True,
        ),
        # Case 2: missing data
        (
            np.nan,
            -79.92741,
            "Error",
            False,
        ),
        # Add more test cases as needed
    ],
)
@pytest.mark.secrets
def test_get_nearest_road_point(lat, lon, return_tuple, status):
    if status:
        assert np.allclose(
            get_nearest_road_point(lat, lon),
            return_tuple,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        with pytest.raises(ValueError):
            get_nearest_road_point(lat, lon)


@pytest.mark.parametrize(
    "lat, lon, return_value, status",
    [  # Case 1: pass case
        (40.4309, -79.739699, 0.033497882662451864, True),
        # Case 2: missing data
        (
            np.nan,
            -79.92741,
            "Error",
            False,
        ),
        # Add more test cases as needed
    ],
)
@pytest.mark.secrets
def test_accessibility(lat, lon, return_value, status):
    if status:
        assert np.isclose(
            accessibility(lat, lon),
            return_value,
            rtol=1e-5,
            atol=1e-8,
        )
    else:
        with pytest.raises(ValueError):
            accessibility(lat, lon)


# TODO: Need a small test raster file for testing the get_elevation
# and get_elevation_delta methods.
