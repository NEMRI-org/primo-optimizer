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

"""
GitHub secrets are not passed to PRs: Therefore all tests that rely on secrets
must be turned off when Actions are run for an incoming PR. This in turn leads to
codecov complaining about reduced coverage with a patch. 

Thus all tests that rely on secrets are implemented in this single file which is 
ignored for codecov analysis by suitably setting ignore paths in codecov.yml
"""

# Installed libs
import numpy as np
import pytest

# User defined lib
from primo.utils.census_utils import get_census_key, CensusClient
from primo.utils.demo_utils import get_population_by_state
from primo.utils.elevation_utils import (
    accessibility,
    get_bing_maps_api_key,
    get_nearest_road_point,
    get_route,
)

# Sample state code for testing. 37 stands for North Carolina
STATE_CODE = 37
STATE_CODE_FAKE = 60


@pytest.mark.secrets
def test_generate_geo_identifiers():
    census_key = get_census_key()
    client = CensusClient(census_key)
    generate_identifiers = client._generate_geo_identifiers
    assert generate_identifiers("42") == ("state:42", "")
    assert generate_identifiers("") == ("", "")
    assert generate_identifiers("42079") == ("county:079", "state:42")
    assert generate_identifiers("42079216601") == (
        "tract:216601",
        "state:42 county:079",
    )


@pytest.mark.secrets
def test_get_population_by_state():
    result_df = get_population_by_state(STATE_CODE)
    assert "Total Population" in result_df.columns
    assert result_df["state"].dtypes == int
    assert result_df["county"].dtypes == int
    assert result_df["tract"].dtypes == int
    assert result_df["Total Population"].dtypes == int

    with pytest.raises(AssertionError):
        get_population_by_state(STATE_CODE_FAKE)


@pytest.mark.secrets
def test_get_bing_maps_api_key():
    key = get_bing_maps_api_key()
    assert len(key) == 64


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
