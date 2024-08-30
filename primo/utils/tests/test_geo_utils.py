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
import pytest

# User-defined libs
from primo.utils.geo_utils import get_distance, is_in_bounds


@pytest.mark.parametrize(
    "origin,dest,distance",
    [
        (
            (45.7595, 4.8422),  # lyon
            (48.8567, 2.3508),  # Paris
            243.712506,
        ),  # Distance in miles
        (
            (39.9525, -75.1652),  # Philadelphia
            (40.4406, -79.9958),  # Pittsburgh
            257.126,
        ),  # Distance in miles
    ],
)
def test_get_distance(origin, dest, distance):
    assert get_distance(origin, dest) == pytest.approx(distance, 0.001)


def test_is_in_bounds():
    assert is_in_bounds(2, "integer", 0, 4, False)
    assert not is_in_bounds(3, "integer", None, 2, False)

    with pytest.raises(ValueError):
        assert is_in_bounds(3.2, "float", None, 2, True)
        assert is_in_bounds(2.7, "float", 3.0, None, True)


# TODO: Add tests for is_valid_lat, is_valid_long, is_valid_geopoint, is_valid_args
