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
import pandas as pd
import pytest

# User-defined libs
from primo.utils.proximity_utils import nearby_hospital_count, nearby_total_school_count

# Sample data for testing
WELL_DATA = [
    {"Well_Latitude": 40.0, "Well_Longitude": -73.0},
    {"Well_Latitude": 41.0, "Well_Longitude": -74.0},
    {"Well_Latitude": 42.0, "Well_Longitude": -75.0},
]

SCHOOL_DATA = [
    {"School_Latitude": 40.0001, "School_Longitude": -73.0001},
    {"School_Latitude": 41.0001, "School_Longitude": -74.0001},
    {"School_Latitude": 42.0001, "School_Longitude": -75.0001},
]

HOSPITAL_DATA = [
    {"Hospital_Latitude": 39.9999, "Hospital_Longitude": -72.9999},
    {"Hospital_Latitude": 40.9999, "Hospital_Longitude": -73.9999},
    {"Hospital_Latitude": 41.9999, "Hospital_Longitude": -74.9999},
]


@pytest.mark.parametrize(
    "well_data, school_data, distance",
    [
        (WELL_DATA, SCHOOL_DATA, 1),
        # Add more test cases as needed
    ],
)
def test_nearby_total_school_count(well_data, school_data, distance):
    well_df = pd.DataFrame(well_data)
    school_df = pd.DataFrame(school_data)
    result_df = nearby_total_school_count(well_df, school_df, distance_miles=distance)
    assert "Schools Within Distance" in result_df.columns
    assert all(result_df["Schools Within Distance"] == [1, 1, 1])


@pytest.mark.parametrize(
    "well_data, hospital_data, distance",
    [
        (WELL_DATA, HOSPITAL_DATA, 1),
    ],
)
def test_nearby_hospital_count(well_data, hospital_data, distance):
    well_df = pd.DataFrame(well_data)
    hospital_df = pd.DataFrame(hospital_data)
    result_df = nearby_hospital_count(well_df, hospital_df, distance_miles=distance)
    assert "Hospitals Within Distance" in result_df.columns
    assert all(result_df["Hospitals Within Distance"] == [1, 1, 1])


# Run tests
if __name__ == "__main__":
    pytest.main()
