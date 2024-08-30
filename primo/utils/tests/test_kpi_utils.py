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
from pandas.testing import assert_frame_equal

# User-defined libs
from primo.utils.kpi_utils import (
    calculate_average,
    calculate_number_of_owners,
    calculate_range,
    calculate_well_number,
    process_data,
)

# Sample data for testing
WELL_DATA_BASE_CASE = {
    "Priority Score [0-100]": [10, 20, 30, 40, 50, 60],
    "Age [Years]": [30, 42, 18, 66, 120, 72],
    "Depth [ft]": [1576, 4567, 794, 1349, 600, 845],
    "Elevation Delta [m]": [-34.39, 54.27, 78.34, 84.52, 34.67, 29.71],
    "Distance to Road [miles]": [3.23, 2.45, 1.67, -0.89, -1.24, 0.36],
    "Distance to Centroid [miles]": [2.45, 5.78, -3.67, 8.56, -5.23, 0.33],
    "Operator Name": ["John", "Tim", "Tim", "Zack", "John", "chris"],
    "Project": [
        "Project 1",
        "Project 1",
        "Project 21",
        "Project 21",
        "Project 1",
        "Project 21",
    ],
}

WELL_DATA_MISSING_VALUES = {
    "Priority Score [0-100]": [np.nan, 58, 62.35, 76.48, 45.23, 23.42],
    "Age [Years]": [23, 46, 107, 85, 21, pd.NaT],
    "Depth [ft]": [1576, 4567, 794, np.nan, 600, 845],
    "Elevation Delta [m]": [-34.39, 54.27, np.nan, 84.52, 34.67, 29.71],
    "Distance to Road [miles]": [3.23, np.nan, 1.67, -0.89, -1.24, 0.36],
    "Distance to Centroid [miles]": [2.45, 5.78, np.nan, 8.56, -5.23, 0.33],
    "Operator Name": ["John", "Tim", "Tim", "Zack", "", "John"],
    "Project": [
        "Project 1",
        "Project 1",
        "Project 21",
        "Project 21",
        "Project 1",
        "Project 21",
    ],
}

PROJECT_EFFICIENCY_METRIC_SCORE = {
    "Project": ["Project 1", "Project 21"],
    "Project Centroid": [(42.6448, -73.90768), (42.13731, -77.89755)],
    "Number of Wells": [3, 3],
    "Number of Unique Owners": [2, 3],
    "Average Elevation Delta [m]": [41.11, 64.19],
    "Average Distance to Road [miles]": [2.31, 0.97],
    "Distance to Centroid [miles]": [4.49, 4.19],
    "Age Range [Years]": [90, 54],
    "Average Age [Years]": [64.00, 52.00],
    "Depth Range [ft]": [3967, 555],
    "Average Depth [ft]": [2247.67, 996.00],
    "Impact Score [0-100]": [26.67, 43.33],
}
PROJECT_CENTROIDS = {
    "Project 1": (42.6448, -73.90768),
    "Project 21": (42.13731, -77.89755),
}


@pytest.mark.parametrize(
    "well_data, avg_priority, avg_age,avg_depth, age_range, depth_range, "
    "avg_elev_delta, number_of_wells, avg_dist_to_rd, avg_dist_to_centroid, "
    "avg_unique_operator_name",
    [
        (WELL_DATA_BASE_CASE, 35, 58, 1621.833, 102, 3967, 52.650, 6, 1.64, 4.337, 4),
        # Add more test cases as needed
    ],
)
def test_core_calculation_methods(
    well_data,
    avg_priority,
    avg_age,
    avg_depth,
    age_range,
    depth_range,
    avg_elev_delta,
    number_of_wells,
    avg_dist_to_rd,
    avg_dist_to_centroid,
    avg_unique_operator_name,
):

    well_df = pd.DataFrame(well_data)

    # Tests for calculate_average_priority_score
    result = calculate_average(well_df, column_name="Priority Score [0-100]")
    assert result == pytest.approx(avg_priority, 0.001)

    # Tests for calculate_average_age
    result = calculate_average(well_df, column_name="Age [Years]")
    assert result == pytest.approx(avg_age, 0.001)

    # Tests for calculate_average_depth
    result = calculate_average(well_df, column_name="Depth [ft]")
    assert result == pytest.approx(avg_depth, 0.001)

    # Tests for calculate_age_range
    result = calculate_range(well_df, column_name="Age [Years]")
    assert result == pytest.approx(age_range, 0.001)

    # Tests for calculate_depth_range
    result = calculate_range(well_df, column_name="Depth [ft]")
    assert result == pytest.approx(depth_range, 0.001)

    # Tests for calculate_well_number
    result = calculate_well_number(well_df)
    assert result == pytest.approx(number_of_wells, 0.001)

    # Tests for calculate_elevation_average
    result = calculate_average(
        well_df, column_name="Elevation Delta [m]", estimation_method="yes"
    )
    assert result == pytest.approx(avg_elev_delta, 0.001)

    # Tests for calculate_road_distance_average
    result = calculate_average(well_df, column_name="Distance to Road [miles]")
    assert result == pytest.approx(avg_dist_to_rd, 0.001)

    # Tests for calculate_centroid_distance_average
    result = calculate_average(well_df, column_name="Distance to Centroid [miles]")
    assert result == pytest.approx(avg_dist_to_centroid, 0.001)

    # Tests for calculate_number_of_owners
    result = calculate_number_of_owners(well_df)
    assert result == pytest.approx(avg_unique_operator_name, 0.001)


@pytest.mark.parametrize(
    "well_data_missing_values, average_elevation, status",
    [
        (WELL_DATA_MISSING_VALUES, 47.512, True),
        (
            {
                "Elevation Delta [m]": [np.nan],
            },
            np.nan,
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_exceptions_calculation_methods(
    well_data_missing_values, average_elevation, status
):
    well_df = pd.DataFrame(well_data_missing_values)

    if status:
        result = calculate_average(
            well_df, column_name="Elevation Delta [m]", estimation_method="yes"
        )
        assert result == pytest.approx(average_elevation, 0.001)

        with pytest.raises(ValueError):
            calculate_average(well_df, column_name="Priority Score [0-100]")

        # Tests for calculate_age_range exceptions
        with pytest.raises(ValueError):
            calculate_range(well_df, column_name="Age [Years]")

        # Tests for calculate_number_of_owners exceptions
        with pytest.raises(ValueError):
            calculate_number_of_owners(well_df)
    else:
        result = calculate_average(
            well_df, column_name="Elevation Delta [m]", estimation_method="yes"
        )
        assert np.isnan(result)


@pytest.mark.parametrize(
    "well_data, project_scores, centroids",
    [
        (WELL_DATA_BASE_CASE, PROJECT_EFFICIENCY_METRIC_SCORE, PROJECT_CENTROIDS),
        # Add more test cases as needed
    ],
)
def test_process_data(well_data, project_scores, centroids):
    well_df = pd.DataFrame(well_data)
    project_score_df = pd.DataFrame(project_scores)
    result = process_data(well_df, centroids)
    assert_frame_equal(result, project_score_df, check_exact=False, atol=0.01)


# Run tests
if __name__ == "__main__":
    pytest.main()
