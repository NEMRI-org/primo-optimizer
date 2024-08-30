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
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# User-defined libs
from primo.utils.estimation_utils import age_estimation, get_record_completeness

# Sample data for testing
WELL_DATA_SET_1 = [
    {
        "Well_Latitude": 40.0,
        "Well_Longitude": -73.0,
        "Well_Age": 50,
        "Well_depth": 2300,
        "5-Year Gas Production [Mcf]": 102,
        "State Wetlands (300 - 1,320 ft)": "FALSE",
        "Compliance [Yes/No]": "No",
        "Production date": "2020-01-01",
    }
]

WELL_DATA_SET_2 = [
    {
        "Well_Latitude": 40.0,
        "Well_Longitude": -73.0,
        "Well_Age": pd.NaT,
        "Well_depth": np.nan,
        "5-Year Gas Production [Mcf]": 102,
        "State Wetlands (300 - 1,320 ft)": "FALSE",
        "Compliance [Yes/No]": "NULL",
        "Production date": "2020-01-01",
    }
]

WELL_DATA_SET_3 = [
    {
        "Well_Latitude": np.nan,
        "Well_Longitude": np.nan,
        "Well_Age": pd.NA,
        "Well_depth": np.nan,
        "5-Year Gas Production [Mcf]": pd.NA,
        "State Wetlands (300 - 1,320 ft)": "NULL",
        "Compliance [Yes/No]": "NULL",
        "Production date": pd.NaT,
    }
]

WELL_DATA_ERROR_1 = [
    {
        "Well_Latitude": 40.0,
        "Well_Longitude": -73.0,
        "Well_Age": 2300,
        "5-Year Gas Production [Mcf]": 102,
        "State Wetlands (300 - 1,320 ft)": "FALSE",
        "Compliance [Yes/No]": "No",
        "Production year": "2020-01-01",
    }
]


COLUMN_CRITERIA_ERROR_1 = [
    "Well_Latitude",
    "Well_Longitude",
    "Well_Age",
    "Well_depth",
    "5-Year Gas Production [Mcf]",
    "State Wetlands (300 - 1,320 ft)",
    "Random",
]

COLUMN_CRITERIA_ERROR_2 = [
    "Well_Latitude",
    "Well_Longitude",
    "Well_Age",
    "Well_depth",
    "5-Year Gas Production [Mcf]",
    "State Wetlands (300 - 1,320 ft)",
    "Compliance [Yes/No]",
    "Production date",
    "Random",
]

COLUMN_CRITERIA_ERROR_3 = [
    "Well_Latitude",
    "Well_Longitude",
    "Well_Age",
    "Well_Age",
    "5-Year Gas Production [Mcf]",
    "State Wetlands (300 - 1,320 ft)",
    "Compliance [Yes/No]",
    "Production date",
]

COLUMN_CRITERIA = [
    "Well_Latitude",
    "Well_Longitude",
    "Well_Age",
    "Well_depth",
    "5-Year Gas Production [Mcf]",
    "State Wetlands (300 - 1,320 ft)",
    "Compliance [Yes/No]",
    "Production date",
]
WELL_DATA_AGE = {
    "API number": [1, 7, 8, 12, 3, 10, 22, 25, 4, 17],
    "Well Age": [np.nan, 2, 0, 8, 34, 8, np.nan, np.nan, 2, 14],
}
WELL_DATA_AGE_NEG_ERROR = {
    "API number": [1, 7, 8, 12, 3, 10, 22, 25, 4, 17],
    "Age": [-1, 2, 0, 8, 34, 8, np.nan, np.nan, 2, 14],
}

WELL_DATA_AGE_STR_ERROR = {
    "API number": [1, 7, 8, 12, 3, 10, 22, 25, 4, 17],
    "Age": ["1", 2, 0, 8, 34, 8, np.nan, np.nan, 2, 14],
}

WELL_DATA_AGE_RESULTS = {
    "API number": [1, 7, 8, 12, 3, 10, 22, 25, 4, 17],
    "Well Age": [18.0, 2.0, 5.0, 8.0, 34.0, 8.0, 11.0, 11.0, 2.0, 14.0],
}


WELL_DATA_DEPTH = {
    "latitude": [
        42.878809,
        42.856099,
        42.873682,
        42.873682,
        43.207792,
        43.213305,
        43.252701,
    ],
    "longitude": [
        -75.869512,
        -75.810540,
        -75.791549,
        -75.791600,
        -76.188250,
        -76.176483,
        -77.242657,
    ],
    "Depth [ft]": [np.nan, np.nan, 4500, 2300, np.nan, 5500, 3000],
}

WELL_DATA_DEPTH_RESULT = {
    "latitude": [
        42.878809,
        42.856099,
        42.873682,
        42.873682,
        43.207792,
        43.213305,
        43.252701,
    ],
    "longitude": [
        -75.869512,
        -75.810540,
        -75.791549,
        -75.791600,
        -76.188250,
        -76.176483,
        -77.242657,
    ],
    "Depth [ft]": [0.0, 3392.0, 4500, 2300, 5496.0, 5500, 3000],
}

# TIF_FILE= "orig_dem.csv"
# if not os.path.exists(TIF_FILE):
#     TIF_FILE = os.path.join("primo", "utils","tests", TIF_FILE)

WELL_DATA_DEPTH_STR_ERROR = {
    "latitude": [
        42.878809,
        42.856099,
        42.873682,
        42.873682,
        43.207792,
        43.213305,
        43.252701,
    ],
    "longitude": [
        -75.869512,
        -75.810540,
        -75.791549,
        -75.791600,
        -76.188250,
        -76.176483,
        -77.242657,
    ],
    "Depth [ft]": ["2000", np.nan, 4500, 2300, np.nan, 5500, 3000],
}

WELL_DATA_DEPTH__LAT_ERROR = {
    "latitude": [
        -500,
        42.856099,
        42.873682,
        42.873682,
        43.207792,
        43.213305,
        43.252701,
    ],
    "longitude": [
        -75.869512,
        -75.810540,
        -75.791549,
        -75.791600,
        -76.188250,
        -76.176483,
        -77.242657,
    ],
    "Depth [ft]": [np.nan, np.nan, 4500, 2300, np.nan, 5500, 3000],
}

WELL_DATA_DEPTH__LON_ERROR = {
    "latitude": [
        42.878809,
        42.856099,
        42.873682,
        42.873682,
        43.207792,
        43.213305,
        43.252701,
    ],
    "longitude": [
        500,
        -75.810540,
        -75.791549,
        -75.791600,
        -76.188250,
        -76.176483,
        -77.242657,
    ],
    "Depth [ft]": [np.nan, np.nan, 4500, 2300, np.nan, 5500, 3000],
}


@pytest.mark.parametrize(
    "well_data_set_1, well_data_set_2, well_data_set_3, column_criteria",
    [
        (WELL_DATA_SET_1, WELL_DATA_SET_2, WELL_DATA_SET_3, COLUMN_CRITERIA),
        # Add more test cases as needed
    ],
)
@pytest.mark.parametrize(
    "well_data_error_1,column_criteria_error_1, column_criteria_error_2,column_criteria_error_3",
    [
        (
            WELL_DATA_ERROR_1,
            COLUMN_CRITERIA_ERROR_1,
            COLUMN_CRITERIA_ERROR_2,
            COLUMN_CRITERIA_ERROR_3,
        ),
        # Add more test cases as needed
    ],
)
def test_get_record_completeness(
    well_data_set_1,
    well_data_set_2,
    well_data_set_3,
    well_data_error_1,
    column_criteria,
    column_criteria_error_1,
    column_criteria_error_2,
    column_criteria_error_3,
):

    well_set_1_df = pd.DataFrame(well_data_set_1)
    well_set_2_df = pd.DataFrame(well_data_set_2)
    well_set_3_df = pd.DataFrame(well_data_set_3)
    result_1_df = get_record_completeness(well_set_1_df, column_criteria)
    result_2_df = get_record_completeness(well_set_2_df, column_criteria)
    result_3_df = get_record_completeness(well_set_3_df, column_criteria)
    assert "record completeness" in result_1_df.columns
    assert all(result_1_df["record completeness"] == 8)
    assert "record completeness" in result_2_df.columns
    assert all(result_2_df["record completeness"] == 5)
    assert "record completeness" in result_3_df.columns
    assert all(result_3_df["record completeness"] == 0)

    # An entry (entries) in column_criteria is not present in the well DataFrame.
    with pytest.raises(ValueError):
        get_record_completeness(well_set_1_df, column_criteria_error_1)

    # Column_Criteria is larger than the well_set_1_df
    with pytest.raises(ValueError):
        get_record_completeness(well_set_1_df, column_criteria_error_2)

    # Duplicate columns in the criteria column list
    with pytest.raises(ValueError):
        get_record_completeness(well_set_1_df, column_criteria_error_3)

    # Duplicate columns in the input DataFrame list
    with pytest.raises(ValueError):
        well_data_error_1_df = pd.DataFrame(well_data_error_1)
        get_record_completeness(well_data_error_1_df, column_criteria)


@pytest.mark.parametrize(
    "well_age_data_set_1,well_age_data_neg_error,well_age_data_str_error, well_age_est_results",
    [
        (
            WELL_DATA_AGE,
            WELL_DATA_AGE_NEG_ERROR,
            WELL_DATA_AGE_STR_ERROR,
            WELL_DATA_AGE_RESULTS,
        ),
        # Add more test cases as needed
    ],
)
def test_age_estimation(
    well_age_data_set_1,
    well_age_data_neg_error,
    well_age_data_str_error,
    well_age_est_results,
):
    well_age_data_set = pd.DataFrame(well_age_data_set_1)
    well_age_data_neg_error_set = pd.DataFrame(well_age_data_neg_error)
    well_age_data_str_error_set = pd.DataFrame(well_age_data_str_error)
    actual_results = pd.DataFrame(well_age_est_results).sort_values(by="API number")
    calculated_results = age_estimation(
        well_age_data_set, age_col_name="Well Age", num_values=2
    )

    # Test to make sure age was estimated properly and NaN values were correctly spotted.
    print(f" calculated results {calculated_results}")
    print(f" actual results {actual_results}")
    assert_frame_equal(calculated_results, actual_results, check_exact=False, atol=0.01)

    # Test the exception that age values are within acceptable bounds and has numerical values
    with pytest.raises(ValueError):
        age_estimation(well_age_data_str_error_set, age_col_name="Age", num_values=2)

    with pytest.raises(ValueError):
        age_estimation(well_age_data_neg_error_set, age_col_name="Age", num_values=2)

    # Test the exception if column_name is not ana actual name of a column in the data set.
    with pytest.raises(ValueError):
        age_estimation(well_age_data_set, age_col_name="Age", num_values=2)


# TODO: Need to create a small test raster file for retrieving elevation data from
#       in order to test the depth estimation and get elevation methods.

# @pytest.mark.parametrize(
#     "well_depth_data_set_1, well_depth_results, tif_file_path",
#     [
#         (
#             WELL_DATA_DEPTH,
#             WELL_DATA_DEPTH_RESULT,
#         ),
#         # Add more test cases as needed
#     ],
# )

# def test_depth_estimation(
#     well_depth_data_set_1,
#     well_depth_results,
#     tif_file_path,
# ):
#     well_depth_data_set_1 = pd.DataFrame(well_depth_data_set_1)
#     actual_results= pd.DataFrame(well_depth_results)
#     calculated_results = depth_estimation(well_depth_data_set_1,
#                                           tif_file_path,
#                                           "latitude",
#                                           "longitude",
#                                           "Depth [ft]",
#                                           distance_miles=2.0)
#     assert_frame_equal(calculated_results, actual_results, check_exact=False, atol=0.01)

# Run tests
if __name__ == "__main__":
    pytest.main()
