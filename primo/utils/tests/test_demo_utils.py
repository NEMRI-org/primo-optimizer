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

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.utils.demo_utils import (
    file_path_widget,
    file_upload_widget,
    generate_configurations,
    get_well_depth,
    get_well_type,
    priority_by_value,
    sort_by_disadvantaged_community_impact,
    sort_columns,
    weight_display,
)

# Sample data for testing
# When converting from Oil [bbl/year] to Oil [Mcf/year], multiply by 5.614583.
# After conversion, the data appear as follows:
# WELL_DATA = [
#     {"Gas [Mcf/year]": 100, "Oil [Mcf/year]": 280.73, "Geoid": 1001021100},
#     {"Gas [Mcf/year]": 150, "Oil [Mcf/year]": 112.29, "Geoid": 1003010100},
#     {"Gas [Mcf/year]": 80, "Oil [Mcf/year]": 505.31, "Geoid": 1003010200},
#     {"Gas [Mcf/year]": 200, "Oil [Mcf/year]": 0, "Geoid": 1003010300},
#     {"Gas [Mcf/year]": 561.4583, "Oil [Mcf/year]": 561.4583, "Geoid": 1003010400},
# ]
WELL_DATA = [
    {"Gas [Mcf/Year]": 100, "Oil [bbl/Year]": 50, "Geoid": 1001021100},
    {"Gas [Mcf/Year]": 150, "Oil [bbl/Year]": 20, "Geoid": 1003010100},
    {"Gas [Mcf/Year]": 80, "Oil [bbl/Year]": 90, "Geoid": 1003010200},
    {"Gas [Mcf/Year]": 200, "Oil [bbl/Year]": 0, "Geoid": 1003010300},
    {"Gas [Mcf/Year]": 561.4583, "Oil [bbl/Year]": 100, "Geoid": 1003010400},
]


TYPE_DATA = ["Oil", "Gas", "Oil", "Gas", "Oil"]


WELL_DATA_ERROR = [
    {"Length": 100, "Height": 50},
    {"Gas [Mcf/year]": 561.4584, "Geoid": 1003010400},
]


@pytest.mark.parametrize(
    "well_data, type_data",
    [
        (WELL_DATA, TYPE_DATA),
        # Add more test cases as needed
    ],
)
@pytest.mark.parametrize(
    "well_data_error",
    [
        (WELL_DATA_ERROR),
        # Add more test cases as needed
    ],
)
def test_get_well_type(well_data, type_data, well_data_error):
    well_df = pd.DataFrame(well_data)
    result_df = get_well_type(well_df)
    assert "Oil [Mcf/year]" not in result_df.columns
    assert "Well Type" in result_df.columns
    assert all(result_df["Well Type"] == type_data)

    with pytest.raises(ValueError):
        well_error_df = pd.DataFrame(well_data_error)
        get_well_type(well_error_df)


# Sample data for testing the sort_columns function
COLUMN_DATA = [
    {
        "Well ID": 1,
        "Incident [Yes/No]": "No",
        "Violation [Yes/No]": "No",
        "Compliance [Yes/No]": "Yes",
        "Leak [Yes/No]": "No",
    },
    {
        "Well ID": 2,
        "Incident [Yes/No]": "Yes",
        "Violation [Yes/No]": "Yes",
        "Compliance [Yes/No]": "Yes",
        "Leak [Yes/No]": "No",
    },
    {
        "Well ID": 3,
        "Incident [Yes/No]": "No",
        "Compliance [Yes/No]": "No",
        "Leak [Yes/No]": "Yes",
    },
    {
        "Well ID": 4,
        "Incident [Yes/No]": "Yes",
        "Compliance [Yes/No]": "Yes",
        "Leak [Yes/No]": "Yes",
    },
]


COLUMN_DATA_MIXED_DATA_TYPES = [
    {
        "Well ID": 1,
        "Incident [Yes/No]": "No",
        "Violation [Yes/No]": np.nan,
        "Compliance [Yes/No]": float(0),
        "Leak [Yes/No]": "Yes",
    },
    {
        "Well ID": 2,
        "Incident [Yes/No]": "No",
        "Violation [Yes/No]": "No",
        "Compliance [Yes/No]": "Yes",
        "Leak [Yes/No]": "No",
    },
    {
        "Well ID": 3,
        "Incident [Yes/No]": 1,
        "Violation [Yes/No]": 0,
        "Compliance [Yes/No]": "No",
        "Leak [Yes/No]": "Yes",
    },
]


COLUMN_DATA_EMPTY = [
    {
        "Well ID": 1,
        "Incident [Yes/No]": np.nan,
        "Violation [Yes/No]": np.nan,
        "Compliance [Yes/No]": np.nan,
        "Leak [Yes/No]": np.nan,
    },
    {
        "Well ID": 2,
        "Incident [Yes/No]": np.nan,
        "Violation [Yes/No]": np.nan,
        "Compliance [Yes/No]": np.nan,
        "Leak [Yes/No]": np.nan,
    },
    {
        "Well ID": 3,
        "Incident [Yes/No]": np.nan,
        "Violation [Yes/No]": np.nan,
        "Compliance [Yes/No]": np.nan,
        "Leak [Yes/No]": np.nan,
    },
]


COLUMN = {
    "Incident [Yes/No]": True,
    "Violation [Yes/No]": False,
    "Compliance [Yes/No]": False,
    "Leak [Yes/No]": True,
}

# Ranking results corresponding to the sample data above
COLUMN_RANKING = [4, 2, 1, 3]
COLUMN_RANKING_MIXED_DATA_TYPES = [2, 1, 3]
COLUMN_RANKING_EMPTY = [1, 2, 3]


@pytest.mark.parametrize(
    "column_data, ranking",
    [
        (COLUMN_DATA, COLUMN_RANKING),
        (COLUMN_DATA_MIXED_DATA_TYPES, COLUMN_RANKING_MIXED_DATA_TYPES),
        (COLUMN_DATA_EMPTY, COLUMN_RANKING_EMPTY),
        # Add more test cases as needed
    ],
)
def test_sort_columns(column_data, ranking):
    column_df = pd.DataFrame(column_data)
    result_df = sort_columns(column_df, COLUMN)
    assert "NULL" in result_df.values
    assert all(result_df["Well ID"] == ranking)


def test_sort_by_disadvantaged_community_impact():
    well_df = pd.DataFrame(WELL_DATA)
    result_df = sort_by_disadvantaged_community_impact(well_df)
    assert "Disadvantaged area %" not in result_df.columns
    assert "Disadvantaged pop %" not in result_df.columns
    assert result_df["Disadvantaged Community"].idxmax() == 0


def test_file_path_widget():
    file_path_widget(
        "Test Widget",
        ".csv",
        "File Selected",
        "primo\\utils\\tests\\screening_data.csv",
    )


def test_file_upload_widget():
    file_upload_widget(
        "Test Widget",
        ".csv",
        "File Uploaded",
    )


@pytest.mark.parametrize(
    "input_value, return_value, status",
    [
        (50.0, 50, True),  # Case 1: the input float is rounded and returned
        (
            30.8,
            30.8,
            True,
        ),  # Case 2: the input float is not rounded and the float is returned
        (0.0, 0, True),  # Case 3: test with zero
        (np.nan, "Error", False),  # Case 4: test with missing data
        # Add more test cases as needed
    ],
)
def test_weight_display(input_value, return_value, status):
    if status:
        assert np.allclose(
            weight_display(input_value), return_value, rtol=1e-5, atol=1e-8
        )
    else:
        with pytest.raises(ValueError):
            weight_display(input_value)


BINARY_INPUT = ["No", "Yes", "NULL"]
BINARY_OUTPUT = [3, 1, 2]
VALUE_INPUT = [10, 35.5, 0, np.nan]
VALUE_OUTPUT = [10, 35.5, 0, np.nan]
MIX_INPUT = ["A", 35.5, "Yes", np.nan]
MIX_OUTPUT = [np.nan, np.nan, 1.0, np.nan]
DATA_MISSING_INPUT = [np.nan, np.nan, np.nan]
DATA_MISSING_OUTPUT = [np.nan, np.nan, np.nan]


@pytest.mark.parametrize(
    "input_column, return_column",
    [
        (BINARY_INPUT, BINARY_OUTPUT),
        (VALUE_INPUT, VALUE_OUTPUT),
        (MIX_INPUT, MIX_OUTPUT),
        (DATA_MISSING_INPUT, DATA_MISSING_OUTPUT),
        # Add more test cases as needed
    ],
)
def test_priority_by_value(input_column, return_column):
    input_column = pd.Series(input_column)
    return_column = pd.Series(return_column)
    pd.testing.assert_series_equal(
        priority_by_value(input_column), return_column, rtol=1e-3, atol=1e-6
    )


@pytest.mark.parametrize(
    "input_dict, threshold, output_dict, status",
    [  # Case 1: Pass case
        (
            [{"Well ID": 1, "Depth [ft]": 2000}],  # Input
            1000.0,  # Threshold
            [{"Well ID": 1, "Depth [ft]": 2000, "Well Depth Type": "Deep"}],  # Output
            True,  # Status
        ),
        # Case 2: The "Depth [ft]" column is not provided
        (
            [{"Well ID": 2, "Latitude": 40.3}],  # Input
            1000,  # Threshold
            "Error",
            False,  # Status
        ),
        # Case 3: The depth of a well is missing
        (
            [{"Well ID": 3, "Depth [ft]": np.nan}],  # Input
            1000,  # Threshold
            [{"Well ID": 3, "Depth [ft]": np.nan, "Well Depth Type": "NULL"}],
            True,  # Status
        ),
        # Case 4: A negative value provided for the threshold
        (
            [{"Well ID": 5, "Depth [ft]": 2000}],  # Input
            -1000,  # Threshold
            "Error",
            False,  # Status
        ),
        # Add more test cases as needed
    ],
)
def test_get_well_depth(input_dict, threshold, output_dict, status):
    input_df = pd.DataFrame(input_dict)
    if status:
        output_df = pd.DataFrame(output_dict)
        pd.testing.assert_frame_equal(
            get_well_depth(input_df, threshold), output_df, rtol=1e-5, atol=1e-8
        )
    else:
        with pytest.raises(ValueError):
            get_well_depth(input_df, threshold)


@pytest.mark.parametrize(
    "weights_file_path, config_path, output_folder_path, life_prod_values,"
    "owner_well_counts, status",
    [  # Case 1: Pass case
        (
            "weights_toy_example.xlsx",
            "config_test.json",
            "..\\config_output",
            [1000],
            [5],
            True,
        ),
        # Case 2: Wrong path to the weight scenario Excel file
        (
            "wrong_file.xlsx",
            "config_test.json",
            "..\\config_output",
            [1000],
            [5],
            False,
        ),
        # Case 3: Wrong path to the base config file
        (
            "weights_toy_example.xlsx",
            "config_wrong.json",
            "..\\config_output",
            [1000],
            [5],
            False,
        ),
        # Case 4: Pass case - no lifelong production constraint
        (
            "weights_toy_example.xlsx",
            "config_test.json",
            "..\\config_output",
            [],
            [5],
            True,
        ),
        # Case 5: No "default" and "owc constraint" sheet in the provided weight scenario Excel file
        (
            "identifiers.csv",
            "config_test.json",
            "..\\config_output",
            [],
            [5],
            None,
        ),
    ],
)
def test_generate_configurations(
    weights_file_path,
    config_path,
    output_folder_path,
    life_prod_values,
    owner_well_counts,
    status,
):
    directory = os.path.dirname(os.path.abspath(__file__))
    weights_file_full_path = os.path.join(directory, weights_file_path)
    config_full_path = os.path.join(directory, config_path)
    if status:
        generate_configurations(
            weights_file_full_path,
            config_full_path,
            output_folder_path,
            life_prod_values,
            owner_well_counts,
        )
    elif status is None:
        with pytest.raises(ValueError):
            generate_configurations(
                weights_file_full_path,
                config_full_path,
                output_folder_path,
                life_prod_values,
                owner_well_counts,
            )
    else:
        with pytest.raises(FileNotFoundError):
            generate_configurations(
                weights_file_full_path,
                config_full_path,
                output_folder_path,
                life_prod_values,
                owner_well_counts,
            )
