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
import logging
import os

# Installed libs
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.data_parser.well_data_columns import WellDataColumnNames

LOGGER = logging.getLogger(__name__)


INCOMPLETE_ROWS = {
    "API Well Number": [5, 10, 15, 20],
    "Age": [2, 4, 6, 8],
    "Depth": [10, 12, 14, 16],
    "x": [2, 3, 4, 5],
    "y": [20, 21, 22],
    "Leak": [7, 17, 27, 37],
    "Compliance": [6, 16, 26, 36],
    "Oil": [15, 20, 21, 25, 26, 28, 38, 41, 42, 49, 50],
    "Gas": [5, 11, 28, 29, 32, 33, 36, 37, 44, 47],
}


# pylint: disable = missing-function-docstring
@pytest.fixture(name="get_well_data_from_csv", scope="function")
def get_well_data_from_csv_fixture():
    """Returns well data from a csv file"""

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Operator Name",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
    )

    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "random_well_data_common.csv",
    )

    wd = WellData(
        data=filename,
        column_names=col_names,
        preliminary_data_check=False,
    )

    return wd


def test_excel_reader(tmp_path, get_well_data_from_csv):
    # Build an excel file in a temp folder
    filename = tmp_path / "my_data.xlsx"
    wd_csv = get_well_data_from_csv
    wd_csv.data.to_excel(filename)

    # Read the excel file from the temp folder
    wd_xlsx = WellData(
        data=str(filename),
        column_names=wd_csv.column_names,
        preliminary_data_check=False,
    )

    pd.testing.assert_frame_equal(wd_csv.data, wd_xlsx.data)
    assert len(wd_csv) == 50
    assert len(wd_xlsx) == 50


def test_unsupported_file_error():
    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Operator Name",
    )

    with pytest.raises(
        TypeError,
        match="Unsupported input file format. Only .xlsx, .xls, and .csv are supported.",
    ):
        WellData(data="file.foo", column_names=col_names)

    with pytest.raises(TypeError, match="Unknown variable type for input data"):
        WellData(data=5, column_names=col_names)


# Test dunder methods
def test_dunder_methods(get_well_data_from_csv):
    wd = get_well_data_from_csv
    col_names = wd.column_names

    # Testing the __contains__ dunder method
    for col in col_names.values():
        assert col in wd

    # The data file has two additional columns. Ensure that
    # those columns are not read
    assert "Elevation Delta [m]" not in wd
    assert "Distance to Road [miles]" not in wd

    # Testing the __iter__ dunder method
    # list(wd) generates the list of rows in wd.data
    assert list(wd) == list(range(2, 52))

    # Testing the __getitem__ dunder method
    assert wd[col_names.age] is wd.data[col_names.age]
    assert wd[col_names.depth] is wd.data[col_names.depth]

    # Testing the __str__, __repr__, and _repr_html_ methods
    # These methods are well tested in pandas, so just calling
    # the method for code coverage
    assert isinstance(str(wd), str)
    assert isinstance(repr(wd), str)
    # pylint: disable = protected-access
    assert isinstance(wd._repr_html_(), str)


def test_has_incomplete_data(caplog, get_well_data_from_csv):
    """Tests the has_incomplete_data method"""
    wd = get_well_data_from_csv
    col_names = wd.column_names

    # Test has_incomplete_data method
    flag, empty_cells = wd.has_incomplete_data(col_names.well_id)
    assert flag
    assert empty_cells == INCOMPLETE_ROWS["API Well Number"]
    assert f"Found empty cells in column {col_names.well_id}" in caplog.text

    flag, empty_cells = wd.has_incomplete_data(col_names.latitude)
    assert flag
    assert empty_cells == INCOMPLETE_ROWS["x"]
    assert f"Found empty cells in column {col_names.latitude}" in caplog.text

    flag, empty_cells = wd.has_incomplete_data(col_names.operator_name)
    assert not flag
    assert empty_cells == []
    assert f"Found empty cells in column {col_names.operator_name}" not in caplog.text


def test_drop_incomplete_data(caplog, get_well_data_from_csv):
    """Tests drop_incomplete_data method"""
    wd = get_well_data_from_csv
    col_names = wd.column_names

    wd.drop_incomplete_data(col_names.well_id, "well_id")
    for i in INCOMPLETE_ROWS["API Well Number"]:
        assert i not in wd.data.index

    # Ensure that the well ids are read as strings
    for i in wd:
        assert isinstance(wd.data.loc[i, col_names.well_id], str)

    assert (
        f"Removed wells because {col_names.well_id} information "
        f"is not available for them."
    ) in caplog.text

    assert (
        wd.get_removed_wells_with_reason["well_id"]
        == INCOMPLETE_ROWS["API Well Number"]
    )
    assert wd.get_removed_wells == INCOMPLETE_ROWS["API Well Number"]
    assert wd.get_removed_wells_with_reason == {
        "well_id": INCOMPLETE_ROWS["API Well Number"]
    }

    # Test existing dict_key when removing rows
    wd.get_removed_wells_with_reason["x"] = []

    wd.drop_incomplete_data(col_names.latitude, "x")
    for i in INCOMPLETE_ROWS["x"]:
        assert i not in wd.data.index

    assert (
        f"Removed wells because {col_names.latitude} information "
        f"is not available for them."
    ) in caplog.text

    # Row 5 has already been removed before, so the list will only have 3 elements
    assert wd.get_removed_wells_with_reason["x"] == [2, 3, 4]

    removed_wells = list(set(INCOMPLETE_ROWS["API Well Number"] + INCOMPLETE_ROWS["x"]))
    removed_wells.sort()
    assert wd.get_removed_wells == removed_wells
    assert wd.get_removed_wells_with_reason == {
        "well_id": INCOMPLETE_ROWS["API Well Number"],
        "x": [2, 3, 4],
    }

    # Operator Name column does not have any empty cells, so no warnings in this case
    wd.drop_incomplete_data(col_names.operator_name, "operator_name")
    assert (
        f"Removed a few wells because {col_names.operator_name} information "
        f"is not available for them."
    ) not in caplog.text
    assert "operator_name" not in wd.get_removed_wells_with_reason

    assert wd.get_removed_wells == removed_wells
    assert wd.get_removed_wells_with_reason == {
        "well_id": INCOMPLETE_ROWS["API Well Number"],
        "x": [2, 3, 4],
    }


def test_fill_incomplete_data(caplog, get_well_data_from_csv):
    """
    Tests fill_incomplete_data method.
    This test also covers the flag_wells_method, and is_data_numeric method.
    """
    wd = get_well_data_from_csv
    col_names = wd.column_names

    flag, non_numeric_rows = wd.is_data_numeric(col_names.ann_gas_production)
    assert not flag
    assert non_numeric_rows == INCOMPLETE_ROWS["Gas"]

    wd.fill_incomplete_data(
        col_name=col_names.ann_gas_production,
        value=wd.config.fill_ann_gas_production,
        flag_col_name="gas_prod_flag",
    )
    assert (
        f"Found empty cells in column {col_names.ann_gas_production}"
    ) in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_gas_production} "
        f"with value {0.0}."
    ) in caplog.text

    # This should have added a new column called gas_prod_flag
    assert "gas_prod_flag" in wd
    for i in wd:
        if i in INCOMPLETE_ROWS["Gas"]:
            assert wd.data.loc[i, "gas_prod_flag"] == 1
            assert np.isclose(wd.data.loc[i, col_names.ann_gas_production], 0)

        else:
            assert wd.data.loc[i, "gas_prod_flag"] == 0

    assert wd.get_flag_columns == ["gas_prod_flag"]

    # After filling the data, the column must be numeric
    flag, non_numeric_rows = wd.is_data_numeric(col_names.ann_gas_production)
    assert flag
    assert non_numeric_rows == []

    flag, non_numeric_rows = wd.is_data_numeric(col_names.ann_oil_production)
    assert not flag
    assert non_numeric_rows == INCOMPLETE_ROWS["Oil"]

    wd.fill_incomplete_data(
        col_name=col_names.ann_oil_production,
        value=wd.config.fill_ann_oil_production,
        flag_col_name="oil_prod_flag",
    )
    assert (
        f"Found empty cells in column {col_names.ann_oil_production}"
    ) in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.ann_oil_production} "
        f"with value {0.0}."
    ) in caplog.text

    assert "oil_prod_flag" in wd
    for i in wd:
        if i in INCOMPLETE_ROWS["Oil"]:
            assert wd.data.loc[i, "oil_prod_flag"] == 1
            assert np.isclose(wd.data.loc[i, col_names.ann_oil_production], 0)

        else:
            assert wd.data.loc[i, "oil_prod_flag"] == 0

    assert wd.get_flag_columns == ["gas_prod_flag", "oil_prod_flag"]

    # After filling the data, the column must be numeric
    flag, non_numeric_rows = wd.is_data_numeric(col_names.ann_oil_production)
    assert flag
    assert non_numeric_rows == []

    # Operator name column has no empty cells, so it should not fill anything
    wd.fill_incomplete_data(
        col_name=col_names.operator_name,
        value="Foo",
        flag_col_name="operator_flag",
    )
    assert f"Found empty cells in column {col_names.operator_name}" not in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.operator_name} "
        f"with value Foo."
    ) not in caplog.text

    assert "operator_flag" not in wd
    assert wd.get_flag_columns == ["gas_prod_flag", "oil_prod_flag"]

    # This file does not calculate any priority scores
    assert wd.get_priority_score_columns == []


def test_replace_data(caplog, get_well_data_from_csv):
    wd = get_well_data_from_csv
    col_names = wd.column_names

    flag, non_numeric_rows = wd.is_data_numeric(col_names.leak)
    assert not flag
    # All rows must be non-numeric
    assert non_numeric_rows == list(range(2, 52))

    # Fill incomplete cells
    wd.fill_incomplete_data(col_names.leak, False, "leak_flag")
    assert f"Found empty cells in column {col_names.leak}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.leak} with value {False}."
    ) in caplog.text

    wd.replace_data(
        col_names.leak,
        {"Yes": 1, "No": 0, True: 1, False: 0},
    )

    flag, non_numeric_rows = wd.is_data_numeric(col_names.leak)
    assert flag
    # All rows must numeric now
    assert non_numeric_rows == []

    for i in wd:
        if i in INCOMPLETE_ROWS["Leak"]:
            assert wd.data.loc[i, "leak_flag"] == 1
        else:
            assert wd.data.loc[i, "leak_flag"] == 0


def test_convert_data_to_binary(caplog, get_well_data_from_csv):
    wd = get_well_data_from_csv
    col_names = wd.column_names

    # Fill incomplete cells
    wd.fill_incomplete_data(col_names.leak, False, "leak_flag")
    assert f"Found empty cells in column {col_names.leak}" in caplog.text
    assert (
        f"Filling any empty cells in column {col_names.leak} with value {False}."
    ) in caplog.text

    wd.convert_data_to_binary(col_names.leak)

    flag, non_numeric_rows = wd.is_data_numeric(col_names.leak)
    assert flag
    # All rows must numeric now
    assert non_numeric_rows == []

    # Test the non-boolean-type error
    for i in INCOMPLETE_ROWS["Compliance"]:
        wd.data.loc[i, col_names.compliance] = "NULL"

    with pytest.raises(
        ValueError,
        match=(
            "Column Compliance \\[Yes/No\\] is expected to contain boolean-type "
            "data. Received a non-boolean value for some/all rows."
        ),
    ):
        wd.convert_data_to_binary(col_names.compliance)


def test_check_data_in_range(get_well_data_from_csv):
    wd = get_well_data_from_csv
    col_names = wd.column_names

    wd.fill_incomplete_data(col_names.age, wd.config.fill_age, "age_flag")
    with pytest.raises(
        ValueError,
        match=(
            "Values in column Age \\[Years\\] are expected to be in the interval "
            "\\[0, 110\\]. However, the value in one or more rows "
            "lies outside the interval."
        ),
    ):
        wd.check_data_in_range(col_names.age, 0, 110)

    assert wd.check_data_in_range(col_names.age, 0, 200) is None


def test_add_new_columns_method(get_well_data_from_csv):
    wd = get_well_data_from_csv
    with pytest.raises(
        NotImplementedError,
        match=(
            "This method is not supported currently. Ensure that all the required columns "
            "are specified in the WellDataColumnNames object. To read unsupported data, "
            "register the corresponding column using the `register_new_columns` method."
        ),
    ):
        wd.add_new_columns()
