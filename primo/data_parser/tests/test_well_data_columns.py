#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the DOE's National Emission Reduction
# Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for
# itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit others to do so.
#
#################################################################################

# Standard libs
import logging

# Installed libs
import pytest

# User-defined libs
from primo.data_parser import ImpactMetrics, WellDataColumnNames

LOGGER = logging.getLogger(__name__)


# pylint: disable = missing-function-docstring, no-member
def test_well_data_column_names():
    wcn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        additional_columns={
            "c1": "Custom Column 1",
            "c2": "Custom Column 2",
        },
    )

    cols = ["well_id", "latitude", "longitude", "age", "depth", "c1", "c2"]

    assert wcn.well_id == "Well API"
    assert wcn.latitude == "Latitude"
    assert wcn.longitude == "Longitude"
    assert wcn.age == "Age [Years]"
    assert wcn.depth == "Depth [ft]"
    assert wcn.c1 == "Custom Column 1"
    assert wcn.c2 == "Custom Column 2"

    # Testing contains, keys, and values methods
    for key in cols:
        assert key in wcn

    assert wcn.keys() == cols
    assert wcn.values() == [
        "Well API",
        "Latitude",
        "Longitude",
        "Age [Years]",
        "Depth [ft]",
        "Custom Column 1",
        "Custom Column 2",
    ]

    # Testing items method
    for key, val in wcn.items():
        assert key in cols
        assert val is not None

    # Testing iterator method
    for key in wcn:
        if key not in cols:
            assert getattr(wcn, key) is None

    # Test register new column method
    wcn.register_new_columns({"new_col_1": "New Column 1"})
    assert wcn.new_col_1 == "New Column 1"

    with pytest.raises(
        AttributeError,
        match=("Attribute new_col_1 is already defined. Use a different name."),
    ):
        wcn.register_new_columns({"new_col_1": "New Column 1"})

    with pytest.raises(
        ValueError, match="Key new col 3 is not a valid python variable name!"
    ):
        wcn.register_new_columns({"new col 3": "New Column 3"})


@pytest.fixture(name="get_well_data_cols", scope="function")
def get_well_data_cols_fixture():
    im_mt = ImpactMetrics()
    # Work with fewer metrics for convenience
    im_mt.delete_metric("environment")
    im_mt.delete_metric("other_emissions")
    im_mt.delete_metric("well_age")
    im_mt.delete_metric("well_count")
    im_mt.delete_submetric("violation")
    im_mt.delete_submetric("incident")
    im_mt.delete_submetric("buildings_near")
    im_mt.delete_submetric("buildings_far")

    # Now, the object has five metrics
    # Set weights for all metrics
    im_mt.set_weight(
        {
            "ch4_emissions": 20,
            "dac_impact": 20,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_integrity": 20,
            # submetrics
            "leak": 50,
            "compliance": 50,
            "fed_dac": 40,
            "state_dac": 60,
            "hospitals": 50,
            "schools": 50,
            "ann_gas_production": 40,
            "ann_oil_production": 60,
        }
    )

    wcn = WellDataColumnNames(
        well_id="Well API",
        latitude="Latitude",
        longitude="Longitude",
        age="Age [Years]",
        depth="Depth [ft]",
        operator_name="Operator Name",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        state_dac="State DAC Score",
        hospitals="Num Hospitals Nearby",
        schools="Num Schools nearby",
        ann_gas_production="Gas [Mcf/yr]",
        ann_oil_production="Oil [bbl/yr]",
        well_integrity="Well Integrity Status",
    )

    return im_mt, wcn


def test_no_warnings_case(get_well_data_cols):
    """
    Checks if the `check_columns_available` method works as expected
    """
    im_mt, wcn = get_well_data_cols
    assert im_mt.check_validity() is None

    # The object has all the required inputs, so this should
    # not raise any warnings or errors.
    assert wcn.check_columns_available(im_mt) is None
    # This should register `the data_col_name` attribute
    assert im_mt.leak.data_col_name == wcn.leak
    assert im_mt.compliance.data_col_name == wcn.compliance
    assert im_mt.fed_dac.data_col_name is None
    assert im_mt.state_dac.data_col_name == wcn.state_dac
    assert im_mt.hospitals.data_col_name == wcn.hospitals
    assert im_mt.schools.data_col_name == wcn.schools
    assert im_mt.ann_production_volume.data_col_name is None
    assert im_mt.five_year_production_volume.data_col_name is None


def test_unsupported_metric_warn(caplog, get_well_data_cols):

    im_mt, wcn = get_well_data_cols
    # Test unsupported metric warning
    im_mt.register_new_metric(name="metric_1", full_name="Metric One")
    wcn.check_columns_available(im_mt)

    assert (
        "Metric/submetric metric_1/Metric One is not supported. "
        "Users are required to process the data for this metric, and "
        "assign the name of the column to the attribute `data_col_name` "
        "in the metric/submetric metric_1 object."
    ) in caplog.text


def test_missing_col_error(get_well_data_cols):
    im_mt, wcn = get_well_data_cols

    # Suppose well_integrity data is not provided
    wcn.well_integrity = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric well_integrity is nonzero, so attribute "
            "well_integrity is an essential input in the "
            "WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    # Repeat the test with a submetric
    wcn.well_integrity = "Well Integrity Status"
    wcn.hospitals = None

    with pytest.raises(
        AttributeError,
        match=(
            "Weight of the metric hospitals is nonzero, so attribute "
            "hospitals is an essential input in the "
            "WellDataColumnNames object."
        ),
    ):
        wcn.check_columns_available(im_mt)

    wcn.hospitals = "Hospitals"
