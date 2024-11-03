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

# Installed libs
import pytest
from pyomo.common.config import Bool, NonNegativeFloat

# User-defined libs
from primo.data_parser.metric_data import (
    EfficiencyMetrics,
    ImpactMetrics,
    Metric,
    SetOfMetrics,
    SubMetric,
)

LOGGER = logging.getLogger(__name__)


# pylint: disable = no-member, missing-function-docstring
# pylint: disable = protected-access
# pylint: disable = too-many-statements
def test_metric_class(caplog):
    z = Metric(name="met_1", weight=50, full_name="Metric One")

    assert z.name == "met_1"
    assert z.full_name == "Metric One"
    assert (z.min_weight, z.weight, z.max_weight) == (0, 50, 100)
    assert z.effective_weight == 50
    assert not z.is_submetric
    print_format = (
        "Metric name: Metric One, Metric weight: 50 \n    Admissible range: [0, 100]"
    )
    assert str(z) == print_format

    assert z.data_col_name is None
    assert z.score_col_name is None
    assert z.score_attribute is None

    assert z.is_binary_type is False
    assert z.has_inverse_priority is False
    assert z.fill_missing_value is None
    assert z._fill_missing_value.name() == "met_1"
    z.fill_missing_value = "foo"
    assert z.fill_missing_value == "foo"

    z.data_col_name = "Metric One"
    assert z.score_col_name == "Metric One Score [0-50]"
    assert z.score_attribute == "met_1_eff_score_0_50"

    # Add test to capture the name modification error
    with pytest.raises(
        ValueError,
        match=("Metric's key name cannot be modified after it is defined."),
    ):
        z.name = "modified_met_1"

    # Add a test to catch a metric name that is not a valid Python variable name
    with pytest.raises(
        ValueError,
        match=(
            "Received met 2 for Metric's key name, which is not a valid python variable name!"
        ),
    ):
        print(Metric("met 2", 100))

    # Add test to capture the weight out-of-range error
    with pytest.raises(
        ValueError,
        match=(
            "Attempted to assign -50 for metric met_1, which "
            "lies outside the admissible range \\[0, 100\\]."
        ),
    ):
        z.weight = -50

    # Test warning for assigning non-integer value
    z.weight = 50.7
    assert (
        "Received 50.7, a non-integer value for weight. "
        "Rounding it to 51, the nearest integer value."
    ) in caplog.text

    # Test the _configure_fill_missing_value method
    z._configure_fill_missing_value(domain=NonNegativeFloat, default=3)
    assert z.fill_missing_value == 3.0
    z.fill_missing_value = 5
    assert z.fill_missing_value == 5.0
    assert not z.is_binary_type

    with pytest.raises(
        ValueError,
        match=(
            "invalid value for configuration 'met_1':\n"
            "\tFailed casting foo\n\tto NonNegativeFloat\n"
            "\tError: could not convert string to float: 'foo'"
        ),
    ):
        z.fill_missing_value = "foo"

    z._configure_fill_missing_value(domain=Bool, default="Yes")
    assert z.is_binary_type
    assert z.fill_missing_value is True


def test_submetric_class():
    z_par = Metric("par_metric", 40, full_name="Parent Metric One")
    z_sub = SubMetric("sub_metric", z_par, 50, full_name="SubMetric One")

    assert z_sub.is_submetric
    assert z_sub.parent_metric is z_par
    assert z_sub.effective_weight == 20
    print_format = (
        "Submetric name: SubMetric One, Submetric weight: 50 \n"
        "    Admissible range: [0, 100] \n"
        "    Is a submetric of Parent Metric One"
    )
    assert str(z_sub) == print_format


def test_set_of_metrics_class():
    z = SetOfMetrics()

    z.register_new_metric("met_1", 33, "Metric One")
    z.register_new_metric("met_2", 33, "Metric Two")
    z.register_new_metric("met_3", 34, "Metric Three")

    print_format = (
        "        Metric Name  Metric weight\n"
        "met_1    Metric One             33\n"
        "met_2    Metric Two             33\n"
        "met_3  Metric Three             34\n"
        "              Total            100"
    )

    assert str(z) == print_format

    z.register_new_submetric("sub_met_1_1", z.met_1, 50, "Sub Metric One-One")
    z.register_new_submetric("sub_met_1_2", z.met_1, 50, "Sub Metric One-Two")
    z.register_new_submetric("sub_met_3_1", z.met_3, 25, "Sub Metric Three-One")
    z.register_new_submetric("sub_met_3_2", z.met_3, 75, "Sub Metric Three-Two")

    print_format = (
        "        Metric Name  Metric weight\n"
        "met_1    Metric One             33\n"
        "met_2    Metric Two             33\n"
        "met_3  Metric Three             34\n"
        "              Total            100\n\n\n"
        "Primary metric Metric One, with weight 33, has submetrics:\n"
        "================================================================================\n"
        "                 Submetric Name  Submetric weight\n"
        "sub_met_1_1  Sub Metric One-One                50\n"
        "sub_met_1_2  Sub Metric One-Two                50\n"
        "                          Total               100\n\n\n"
        "Primary metric Metric Three, with weight 34, has submetrics:\n"
        "================================================================================\n"
        "                   Submetric Name  Submetric weight\n"
        "sub_met_3_1  Sub Metric Three-One                25\n"
        "sub_met_3_2  Sub Metric Three-Two                75\n"
        "                            Total               100"
    )

    assert str(z) == print_format

    assert z.check_validity() is None
    assert isinstance(z.met_1, Metric)
    assert isinstance(z.met_2, Metric)
    assert isinstance(z.met_3, Metric)
    assert isinstance(z.sub_met_1_1, SubMetric)
    assert isinstance(z.sub_met_1_2, SubMetric)
    assert isinstance(z.sub_met_3_1, SubMetric)
    assert isinstance(z.sub_met_3_2, SubMetric)

    assert z.met_1.submetrics == {
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
    }
    assert not hasattr(z.met_2, "submetrics")
    assert z.met_3.submetrics == {
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
    }

    _all_metrics = {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
    }
    assert dict(z.items()) == _all_metrics
    assert z.get_primary_metrics == {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
    }
    assert z.get_submetrics == {
        "met_1": {"sub_met_1_1": z.sub_met_1_1, "sub_met_1_2": z.sub_met_1_2},
        "met_3": {"sub_met_3_1": z.sub_met_3_1, "sub_met_3_2": z.sub_met_3_2},
    }
    assert z._get_all_metrics_extended == {
        "met_1": z.met_1,
        "met_2": z.met_2,
        "met_3": z.met_3,
        "sub_met_1_1": z.sub_met_1_1,
        "sub_met_1_2": z.sub_met_1_2,
        "sub_met_3_1": z.sub_met_3_1,
        "sub_met_3_2": z.sub_met_3_2,
        "Metric One": z.met_1,
        "Metric Two": z.met_2,
        "Metric Three": z.met_3,
        "Sub Metric One-One": z.sub_met_1_1,
        "Sub Metric One-Two": z.sub_met_1_2,
        "Sub Metric Three-One": z.sub_met_3_1,
        "Sub Metric Three-Two": z.sub_met_3_2,
    }

    # Test iterator method and contains method
    for obj in z:
        assert obj in _all_metrics.values()
        assert obj.name in z
        assert obj.full_name in z

    # Test for receiving update error: Enter a typo in the input dict
    with pytest.raises(
        KeyError,
        match=("Metrics/submetrics \\['met 1'\\] are not recognized/registered."),
    ):
        z.set_weight({"met 1": 33, "Metric Two": 33})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError, match=("Sum of weights of primary metrics does not add up to 100")
    ):
        z.set_weight({"met_1": 33, "met_2": 30})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Weight of the primary metric met_1 is zero, but the sum of "
            "weights of its submetrics is 100, which is nonzero."
        ),
    ):
        z.set_weight({"met_1": 0, "met_2": 66})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Sum of weights of submetrics of the primary metric met_1 does not add up to 100."
        ),
    ):
        z.set_weight(
            {"met_1": 33, "met_2": 33}, submetrics={"met_1": {"sub_met_1_1": 0}}
        )

    z.set_weight({"sub_met_1_1": 50})

    # Test for receiving error raised when registering non-Metric instances
    with pytest.raises(
        TypeError,
        match=(
            "Attributes of SetOfMetrics must be instances of Metric. "
            "Attempted to register Foo."
        ),
    ):
        z.foo = "Foo"

    # Test for receiving error when an existing metric is overwritten
    with pytest.raises(
        ValueError,
        match=(
            "Metric/submetric met_2 has already been registered. "
            "Attempting to register a new metric with the same name."
        ),
    ):
        z.register_new_metric("met_2")

    # Try deleting primary metric on z.met_1. This should automatically delete submetrics
    z.delete_metric("met_1")
    assert not hasattr(z, "met_1")
    assert not hasattr(z, "sub_met_1_1")
    assert not hasattr(z, "sub_met_1_2")
    assert len(list(z)) == 4

    # Test for receiving error when deleting a metric that does not exist
    with pytest.raises(
        AttributeError, match=("Metric/submetric met_1 does not exist.")
    ):
        z.delete_metric("met_1")

    # Try deleting submetric for z.sub_met_3_1.
    z.delete_submetric("sub_met_3_1")
    assert not hasattr(z, "sub_met_3_1")

    # Test for receiving error when deleting a metric that does not exist
    with pytest.raises(AttributeError, match="Submetric sub_met_3_1 does not exist."):
        z.delete_submetric("sub_met_3_1")


def test_impact_metrics_class():
    im_wt = ImpactMetrics()

    assert hasattr(im_wt, "ch4_emissions")
    assert hasattr(im_wt, "dac_impact")
    assert hasattr(im_wt, "sensitive_receptors")
    assert hasattr(im_wt, "ann_production_volume")
    assert hasattr(im_wt, "five_year_production_volume")
    assert hasattr(im_wt, "well_age")
    assert hasattr(im_wt, "well_count")
    assert hasattr(im_wt, "other_emissions")
    assert hasattr(im_wt, "well_integrity")
    assert hasattr(im_wt, "environment")

    assert hasattr(im_wt, "leak")
    assert hasattr(im_wt, "compliance")
    assert hasattr(im_wt, "violation")
    assert hasattr(im_wt, "incident")
    assert im_wt.ch4_emissions.submetrics == {
        "leak": im_wt.leak,
        "compliance": im_wt.compliance,
        "violation": im_wt.violation,
        "incident": im_wt.incident,
    }
    assert im_wt.compliance.is_binary_type
    assert im_wt.compliance.has_inverse_priority
    assert im_wt.compliance._fill_missing_value._domain is Bool
    assert im_wt.compliance.fill_missing_value

    assert hasattr(im_wt, "fed_dac")
    assert hasattr(im_wt, "state_dac")
    assert im_wt.dac_impact.submetrics == {
        "fed_dac": im_wt.fed_dac,
        "state_dac": im_wt.state_dac,
    }

    assert hasattr(im_wt, "hospitals")
    assert hasattr(im_wt, "schools")
    assert hasattr(im_wt, "buildings_near")
    assert hasattr(im_wt, "buildings_far")
    assert im_wt.sensitive_receptors.submetrics == {
        "hospitals": im_wt.hospitals,
        "schools": im_wt.schools,
        "buildings_near": im_wt.buildings_near,
        "buildings_far": im_wt.buildings_far,
    }

    assert hasattr(im_wt, "fed_wetlands_near")
    assert hasattr(im_wt, "fed_wetlands_far")
    assert hasattr(im_wt, "state_wetlands_near")
    assert hasattr(im_wt, "state_wetlands_far")
    assert im_wt.environment.submetrics == {
        "fed_wetlands_near": im_wt.fed_wetlands_near,
        "state_wetlands_near": im_wt.state_wetlands_near,
        "fed_wetlands_far": im_wt.fed_wetlands_far,
        "state_wetlands_far": im_wt.state_wetlands_far,
    }

    assert hasattr(im_wt, "ann_gas_production")
    assert hasattr(im_wt, "ann_oil_production")
    assert hasattr(im_wt, "five_year_gas_production")
    assert hasattr(im_wt, "five_year_oil_production")
    assert im_wt.ann_production_volume.submetrics == {
        "ann_gas_production": im_wt.ann_gas_production,
        "ann_oil_production": im_wt.ann_oil_production,
    }
    assert im_wt.five_year_production_volume.submetrics == {
        "five_year_gas_production": im_wt.five_year_gas_production,
        "five_year_oil_production": im_wt.five_year_oil_production,
    }

    assert hasattr(im_wt, "h2s_leak")
    assert hasattr(im_wt, "brine_leak")
    assert im_wt.other_emissions.submetrics == {
        "h2s_leak": im_wt.h2s_leak,
        "brine_leak": im_wt.brine_leak,
    }

    assert not hasattr(im_wt.well_age, "submetrics")
    assert not hasattr(im_wt.well_count, "submetrics")
    assert not hasattr(im_wt.well_integrity, "submetrics")

    im_wt.set_weight(
        primary_metrics={
            "ch4_emissions": 25,
            "sensitive_receptors": 30,
            "environment": 20,
            "dac_impact": 25,
        },
        submetrics={
            "ch4_emissions": {
                "leak": 30,
                "compliance": 20,
                "violation": 50,
            },
            "dac_impact": {
                "fed_dac": 60,
                "state_dac": 40,
            },
            "environment": {
                "fed_wetlands_near": 50,
                "state_wetlands_near": 50,
            },
            "sensitive_receptors": {
                "buildings_near": 20,
                "hospitals": 40,
                "schools": 40,
            },
        },
    )

    assert im_wt.check_validity() is None

    im_widget = im_wt.build_widget()
    im_widget.confirm_weights(None)
    im_wt.set_weight_from_widget(im_widget)
    assert im_wt.check_validity() is None


def test_efficiency_metrics_class():
    ef_wt = EfficiencyMetrics()

    assert hasattr(ef_wt, "num_wells")
    assert hasattr(ef_wt, "num_unique_owners")
    assert hasattr(ef_wt, "avg_elevation_delta")
    assert hasattr(ef_wt, "age_range")
    assert hasattr(ef_wt, "depth_range")
    assert hasattr(ef_wt, "record_completeness")
    assert hasattr(ef_wt, "avg_dist_to_road")
    assert len(ef_wt.get_primary_metrics) == 7

    ef_wt.set_weight(
        {
            "num_wells": 20,
            "age_range": 20,
            "depth_range": 60,
        }
    )
    assert ef_wt.check_validity() is None
