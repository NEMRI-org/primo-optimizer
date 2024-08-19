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
from primo.data_parser.metric_data import (
    EfficiencyMetrics,
    ImpactMetrics,
    Metric,
    SetOfMetrics,
    SubMetric,
)


# pylint: disable = no-member, missing-function-docstring
# pylint: disable = f-string-without-interpolation
# pylint: disable = protected-access, unused-variable
# pylint: disable = too-many-statements
def test_metric_class():
    z = Metric(name="met_1", value=50, full_name="Metric One")

    assert z.name == "met_1"
    assert z.full_name == "Metric One"
    assert (z.min_value, z.value, z.max_value) == (0, 50, 100)
    assert z.weight == 50
    assert not z.is_submetric
    print_format = (
        f"Metric name: Metric One, Metric value: 50 \n"
        f"    Admissible range: [0, 100]"
    )
    assert str(z) == print_format

    assert z.data_col_name is None
    assert z.score_col_name is None

    z.data_col_name = "Metric One"
    assert z.score_col_name == "Metric One Score [0-50]"

    # Add test to capture the name modification error
    with pytest.raises(
        ValueError,
        match=("Metric's key name cannot be modified after it is defined."),
    ):
        z.name = "modified_met_1"

    # Add a test to catch non isidentifier metric
    with pytest.raises(
        ValueError,
        match=(
            "Received met 2 for Metric's key name, which is not a valid python variable name!"
        ),
    ):
        y = Metric("met 2", 100)

    # Add test to capture the value out-of-range error
    with pytest.raises(
        ValueError,
        match=(
            f"Attempted to assign -50 for metric met_1, which "
            f"lies outside the admissible range \\[0, 100\\]."
        ),
    ):
        z.value = -50


def test_submetric_class():
    z_par = Metric("par_metric", 40, full_name="Parent Metric One")
    z_sub = SubMetric("sub_metric", z_par, 50, full_name="SubMetric One")

    assert z_sub.is_submetric
    assert z_sub.parent_metric is z_par
    assert z_sub.weight == 20
    print_format = (
        "Submetric name: SubMetric One, Submetric value: 50 \n"
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
        "        Metric Name  Metric Value\n"
        "met_1    Metric One            33\n"
        "met_2    Metric Two            33\n"
        "met_3  Metric Three            34\n"
        "              Total           100"
    )

    assert str(z) == print_format

    z.register_new_submetric("sub_met_1_1", z.met_1, 50, "Sub Metric One-One")
    z.register_new_submetric("sub_met_1_2", z.met_1, 50, "Sub Metric One-Two")
    z.register_new_submetric("sub_met_3_1", z.met_3, 25, "Sub Metric Three-One")
    z.register_new_submetric("sub_met_3_2", z.met_3, 75, "Sub Metric Three-Two")

    print_format = (
        "        Metric Name  Metric Value\n"
        "met_1    Metric One            33\n"
        "met_2    Metric Two            33\n"
        "met_3  Metric Three            34\n"
        "              Total           100\n\n\n"
        "Primary metric Metric One, with weight 33, has submetrics:\n"
        "================================================================================\n"
        "                 Submetric Name  Submetric Value\n"
        "sub_met_1_1  Sub Metric One-One               50\n"
        "sub_met_1_2  Sub Metric One-Two               50\n"
        "                          Total              100\n\n\n"
        "Primary metric Metric Three, with weight 34, has submetrics:\n"
        "================================================================================\n"
        "                   Submetric Name  Submetric Value\n"
        "sub_met_3_1  Sub Metric Three-One               25\n"
        "sub_met_3_2  Sub Metric Three-Two               75\n"
        "                            Total              100"
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

    # Test iter method and contains method
    for obj in z:
        assert obj in _all_metrics.values()
        assert obj.name in z
        assert obj.full_name in z

    # Test for receiving update error: Enter a typo in the input dict
    with pytest.raises(
        KeyError,
        match=("Metrics/submetrics \\['met 1'\\] are not recognized/registered."),
    ):
        z.set_value({"met 1": 33, "Metric Two": 33})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError, match=("Sum of weights of primary metrics does not add up to 100")
    ):
        z.set_value({"met_1": 33, "met_2": 30})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Weight of the primary metric met_1 is zero, but the sum of "
            "weights of its submetrics is 100, which is nonzero."
        ),
    ):
        z.set_value({"met_1": 0, "met_2": 66})

    # Test for receiving check_update error
    with pytest.raises(
        ValueError,
        match=(
            "Sum of weights of submetrics of the primary metric met_1 does not add up to 100."
        ),
    ):
        z.set_value(
            {"met_1": 33, "met_2": 33}, submetrics={"met_1": {"sub_met_1_1": 0}}
        )

    z.set_value({"sub_met_1_1": 50})

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
            f"Metric/submetric met_2 has already been registered. "
            f"Attempting to register a new metric with the same name."
        ),
    ):
        z.register_new_metric("met_2")

    # Try deleting primary metric on z.met_1. This should automatically delete submetrics
    z.delete_metric("met_1")
    assert not hasattr(z, "met_1")
    assert not hasattr(z, "sub_met_1_1")
    assert not hasattr(z, "sub_met_1_2")
    assert len([obj for obj in z]) == 4

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
    assert hasattr(im_wt, "production_volume")
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

    assert hasattr(im_wt, "ann_production_volume")
    assert hasattr(im_wt, "five_year_production_volume")
    assert im_wt.production_volume.submetrics == {
        "ann_production_volume": im_wt.ann_production_volume,
        "five_year_production_volume": im_wt.five_year_production_volume,
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

    im_wt.set_value(
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


def test_efficiency_metrics_class():
    ef_wt = EfficiencyMetrics()

    assert hasattr(ef_wt, "num_wells")
    assert hasattr(ef_wt, "num_unique_owners")
    assert hasattr(ef_wt, "dist_centroid")
    assert hasattr(ef_wt, "elevation_delta")
    assert hasattr(ef_wt, "age_range")
    assert hasattr(ef_wt, "depth_range")
    assert hasattr(ef_wt, "record_completeness")
    assert len(ef_wt.get_primary_metrics) == 7

    ef_wt.set_value(
        {
            "num_wells": 20,
            "dist_centroid": 30,
            "age_range": 20,
            "depth_range": 30,
        }
    )
    assert ef_wt.check_validity() is None
