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
import pyomo.environ as pyo
import pytest

# User-defined libs
# pylint: disable = no-name-in-module
from primo.opt_model.efficiency_block import EfficiencyBlock


@pytest.fixture(name="get_dummy_model", scope="function")
def get_dummy_model_fixture():
    """
    Pytest fixture to create a dummy model
    """
    m = pyo.ConcreteModel()
    m.set_wells = pyo.RangeSet(5)

    m.select_well = pyo.Var(m.set_wells, within=pyo.Binary)
    m.num_wells_var = pyo.Var(m.set_wells, within=pyo.Binary)
    m.select_cluster = pyo.Var(within=pyo.Binary)
    m.efficiency_model = EfficiencyBlock()

    return m


def test_eff_block_no_input_var(get_dummy_model):
    """Tests the efficiency block class"""

    m = get_dummy_model
    eff_blk = m.efficiency_model
    eff_blk.append_cluster_eff_vars()

    # Test the existence of variables
    assert isinstance(eff_blk.cluster_efficiency, pyo.Var)
    assert isinstance(eff_blk.aggregated_efficiency, pyo.Var)
    assert isinstance(eff_blk.calculate_aggregated_efficiency_1, pyo.Constraint)
    assert isinstance(eff_blk.calculate_aggregated_efficiency_2, pyo.Constraint)
    assert isinstance(eff_blk.calculate_cluster_efficiency, pyo.Constraint)
    assert isinstance(eff_blk.cluster_efficiency_score, pyo.Expression)

    # Check if constraints are implemented correctly
    assert (
        str(eff_blk.calculate_aggregated_efficiency_1[1].expr)
        == "efficiency_model.aggregated_efficiency[1]  <=  100*num_wells_var[1]"
    )
    assert (
        str(eff_blk.calculate_aggregated_efficiency_1[5].expr)
        == "efficiency_model.aggregated_efficiency[5]  <=  100*num_wells_var[5]"
    )

    assert str(eff_blk.calculate_aggregated_efficiency_2[1].expr) == (
        "efficiency_model.aggregated_efficiency[1]  <=  "
        "efficiency_model.cluster_efficiency"
    )
    assert str(eff_blk.calculate_aggregated_efficiency_2[5].expr) == (
        "efficiency_model.aggregated_efficiency[5]  <=  "
        "efficiency_model.cluster_efficiency"
    )

    assert (
        str(eff_blk.calculate_cluster_efficiency.expr)
        == "efficiency_model.cluster_efficiency  ==  0"
    )

    eff_score_expr = "".join(
        [f"{n}*efficiency_model.aggregated_efficiency[{n}] + " for n in m.set_wells]
    )[2:-3]
    assert str(eff_blk.cluster_efficiency_score.expr) == eff_score_expr

    eff_blk.sub_blk = pyo.Block()
    eff_blk.sub_blk.score = pyo.Var(initialize=20)
    assert eff_blk.get_efficiency_scores() == {"sub_blk": 20}


def test_eff_block_with_input_var(get_dummy_model):
    """Tests the efficiency block class"""

    eff_blk = get_dummy_model.efficiency_model
    eff_blk.v1 = pyo.Var()
    eff_blk.v2 = pyo.Var()

    eff_blk.append_cluster_eff_vars(eff_vars=[eff_blk.v1, eff_blk.v2])

    # Check if constraints are implemented correctly
    assert str(eff_blk.calculate_cluster_efficiency.expr) == (
        "efficiency_model.cluster_efficiency  ==  "
        "efficiency_model.v1 + efficiency_model.v2"
    )

    eff_blk.v2.value = 10
    assert eff_blk.get_efficiency_scores() == {"v1": None, "v2": 10}
