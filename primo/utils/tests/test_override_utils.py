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

# User-defined libs
from primo.data_parser import EfficiencyMetrics
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.result_parser import Campaign
from primo.opt_model.tests.test_model_options import (  # pylint: disable=unused-import
    get_column_names_fixture,
)
from primo.utils.config_utils import (
    OverrideAddInfo,
    OverrideRemoveLockInfo,
    OverrideSelections,
)
from primo.utils.override_utils import AssessFeasibility, OverrideCampaign
from primo.utils.tests.test_config_utils import (  # pylint: disable=unused-import
    efficiency_metrics_fixture,
    get_model_fixture,
)


@pytest.fixture(name="or_infeasible_selection")
def or_infeasible_selection_fixture():
    """
    Pytest fixture for constructing an override selection return which
    will lead to infeasible P&A projects.
    """
    project_remove = [13]
    well_remove = {1: [851, 858]}
    well_add_existing_cluster = {
        1: [851],
        6: [600],
        11: [80],
        10: [734],
        40: [601],
    }
    well_add_new_cluster = {11: [851, 80], 6: [600], 10: [734], 40: [601]}
    project_lock = [19]
    well_lock = {19: [21, 83, 182, 280, 981]}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_override_campaign(or_infeasible_selection, get_model):
    """
    Test the override campaign class when the new project is infeasible
    after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_infeasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert isinstance(or_camp_class.remove, OverrideRemoveLockInfo)
    assert isinstance(or_camp_class.add, OverrideAddInfo)
    assert isinstance(or_camp_class.lock, OverrideRemoveLockInfo)
    assert isinstance(or_camp_class.opt_inputs, OptModelInputs)
    assert or_camp_class.eff_metrics == eff_metrics
    assert isinstance(or_camp_class.eff_metrics, EfficiencyMetrics)
    assert hasattr(or_camp_class.remove, "cluster")
    assert hasattr(or_camp_class.remove, "well")

    # Update the optimal campaign based on the override selection
    assert 13 not in or_camp_class.new_campaign
    assert 851 not in or_camp_class.new_campaign[1]
    assert 6 in or_camp_class.new_campaign
    assert or_camp_class.new_campaign[6] == [600]
    assert or_camp_class.new_campaign[19] == [21, 83, 182, 280, 981]

    assert isinstance(or_camp_class.well_data, WellData)
    assert 210 not in or_camp_class.well_data.data.index
    assert len(or_camp_class.plug_list) == 37

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() > 0
    assert or_camp_class.feasibility.assess_owner_well_count()
    assert or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_dac() == 0

    violation_info_dict = or_camp_class.violation_info()
    assert len(violation_info_dict) == 4
    assert violation_info_dict["Project Status:"] == "INFEASIBLE"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], str)
    assert isinstance(violation_info_dict[key_list[2]], pd.DataFrame)
    assert isinstance(violation_info_dict[key_list[3]], pd.DataFrame)

    override_campaign = or_camp_class.recalculate()
    assert isinstance(override_campaign, Campaign)
    assert hasattr(override_campaign, "set_efficiency_weights")
    assert hasattr(override_campaign, "compute_efficiency_scores")
    project = override_campaign.projects[1]
    assert np.isclose(project.impact_score, 68.88, rtol=1e-2, atol=1e-2)
    assert np.isclose(project.efficiency_score, 32.006, rtol=1e-2, atol=1e-2)
    assert len(override_campaign.wd) == len(opt_mdl_inputs.config.well_data)

    override_campaign_dict = or_camp_class.recalculate_scores()
    assert isinstance(override_campaign_dict, dict)
    assert 13 not in override_campaign_dict
    assert all(
        [63.55375464985779, 46.03383590762232][i]
        == pytest.approx(override_campaign_dict[11][i], rel=1e-2)
        for i in range(2)
    )

    override_cluster_dict, override_well_dict = or_camp_class.re_optimize_dict()
    assert override_cluster_dict == {19: 1}
    assert override_well_dict == {
        1: {851: 0, 858: 0},
        11: {851: 1, 80: 1},
        6: {600: 1},
        19: {21: 1, 83: 1, 182: 1, 280: 1, 981: 1},
        10: {734: 1},
        40: {601: 1},
    }


@pytest.fixture(name="or_feasible_selection")
def or_feasible_selection_fixture():
    """
    Pytest fixture for constructing an override selection return which
    will lead to feasible P&A projects.
    """
    project_remove = [13]
    well_remove = {}
    well_add_existing_cluster = {}
    well_add_new_cluster = {}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_feasible_override_campaign(or_feasible_selection, get_model):
    """
    Test the override campaign class when the new project is feasible
    after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_feasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    # Assign all wells as disadvantaged wells for testing purpose
    or_camp_class.well_data.data["is_disadvantaged"] = 1
    or_camp_class.feasibility.opt_inputs.config.perc_wells_in_dac = 40

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_dac() < 0

    violation_info_dict = or_camp_class.violation_info()
    assert len(violation_info_dict) == 1
    assert violation_info_dict["Project Status:"] == "FEASIBLE"


def test_infeasible_dac(or_feasible_selection, get_model):
    """
    Test the override campaign class when the DAC for the
    new project is infeasible after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_feasible_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    # Assign all wells as not disadvantaged wells for testing purpose
    or_camp_class.well_data.data["is_disadvantaged"] = 0
    or_camp_class.feasibility.opt_inputs.config.perc_wells_in_dac = 40

    assert isinstance(or_camp_class.feasibility, AssessFeasibility)
    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_dac() > 0

    violation_info_dict = or_camp_class.violation_info()
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "INFEASIBLE"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], str)


@pytest.fixture(name="or_infeasible_owc_selection")
def or_infeasible_owc_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the owner well count constraint.
    """
    project_remove = [13]
    well_remove = {}
    well_add_existing_cluster = {19: [86]}
    well_add_new_cluster = {19: [86]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_owc(or_infeasible_owc_selection, get_model):
    """
    Test the override campaign class where the new projects violate
    the owner well count constraint after the override step
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_infeasible_owc_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert or_camp_class.feasibility.assess_owner_well_count()
    assert not or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_dac() == 0

    violation_info_dict = or_camp_class.violation_info()
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "INFEASIBLE"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)
    violated_operators = violation_info_dict[key_list[1]]
    assert violated_operators["Owner"].loc[violated_operators.index[0]] == "Owner 7"
    assert violated_operators["Number of wells"].loc[violated_operators.index[0]] == 2
    assert violated_operators["Wells"].loc[violated_operators.index[0]] == [
        "33912",
        "69687",
    ]


@pytest.fixture(name="or_feasible_distance_selection")
def or_infeasible_distance_selection_fixture():
    """
    Pytest fixture for constructing an override selection return that
    results in new P&A projects violating the distance constraint.
    """
    project_remove = [13]
    well_remove = {1: [851]}
    well_add_existing_cluster = {1: [851, 807]}
    well_add_new_cluster = {11: [851], 36: [807]}
    project_lock = []
    well_lock = {}
    remove_widget_return = OverrideRemoveLockInfo(project_remove, well_remove)
    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    lock_widget_return = OverrideRemoveLockInfo(project_lock, well_lock)
    or_selection = OverrideSelections(
        remove_widget_return, add_widget_return, lock_widget_return
    )
    return or_selection


def test_infeasible_distance(or_feasible_distance_selection, get_model):
    """
    Test the override campaign class where (1) the new projects violate
    the distance constraint after the override step (2) A well is not removed
    from the recommended projects before being reassigned to another project
    """
    opt_campaign, opt_mdl_inputs, eff_metrics = get_model
    or_selection = or_feasible_distance_selection

    or_camp_class = OverrideCampaign(
        or_selection, opt_mdl_inputs, opt_campaign.clusters_dict, eff_metrics
    )

    # Test if there is any duplication in the plug_list
    assert 807 in or_camp_class.plug_list
    assert or_camp_class.plug_list.count(807) == 1

    assert not or_camp_class.feasibility.assess_feasibility()
    assert or_camp_class.feasibility.assess_budget() < 0
    assert not or_camp_class.feasibility.assess_owner_well_count()
    assert or_camp_class.feasibility.assess_distances()
    assert or_camp_class.feasibility.assess_dac() == 0

    violation_info_dict = or_camp_class.violation_info()
    assert len(violation_info_dict) == 2
    assert violation_info_dict["Project Status:"] == "INFEASIBLE"
    key_list = list(violation_info_dict.keys())
    assert isinstance(violation_info_dict[key_list[1]], pd.DataFrame)
    violated_operators = violation_info_dict[key_list[1]].head(1)
    assert violated_operators["Project"].loc[violated_operators.index[0]] == 11
    assert violated_operators["Well 1"].loc[violated_operators.index[0]] == "62199"
    assert violated_operators["Well 2"].loc[violated_operators.index[0]] == "58876"
    assert np.isclose(
        violated_operators["Distance between Well 1 and 2 [Miles]"].loc[
            violated_operators.index[0]
        ],
        98.30642202632235,
    )
