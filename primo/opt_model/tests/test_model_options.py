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
import copy
import logging
import pathlib

# Installed libs
import numpy as np
import pyomo.environ as pe
import pytest

# User-defined libs
from primo.data_parser import ImpactMetrics, WellDataColumnNames
from primo.data_parser.well_data import WellData
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.model_with_clustering import (  # pylint: disable=no-name-in-module
    IndexedClusterBlock,
    PluggingCampaignModel,
)
from primo.opt_model.result_parser import Campaign, Project
from primo.utils.config_utils import OverrideAddInfo

LOGGER = logging.getLogger(__name__)


# pylint: disable=duplicate-code


# pylint: disable=missing-function-docstring
@pytest.fixture(name="get_column_names", scope="function")
def get_column_names_fixture():
    """
    Pytest fixture to set up the impact metric, assign
    column names, and read the test data.
    """

    # Define impact metrics by creating an instance of ImpactMetrics class
    im_metrics = ImpactMetrics()

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "ch4_emissions": 35,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 15,
            "well_count": 10,
        },
        submetrics={
            "ch4_emissions": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    # Construct an object to store column names
    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        hospitals="Number of Nearby Hospitals",
        schools="Number of Nearby Schools",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        # These are user-specific columns
        elevation_delta="Elevation Delta [m]",
        dist_to_road="Distance to Road [miles]",
    )

    current_file = pathlib.Path(__file__).resolve()
    # primo folder is 2 levels up the current folder
    data_file = str(current_file.parents[2].joinpath("demo", "Example_1_data.csv"))
    return im_metrics, col_names, data_file


def test_opt_model_inputs(get_column_names):
    """
    Test that the optimization model is constructed and solved correctly.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Catch inputs missing error
    with pytest.raises(
        ValueError,
        match=(
            "One or more essential input arguments in \\[well_data, total_budget, "
            "mobilization_cost\\] are missing while instantiating the object. "
            "WellData object containing information on all wells, the total budget, "
            "and the mobilization cost are essential inputs for the optimization model. "
        ),
    ):
        opt_mdl_inputs = OptModelInputs()

    # Catch priority score missing error
    with pytest.raises(
        ValueError,
        match=(
            "Unable to find priority scores in the WellData object. Compute the scores "
            "using the compute_priority_scores method."
        ),
    ):
        opt_mdl_inputs = OptModelInputs(
            well_data=wd_gas,
            total_budget=3250000,  # 3.25 million USD
            mobilization_cost=mobilization_cost,
        )

    # Compute priority scores
    # Test the model and options
    wd_gas.compute_priority_scores()

    assert "Clusters" not in wd_gas

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
        min_budget_usage=50,
        penalize_unused_budget=True,
    )

    # Ensure that clustering is performed internally
    assert "Clusters" in wd_gas

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    opt_mdl = opt_mdl_inputs.optimization_model

    assert hasattr(opt_mdl_inputs, "config")
    assert "Clusters" in wd_gas  # Column is added after clustering
    assert hasattr(opt_mdl_inputs, "campaign_candidates")
    assert hasattr(opt_mdl_inputs, "pairwise_distance")
    assert hasattr(opt_mdl_inputs, "pairwise_age_difference")
    assert hasattr(opt_mdl_inputs, "pairwise_depth_difference")
    assert hasattr(opt_mdl_inputs, "owner_well_count")

    assert opt_mdl_inputs.get_max_cost_project is None
    assert opt_mdl_inputs.get_total_budget == 3.25

    scaled_mobilization_cost = {1: 0.12, 2: 0.21, 3: 0.28, 4: 0.35}
    for n_wells in range(5, len(wd_gas.data) + 1):
        scaled_mobilization_cost[n_wells] = n_wells * 0.084

    get_mobilization_cost = opt_mdl_inputs.get_mobilization_cost
    for well, cost in scaled_mobilization_cost.items():
        assert np.isclose(get_mobilization_cost[well], cost)

    assert isinstance(opt_mdl, PluggingCampaignModel)
    assert isinstance(opt_campaign, Campaign)
    assert isinstance(opt_campaign.projects[1], Project)

    # Four or five projects are chosen in the optimal campaign
    # TODO: Confirm degeneracy
    assert len(opt_campaign.projects) in [4, 5]

    # Test the structure of the optimization model
    num_clusters = len(set(wd_gas["Clusters"]))
    assert hasattr(opt_mdl, "cluster")
    assert len(opt_mdl.cluster) == num_clusters
    assert isinstance(opt_mdl.cluster, IndexedClusterBlock)
    assert not hasattr(opt_mdl, "min_wells_in_dac_constraint")
    assert hasattr(opt_mdl, "max_well_owner_constraint")
    assert hasattr(opt_mdl, "total_priority_score")

    # Check if the scaling factor for unused budget variable is correctly built
    # pylint: disable=protected-access
    scaling_factor, budget_sufficient = opt_mdl._unused_budget_variable_scaling()
    assert np.isclose(scaling_factor, 955.6699386511185)
    assert not budget_sufficient

    # Check if all the cluster sets are defined
    assert hasattr(opt_mdl.cluster[1], "set_wells")
    assert hasattr(opt_mdl.cluster[1], "set_wells_dac")
    assert hasattr(opt_mdl.cluster[1], "set_well_pairs_remove")
    assert hasattr(opt_mdl.cluster[1], "set_well_pairs_keep")

    # Check if all the required variables are defined
    assert not opt_mdl.cluster[1].select_cluster.is_indexed()
    assert opt_mdl.cluster[1].select_cluster.is_binary()
    assert opt_mdl.cluster[1].select_well.is_indexed()
    for j in opt_mdl.cluster[1].select_well:
        assert opt_mdl.cluster[1].select_well[j].domain == pe.Binary
    assert opt_mdl.cluster[1].num_wells_var.is_indexed()
    for j in opt_mdl.cluster[1].num_wells_var:
        assert opt_mdl.cluster[1].num_wells_var[j].domain == pe.Binary
    assert not opt_mdl.cluster[1].plugging_cost.is_indexed()
    assert opt_mdl.cluster[1].plugging_cost.domain == pe.NonNegativeReals
    assert opt_mdl.cluster[1].num_wells_chosen.domain == pe.NonNegativeReals
    assert opt_mdl.cluster[1].num_wells_dac.domain == pe.NonNegativeReals
    # pylint: disable=no-member
    assert opt_mdl.unused_budget.domain == pe.NonNegativeReals

    # Check if upper bound of the unused budget is defined correctly
    assert opt_mdl.unused_budget.upper is not None

    # Check if the required expressions are defined
    assert hasattr(opt_mdl.cluster[1], "cluster_priority_score")

    # Check if the required constraints are defined
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_chosen")
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_in_dac")
    assert hasattr(opt_mdl.cluster[1], "calculate_plugging_cost")
    assert hasattr(opt_mdl.cluster[1], "campaign_length")
    assert hasattr(opt_mdl.cluster[1], "num_well_uniqueness")
    assert not hasattr(opt_mdl.cluster[1], "ordering_num_wells_vars")
    assert hasattr(opt_mdl.cluster[1], "skip_distant_well_cuts")
    assert len(opt_mdl.cluster[1].skip_distant_well_cuts) == 0

    # Test activate and deactivate methods
    opt_mdl.cluster[1].deactivate()
    assert opt_mdl.cluster[1].select_cluster.value == 0
    assert opt_mdl.cluster[1].num_wells_chosen.value == 0
    assert opt_mdl.cluster[1].num_wells_dac.value == 0
    assert opt_mdl.cluster[1].plugging_cost.value == 0
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].num_wells_chosen.is_fixed()
    assert opt_mdl.cluster[1].num_wells_dac.is_fixed()
    assert opt_mdl.cluster[1].plugging_cost.is_fixed()

    opt_mdl.cluster[1].activate()
    assert not opt_mdl.cluster[1].select_cluster.is_fixed()
    assert not opt_mdl.cluster[1].num_wells_chosen.is_fixed()
    assert not opt_mdl.cluster[1].num_wells_dac.is_fixed()
    assert not opt_mdl.cluster[1].plugging_cost.is_fixed()

    # Test fix and unfix methods
    opt_mdl.cluster[1].fix(0)
    # since no arguments are specified only cluster variable is fixed
    # at its incumbent value, which is zero based on earlier operations
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 0
    for j in opt_mdl.cluster[1].select_well:
        assert not opt_mdl.cluster[1].select_well[j].is_fixed()

    opt_mdl.cluster[1].unfix()

    # fix method with only cluster argument
    opt_mdl.cluster[1].fix(cluster=1)
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 1
    for j in opt_mdl.cluster[1].select_well:
        assert not opt_mdl.cluster[1].select_well[j].is_fixed()

    opt_mdl.cluster[1].unfix()

    # fix method with both cluster and well arguments
    opt_mdl.cluster[1].fix(
        cluster=1,
        wells={i: 1 for i in opt_mdl.cluster[1].set_wells},
    )
    assert opt_mdl.cluster[1].select_cluster.is_fixed()
    assert opt_mdl.cluster[1].select_cluster.value == 1
    for j in opt_mdl.cluster[1].select_well:
        assert opt_mdl.cluster[1].select_well[j].is_fixed()
        assert opt_mdl.cluster[1].select_well[j].value == 1


def test_incremental_formulation(get_column_names):
    """
    Test that the incremental formulation of the optimization model.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
        num_wells_model_type="incremental",
    )

    opt_mdl = opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    assert isinstance(opt_mdl, PluggingCampaignModel)
    assert isinstance(opt_campaign, Campaign)
    assert isinstance(opt_campaign.projects[1], Project)
    # assert isinstance(opt_campaign[1], dict)

    # Check if the scaling factor for budget slack variable is correctly built
    # pylint: disable=protected-access
    _, budget_sufficient = opt_mdl._unused_budget_variable_scaling()
    # pylint: disable=no-member
    assert np.isclose(opt_mdl.unused_budget_scaling.value, 0)
    assert not budget_sufficient

    # Four or five projects are chosen in the optimal campaign
    # TODO: Confirm degeneracy

    assert len(opt_campaign.projects) in [4, 5]

    # Check if the required constraints are defined
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_chosen")
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_in_dac")
    assert hasattr(opt_mdl.cluster[1], "calculate_plugging_cost")
    assert hasattr(opt_mdl.cluster[1], "campaign_length")
    assert not hasattr(opt_mdl.cluster[1], "num_well_uniqueness")
    assert hasattr(opt_mdl.cluster[1], "ordering_num_wells_vars")


def test_unused_budget_variable_scaling(get_column_names):
    """
    Test the optimization model when there is enough budget for plugging all wells.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=325000000,  # 325 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
        min_budget_usage=50,
        penalize_unused_budget=True,
    )

    opt_mdl = opt_mdl_inputs.build_optimization_model()

    # Check if the scaling factor for budget slack variable is correctly built
    # pylint: disable=protected-access
    scaling_factor, budget_sufficient = opt_mdl._unused_budget_variable_scaling()
    assert np.isclose(scaling_factor, 105.71767887503083)
    assert budget_sufficient

    # Check if the upper bound of the unused budget is set
    assert opt_mdl.unused_budget.upper is None


# pylint: disable=too-many-locals
def test_override_re_optimization(get_column_names):
    """
    Test that the optimization model is constructed and solved correctly
    when an override choice is made.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    override_dict = (
        {13: 0, 19: 1},
        {
            1: {851: 0, 858: 0},
            11: {80: 1},
            6: {600: 1},
            19: {21: 1, 83: 1, 182: 1, 280: 1, 981: 1},
            10: {734: 1},
            40: {601: 1},
        },
    )

    well_add_existing_cluster = {
        6: [600],
        11: [80],
        10: [734],
        40: [601],
    }
    well_add_new_cluster = {11: [80], 6: [600], 10: [734], 40: [601]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)
    initial_opt_mdl_inputs = copy.deepcopy(opt_mdl_inputs)

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    assert hasattr(opt_mdl_inputs, "update_cluster")
    assert 13 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(add_widget_return)

    assert (
        opt_mdl_inputs.campaign_candidates == initial_opt_mdl_inputs.campaign_candidates
    )
    assert opt_mdl_inputs.owner_well_count == initial_opt_mdl_inputs.owner_well_count
    assert opt_mdl_inputs.pairwise_distance == initial_opt_mdl_inputs.pairwise_distance

    # Build the new optimization model based on the override selection
    or_opt_mdl = opt_mdl_inputs.build_optimization_model(override_dict)
    or_opt_campaign = opt_mdl_inputs.solve_model(solver="highs")

    assert hasattr(or_opt_mdl, "fix_var")

    # Ensure clusters and wells are fixed based on the override selection
    assert or_opt_mdl.cluster[13].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[13].select_cluster.value == 0
    for j in or_opt_mdl.cluster[13].select_well:
        assert not or_opt_mdl.cluster[13].select_well[j].is_fixed()

    assert or_opt_mdl.cluster[19].select_cluster.is_fixed()
    assert or_opt_mdl.cluster[19].select_cluster.value == 1
    for j in override_dict[1][19]:
        assert or_opt_mdl.cluster[19].select_well[j].is_fixed()
        assert or_opt_mdl.cluster[19].select_well[j].value == override_dict[1][19][j]

    assert not or_opt_mdl.cluster[1].select_cluster.is_fixed()

    # Test the re-optimization results
    assert 13 not in or_opt_campaign.projects
    assert 80 in or_opt_campaign.projects[11].well_data.data.index
    assert 600 in or_opt_campaign.projects[6].well_data.data.index
    assert 851 not in or_opt_campaign.projects[1].well_data.data.index
    assert or_opt_campaign.clusters_dict[19] == [21, 83, 182, 280, 981]


# pylint: disable=too-many-locals
def test_re_cluster(get_column_names):
    """
    Test the re_cluster function to ensure that the optimization model
    inputs are accurately updated based on the override choice.
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    well_add_existing_cluster = {
        6: [600],
        11: [80],
        19: [981],
        40: [601],
    }
    well_add_new_cluster = {11: [80], 6: [600], 10: [981], 40: [601]}

    add_widget_return = OverrideAddInfo(well_add_existing_cluster, well_add_new_cluster)

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="highs")
    assert hasattr(opt_mdl_inputs, "update_cluster")
    assert 13 in opt_campaign.projects

    # Update the model input based on the override selection
    opt_mdl_inputs.update_cluster(add_widget_return)

    assert 981 not in opt_mdl_inputs.campaign_candidates[19]
    assert 981 in opt_mdl_inputs.campaign_candidates[10]
    assert (10, 981) in opt_mdl_inputs.owner_well_count["Owner 104"]
    assert (19, 981) not in opt_mdl_inputs.owner_well_count["Owner 104"]
    assert any(981 in key for key in opt_mdl_inputs.pairwise_distance[10])
    assert not any(981 in key for key in opt_mdl_inputs.pairwise_distance[19])


def test_dictionary_instantiation(get_column_names):
    """
    Test using a dictionary to instantiate the OptModelInputs object
    and avoid re-clustering
    """
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names, impact_metrics=im_metrics)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    gas_oil_wells_replica = wd.get_gas_oil_wells
    wd_gas_replica = gas_oil_wells_replica["gas"]
    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores()
    wd_gas_replica.compute_priority_scores()

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    assert "Clusters" not in wd_gas_replica

    clustering_dictionary = opt_mdl_inputs.campaign_candidates

    OptModelInputs(
        cluster_mapping=clustering_dictionary,
        well_data=wd_gas_replica,
        total_budget=3210000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    assert "Clusters" in wd_gas_replica
