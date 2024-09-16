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
import pyomo.environ as pe
import pytest
from pyomo.solvers.plugins.solvers.SCIPAMPL import SCIPAMPL

# User-defined libs
from primo.data_parser import ImpactMetrics, WellData, WellDataColumnNames
from primo.opt_model.model_options import OptModelInputs
from primo.opt_model.model_with_clustering import (
    IndexedClusterBlock,
    PluggingCampaignModel,
)

LOGGER = logging.getLogger(__name__)


@pytest.fixture(name="get_column_names", scope="function")
def get_column_names_fixture():
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
        additional_columns={
            "elevation_delta": "Elevation Delta [m]",
            "dist_to_road": "Distance to Road [miles]",
        },
    )

    filename = os.path.dirname(os.path.realpath(__file__))[:-16]  # Primo folder
    filename += r"/demo/Example_1_data.csv"

    return im_metrics, col_names, filename


def test_opt_model_inputs(get_column_names):
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names)

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
    wd_gas.compute_priority_scores(impact_metrics=im_metrics)

    assert "Clusters" not in wd_gas

    # Formulate the optimization problem
    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
    )

    # Ensure that clustering is performed internally
    assert "Clusters" in wd_gas

    opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="scip")
    opt_mdl = opt_mdl_inputs.optimization_model
    solver = opt_mdl_inputs.solver

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
    for well in scaled_mobilization_cost:
        assert np.isclose(get_mobilization_cost[well], scaled_mobilization_cost[well])

    assert isinstance(opt_mdl, PluggingCampaignModel)
    assert isinstance(solver, SCIPAMPL)
    assert isinstance(opt_campaign, tuple)
    assert isinstance(opt_campaign[0], dict)
    assert isinstance(opt_campaign[1], dict)

    # Four projects are chosen in the optimal campaign
    assert len(opt_campaign[0]) == 4

    # Test the structure of the optimization model
    num_clusters = len(set(wd_gas["Clusters"]))
    assert hasattr(opt_mdl, "cluster")
    assert len(opt_mdl.cluster) == num_clusters
    assert isinstance(opt_mdl.cluster, IndexedClusterBlock)
    assert not hasattr(opt_mdl, "min_wells_in_dac_constraint")
    assert hasattr(opt_mdl, "max_well_owner_constraint")
    assert hasattr(opt_mdl, "total_priority_score")

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
    opt_mdl.cluster[1].fix()
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
    im_metrics, col_names, filename = get_column_names

    # Create the well data object
    wd = WellData(data=filename, column_names=col_names)

    # Partition the wells as gas/oil
    gas_oil_wells = wd.get_gas_oil_wells
    wd_gas = gas_oil_wells["gas"]

    # Mobilization cost
    mobilization_cost = {1: 120000, 2: 210000, 3: 280000, 4: 350000}
    for n_wells in range(5, len(wd_gas.data) + 1):
        mobilization_cost[n_wells] = n_wells * 84000

    # Test the model and options
    wd_gas.compute_priority_scores(impact_metrics=im_metrics)

    opt_mdl_inputs = OptModelInputs(
        well_data=wd_gas,
        total_budget=3250000,  # 3.25 million USD
        mobilization_cost=mobilization_cost,
        threshold_distance=10,
        max_wells_per_owner=1,
        num_wells_model_type="incremental",
    )

    opt_mdl = opt_mdl_inputs.build_optimization_model()
    opt_campaign = opt_mdl_inputs.solve_model(solver="scip")

    assert isinstance(opt_mdl, PluggingCampaignModel)
    assert isinstance(opt_campaign, tuple)
    assert isinstance(opt_campaign[0], dict)
    assert isinstance(opt_campaign[1], dict)

    # Four projects are chosen in the optimal campaign
    assert len(opt_campaign[0]) == 4

    # Check if the required constraints are defined
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_chosen")
    assert hasattr(opt_mdl.cluster[1], "calculate_num_wells_in_dac")
    assert hasattr(opt_mdl.cluster[1], "calculate_plugging_cost")
    assert hasattr(opt_mdl.cluster[1], "campaign_length")
    assert not hasattr(opt_mdl.cluster[1], "num_well_uniqueness")
    assert hasattr(opt_mdl.cluster[1], "ordering_num_wells_vars")
