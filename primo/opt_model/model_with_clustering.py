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
from typing import Dict, Optional

# Installed libs
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeReals,
    NonNegativeIntegers,
    Objective,
    Param,
    Set,
    Var,
    maximize,
)

LOGGER = logging.getLogger(__name__)


def build_cluster_model(model_block, cluster):
    """
    Builds the model block (adds essential variables and constraints)
    for a given cluster `cluster`
    """
    # Parameters are located in the parent block
    params = model_block.parent_block().model_inputs
    wd = params.config.well_data
    well_index = params.campaign_candidates[cluster]
    pairwise_distance = params.pairwise_distance[cluster]

    # Get well pairs which violate the distance threshold
    well_dac = []
    # Update the column name after federal DAC info is added
    if "is_disadvantaged" in wd:
        for well in well_index:
            if wd.data.loc[well, "is_disadvantaged"]:
                well_dac.append(well)

    well_pairs_remove = [
        key
        for key, val in pairwise_distance.items()
        if val > params.config.threshold_distance
    ]
    well_pairs_keep = [key for key in pairwise_distance if key not in well_pairs_remove]

    # Essential model sets
    model_block.set_wells = Set(
        initialize=well_index,
        doc="Set of wells in cluster c",
    )
    model_block.set_wells_dac = Set(
        initialize=well_dac,
        doc="Set of wells that are in disadvantaged communities",
    )
    model_block.set_well_pairs_remove = Set(
        initialize=well_pairs_remove,
        doc="Well-pairs which cannot be a part of the project",
    )
    model_block.set_well_pairs_keep = Set(
        initialize=well_pairs_keep,
        doc="Well-pairs which can be a part of the project",
    )

    # Essential variables
    model_block.select_cluster = Var(
        within=Binary,
        doc="1, if wells from the cluster are chosen for plugging, 0 Otherwise",
    )
    model_block.select_well = Var(
        model_block.set_wells,
        within=Binary,
        doc="1, if the well is selected for plugging, 0 otherwise",
    )
    model_block.num_wells_var = Var(
        range(1, len(model_block.set_wells) + 1),
        within=Binary,
        doc="Variables to track the total number of wells chosen",
    )
    model_block.plugging_cost = Var(
        within=NonNegativeReals,
        doc="Total cost for plugging wells in this cluster",
    )

    # Although the following two variables are of type Integer, they
    # can be declared as continuous. The optimal solution is guaranteed to have
    # integer values.

    model_block.num_wells_chosen = Var(
        within=NonNegativeReals,
        doc="Total number of wells chosen in the project",
    )
    model_block.num_wells_dac = Var(
        within=NonNegativeReals,
        doc="Number of wells chosen in disadvantaged communities",
    )

    # Set the maximum cost and size of the project: default is None.
    model_block.plugging_cost.setub(params.get_max_cost_project)
    model_block.num_wells_chosen.setub(params.config.max_size_project)

    # Useful expressions
    priority_score = wd["Priority Score [0-100]"]
    model_block.cluster_priority_score = Expression(
        expr=(
            sum(
                priority_score[w] * model_block.select_well[w]
                for w in model_block.set_wells
            )
        ),
        doc="Computes the total priority score for the cluster",
    )

    pairwise_age_range = params.pairwise_age_range[cluster]
    model_block.pairwise_age_range = Param(
        model_block.set_well_pairs_keep,
        initialize=pairwise_age_range,
        doc="Pairwise age range for well pairs in the cluster.",
    )

    pairwise_depth_range = params.pairwise_depth_range[cluster]
    model_block.pairwise_depth_range = Param(
        model_block.set_well_pairs_keep,
        initialize=pairwise_depth_range,
        doc="Pairwise depth range for well pairs in the cluster.",
    )

    # Essential constraints
    model_block.calculate_num_wells_chosen = Constraint(
        expr=(
            sum(model_block.select_well[w] for w in model_block.set_wells)
            == model_block.num_wells_chosen
        ),
        doc="Calculate the total number of wells chosen",
    )
    model_block.calculate_num_wells_in_dac = Constraint(
        expr=(
            sum(model_block.select_well[w] for w in model_block.set_wells_dac)
            == model_block.num_wells_dac
        ),
        doc="Calculate the number of wells chosen that are in dac",
    )

    # This is to test which formulation is faster. If there is no
    # benefit in terms of computational time, then delete this method.
    if params.config.num_wells_model_type == "incremental":
        num_wells_incremental_formulation(model_block)
        return

    # Using the multicommodity formulation
    mob_cost = params.get_mobilization_cost
    model_block.calculate_plugging_cost = Constraint(
        expr=(
            sum(
                mob_cost[i] * model_block.num_wells_var[i]
                for i in model_block.num_wells_var
            )
            == model_block.plugging_cost
        ),
        doc="Calculates the total plugging cost for the cluster",
    )
    model_block.campaign_length = Constraint(
        expr=(
            sum(i * model_block.num_wells_var[i] for i in model_block.num_wells_var)
            == model_block.num_wells_chosen
        ),
        doc="Determines the number of wells chosen",
    )
    model_block.num_well_uniqueness = Constraint(
        expr=(
            sum(model_block.num_wells_var[i] for i in model_block.num_wells_var)
            == model_block.select_cluster
        ),
        doc="Ensures at most one num_wells_var is selected",
    )

    if params.config.objective_type == "Impact":
        return
    else:
        pass

    #### Efficiency terms
    model_block.eff_num_wells = Var(within=NonNegativeReals, doc="")

    model_block.eff_age_range = Var(within=NonNegativeReals, doc="")

    model_block.eff_depth_range = Var(within=NonNegativeReals, doc="")

    model_block.eff_dist_road = Var(within=NonNegativeReals, doc="")

    model_block.eff_elev_delta = Var(within=NonNegativeReals, doc="")

    model_block.eff_unique_owner = Var(within=NonNegativeReals, doc="")

    model_block.eff_rec_comp = Var(within=NonNegativeReals, doc="")

    model_block.eff_pop_den = Var(within=NonNegativeReals, doc="")

    model_block.eff_dist_range = Var(
        within=NonNegativeReals, doc=""
    )  # change to dist_range

    model_block.eff_age_depth_dist = Var(within=NonNegativeReals, doc="")

    model_block.eff_pop_road_elev = Var(within=NonNegativeReals, doc="")

    max_dist_road = params.config.max_distance_to_road
    max_elev_delta = params.config.max_elevation_delta
    max_size_project = params.config.max_size_project
    max_unique_owners = params.config.max_number_of_unique_owners
    max_age_range = params.config.max_age_range
    max_depth_range = params.config.max_depth_range
    max_rec_comp = params.config.record_completeness
    max_pop_den = params.config.max_population_density
    max_dist_range = params.config.max_well_distance
    max_num_proj = params.config.max_num_project

    eff_metrics = wd.efficiency_metrics
    w_age_range = eff_metrics.age_range.effective_weight
    w_depth_range = eff_metrics.depth_range.effective_weight
    w_rec_comp = eff_metrics.rec_comp.effective_weight
    w_pop_den = eff_metrics.pop_den.effective_weight
    w_well_dist = eff_metrics.well_dist.effective_weight
    w_unique_owners = eff_metrics.unique_owners.effective_weight
    w_num_wells = eff_metrics.num_wells.effective_weight
    w_elev_delta = eff_metrics.elev_delta.effective_weight
    w_dist_road = eff_metrics.dist_road.effective_weight

    model_block.eff_num_wells = Constraint(
        expr=(
            model_block.eff_num_wells
            == w_num_wells * (model_block.num_wells_chosen / max_size_project)
        )
    )

    @model_block.Constraint(
        model_block.set_well_pairs_keep,
        doc="Combined constraint for age, depth, distance",
    )
    def rule(model_block, w1, w2):
        cluster = model_block.parent_block().cluster
        return (
            w_age_range + w_depth_range + w_well_dist
        ) * model_block.select_cluster - model_block.eff_age_depth_distance >= (
            model_block.select_well[w1]
            + model_block.select_well[w2]
            - model_block.select_cluster
        ) * (
            (w_age_range * model_block.pairwise_age_range[w1, w2] / max_age_range)
            + (
                w_depth_range
                * model_block.pairwise_depth_range[w1, w2]
                / max_depth_range
            )
            + (w_well_dist * model_block.pairwise_dist[w1, w2] / max_dist_range)
        )

    @model_block.Constraint(
        model_block.set_wells,
        doc="Combined constraint for popoulation density, distance to road, elevation",
    )
    def rule(model_block, w):
        cluster = model_block.parent_block().cluster
        return (
            w_pop_den + w_dist_road + w_elev_delta
        ) * model_block.select_cluster - model_block.eff_pop_road_elev >= (
            model_block.select_well[w] - model_block.select_cluster
        ) * (
            (w_pop_den * wd.col_names.population_density[w] / max_pop_den)
            + (w_dist_road * wd.col_names.dist_road[w] / max_dist_road)
            + (w_elev_delta * wd.col_names.elevation_delta[w] / max_elev_delta)
        )

    # @model_block.Constraint(
    #     model_block.set_wells, doc="Unique Owner constraint for all wells"
    # )
    # def unique_owner_rule(model_block, w1):
    #     cluster = model_block.parent_block().cluster
    #     return (
    #         1 - model_block.eff_unique_owner
    #         >= model_block.unique_owner[w1]
    #         * (model_block.select_well[w1])
    #         / max_unique_owners
    #     )

    @model_block.Constraint(
        model_block.set_wells, doc="Unique Owner constraint for all wells"
    )
    def rec_comp_rule(model_block, w1):
        cluster = model_block.parent_block().cluster
        return model_block.eff_rec_comp <= w_rec_comp * (
            model_block.rec_comp[w1] * (model_block.select_well[w1]) / max_rec_comp
        )

    @model_block.Constraint(doc="Maximum number of projects constraint")
    def max_proj_rule(model_block):
        return (
            sum(
                model_block.select_cluster[i]
                for i in model_block.parent_block().cluster
            )
            <= max_num_proj
        )


def num_wells_incremental_formulation(model_block):
    """
    Models the number of wells constraint using the incremental cost
    formulation.
    """
    mob_cost = model_block.parent_block().model_inputs.get_mobilization_cost
    model_block.calculate_plugging_cost = Constraint(
        expr=(
            mob_cost[1] * model_block.num_wells_var[1]
            + sum(
                (mob_cost[i] - mob_cost[i - 1]) * model_block.num_wells_var[i]
                for i in model_block.num_wells_var
                if i != 1
            )
            == model_block.plugging_cost
        ),
        doc="Calculates the total plugging cost for the cluster",
    )
    model_block.campaign_length = Constraint(
        expr=(
            sum(model_block.num_wells_var[i] for i in model_block.num_wells_var)
            == model_block.num_wells_chosen
        ),
        doc="Computes the number of wells chosen",
    )

    @model_block.Constraint(
        model_block.num_wells_var.index_set(),
        doc="Ordering num_wells_var variables",
    )
    def ordering_num_wells_vars(model_block, well_idx):
        if well_idx == 1:
            return model_block.num_wells_var[well_idx] == model_block.select_cluster

        return (
            model_block.num_wells_var[well_idx]
            <= model_block.num_wells_var[well_idx - 1]
        )


@declare_custom_block("ClusterBlock")
class ClusterBlockData(BlockData):
    """
    A custom block class for storing variables and constraints
    belonging to a cluster.
    Essential variables and constraints will be added via "rule"
    argument. Here, define methods only for optional cluster-level
    constraints and expressions.
    """

    def deactivate(self):
        """
        Deactivates the constraints present in this block.
        The variables will not be passed to the solver, unless
        they are used in other active constraints.
        """
        super().deactivate()
        self.select_cluster.fix(0)
        self.plugging_cost.fix(0)
        self.num_wells_dac.fix(0)
        self.num_wells_chosen.fix(0)

    def activate(self):
        super().activate()
        self.select_cluster.unfix()
        self.plugging_cost.unfix()
        self.num_wells_dac.unfix()
        self.num_wells_chosen.unfix()

    def fix(
        self,
        cluster: Optional[int] = None,
        wells: Optional[Dict[int, int]] = None,
    ):
        """
        Fixes the binary variables associated with the cluster
        and/or the wells with in the cluster. To fix all variables
        within the cluster, use the fix_all_vars() method.

        Parameters
        ----------
        cluster : 0 or 1, default = None
            `select_cluster` variable will be fixed to this value.
            If None, select_cluster will be fixed to its incumbent value.

        wells : dict, default = None
            key => index of the well, value => value of `select_well`
            binary variable.
        """
        if cluster is None:
            # Nothing is specified, so fixing it to its incumbent value
            self.select_cluster.fix()

        elif cluster in [0, 1]:
            self.select_cluster.fix(cluster)

        if wells is not None:
            # Need to fix a few wells within the cluster
            for w in self.set_wells:
                if w in wells:
                    self.select_well.fix(wells[w])

    def unfix(self):
        """
        Unfixes all the variables within the cluster.
        """
        self.unfix_all_vars()

    def add_distant_well_cuts(self):
        """
        Delete well pairs which are farther than the threshold distance
        """

        @self.Constraint(
            self.set_well_pairs_remove,
            doc="Removes well pairs which are far apart",
        )
        def skip_distant_well_cuts(b, w1, w2):
            return b.select_well[w1] + b.select_well[w2] <= b.select_cluster


# pylint: disable-next = too-many-ancestors
class PluggingCampaignModel(ConcreteModel):
    """
    Builds the optimization model
    """

    def __init__(self, model_inputs, *args, **kwargs):
        """
        Builds the optimization model for identifying the set of projects that
        maximize the overall impact and/or efficiency of plugging.

        Parameters
        ----------
        model_inputs : OptModelInputs
            Object containing the necessary inputs for the optimization model
        """
        super().__init__(*args, **kwargs)

        self.model_inputs = model_inputs
        self.set_clusters = Set(
            initialize=list(model_inputs.campaign_candidates.keys())
        )

        # Define only those parameters which are useful for sensitivity analysis
        self.total_budget = Param(
            initialize=model_inputs.get_total_budget,
            mutable=True,
            doc="Total budget available [Million USD]",
        )
        # Define essential variables and constraints for each cluster
        self.cluster = ClusterBlock(self.set_clusters, rule=build_cluster_model)

        # Add total budget constraint
        self.total_budget_constraint = Constraint(
            expr=(
                sum(self.cluster[c].plugging_cost for c in self.set_clusters)
                <= self.total_budget
            ),
            doc="Total cost of plugging must be within the total budget",
        )

        # Add optional constraints:
        if model_inputs.config.perc_wells_in_dac is not None:
            self.add_min_wells_in_dac()

        if (
            model_inputs.config.threshold_distance is not None
            and not model_inputs.config.lazy_constraints
        ):
            for c in self.set_clusters:
                self.cluster[c].add_distant_well_cuts()

        if model_inputs.config.max_wells_per_owner is not None:
            self.add_owner_well_count()

        # Append the objective function
        self.append_objective()

    def add_min_wells_in_dac(self):
        """
        Adds a constraint that ensures that a certain percentage of wells
        are chosen from disadvantaged communities.
        """
        self.min_wells_in_dac_constraint = Constraint(
            expr=(
                sum(self.cluster[c].num_wells_dac for c in self.set_clusters)
                >= (self.model_inputs.config.perc_wells_in_dac / 100)
                * sum(self.cluster[c].num_wells_chosen for c in self.set_clusters)
            ),
            doc="Ensure that a certain percentage of wells are in dac",
        )

    def add_owner_well_count(self):
        """
        Constrains the maximum number of wells belonging to a specific owner
        chosen for plugging.
        """
        max_owc = self.model_inputs.config.max_wells_per_owner
        owner_dict = self.model_inputs.owner_well_count

        @self.Constraint(
            owner_dict.keys(),
            doc="Limit number of wells belonging to each owner",
        )
        def max_well_owner_constraint(b, owner):
            return (
                sum(b.cluster[c].select_well[w] for c, w in owner_dict[owner])
                <= max_owc
            )

    def append_objective(self):
        """ "
        Appends objective function to the model
        """
        self.total_priority_score = Objective(
            expr=(
                sum(self.cluster[c].cluster_priority_score for c in self.set_clusters)
            ),
            sense=maximize,
            doc="Total Priority score",
        )

    def get_optimal_campaign(self):
        """
        Extracts the optimal choice of wells from the solved model
        """
        optimal_campaign = {}
        plugging_cost = {}

        for c in self.set_clusters:
            blk = self.cluster[c]
            if blk.select_cluster.value < 0.05:
                # Cluster c is not chosen, so continue
                continue

            # Wells in cluster c are chosen
            optimal_campaign[c] = []
            plugging_cost[c] = blk.plugging_cost.value
            for w in blk.set_wells:
                if blk.select_well[w].value > 0.95:
                    # Well w is chosen, so store it in the dict
                    optimal_campaign[c].append(w)

        # Well data
        # TODO: Uncomment when result_parser.py is completed
        # wd = self.model_inputs.config.well_data
        # return OptimalCampaign(wd, optimal_campaign, plugging_cost)
        return (optimal_campaign, plugging_cost)

    def get_solution_pool(self, solver):
        """
        Extracts solutions from the solution pool

        Parameters
        ----------
        solver : Pyomo solver object
        """
        pm = self  # This is the pyomo model
        # pylint: disable=protected-access
        gm = solver._solver_model  # This is the gurobipy model
        # Get Pyomo var to Gurobipy var map.
        # Gurobi vars can be accessed as pm_to_gm[<pyomo var>]
        pm_to_gm = solver._pyomo_var_to_solver_var_map

        # Number of solutions found
        num_solutions = gm.SolCount
        solution_pool = {}
        # Well data
        # wd = self.model_inputs.config.well_data

        for i in range(num_solutions):
            gm.Params.SolutionNumber = i

            optimal_campaign = {}
            plugging_cost = {}

            for c in pm.set_clusters:
                blk = pm.cluster[c]
                if pm_to_gm[blk.select_cluster].Xn < 0.05:
                    # Cluster c is not chosen, so continue
                    continue

                # Wells in cluster c are chosen
                optimal_campaign[c] = []
                plugging_cost[c] = pm_to_gm[blk.plugging_cost].Xn
                for w in blk.set_wells:
                    if pm_to_gm[blk.select_well[w]].Xn > 0.95:
                        # Well w is chosen, so store it in the dict
                        optimal_campaign[c].append(w)

            # TODO: Uncomment the following lines after result_parser is merged
            # solution_pool[i + 1] = OptimalCampaign(
            #     wd=wd, clusters_dict=optimal_campaign, plugging_cost=plugging_cost
            # )
            solution_pool[i + 1] = (optimal_campaign, plugging_cost)

        return solution_pool
