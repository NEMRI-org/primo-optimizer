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
    Objective,
    Param,
    Set,
    Var,
    maximize,
)

# User-defined libs
from primo.opt_model.efficiency import build_efficiency_model
from primo.opt_model.result_parser import Campaign

LOGGER = logging.getLogger(__name__)


def build_cluster_model(model_block, cluster):
    """
    Builds the model block (adds essential variables and constraints)
    for a given cluster `cluster`
    """
    # Parameters are located in the parent block
    params = model_block.parent_block().model_inputs  # OptModelInputs object

    # Essential model sets
    model_block.set_wells = Set(
        initialize=params.campaign_candidates[cluster],
        doc="Set of wells in cluster c",
    )
    model_block.set_wells_dac = Set(
        initialize=params.wells_in_dac[cluster],
        doc="Set of wells that are in disadvantaged communities",
    )
    model_block.set_well_pairs_remove = Set(
        initialize=params.well_pairs_remove[cluster],
        doc="Well-pairs which cannot be a part of the project",
    )
    model_block.set_well_pairs_keep = Set(
        initialize=params.well_pairs_keep[cluster],
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

    # Useful expressions
    wd = params.config.well_data
    wt_impact = params.config.objective_weight_impact / 100

    priority_score = wd[wd.column_names.priority_score]

    @model_block(model_block.set_wells, doc="(Combined) priority score for each well")
    def well_priority_score(blk, w):
        return wt_impact * priority_score[w] * blk.select_well[w]

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
        scaling_factor, budget_sufficient = self.budget_slack_variable_scaling()
        self.slack_var_scaling = Param(
            initialize=scaling_factor,
            mutable=True,
            within=NonNegativeReals,
            doc="Budget slack variable coefficient in the objective function",
        )

        # Define essential variables and constraints for each cluster
        self.cluster = ClusterBlock(self.set_clusters, rule=build_cluster_model)

        # Define slack variable for unutilized budget

        self.unused_budget = Var(
            within=NonNegativeReals,
            doc="Unused budget from the total budget",
        self.budget_slack_var = Var(
            within=NonNegativeReals,
            doc="Slack variable for the unutilized amount of total budget",
        )

        # Add total budget constraint
        self.total_budget_constraint = Constraint(
            expr=(
                sum(self.cluster[c].plugging_cost for c in self.set_clusters)
                + self.unused_budget
                <= self.total_budget
            ),
            doc="Total cost of plugging must be within the total budget",
        )


        if model_inputs.objective_type in ["Efficiency", "Combined"]:
            LOGGER.info("Building the efficiency model.")
            for c in self.set_clusters:
                build_efficiency_model(self.cluster[c], cluster=c)
        # Add a constraint to calculate the unutilized amount of budget
        self.budget_constraint_slack = Constraint(
            expr=(
                self.total_budget
                - sum(self.cluster[c].plugging_cost for c in self.set_clusters)
                <= self.budget_slack_var
            ),
            doc="Calculate the unutilized amount of budget",
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

        if model_inputs.config.max_num_projects is not None:
            self.add_max_num_projects()

        for c in self.set_clusters:
            self.cluster[c].plugging_cost.setub(model_inputs.get_max_cost_project)
            self.cluster[c].num_wells_chosen.setub(model_inputs.config.max_num_wells)

        if model_inputs.config.min_budget_usage is not None:
            if budget_sufficient is False:
                self.add_min_budget_usage()
        else:
            self.slack_var_scaling = 0

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
            list(owner_dict.keys()),
            doc="Limit number of wells belonging to each owner",
        )
        def max_well_owner_constraint(b, owner):
            return (
                sum(b.cluster[c].select_well[w] for c, w in owner_dict[owner])
                <= max_owc
            )

    def add_max_num_projects(self):
        """Constrains the maximum number of projects a campaign can have."""
        self.max_num_projects_constraint = Constraint(
            expr=(
                sum(self.cluster[c].select_cluster for c in self.cluster)
                <= self.model_inputs.config.max_num_projects
            ),
            doc="Maximum number of projects constraint",
        )

    def add_min_budget_usage(self):
        """
        Implements a constraint to ensure at least 50% of the
        budget is used when the budget is sufficient to
        plug all wells.
        """
        min_percent = self.model_inputs.config.min_budget_usage / 100
        self.min_budget_usage_con = Constraint(
            expr=(
                sum(self.cluster[c].plugging_cost for c in self.set_clusters)
                >= min_percent * self.total_budget
            ),
            doc="Enforces a minimum budget usage",
        )

    def append_objective(self):
        """
        Appends objective function to the model
        """
        self.total_priority_score = Objective(
            expr=sum(
                self.cluster[c].well_priority_score[w]
                for c in self.set_clusters
                for w in self.cluster[c].set_wells
            )
            - self.slack_var_scaling * self.unused_budget,
            expr=(
                sum(self.cluster[c].cluster_priority_score for c in self.set_clusters)
                - self.slack_var_scaling * self.budget_slack_var
            ),
            sense=maximize,
            doc="Total Priority score minus scaled slack variable for unutilized budget",
        )

    def budget_slack_variable_scaling(self):
        """
        Check whether the budget is sufficient to plug all wells and
        calculate the scaling factor for the budget slack variable based on the
        corresponding scenario.

        """
        # estimate the maximum number of wells can be plugged with the budget.
        unit_cost = max(self.model_inputs.get_mobilization_cost.values()) / max(
            self.model_inputs.get_mobilization_cost
        )
        max_well_num = self.model_inputs.get_total_budget / unit_cost

        # calculate the scaling factor for the budget slack variable if the
        # budget is not sufficient to plug all wells
        if max_well_num < len(self.model_inputs.config.well_data):
            scaling_budget_slack = (
                max_well_num
                * max(self.model_inputs.config.well_data["Priority Score [0-100]"])
            ) / self.model_inputs.get_total_budget
            budget_sufficient = False
        # calculate the scaling factor for the budget slack variable if the
        # budget is sufficient to plug all wells
        else:
            scaling_budget_slack = (
                len(self.model_inputs.config.well_data)
                * max(self.model_inputs.config.well_data["Priority Score [0-100]"])
            ) / self.model_inputs.get_total_budget
            budget_sufficient = True
        return scaling_budget_slack, budget_sufficient

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

        wd = self.model_inputs.config.well_data
        return Campaign(wd, optimal_campaign, plugging_cost)

    def get_solution_pool(self, solver):
        """
        Extracts solutions from the solution pool

        Parameters
        ----------
        solver : Pyomo solver object
        """
        pm = self  # This is the Pyomo model
        # pylint: disable=protected-access
        gm = solver._solver_model  # This is the Gurobipy model
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

            solution_pool[i + 1] = Campaign(
                wd=self.model_inputs.config.well_data,
                clusters_dict=optimal_campaign,
                plugging_cost=plugging_cost,
            )

        return solution_pool
