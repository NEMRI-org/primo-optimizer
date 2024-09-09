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

LOGGER = logging.getLogger(__name__)


def build_cluster_model(m, c):
    """
    Builds the model (adds essential variables and constraints)
    for a given cluster c
    """
    # Parameters are located in the parent block
    params = m.parent_block().model_inputs
    wd = params.config.well_data
    well_index = params.campaign_candidates[c]
    pairwise_distance = params.pairwise_distance[c]
    # pairwise_age_range = params.pairwise_age_range[c]
    # pairwise_depth_range = params.pairwise_depth_range[c]

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
    m.set_wells = Set(
        initialize=well_index,
        doc="Set of wells in cluster c",
    )
    m.set_wells_dac = Set(
        initialize=well_dac,
        doc="Set of wells that are in disadvantaged communities",
    )
    m.set_well_pairs_remove = Set(
        initialize=well_pairs_remove,
        doc="Well-pairs which cannot be a part of the project",
    )
    m.set_well_pairs_keep = Set(
        initialize=well_pairs_keep,
        doc="Well-pairs which can be a part of the project",
    )

    # Essential variables
    m.select_cluster = Var(
        within=Binary,
        doc="1, if wells from the cluster are chosen for plugging, 0 Otherwise",
    )
    m.select_well = Var(
        m.set_wells,
        within=Binary,
        doc="1, if the well is selected for plugging, 0 otherwise",
    )
    m.num_wells_var = Var(
        range(1, len(m.set_wells) + 1),
        within=Binary,
        doc="Variables to track the total number of wells chosen",
    )
    m.plugging_cost = Var(
        within=NonNegativeReals,
        doc="Total cost for plugging wells in this cluster",
    )

    # Although the following two variables are of type Integer, they
    # can be declared as continuous. The optimal solution is guaranteed to have
    # integer values.
    m.num_wells_chosen = Var(
        within=NonNegativeReals,
        doc="Total number of wells chosen in the project",
    )
    m.num_wells_dac = Var(
        within=NonNegativeReals,
        doc="Number of wells chosen in disadvantaged communities",
    )

    # Set the maximum cost and size of the project: default is None.
    m.plugging_cost.setub(params.get_max_cost_project)
    m.num_wells_chosen.setub(params.config.max_size_project)

    # Useful expressions
    priority_score = wd.data["Priority Score [0-100]"]
    m.cluster_priority_score = Expression(
        expr=(sum(priority_score[w] * m.select_well[w] for w in m.set_wells)),
        doc="Computes the total priority score for the cluster",
    )

    # Essential constraints
    m.calculate_num_wells_chosen = Constraint(
        expr=(sum(m.select_well[w] for w in m.set_wells) == m.num_wells_chosen),
        doc="Calculate the total number of wells chosen",
    )
    m.calculate_num_wells_in_dac = Constraint(
        expr=(sum(m.select_well[w] for w in m.set_wells_dac) == m.num_wells_dac),
        doc="Calculate the number of wells chosen that are in dac",
    )

    # This is to test which formulation is faster. If there is no
    # benefit in terms of computational time, then delete this method.
    if params.config.num_wells_model_type == "incremental":
        num_wells_incremental_formulation(m)
        return

    # Using the multicommodity formulation
    mob_cost = params.get_mobilization_cost
    m.calculate_plugging_cost = Constraint(
        expr=(
            sum(mob_cost[i] * m.num_wells_var[i] for i in m.num_wells_var)
            == m.plugging_cost
        ),
        doc="Calculates the total plugging cost for the cluster",
    )
    m.campaign_length = Constraint(
        expr=(
            sum(i * m.num_wells_var[i] for i in m.num_wells_var) == m.num_wells_chosen
        ),
        doc="Determines the number of wells chosen",
    )
    m.num_well_uniqueness = Constraint(
        expr=(sum(m.num_wells_var[i] for i in m.num_wells_var) == m.select_cluster),
        doc="Ensures at most one num_wells_var is selected",
    )


def num_wells_incremental_formulation(m):
    """
    Models the number of wells constraint using the incremental cost
    formulation.
    """
    mob_cost = m.parent_block().model_inputs.get_mobilization_cost
    m.calculate_plugging_cost = Constraint(
        expr=(
            mob_cost[1] * m.num_wells_var[1]
            + sum(
                (mob_cost[i] - mob_cost[i - 1]) * m.num_wells_var[i]
                for i in m.num_wells_var
                if i != 1
            )
            == m.plugging_cost
        ),
        doc="Calculates the total plugging cost for the cluster",
    )
    m.campaign_length = Constraint(
        expr=(sum(m.num_wells_var[i] for i in m.num_wells_var) == m.num_wells_chosen),
        doc="Computes the number of wells chosen",
    )

    @m.Constraint(
        m.num_wells_var.index_set(),
        doc="Ordering num_wells_var variables",
    )
    def ordering_num_wells_vars(b, i):
        if i == 1:
            return b.num_wells_var[i] == b.select_cluster

        return b.num_wells_var[i] <= b.num_wells_var[i - 1]


# pylint: disable = trailing-whitespace, attribute-defined-outside-init
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
        self.num_wells_chosen.fix(0)

    def activate(self):
        super().activate()
        self.select_cluster.unfix()
        self.pluggin_cost.unfix()
        self.num_wells_chosen.unfix()

    def fix(self, cluster=1, wells=None):
        """
        Fixes the binary variables associated with the cluster
        and/or the wells with in the cluster.

        Parameters
        ----------
        cluster : 0 or 1, default = 1
            `select_cluster` variable will be fixed to this value.

        wells : dict, default = None
            key => index of the well, value => value of `select_well`
            binary variable.
        """
        # Fix the cluster binary variable
        self.select_cluster.fix(cluster)

        if wells is None:
            # Well binary variables are not selected, so return
            return

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
        # wd = self.model_inputs.config.well_data
        # return OptimalCampaign(wd, optimal_campaign, plugging_cost)
        return (optimal_campaign, plugging_cost)
