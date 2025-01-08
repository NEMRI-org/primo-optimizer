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
from itertools import combinations

# Installed libs
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    IsInstance,
    NonNegativeFloat,
    NonNegativeInt,
    document_kwargs_from_configdict,
)
from pyomo.environ import SolverFactory

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.utils import get_solver
from primo.utils.clustering_utils import (
    distance_matrix,
    perform_agglomerative_clustering,
    perform_louvain_clustering,
)
from primo.utils.domain_validators import InRange, validate_mobilization_cost
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def model_config() -> ConfigDict:
    """
    Returns a Pyomo ConfigDict object that includes all user options
    associated with optimization modeling
    """
    # Container for storing and performing domain validation
    # of the inputs of the optimization model.
    # ConfigValue automatically performs domain validation.
    config = ConfigDict()
    config.declare(
        "well_data",
        ConfigValue(
            domain=IsInstance(WellData),
            doc="WellData object containing the entire dataset",
        ),
    )
    config.declare(
        "total_budget",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Total budget for plugging [in USD]",
        ),
    )
    config.declare(
        "mobilization_cost",
        ConfigValue(
            domain=validate_mobilization_cost,
            doc="Cost of plugging wells [in USD]",
        ),
    )
    config.declare(
        "perc_wells_in_dac",
        ConfigValue(
            domain=InRange(0, 100),
            doc="Minimum percentage of wells in disadvantaged communities",
        ),
    )
    config.declare(
        "threshold_distance",
        ConfigValue(
            default=10.0,
            domain=NonNegativeFloat,
            doc="Maximum distance [in miles] allowed between wells",
        ),
    )
    config.declare(
        "max_wells_per_owner",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells per owner",
        ),
    )
    config.declare(
        "max_cost_project",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum cost per project [in USD]",
        ),
    )
    config.declare(
        "max_size_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells admissible per project",
        ),
    )
    config.declare(
        "num_wells_model_type",
        ConfigValue(
            default="multicommodity",
            domain=In(["multicommodity", "incremental"]),
            doc="Choice of formulation for modeling number of wells",
        ),
    )
    config.declare(
        "model_nature",
        ConfigValue(
            default="linear",
            domain=In(["linear", "quadratic", "aggregated_linear"]),
            doc="Nature of the optimization model: MILP or MIQCQP",
        ),
    )
    config.declare(
        "cluster_formulation",
        ConfigValue(
            default="Agglomerative",
            domain=In(["Agglomerative", "Louvain"]),
            doc="Formulation used for clustering the wells",
        ),
    )
    config.declare(
        "threshold_cluster_size",
        ConfigValue(
            default=300,
            domain=NonNegativeInt,
            doc="Maximum size of clusters for Louvain clustering",
        ),
    )
    config.declare(
        "num_nearest_neighbors",
        ConfigValue(
            default=10,
            domain=NonNegativeInt,
            doc="Nearest neighbors while constructing graph for Louvain clustering",
        ),
    )
    config.declare(
        "max_resolution",
        ConfigValue(
            default=10,
            domain=NonNegativeFloat,
            doc="Maximum resolution for Louvain clustering",
        ),
    )
    config.declare(
        "lazy_constraints",
        ConfigValue(
            default=False,
            domain=Bool,
            doc="If True, some constraints will be added as lazy constraints",
        ),
    )
    config.declare(
        "min_budget_usage",
        ConfigValue(
            default=None,
            domain=InRange(0, 100),
            doc="Minimum percentage of the total budget to be used for plugging",
        ),
    )
    config.declare(
        "penalize_unused_budget",
        ConfigValue(
            default=False,
            domain=Bool,
            doc=(
                "If True, unused budget will be penalized in the objective function\n"
                "with suitably chosen weight factor"
            ),
        ),
    )

    return config


class OptModelInputs:  # pylint: disable=too-many-instance-attributes
    """
    Assembles all the necessary inputs for the optimization model.
    """

    # Using ConfigDict from Pyomo for domain validation.
    CONFIG = model_config()

    @document_kwargs_from_configdict(CONFIG)
    def __init__(self, cluster_mapping=None, **kwargs):
        # Update the values of all the inputs
        # ConfigDict handles KeyError, other input errors, and domain errors
        LOGGER.info("Processing optimization model inputs.")
        self.config = self.CONFIG(kwargs)

        # Raise an error if the essential inputs are not provided
        wd = self.config.well_data
        if None in [wd, self.config.total_budget, self.config.mobilization_cost]:
            msg = (
                "One or more essential input arguments in [well_data, total_budget, "
                "mobilization_cost] are missing while instantiating the object. "
                "WellData object containing information on all wells, the total budget, "
                "and the mobilization cost are essential inputs for the optimization model. "
            )
            raise_exception(msg, ValueError)

        # Raise an error if priority scores are not calculated.
        if "Priority Score [0-100]" not in wd:
            msg = (
                "Unable to find priority scores in the WellData object. Compute the scores "
                "using the compute_priority_scores method."
            )
            raise_exception(msg, ValueError)

        col_names = wd.col_names
        if cluster_mapping is None:
            logging.info("Clustering Data in Opt Model Inputs")
            # Construct campaign candidates
            # Step 1: Perform clustering, Should distance_threshold be a user argument?
            if self.config.cluster_formulation == "Agglomerative":
                perform_agglomerative_clustering(
                    wd, threshold_distance=self.config.threshold_distance
                )
            else:
                perform_louvain_clustering(
                    wd,
                    threshold_distance=self.config.threshold_distance,
                    threshold_cluster_size=self.config.threshold_cluster_size,
                    nearest_neighbors=self.config.num_nearest_neighbors,
                    max_resolution=self.config.max_resolution,
                )
            # Step 2: Identify list of wells belonging to each cluster
            # Structure: {cluster_1: [index_1, index_2,..], cluster_2: [], ...}
            set_clusters = set(wd[col_names.cluster])
            self.campaign_candidates = {
                cluster: list(wd.data[wd[col_names.cluster] == cluster].index)
                for cluster in set_clusters
            }
            self.pairwise_distance = self._pairwise_matrix(metric="distance")
            self.pairwise_age_difference = self._pairwise_matrix(metric="age")
            self.pairwise_depth_difference = self._pairwise_matrix(metric="depth")

        else:
            logging.info("Skipping clustering step in Opt Model Inputs")
            self.campaign_candidates = cluster_mapping
            well_cluster_map = {index: "" for index in wd.data.index}
            for cluster, wells in self.campaign_candidates.items():
                for well in wells:
                    well_cluster_map[well] = cluster
            for _, value in well_cluster_map.items():
                assert value != ""
            cluster_col_values = [cluster for _, cluster in well_cluster_map.items()]
            wd.data["Clusters"] = cluster_col_values
            col_names.register_new_columns({"cluster": "Clusters"})

            self.pairwise_distance = {}
            self.pairwise_age_difference = {}
            self.pairwise_depth_difference = {}
        # Step 3: Construct pairwise-metrics between wells in each cluster.
        # Structure: {cluster: {(index_1, index_2): distance_12, ...}...}

        # Construct owner well count data
        if wd.config.verify_operator_name:
            operator_list = set(wd[col_names.operator_name])
            self.owner_well_count = {owner: [] for owner in operator_list}
            for well in wd:
                # {Owner 1: [(c1, i2), (c1, i3), (c4, i7), ...], ...}
                # Key => Owner name, Tuple[0] => cluster, Tuple[1] => index
                self.owner_well_count[
                    wd.data.loc[well, col_names.operator_name]
                ].append((wd.data.loc[well, col_names.cluster], well))

        # NOTE: Attributes _opt_model and _solver are defined in
        # build_optimization_model and solve_model methods, respectively.
        self._opt_model = None
        self._solver = None
        LOGGER.info("Finished optimization model inputs.")

    @property
    def get_total_budget(self):
        """Returns scaled total budget [in million USD]"""
        # Optimization model uses scaled total budget value to avoid numerical issues
        return self.config.total_budget / 1e6

    @property
    def get_mobilization_cost(self):
        """Returns scaled mobilization cost [in million USD]"""
        # Optimization model uses Scaled mobilization costs to avoid numerical issues
        return {
            num_wells: cost / 1e6
            for num_wells, cost in self.config.mobilization_cost.items()
        }

    @property
    def get_max_cost_project(self):
        """Returns scaled maximum cost of the project [in million USD]"""
        if self.config.max_cost_project is None:
            return None

        return self.config.max_cost_project / 1e6

    @property
    def optimization_model(self):
        """Returns the Pyomo optimization model"""
        return self._opt_model

    @property
    def solver(self):
        """Returns the solver object"""
        return self._solver

    def _pairwise_matrix(self, metric: str):
        wd = self.config.well_data  # WellData object
        # distance_matrix returns a numpy array.
        metric_array = distance_matrix(wd, {metric: 1})

        # DataFrame index -> metric_array index map
        df_to_array = {
            df_index: array_index for array_index, df_index in enumerate(wd.data.index)
        }

        # NOTE: Storing the entire matrix may require a lot of memory.
        # So, constructing the following dict of dicts
        # {cluster: {(w1, w2): metric, (w1, w3): metric,...}, ...}
        return {
            cluster: {
                (w1, w2): metric_array[df_to_array[w1], df_to_array[w2]]
                for w1, w2 in combinations(well_list, 2)
            }
            for cluster, well_list in self.campaign_candidates.items()
        }

    def build_optimization_model(self, override_dict=None):
        """Builds the optimization model"""
        LOGGER.info("Beginning to construct the optimization model.")
        self._opt_model = PluggingCampaignModel(self, override_dict)
        LOGGER.info("Completed the construction of the optimization model.")
        return self._opt_model

    def solve_model(self, **kwargs):
        """Solves the optimization"""

        # Adding support for pool search if gurobi_persistent is available
        # To get n-best solutions, pass pool_search_mode = 2 and pool_size = n
        pool_search_mode = kwargs.pop("pool_search_mode", 0)
        pool_size = kwargs.pop("pool_size", 10)

        # If a solver is specified, use it.
        if "solver" in kwargs:
            solver = get_solver(**kwargs)
            solver_name = kwargs["solver"]
        else:
            # Otherwise, auto-detect solver in order of priority
            for solver_name in ("gurobi_persistent", "gurobi", "scip", "glpk", "highs"):
                if SolverFactory(solver_name).available(exception_flag=False):
                    LOGGER.info(
                        f"Optimization solver is not specified. "
                        f"Using {solver_name} as the optimization solver."
                    )
                    solver = get_solver(solver=solver_name, **kwargs)
                    break

        self._solver = solver
        if solver_name == "gurobi_persistent":
            # For persistent solvers, model instance need to be set manually
            solver.set_instance(self._opt_model)
            solver.set_gurobi_param("PoolSearchMode", pool_search_mode)
            solver.set_gurobi_param("PoolSolutions", pool_size)

        # Solve the optimization problem
        solver.solve(self._opt_model)

        # Return the solution pool, if it is requested
        if solver_name == "gurobi_persistent" and pool_search_mode == 2:
            # Return the solution pool if pool_search_mode is active
            return self._opt_model.get_solution_pool(self._solver)

        # In all other cases, return the optimal campaign
        return self._opt_model.get_optimal_campaign()

    def update_cluster(self, add_widget_return):
        """
        Updates the campaign candidates by changing the cluster numbers for specific wells.

        Parameters
        ---------
        add_widget_return : OverrideAddInfo
            An OverrideAddInfo object which includes information on wells selected to add
            to the existing optimal P&A projects
        """
        existing_clusters = add_widget_return.existing_clusters
        new_clusters = add_widget_return.new_clusters
        wd = self.config.well_data
        col_names = wd.col_names

        if existing_clusters != new_clusters:
            # Remove wells from existing clusters and update owner well counts
            for existing_cluster, existing_wells in existing_clusters.items():
                for well in existing_wells:
                    self.campaign_candidates[existing_cluster].remove(well)
                    self.owner_well_count[
                        wd.data.loc[well, col_names.operator_name]
                    ].remove((existing_cluster, well))

            # Add wells to new clusters and update the well data and owner well counts
            for new_cluster, wells in new_clusters.items():
                for well in wells:
                    self.campaign_candidates[new_cluster].append(well)
                    self.config.well_data.data.loc[well, col_names.cluster] = (
                        new_cluster
                    )
                    self.owner_well_count[
                        wd.data.loc[well, col_names.operator_name]
                    ].append((new_cluster, well))

            # Update pairwise distances based on the new well data
            self.pairwise_distance = self._pairwise_matrix(metric="distance")
