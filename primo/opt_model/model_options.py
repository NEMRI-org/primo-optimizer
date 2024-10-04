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
from primo.data_parser import WellData
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.utils import check_optimal_termination, get_solver
from primo.utils.clustering_utils import distance_matrix, perform_clustering
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
        "objective_type",
        ConfigValue(
            default="Impact",
            domain=In(["Impact", "Efficiency", "Combined"]),
            doc="Objective Type",
        ),
    )
    config.declare(
        "objective_weight_impact",
        ConfigValue(
            domain=InRange(0, 100),
            doc="Weight of Impact in Objective Function",
        ),
    )
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
            doc="The minimum percent of the budget usage when the budget is insufficient for plugging all wells",
        ),
    )

    config.declare(
        "max_distance_to_road",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum Distance to road of wells selected in the project",
        ),
    )

    config.declare(
        "max_elevation_delta",
        ConfigValue(
            domain=float,
            doc="Maximum Elevation delta from closest road point of wells selected in the project",
        ),
    )

    config.declare(
        "max_number_of_unique_owners",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum Number of unique owners in the selected project",
        ),
    )

    config.declare(
        "max_age_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum Age range of wells in the project",
        ),
    )

    config.declare(
        "max_depth_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum Depth range of wells in the project",
        ),
    )

    config.declare(
        "record_completeness",
        ConfigValue(
            domain=InRange(0, 100),
            doc="Record completeness of wells in well data",
        ),
    )
    config.declare(
        "max_well_distance",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum Pairwise distance of wells in well data",
        ),
    )
    config.declare(
        "max_population_density",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="population density of wells in well data",
        ),
    )
    config.declare(
        "max_num_project",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of projects",
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
    def __init__(self, **kwargs):
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

        # Construct campaign candidates
        # Step 1: Perform clustering, Should distance_threshold be a user argument?
        perform_clustering(wd, distance_threshold=10.0)
        self.get_eff_scaling_factors()

        # Step 2: Identify list of wells belonging to each cluster
        # Structure: {cluster_1: [index_1, index_2,..], cluster_2: [], ...}
        col_names = wd.col_names
        set_clusters = set(wd[col_names.cluster])
        self.campaign_candidates = {
            cluster: list(wd.data[wd[col_names.cluster] == cluster].index)
            for cluster in set_clusters
        }

        # Step 3: Construct pairwise-metrics between wells in each cluster.
        # Structure: {cluster: {(index_1, index_2): distance_12, ...}...}
        self.pairwise_distance = self._pairwise_matrix(metric="distance")
        self.pairwise_age_difference = self._pairwise_matrix(metric="age")
        self.pairwise_depth_difference = self._pairwise_matrix(metric="depth")

        # Construct owner well count data
        operator_list = set(wd[col_names.operator_name])
        self.owner_well_count = {owner: [] for owner in operator_list}
        for well in wd:
            # {Owner 1: [(c1, i2), (c1, i3), (c4, i7), ...], ...}
            # Key => Owner name, Tuple[0] => cluster, Tuple[1] => index
            self.owner_well_count[wd.data.loc[well, col_names.operator_name]].append(
                (wd.data.loc[well, col_names.cluster], well)
            )

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

    def build_optimization_model(self):
        """Builds the optimization model"""
        LOGGER.info("Beginning to construct the optimization model.")
        self._opt_model = PluggingCampaignModel(self)
        LOGGER.info("Completed the construction of the optimization model.")
        return self._opt_model

    def get_eff_scaling_factors(self):
        """Returns scaling factors for efficiency metrics to be used in the opt model"""
        LOGGER.info("Beginning to calculate the scaling factors for efficiency metrics")
        if self.config.objective_type == "Efficiency" or "Combined":
            eff_metrics = wd.efficiency_metrics
            wd = self.config.well_data
            if (
                self.config.max_distance_to_road == None
                and eff_metrics.dist_road.effective_weight > 0
            ):
                self.config.max_distance_to_road = max(wd["Distance to Road [miles]"])
                LOGGER.info(
                    f"Maximum distance to road not provided, setting it to {max(wd['Distance to Road [miles]'])}"
                )
            if (
                self.config.max_elevation_delta == None
                and eff_metrics.elev_delta.effective_weight > 0
            ):
                self.config.max_elevation_delta = max(wd["Elevation Delta [m]"])
                LOGGER.warning(
                    f"Maximum elevation delta not provided, setting it to {max(wd['Elevation Delta [m]'])}"
                )
            if (
                self.config.max_size_project == None
                and eff_metrics.num_wells.effective_weight > 0
            ):
                self.config.max_size_project = 20
                LOGGER.warning(f"Maximum project size not provided, setting it to {20}")
            if (
                self.config.max_number_of_unique_owners == None
                and eff_metrics.num_unique_owners.effective_weight > 0
            ):
                self.config.max_number_of_unique_owners = 20
                LOGGER.warning(
                    f"Maximum number of unique owners not provided, setting it to {20}"
                )
            if (
                self.config.max_age_range == None
                and eff_metrics.age_range.effective_weight > 0
            ):
                self.config.max_age_range = max(
                    max(inner_dict.values())
                    for inner_dict in self.pairwise_age_difference.values()
                )
                LOGGER.warning(
                    f"Maximum age range not provided, setting it to {self.config.max_age_range}"
                )
            if (
                self.config.max_depth_range == None
                and eff_metrics.depth_range.effective_weight > 0
            ):
                self.config.max_depth_range = max(
                    max(inner_dict.values())
                    for inner_dict in self.pairwise_depth_difference.values()
                )
                LOGGER.warning(
                    f"Maximum depth range not provided, setting it to {self.config.max_depth_range}"
                )
            if (
                self.config.max_well_distance == None
                and eff_metrics.well_distance.effective_weight > 0
            ):
                self.config.max_well_distance = max(
                    max(inner_dict.values())
                    for inner_dict in self.pairwise_distance.values()
                )
                LOGGER.warning(
                    f"Maximum pairwise distance not provided, setting it to {self.config.max_well_distance}"
                )
            if (
                self.config.max_record_completeness == None
                and eff_metrics.rec_comp.effective_weight > 0
            ):
                self.config.max_record_completeness = max(wd["Record Completeness"])
                LOGGER.warning(
                    f"Maximum record completeness not provided, setting it to {max(wd['Record Completeness'])}"
                )
            if (
                self.config.max_population_density == None
                and eff_metrics.pop_den.effective_weight > 0
            ):
                self.config.max_population_density = max(wd["Population Density"])
                LOGGER.warning(
                    f"Maximum population density not provided, setting it to {max(wd['Population Density'])}"
                )
            LOGGER.info(
                "Completed calculating the scaling factors for efficiency metrics"
            )
        else:
            return
        return

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
        solver.solve(self._opt_model, tee=kwargs.pop("stream_output", True))

        # Return the solution pool, if it is requested
        if solver_name == "gurobi_persistent" and pool_search_mode == 2:
            # Return the solution pool if pool_search_mode is active
            return self._opt_model.get_solution_pool(self._solver)

        # In all other cases, return the optimal campaign
        return self._opt_model.get_optimal_campaign()
