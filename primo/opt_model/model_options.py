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
from primo.opt_model.efficiency import compute_efficieny_scaling_factors
from primo.opt_model.model_with_clustering import PluggingCampaignModel
from primo.utils import get_solver
from primo.utils.clustering_utils import (
    get_admissible_well_pairs,
    get_pairwise_metrics,
    get_wells_in_dac,
    perform_clustering,
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

    # Essential inputs for the optimization model
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

    # Model type and model nature options
    config.declare(
        "objective_type",
        ConfigValue(
            default="Impact",
            domain=In(["Impact", "Efficiency", "Combined"]),
            doc="Objective Type",
        ),
    )
    config.declare(
        "efficiency_formulation",
        ConfigValue(
            default="Max Scaling",
            domain=In(["Max Scaling", "Zone"]),
            doc="Efficiency Formulation",
        ),
    )
    config.declare(
        "objective_weight_impact",
        ConfigValue(
            default=100,
            domain=InRange(0, 100),
            doc="Weight associated with Impact in the objective function",
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
        "lazy_constraints",
        ConfigValue(
            default=False,
            domain=Bool,
            doc="If True, some constraints will be added as lazy constraints",
        ),
    )

    # Parameters for optional constraints

    config.declare(
        "perc_wells_in_dac",
        ConfigValue(
            domain=InRange(0, 100),
            doc="Minimum percentage of wells in disadvantaged communities",
        ),
    )
    config.declare(
        "max_dist_range",
        ConfigValue(
            default=3.0,
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
        "max_num_wells",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of wells admissible per project",
        ),
    )
    config.declare(
        "max_num_projects",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of projects admissible in a campaign",
        ),
    )
    config.declare(
        "min_budget_usage",
        ConfigValue(
            domain=InRange(0, 100),
            doc=(
                "Minimum percent of the budget that needs to be used "
                "for plugging all wells"
            ),
        ),
    )

    # Parameters for computing efficiency metrics
    config.declare(
        "max_dist_to_road",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum distance to road allowed for selected wells",
        ),
    )
    config.declare(
        "max_elevation_delta",
        ConfigValue(
            domain=float,
            doc=(
                "Maximum elevation delta from the closest road "
                "point allowed for selected wells"
            ),
        ),
    )
    config.declare(
        "max_num_unique_owners",
        ConfigValue(
            domain=NonNegativeInt,
            doc="Maximum number of unique owners allowed in a project",
        ),
    )
    config.declare(
        "max_age_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum age range allowed in a project",
        ),
    )
    config.declare(
        "max_depth_range",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum depth range allowed in a project",
        ),
    )
    config.declare(
        "max_population_density",
        ConfigValue(
            domain=NonNegativeFloat,
            doc="Maximum population density allowed to have near a well",
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
        if not hasattr(wd.column_names, "priority_score"):
            msg = (
                "Unable to find priority scores in the WellData object. Compute the scores "
                "using the compute_priority_scores method."
            )
            raise_exception(msg, ValueError)

        # Construct campaign candidates
        LOGGER.info(
            f"Constructing clusters where the distance between wells "
            f"is less than {self.config.max_dist_range} miles. Pass "
            f"max_dist_range argument to change this value."
        )
        # Clustering returns a dictionary where
        # keys => cluster number, values => list of wells in the cluster
        self.campaign_candidates = perform_clustering(
            wd, distance_threshold=self.config.max_dist_range
        )

        # Obtain well pairs that cannot be a part of the project
        # get_pairwise_metrics returns an object with three attributes:
        # distance, age, and depth; which contain pairwise distances, age range,
        # and depth range, respectively.
        self.pairwise_metrics = get_pairwise_metrics(wd, self.campaign_candidates)
        well_pairs = get_admissible_well_pairs(
            pairwise_metrics=self.pairwise_metrics,
            max_distance=self.config.max_dist_range,
            max_age_range=self.config.max_age_range,
            max_depth_range=self.config.max_depth_range,
        )

        # well_pairs_keep and well_pairs_remove are dictionaries, where
        # keys => cluster number, values => list(Tuples(well pairs))
        self.well_pairs_keep = well_pairs[0]
        self.well_pairs_remove = well_pairs[1]

        # wells_in_dac: {cluster: [list of wells in dac], ...}
        self.wells_in_dac = get_wells_in_dac(wd, self.campaign_candidates)

        # Construct owner well count data
        col_names = wd.column_names
        operator_list = set(wd[col_names.operator_name])
        self.owner_well_count = {owner: [] for owner in operator_list}
        for well in wd:
            # {Owner 1: [(c1, i2), (c1, i3), (c4, i7), ...], ...}
            # Key => Owner name, Tuple[0] => cluster, Tuple[1] => index
            self.owner_well_count[wd.data.loc[well, col_names.operator_name]].append(
                (wd.data.loc[well, col_names.cluster], well)
            )

        # Raise an error if efficiency metrics are not provided when they are required
        if self.config.objective_type in ["Efficiency", "Combined"]:
            if wd.config.efficiency_metrics is None:
                msg = (
                    f"efficiency_metrics is not defined in the WellData object. "
                    f"Effeciency metrics are essential for objective_type "
                    f"{self.config.objective_type}."
                )
                raise_exception(msg, ValueError)

            # Efficiency metrics are provided, so compute scaling factors
            compute_efficieny_scaling_factors(self)

        # NOTE: Attributes _opt_model and _solver are defined in
        # build_optimization_model and solve_model methods, respectively.
        self._opt_model = None
        self._solver = None
        self._opt_campaign = None
        LOGGER.info("Completed processing optimization model inputs.")

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

    @property
    def optimal_campaign(self):
        """Return the optimal campaign"""
        if self._opt_campaign is None:
            LOGGER.warning("Optimal campaign is not available.")

        return self._opt_campaign

    def build_optimization_model(self):
        """Builds the optimization model"""
        LOGGER.info("Beginning to construct the optimization model.")
        self._opt_model = PluggingCampaignModel(self)
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
        solver.solve(self._opt_model, tee=kwargs.pop("stream_output", True))

        # Return the solution pool, if it is requested
        if solver_name == "gurobi_persistent" and pool_search_mode == 2:
            # Return the solution pool if pool_search_mode is active
            return self._opt_model.get_solution_pool(self._solver)

        # In all other cases, return the optimal campaign
        self._opt_campaign = self._opt_model.get_optimal_campaign()
        return self._opt_campaign
