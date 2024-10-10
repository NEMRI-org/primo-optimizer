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
from pyomo.environ import Constraint, NonNegativeReals, Var

# User-defined libs
from primo.data_parser import WellData

LOGGER = logging.getLogger(__name__)


def compute_efficieny_scaling_factors(opt_model_inputs):
    """
    Checks whether scaling factors for efficiency metrics are provided by
    the user or not. If not, computes the scaling factors using the entire
    dataset.

    Parameters
    ----------
    opt_model_inputs : OptModelInputs
        Object containing the necessary inputs for the optimization model
    """
    LOGGER.info("Checking/computing scaling factors for efficiency metrics")
    config = opt_model_inputs.config
    wd = opt_model_inputs.config.well_data
    col_names = wd.column_names
    eff_metrics = wd.config.efficiency_metrics
    eff_weights = wd.config.efficiency_metrics.get_weights

    def log_message(met_name, scale_value):
        """Function for logging warning message"""
        LOGGER.warning(
            f"Scaling factor for the efficiency metric {met_name} is not "
            f"provided, so it is set to {scale_value}. To modify the "
            f"scaling factor, pass argument max_{met_name} while instantiating "
            f"the OptModelInputs object."
        )

    # Setting a scaling factor for num_wells metric
    if config.max_num_wells is None and eff_weights.num_wells > 0:
        log_message(eff_metrics.num_wells.name, 20)
        config.max_size_project = 20

    # Setting a scaling factor for num_unique_owners metric
    if config.max_num_unique_owners is None and eff_weights.num_unique_owners > 0:
        log_message(eff_metrics.num_unique_owners.name, 5)
        config.max_num_unique_owners = 5

    # Computing scaling factor dist_to_road metric
    if config.max_dist_to_road is None and eff_weights.dist_to_road > 0:
        max_dist_to_road = wd[col_names.dist_to_road].max()
        log_message(eff_metrics.dist_to_road.name, max_dist_to_road)
        config.max_dist_to_road = max_dist_to_road

    # Computing scaling factor for elevation_delta metric
    if config.max_elevation_delta is None and eff_weights.elevation_delta > 0:
        max_elevation_delta = wd[col_names.elevation_delta].max()
        log_message(eff_metrics.elevation_delta.name, max_elevation_delta)
        config.max_elevation_delta = max_elevation_delta

    # Setting a scaling factor for population_density metric
    if config.max_population_density is None and eff_weights.population_density > 0:
        max_population_density = wd[col_names.population_density].max()
        log_message(eff_metrics.population_density.name, max_population_density)
        config.max_population_density = max_population_density

    # Computing scaling factors for age_range and depth_range metrics
    # Scaling factor for dist_range = threshold_distance/max_dist_range
    if config.max_age_range is None and eff_weights.age_range > 0:
        max_age_range = max(
            max(inner_dict.values())
            for inner_dict in opt_model_inputs.pairwise_metrics.age.values()
        )
        log_message(eff_metrics.age_range.name, max_age_range)
        config.max_age_range = max_age_range

    if config.max_depth_range is None and eff_weights.depth_range > 0:
        max_depth_range = max(
            max(inner_dict.values())
            for inner_dict in opt_model_inputs.pairwise_metrics.depth.values()
        )
        log_message(eff_metrics.depth_range.name, max_depth_range)
        config.max_depth_range = max_depth_range

    # Compute the aggregated scores for a few metrics
    opt_model_inputs.aggregated_eff_scores = get_aggregated_data(config.well_data)

    LOGGER.info("Completed calculating the scaling factors for efficiency metrics")


def get_aggregated_data(wd: WellData):
    """
    Returns the aggregated data required for efficiency metrics
    record_completeness, population_density, elevation_delta, and
    dist_to_road

    Parameters
    ----------
    wd : WellData
        Object containing the well data

    Returns
    -------
    aggregated_data : pd.DataFrame
        Weighted average of efficiency scores associated with
        population density, distance to road, record completeness,
        and elevation delta.
    """
    col_names = wd.column_names
    eff_weights = wd.config.efficiency_metrics.get_weights

    # Compute the score for record completeness
    data = wd.data[wd.get_flag_columns].sum(axis=1)
    num_columns = len(wd.get_flag_columns)
    if num_columns > 0:
        # Input data has missing cells
        aggregated_data = (eff_weights.record_completeness / num_columns) * data

    else:
        # No data is missing, so elements of data are zeros
        aggregated_data = data

    # Compute the score for population density
    # NOTE: Assuming that the denominator is not going to be zero.
    # Need to update the code to handle that scenario.
    if eff_weights.population_density > 0:
        # Population density data is available
        data = wd[col_names.population_density]
        aggregated_data += (eff_weights.population_density / data.max()) * data

    # Compute the score for distance to road
    if eff_weights.dist_to_road > 0:
        # Distance to road data is available
        data = wd[col_names.dist_to_road]
        aggregated_data += (eff_weights.dist_to_road / data.max()) * data

    # Compute the score for elevation delta
    if eff_weights.elevation_data > 0:
        # Elevation delta is available
        data = abs(wd[col_names.elevation_delta])
        aggregated_data += (eff_weights.elevation_delta / data.max()) * data

    return aggregated_data


def build_efficiency_model(model_block, cluster):
    """
    Declares essential variables and constraints needed for the
    computation of efficiency scores.

    Parameters
    ----------
    model_block : Pyomo Block
        Model for each cluster

    cluster : int
        Cluster number
    """
    params = model_block.parent_block().model_inputs
    sf = params.config  # Scaling factors are located in this object
    weights = sf.well_data.config.efficiency_metrics.get_weights

    # Define all efficiency variables
    model_block.cluster_efficiency = Var(
        within=NonNegativeReals,
        doc="Efficiency of the entire project",
    )
    model_block.eff_num_wells = Var(
        within=NonNegativeReals,
        doc="Score associated with the size of the project",
    )
    model_block.eff_num_unique_owners = Var(
        within=NonNegativeReals,
        doc="Score associated with number of unique owners",
    )
    model_block.eff_age_depth_dist = Var(
        within=NonNegativeReals,
        doc="Score associated with age, depth, distance ranges",
    )
    model_block.eff_pop_road_elevation_rec = Var(
        within=NonNegativeReals,
        doc=(
            "Score associated with population density, "
            "distance to road, elevation delta, record completeness"
        ),
    )
    # Create a copy of efficiency variable for each well
    model_block.well_efficiency = Var(
        model_block.set_wells,
        within=NonNegativeReals,
        doc="Efficiency score associated with each well",
    )

    # Add essential constraints
    model_block.calculate_cluster_efficiency = Constraint(
        expr=(
            model_block.cluster_efficiency
            == model_block.eff_num_wells
            + model_block.eff_age_depth_dist
            + model_block.eff_pop_road_elevation_rec
            + model_block.eff_num_unique_owners
        ),
        doc="Calculates the total efficiency of the project",
    )

    model_block.calculate_eff_num_wells = Constraint(
        expr=(
            model_block.eff_num_wells
            == weights.num_wells * model_block.num_wells_chosen / sf.max_num_wells
        ),
        doc="Calculates efficiency of num_wells metric",
    )

    if weights.num_unique_owners > 0:
        LOGGER.warning(
            "num_unique_owners is not supported currently. Setting its score to zero"
        )
        model_block.eff_num_unique_owners.fix(0)

    # Age range, depth range, and distance metrics
    aggregated_weight = weights.age_range + weights.depth_range + weights.dist_range
    dist_range = params.pairwise_metrics.distance[cluster]
    age_range = params.pairwise_metrics.age[cluster]
    depth_range = params.pairwise_metrics.depth[cluster]

    if aggregated_weight > 0:
        # Calculate the efficiency variable
        @model_block.Constraint(
            model_block.set_well_pairs_keep,
            doc="Combined constraint for age, depth, distance",
        )
        def calculate_age_depth_dist(blk, w1, w2):
            coeff = (
                weights.age_range * age_range[w1, w2] / sf.max_age_range
                + weights.depth_range * depth_range[w1, w2] / sf.max_depth_range
                + weights.dist_range * dist_range[w1, w2] / sf.max_dist_range
            )
            return (
                aggregated_weight * blk.select_cluster - blk.eff_age_depth_dist
                >= coeff
                * (blk.select_well[w1] + blk.select_well[w2] - blk.select_cluster)
            )

    else:
        # Metrics are not chosen, so fix the score to zero
        model_block.eff_age_depth_dist.fix(0)

    # Population density, distance to road, elevation delta, record completeness
    aggregated_weight = (
        weights.population_density
        + weights.dist_to_road
        + weights.elevation_delta
        + weights.record_completeness
    )
    # Aggregated coefficient is computed using the get_aggregated_data function, and
    # set as an attribute in the compute_efficiency_scaling_factors function
    aggregated_coeff = params.aggregated_eff_scores
    if aggregated_weight > 0:

        @model_block.Constraint(
            model_block.set_wells,
            doc=(
                "Score associated with popoulation density, "
                "distance to road, elevation, and record completeness"
            ),
        )
        def calculate_eff_pop_elevation_record(model_block, w):
            return (
                aggregated_weight * model_block.select_cluster
                - model_block.eff_pop_road_elevation_record
                >= aggregated_coeff[w] * model_block.select_well[w]
            )

    else:
        # Metrics are not chosen, so set the efficiency to zero
        model_block.eff_pop_road_elevation_record.fix(0)

    # Compute well efficiency score
    @model_block.Constraint(
        model_block.set_wells,
        doc="Set efficiency to zero is the well is not selected",
    )
    def calculate_well_efficiency_1(blk, w):
        return blk.well_efficiency[w] <= 100 * blk.select_well[w]

    @model_block.Constraint(
        model_block.set_wells,
        doc="Bound well efficiency if the well is selected",
    )
    def calculate_well_efficiency_2(blk, w):
        return blk.well_efficiency[w] <= blk.cluster_efficiency

    # Add the efficiency score to the well priority score expression
    # to obtain the combined/weighted priority score for each well
    wt_impact = sf.objective_weight_impact / 100
    wt_efficiency = 1 - wt_impact

    for w in model_block.set_wells:
        model_block.well_priority_score[w] += (
            wt_efficiency * model_block.well_efficiency[w]
        )