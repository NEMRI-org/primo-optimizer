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
import pandas as pd
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.environ import NonNegativeReals, Set, Var

# User-defined libs
from primo.data_parser.default_data import WELL_BASED_METRICS, WELL_PAIR_METRICS
from primo.utils.clustering_utils import get_pairwise_metrics

LOGGER = logging.getLogger(__name__)


def compute_efficieny_scaling_factors(opt_model):
    """
    Checks whether scaling factors for efficiency metrics are provided by
    the user or not. If not, computes the scaling factors using the entire
    dataset.

    Parameters
    ----------
    opt_model : PluggingCampaignModel
        Object containing the optimization model
    """
    LOGGER.info("Computing scaling factors for efficiency metrics")
    config = opt_model.model_inputs.config
    wd = config.well_data
    eff_metrics = wd.config.efficiency_metrics
    eff_weights = eff_metrics.get_weights

    def set_scaling_factor(metric_name, scale_value):
        """Function for logging warning message"""
        LOGGER.warning(
            f"Scaling factor for the efficiency metric {metric_name} is not "
            f"provided, so it is set to {scale_value}. To modify the "
            f"scaling factor, pass argument max_{metric_name} while instantiating "
            f"the OptModelInputs object."
        )
        setattr(config, "max_" + metric_name, scale_value)

    # Setting a scaling factor for num_wells metric
    if config.max_num_wells is None and eff_weights.num_wells > 0:
        set_scaling_factor("num_wells", 25)

    # Setting a scaling factor for num_unique_owners metric
    if config.max_num_unique_owners is None and eff_weights.num_unique_owners > 0:
        set_scaling_factor("num_unique_owners", 5)

    for metric in WELL_BASED_METRICS:
        if (
            getattr(eff_weights, metric, 0) > 0
            and getattr(config, "max_" + metric) is None
        ):
            # Metric is chosen, but the scaling factor is not specified
            scale_value = wd[getattr(eff_metrics, metric).data_col_name].max()
            set_scaling_factor(metric, scale_value)

    if sum(getattr(eff_weights, metric, 0) for metric in WELL_PAIR_METRICS) == 0:
        # None of the pairwise metrics are selected, so return
        return

    # Append the pairwise metrics to the model
    for c in opt_model.set_clusters:
        cm = opt_model.cluster[c]
        cm.pairwise_metrics = get_pairwise_metrics(wd, cm.set_wells)
        cm.set_well_pairs = Set(initialize=cm.pairwise_metrics.index.to_list())

    for metric in WELL_PAIR_METRICS:
        if (
            getattr(eff_weights, metric, 0) > 0
            and getattr(config, "max_" + metric) is None
        ):
            # Metric is chosen, but the scaling factor is not specified
            scale_value = max(
                opt_model.cluster[c].pairwise_metrics[metric].max()
                for c in opt_model.set_clusters
            )
            set_scaling_factor(metric, scale_value)


@declare_custom_block("MaxFormulationBlock")
class MaxFormulationBlockData(BlockData):
    """
    Block for building max-scaling efficiency model
    """

    @property
    def cluster_model(self):
        """
        Returns a pointer to the cluster model
        """
        return self.parent_block().parent_block()

    def compute_metric_score(
        self,
        weight: int,
        metric_data: pd.Series,
        scaling_factor: float,
        metric_type: str,
    ):
        """
        Builds the efficiency expressions for well-based metrics
        """
        # pylint: disable = attribute-defined-outside-init
        self.score = Var(
            domain=NonNegativeReals,
            bounds=(0, weight),
            doc="Score variable for this efficiency metric",
        )
        well_vars = self.cluster_model.select_well
        select_cluster = self.cluster_model.select_cluster

        norm_metric_data = metric_data / scaling_factor
        norm_metric_data[norm_metric_data >= 1] = 1

        if metric_type == "well_based":

            @self.Constraint(self.cluster_model.set_wells)
            def calculate_score(blk, w):
                return blk.score <= weight * (
                    select_cluster - norm_metric_data[w] * well_vars[w]
                )

        elif metric_type == "well_pair":

            @self.Constraint(self.cluster_model.set_well_pairs)
            def calculate_score(blk, w1, w2):
                return select_cluster - blk.score / weight <= (
                    norm_metric_data[w1, w2]
                    * (well_vars[w1] + well_vars[w2] - select_cluster)
                )

        elif metric_type == "num_wells":

            @self.Constraint()
            def calculate_score(blk):
                return (
                    blk.score
                    <= weight * sum(well_vars[w] for w in well_vars) / scaling_factor
                )

        elif metric_type == "num_unique_owners":
            LOGGER.warning(
                "Efficiency metric num_unique_owners is not supported currently"
            )


def build_cluster_efficiency_model(eff_blk):
    """
    Builds efficiency model for each cluster

    Parameters
    ----------
    cm : ClusterBlock
        Cluster model object
    """
    # # For reference, this is the model Hierarchy
    # PluggingCampaignModel/Pyomo ConcreteModel
    #     |__ClusterBlock
    #         |__EfficiencyBlock
    #             |__MaxFormulationBlock

    # OptModelInputs's config object that contains zone information
    cm = eff_blk.parent_block()  # cluster model block
    pm = cm.parent_block()  # Plugging campaign model/ConcreteModel
    sf = pm.model_inputs.config  # Block containing scaling factors
    wd = sf.well_data  # WellData object
    eff_metrics = wd.config.efficiency_metrics
    weights = eff_metrics.get_weights
    list_wells = list(cm.set_wells)  # List of wells in this cluster

    # Assess well-based metrics
    for metric in WELL_BASED_METRICS:
        if getattr(weights, metric, 0) == 0:
            # Metric is not selected. So, Skip
            continue

        # Construct Efficiency model for the metric
        # pylint: disable = undefined-variable
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            weight=getattr(weights, metric),
            metric_data=wd.data.loc[
                list_wells, getattr(eff_metrics, metric).data_col_name
            ],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type="well_based",
        )

    for metric in WELL_PAIR_METRICS:
        if getattr(weights, metric, 0) == 0:
            # Metric is not selected, so skip
            continue

        # pylint: disable = undefined-variable
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            metric_data=cm.pairwise_metrics[metric],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type="well_pair",
        )

    if weights.num_wells > 0:
        # pylint: disable = undefined-variable
        metric = "num_wells"
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            metric_data=cm.pairwise_metrics[metric],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type=metric,
        )

    if weights.num_wells > 0:
        # pylint: disable = undefined-variable
        metric = "num_unique_owners"
        setattr(eff_blk, metric, MaxFormulationBlock())
        getattr(eff_blk, metric).compute_metric_score(
            metric_data=cm.pairwise_metrics[metric],
            scaling_factor=getattr(sf, "max_" + metric),
            metric_type=metric,
        )
