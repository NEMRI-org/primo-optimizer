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

"""
This module defines a new class called EfficiencyBlock. All the
efficiency related calculations will be contained in this block.

The hierarchy of the entire optimization model is as follows:
PluggingCampaignModel/Pyomo ConcreteModel
    |__ClusterBlock
        |__EfficiencyBlock
            |__EffMetricBlock
"""

# Standard libs
import logging

# Installed libs
from pyomo.core.base.block import Block, BlockData, declare_custom_block
from pyomo.environ import NonNegativeReals, Var

WELL_BASED_METRICS = [
    "elevation_delta",
    "dist_to_road",
    "population_density",
    "record_completeness",
]
WELL_PAIR_METRICS = ["age_range", "depth_range", "dist_range"]

LOGGER = logging.getLogger(__name__)


@declare_custom_block("EfficiencyBlock")
class EfficiencyBlockData(BlockData):
    """Container for storing the efficiency calculations"""

    def __init__(self, component):
        super().__init__(component)

        # The following line is needed for some old legacy reason.
        # So, do not remove it.
        self._suppress_ctypes = {}
        self._has_fixing_var_constraints = False
        self._eff_vars = None  # Efficiency variables (for max_formulation)

    # pylint: disable = attribute-defined-outside-init
    def append_cluster_eff_vars(self, eff_vars: list = None):
        """
        Adds the common constraints

        Parameters
        ----------
        eff_vars : list, optional, default = None
            List of efficiency variables. This is needed for the
            max formulation, and this is not needed for the zone
            formulation
        """

        cm = self.parent_block()  # ClusterBlock Model
        # Declare variables
        self.cluster_efficiency = Var(within=NonNegativeReals)
        self.aggregated_efficiency = Var(
            cm.num_wells_var.index_set(),
            within=NonNegativeReals,
            doc="num_wells_var * cluster_efficiency",
        )

        @self.Constraint(cm.num_wells_var.index_set())
        def calculate_aggregated_efficiency_1(blk, n):
            return blk.aggregated_efficiency[n] <= 100 * cm.num_wells_var[n]

        @self.Constraint(cm.num_wells_var.index_set())
        def calculate_aggregated_efficiency_2(blk, n):
            return blk.aggregated_efficiency[n] <= blk.cluster_efficiency

        @self.Expression(doc="Total Efficiency Score")
        def cluster_efficiency_score(blk):
            return sum(
                n * blk.aggregated_efficiency[n] for n in cm.num_wells_var.index_set()
            )

        if eff_vars is None:
            # Score variables will be picked up automatically
            # for the zone formulation
            eff_vars = []

        self._eff_vars = eff_vars  # Store pointers to efficiency variables

        @self.Constraint(doc="Computes the efficiency of the project")
        def calculate_cluster_efficiency(blk):
            return blk.cluster_efficiency == sum(eff_vars) + sum(
                eff_metric_blk.score
                for eff_metric_blk in blk.component_data_objects(Block)
            )

    def get_efficiency_scores(self):
        """Returns a dictionary containing efficiency scores"""
        scores = {}
        for blk in self.component_data_objects(Block):
            scores[blk.name.split(".")[-1]] = blk.score.value

        for var in self._eff_vars:
            scores[var.name.split(".")[-1]] = var.value

        return scores
