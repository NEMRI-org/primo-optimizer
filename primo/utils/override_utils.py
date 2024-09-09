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
from typing import Dict, List, Optional

# Installed libs
import pandas as pd
from haversine import Unit, haversine_vector

# User-defined libs
from primo.data_parser.data_model import OptInputs
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


class Recalculate:
    """
    Class for assessing whether the overridden P&A projects adhere to the constraints
    defined in the optimization problem

    Parameters
    ----------

    selected_wells : pd.DataFrame
        A DataFrame containing wells selected based on solving the optimization problem
        and/or by manual selections and overrides

    wells_added : List[str]
        A list of wells that the user wishes to add to the P&A projects

    wells_removed : List[str]
        A list of wells that the user wishes to remove from the P&A projects

    well_df : pd.DataFrame
        A DataFrame that includes all candidate wells

    opt_inputs : OptInputs
        Input object for the optimization problem

    dac_weight : int
        An integer for the weight assigned to the DAC priority factor.

    Attributes
    ----------

    original_wells : pd.DataFrame
        List of original wells selected for plugging

    wells_added : List[str]
        List of wells to be added for plugging

    wells_removed : List[str]
        List of wells to be removed from plugging projects

    well_df : pd.DataFrame
        Full data associated with all wells

    opt_inputs : OptInputs
        The inputs associated with the optimization problem

    dac_weight : Union[float, None]
        An integer for the weight assigned to the DAC priority factor.

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        original_wells: pd.DataFrame,
        wells_added: List[str],
        wells_removed: List[str],
        well_df: pd.DataFrame,
        opt_inputs: OptInputs,
        dac_weight: Optional[float] = None,
    ):
        self.original_wells = original_wells
        self.wells_added = [int(well_id) for well_id in wells_added]
        self.wells_removed = [int(well_id) for well_id in wells_removed]
        self.well_df = well_df
        self.opt_inputs = opt_inputs
        self.dac_weight = dac_weight

        # Get list of wells to be plugged (opt results + user overrides)
        added_wells = well_df[well_df["API Well Number"].isin(self.wells_added)]
        self._plugged_list = pd.concat([original_wells.copy(), added_wells])

        # Get list of wells to be removed from plugging
        self._plugged_list = self._plugged_list[
            ~self._plugged_list["API Well Number"].isin(self.wells_removed)
        ]

    def _add_well(self, well_id: int):
        """
        Adds a well in the considerations for plugging
        """
        if well_id in self.wells_added:
            msg = f"Well: {well_id} was already included in plugging list"
            LOGGER.warning(msg)
            print(msg)
            return

        if well_id in self.wells_removed:
            raise_exception(
                f"Well: {well_id} was already included in removal list", ValueError
            )

        self.wells_added.append(well_id)
        self._plugged_list = pd.concat(
            [
                self._plugged_list,
                self.well_df[self.well_df["API Well Number"] == well_id],
            ]
        )
        return

    def _remove_well(self, well_id: int):
        """
        Removes a well from the considerations for plugging
        """
        if well_id in self.wells_removed:
            raise_exception(
                f"Well: {well_id} was already included in removal list", ValueError
            )

        if well_id in self.wells_added:
            self.wells_added.remove(well_id)

        self._plugged_list = self._plugged_list[
            self._plugged_list["API Well Number"] != well_id
        ]

    # TODO: Move these assessments elsewhere as they can be reused
    def assess_budget(self) -> float:
        """
        Assesses the impact on budget after user overrides and returns the
        amount by which the budget is violated. A 0 or negative value indicates
        that we are still under budget
        """
        total_cost = 0
        for _, groups in self._plugged_list.groupby("Project"):
            n_wells = len(groups)
            campaign_cost = self.opt_inputs.mobilization_cost[n_wells]
            total_cost += campaign_cost

        return total_cost - self.opt_inputs.budget

    def assess_dac(self) -> float:
        """
        Assess whether the DAC constraint is violated or not and returns
        the amount by which the DAC constraint is violated. A 0 or negative value
        indicates that constraint is satisfied
        """
        opt_inputs = self.opt_inputs
        if self.dac_weight is None:
            # When the user does not select DAC as a priority factor,
            # this constraint becomes meaninggless
            return 0

        disadvantaged_wells = 0
        col_name = f"DAC Score [0-{int(self.dac_weight)}]"
        threshold = self.dac_weight / 100 * opt_inputs.dac_budget_fraction

        # count number of wells that exceed threshold based on disadvantaged
        # community score
        disadvantaged_wells = (self._plugged_list[col_name] > threshold).sum()
        dac_percent = disadvantaged_wells * 100 / self._plugged_list.shape[0]

        return opt_inputs.dac_budget_fraction - dac_percent

    def assess_owner_well_count(self) -> Dict:
        """
        Assess whether the owner well count constraint is violated or not.
        Returns list of owners and wells selected for each for whom the owner
        well count constraint is violated
        """
        violated_operators = {}
        for operator, groups in self._plugged_list.groupby("Operator Name"):
            n_wells = len(groups)
            if n_wells > self.opt_inputs.max_wells_per_owner:
                violated_operators[operator] = [
                    n_wells,
                    groups["API Well Number"].to_list(),
                ]

        return violated_operators

    def assess_distances(self) -> Dict:
        """
        Assess whether the maximum distance between two wells constraint is violated or not
        """
        distance_threshold = self.opt_inputs.distance_threshold
        distance_violations = {}
        # TODO: Consider calculating these once and for all since back filling
        # might end up causing multiple iterations here
        for cluster_id, groups in self._plugged_list.groupby("Project"):
            well_id = list(groups["API Well Number"])
            num_well = len(well_id)
            groups["coords"] = list(zip(groups.Latitude, groups.Longitude))
            distance_metric_distance = haversine_vector(
                groups["coords"].to_list(),
                groups["coords"].to_list(),
                unit=Unit.MILES,
                comb=True,
            )
            well_distance = {}
            for well_1 in range(num_well - 1):
                for well_2 in range(well_1 + 1, num_well):
                    well_distance = distance_metric_distance[well_1][well_2]
                    if well_distance > distance_threshold:
                        distance_violations[
                            (cluster_id, well_id[well_1], well_id[well_2])
                        ] = [
                            cluster_id,
                            well_id[well_1],
                            well_id[well_2],
                            well_distance,
                        ]

        return distance_violations

    def assess_feasibility(self) -> bool:
        """
        Assesses whether current set of selections is feasible
        """
        if self.assess_budget() > 0:
            return False

        if self.assess_dac() > 0:
            return False

        if self.assess_owner_well_count():
            return False

        if self.assess_distances():
            return False

        return True

    def backfill(self) -> List[str]:
        """
        If current selections (either due to manual overrides or termination of solver
        prior to optimality)
        leave room for budget, the method returns a list of candidates that can
        be added to the plugging projects without violating any constraints

        Parameters:
        ----------
        None

        Returns:
        --------
        List of well API numbers that can be added in the project
        """
        additions = []
        if self.assess_feasibility() is False:
            # The current well selections already lead to infeasibility (presumably
            # due to budget being exceeded)
            # There is no room to add more wells for backfilling
            return additions

        # Sort by descending order
        candidates = self.well_df.sort_values("Priority Score [0-100]", ascending=False)

        # Only include those candidates not considered for plugging
        candidates = candidates[
            ~candidates["API Well Number"].isin(self._plugged_list["API Well Number"])
        ]

        # Remove candidates in removal list
        candidates = candidates[~candidates["API Well Number"].isin(self.wells_removed)]
        # Stick to those projects that already exist
        existing_projects = set(self._plugged_list["Project"])
        candidates = candidates[candidates["Project"].isin(existing_projects)]

        for well_id in candidates["API Well Number"]:
            self._add_well(well_id)
            if self.assess_feasibility():
                additions.append(well_id)
            else:
                self._remove_well(well_id)

        return additions
