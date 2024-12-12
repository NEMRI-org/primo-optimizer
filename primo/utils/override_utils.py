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
import copy
import logging
from itertools import combinations
from typing import Dict

# Installed libs
import pandas as pd

# User-defined libs
from primo.opt_model.result_parser import Campaign
from primo.utils.clustering_utils import distance_matrix

LOGGER = logging.getLogger(__name__)


class AssessFeasibility:
    """
    Class for assessing whether the P&A projects adhere to the constraints
    defined in the optimization problem.

    Parameters
    ----------
    opt_inputs : OptModelInputs
        The optimization model inputs.

    opt_campaign : Dict
        A dictionary where keys are cluster numbers and values
        are list of wells for each cluster in the P&A projects.

    wd : WellData
        The WellData object for wells being plugged in the P&A projects.

    plug_list : List
        A list of wells to be plugged in the P&A projects.

    Attributes
    ----------
    campaign_cost_dict : Dict
        A dictionary that will hold the cost calculations for the projects within the campaign,
        mapped to their respective project identifiers.
    """

    def __init__(self, opt_inputs, opt_campaign: Dict, wd, plug_list):

        self.opt_inputs = opt_inputs
        self.new_campaign = opt_campaign
        self.wd = wd
        self.plug_list = plug_list
        self.campaign_cost_dict = {}

        for cluster, groups in self.new_campaign.items():
            n_wells = len(groups)
            campaign_cost = self.opt_inputs.get_mobilization_cost[n_wells]
            self.campaign_cost_dict[cluster] = campaign_cost

    def assess_budget(self) -> float:
        """
        Assesses whether the budget constraint is violated and returns the
        amount by which the budget is violated. A 0 or negative value indicates
        that we are still under budget
        """
        total_cost = 0

        total_cost = sum(
            project_cost for _, project_cost in self.campaign_cost_dict.items()
        )

        return round((total_cost - self.opt_inputs.get_total_budget) * 1e6)

    def assess_dac(self) -> float:
        """
        Assess whether the DAC constraint is violated or not and returns
        the amount by which the DAC constraint is violated. A 0 or negative value
        indicates that constraint is satisfied
        """
        opt_inputs = self.opt_inputs.config
        dac_weight = opt_inputs.perc_wells_in_dac
        if dac_weight is None:
            # When the user does not select DAC as a priority factor,
            # this constraint becomes meaningless
            return 0

        disadvantaged_wells = 0

        # count number of wells that exceed threshold based on disadvantaged
        # community score
        disadvantaged_wells = sum(
            self.wd.data.loc[well, "is_disadvantaged"] for well in self.plug_list
        )
        dac_percent = disadvantaged_wells / len(self.plug_list) * 100

        return opt_inputs.perc_wells_in_dac - dac_percent

    def assess_owner_well_count(self) -> Dict:
        # pylint: disable=protected-access
        """
        Assess whether the owner well count constraint is violated or not.
        Returns list of owners and wells selected for each for whom the owner
        well count constraint is violated
        """
        violated_operators = {}
        for operator, groups in self.wd.data.groupby(self.wd._col_names.operator_name):
            n_wells = len(groups)
            if n_wells > self.opt_inputs.config.max_wells_per_owner:
                violated_operators.setdefault("Owner", []).append(operator)
                violated_operators.setdefault("Number of wells", []).append(n_wells)
                violated_operators.setdefault("Wells", []).append(
                    groups[self.wd._col_names.well_id].to_list()
                )

        return violated_operators

    def assess_distances(self) -> Dict:
        # pylint: disable=protected-access
        """
        Assess whether the maximum distance between two wells constraint is violated or not
        """
        distance_threshold = self.opt_inputs.config.threshold_distance
        distance_violation = {}
        # Assign weight for distance as 1 to ensure the distance matrix returns physical
        # distance between two well pairs
        metric_array = distance_matrix(self.wd, {"distance": 1})
        df_to_array = {
            df_index: array_index
            for array_index, df_index in enumerate(self.wd.data.index)
        }

        for cluster, well_list in self.new_campaign.items():
            for w1, w2 in combinations(well_list, 2):
                well_distance = metric_array[df_to_array[w1], df_to_array[w2]]
                if well_distance > distance_threshold:
                    distance_violation.setdefault("Project", []).append(cluster)
                    distance_violation.setdefault("Well 1", []).append(
                        self.wd.data.loc[w1][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault("Well 2", []).append(
                        self.wd.data.loc[w2][self.wd._col_names.well_id]
                    )
                    distance_violation.setdefault(
                        "Distance between Well 1 and 2 [Miles]", []
                    ).append(well_distance)

        return distance_violation

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


# pylint: disable=too-many-instance-attributes
class OverrideCampaign:
    """
    Class for constructing new campaigns based on the override results
    and returning infeasibility information.

    Parameters
    ----------
    override_selections : OverrideSelections
        Object containing the override selections

    opt_inputs : OptModelInputs
        Object containing the necessary inputs for the optimization model

    opt_campaign : dict
        A dictionary for the original suggested P&A project
        where keys are cluster numbers and values
        are list of wells for each cluster.

    eff_metrics : EfficiencyMetrics
        The efficiency metrics
    """

    def __init__(
        self,
        override_selections,
        opt_inputs,
        opt_campaign: Dict,
        eff_metrics,
    ):
        opt_campaign_copy = copy.deepcopy(opt_campaign)
        self.new_campaign = opt_campaign_copy
        self.remove = override_selections.remove_widget_return
        self.add = override_selections.add_widget_return
        self.lock = override_selections.lock_widget_return
        self.opt_inputs = opt_inputs
        self.eff_metrics = eff_metrics

        # change well cluster
        self._modify_campaign()
        self.plug_list = []
        for _, well_list in self.new_campaign.items():
            self.plug_list += well_list
        # prevent duplication in plug_list
        self.plug_list = list(set(self.plug_list))
        self.well_data = self.opt_inputs.config.well_data._construct_sub_data(
            self.plug_list
        )

        self.feasibility = AssessFeasibility(
            self.opt_inputs, self.new_campaign, self.well_data, self.plug_list
        )

    def _modify_campaign(self):
        """
        Modify the original suggested P&A project
        """
        # remove clusters
        for cluster in self.remove.cluster:
            del self.new_campaign[cluster]

        # remove wells
        for cluster, well_list in self.remove.well.items():
            if cluster not in self.remove.cluster:
                for well in well_list:
                    self.new_campaign[cluster].remove(well)

        # add well with new cluster
        for cluster, well_list in self.add.new_clusters.items():
            self.new_campaign.setdefault(cluster, []).extend(well_list)

    def violation_info(self):
        """
        Return information on constraints that the new campaign
        have violated.
        """
        violation_info_dict = {}
        if self.feasibility.assess_feasibility() is False:
            violation_info_dict = {"Project Status:": "INFEASIBLE"}
            violate_cost = self.feasibility.assess_budget()
            violate_operator = self.feasibility.assess_owner_well_count()
            violate_distance = self.feasibility.assess_distances()
            violate_dac = self.feasibility.assess_dac()

            if violate_cost > 0:
                msg = (
                    "After the modification, the total budget is over "
                    f"the limit by ${int(violate_cost)}. Please consider modifying "
                    "wells you have selected or "
                    "re-running the optimization problem."
                )

                violation_info_dict[msg] = """"""

            if violate_operator:
                msg = (
                    "After the modification, the following owners have "
                    f"more than {self.opt_inputs.config.max_wells_per_owner} well(s) "
                    "being selected. Please consider modifying wells you have "
                    "selected or re-running "
                    "the optimization problem."
                )

                violate_operator_df = pd.DataFrame.from_dict(violate_operator)
                violation_info_dict[msg] = violate_operator_df

            if violate_distance:
                msg = (
                    "After the modification, the following projects have "
                    "wells are far away from each others. Please consider modifying "
                    "wells you have selected or "
                    "re-running the optimization problem."
                )

                violate_distance_df = pd.DataFrame.from_dict(violate_distance)
                violation_info_dict[msg] = violate_distance_df

            if violate_dac > 0:
                dac_percent = self.opt_inputs.config.perc_wells_in_dac - violate_dac
                msg = (
                    f"After the modification, {int(dac_percent)}% of well "
                    "is in DAC. Please consider modifying wells you have selected."
                )
                violation_info_dict[msg] = """"""
        else:
            violation_info_dict = {"Project Status:": "FEASIBLE"}

        return violation_info_dict

    def override_campaign(self):
        """
        Construct the new Campaign object based on the override selection
        """
        plugging_cost = self.feasibility.campaign_cost_dict
        wd = self.opt_inputs.config.well_data
        return Campaign(wd, self.new_campaign, plugging_cost)

    def recalculate(self):
        """
        Recalculate the efficiency scores and impact scores of the new campaign
        based on the override selection
        """
        override_campaign = self.override_campaign()
        override_campaign.set_efficiency_weights(self.eff_metrics)
        override_campaign.compute_efficiency_scores()
        return override_campaign

    def recalculate_scores(self):
        """
        A function to return the impact score and efficiency score of
        the new campaign based on the override selection
        """
        override_campaign = self.recalculate()
        return {
            project_id: [project.impact_score, project.efficiency_score]
            for project_id, project in override_campaign.projects.items()
        }

    def re_optimize_dict(self):
        """
        Generate dictionaries for clusters and wells to be fixed based on
        the override selection
        """
        re_optimize_cluster_dict = {}
        re_optimize_well_dict = {}

        # Assign 0 to wells being removed
        for cluster, well_list in self.remove.well.items():
            re_optimize_well_dict[cluster] = {well: 0 for well in well_list}

        # Assign 1 to wells being added
        for cluster, well_list in self.add.new_clusters.items():
            if cluster not in re_optimize_well_dict:
                re_optimize_well_dict[cluster] = {}
            for well in well_list:
                re_optimize_well_dict[cluster][well] = 1

        # Assign 1 to clusters being locked
        for cluster in self.lock.cluster:
            re_optimize_cluster_dict[cluster] = 1

        # Assign 1 to wells being locked
        for cluster, well_list in self.lock.well.items():
            if cluster not in re_optimize_well_dict:
                re_optimize_well_dict[cluster] = {}
            for well in well_list:
                re_optimize_well_dict[cluster][well] = 1

        return re_optimize_cluster_dict, re_optimize_well_dict
