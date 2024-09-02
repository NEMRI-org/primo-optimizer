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
from typing import Dict, List, Union

# Installed libs
import ipywidgets as widgets
import pandas as pd
from fast_autocomplete import AutoComplete
from haversine import Unit, haversine_vector
from IPython.display import display

# User-defined libs
from primo.data_parser.data_model import OptInputs
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable = attribute-defined-outside-init
class OverrideWidget:
    """
    Class for displaying an autofill widget in Jupyter Notebook that allows the user select
    wells that they would like to add to or remove from the optimal solution

    Parameters
    ----------

    well_df : pd.DataFrame
        Data frame of all wells

    Attributes
    ----------

    widget : widgets.Combobox

        A text widget with autofill feature for selecting wells for `well_list`

    button : widgets.Button
        A button to confirm and add the selected well to `well_list`

    well_list : list
        A list containing the API well number of wells that the user would like to add to
        or remove from the result of the optimization problem

    """

    def __init__(self, well_df: pd.DataFrame, button_description: str):
        well_dict = {str(well_id): {} for well_id in well_df["API Well Number"]}

        words = well_dict
        self.autocomplete = AutoComplete(words=words)
        self.widget = widgets.Combobox(
            value="",
            placeholder="Select well",
            description="Well",
            disabled=False,
        )
        self.widget.observe(self._on_change, names="value")

        layout = widgets.Layout(width="auto", height="auto")

        self.button_add = widgets.Button(description=button_description, layout=layout)
        self.button_add.on_click(self._on_button_clicked_add)
        self.button_remove = widgets.Button(description="Undo", layout=layout)
        self.button_remove.on_click(self._on_button_clicked_remove)
        self.well_list = []

    def _on_change(self, data) -> None:
        """
        Dynamically update the well option in the drop down list of the widget
        based on the information user has already typed in
        """

        self.text = data["new"]

        values = self.autocomplete.search(self.text, max_cost=3, size=3)

        # convert nested list to flat list
        values = list(sorted(set(str(item) for sublist in values for item in sublist)))

        self.widget.options = values

    def _on_button_clicked_add(self, _) -> None:
        """
        Add the selected well to the well_list and print the corresponding confirmation
        message in the Jupyter Notebook
        """

        if self.text in self.well_list:
            raise_exception(
                f"Well {self.text} has already been added to the override list",
                ValueError,
            )
        else:
            self.well_list.append(self.text)
            LOGGER.info(f"Well {self.text} has been added to the override list.")

    def _on_button_clicked_remove(self, _) -> None:
        """
        Remove a selected well from the well_list and display the corresponding
        confirmation message in the Jupyter Notebook
        """

        if self.text not in self.well_list:
            raise_exception(
                f"Well {self.text} is not in the list",
                ValueError,
            )
        else:
            self.well_list.remove(self.text)
            LOGGER.info(f"Well {self.text} has been removed from the list.")

    def display(self) -> widgets.HBox:
        """
        Display the widget and button in the Jupyter Notebook
        """
        buttons = widgets.HBox([self.button_add, self.button_remove])
        display(self.widget, buttons)

    def return_value(self) -> List:
        """
        Return the list of selected wells
        """
        return self.well_list


def user_input(well_df_add: pd.DataFrame, well_df_remove: pd.DataFrame):
    """
    A wrapper for generating a widget that facilitates the addition and removal of wells

    """
    widget_add = OverrideWidget(well_df_add, "Add the well to the suggested projects")
    widget_remove = OverrideWidget(
        well_df_remove, "Remove the well from the suggested projects"
    )

    return widget_add, widget_remove


class Recalculate:
    """
    Class for assess whether whether the overridden P&A projects adhere to the constraints
    defined in the optimization problem

    Parameters
    ----------

    original_well_list : pd.DataFrame
        A DataFrame containing wells selected based on solving the optimization problem
        without considering the override wells.

    well_add_list : list
        A list of wells that the user wishes to add to the P&A projects

    well_remove_list : list
        A list of wells that the user wishes to remove from the P&A projects

    well_df : pd.DataFrame
        A DataFrame that includes all candidate wells

    mobilization_costs : Dict[int, float]
        A dictionary mapping well IDs to mobilization costs. These costs should match
        those used in the optimization problem.

    budget : int
        An integer for the available budget for plugging wells

    max_wells_per_owner : int
        An integer for the maximum number of wells an owner can have in the overall
        P&A projects. This value should align with the one used in the optimization problem.

    max_distance : int
        An integer for the maximum allowable distance between two wells within the
        same project. This value should match the one used in the optimization problem.

    dac_weight : int
        An integer for the weight assigned to the DAC priority factor.

    dac_budget_fraction : int
        An integer for the minimum percentage of wells to be plugged that should
        be located in DAC areas. This value should align with the one used in the
        optimization problem.

    Attributes
    ----------

    violate_cost : Union[int, bool]
        The actual budget if the budget constraint is violated; otherwise False

    violate_dac : Union[int, bool]
        The actual DAC percentage if the DAC constraint is violated otherwise False

    violate_operator : dict
        A dictionary providing information on owners who exceed the well count constraint.
        Includes owner name, the number of wells belonging to the owner in the project,
        and the specific well IDs.

    violate_distance : dict
        A dictionary detailing wells that breach the maximum distance constraint.
        Includes the project, well pairs, and the actual distances between these well pairs.

    """

    def __init__(
        self,
        original_well_list: pd.DataFrame,
        well_add_list: list,
        well_remove_list: list,
        well_df: pd.DataFrame,
        opt_inputs: OptInputs,
        dac_weight: float = None,
    ):
        self.original_well_list = original_well_list
        self.well_add_list = well_add_list
        self.well_remove_list = well_remove_list
        self.well_df = well_df
        self.mobilization_costs = opt_inputs.mobilization_cost
        self.budget = opt_inputs.budget
        self.dac_budget_fraction = opt_inputs.dac_budget_fraction
        self.max_wells_per_owner = opt_inputs.max_wells_per_owner
        self.max_distance = opt_inputs.distance_threshold
        self.dac_weight = dac_weight

        for well_id in self.well_add_list:
            well_id = int(well_id)
            self.well_add = self.well_df[self.well_df["API Well Number"] == well_id]
            self.original_well_list = pd.concat(
                [self.original_well_list, self.well_add]
            )

        well_remove_list = [int(well_id) for well_id in self.well_remove_list]
        self.original_well_list["drop"] = self.original_well_list.apply(
            lambda row: 0 if row["API Well Number"] in well_remove_list else 1,
            axis=1,
        )
        self.well_return_df = self.original_well_list[
            self.original_well_list["drop"] == 1
        ]

    def budget_assess(self) -> Union[int, bool]:
        """
        Assess whether the budget constraint is violated or not
        """
        total_cost = 0
        for _, groups in self.well_return_df.groupby("Project"):
            n_wells = len(groups)
            campaign_cost = self.mobilization_costs[n_wells]
            total_cost += campaign_cost
        if total_cost > self.budget:
            self.violate_cost = total_cost
        else:
            self.violate_cost = False
        return self.violate_cost

    def dac_assess(self) -> Union[int, bool]:
        """
        Assess whether the DAC constraint is violated or not
        """
        for _, row in self.well_return_df.iterrows():
            well_id = row["API Well Number"]
            if self.dac_weight is not None:
                threshold = self.dac_weight / 100 * self.dac_budget_fraction
                disadvantaged_community_score = row[
                    f"DAC Score [0-{int(self.dac_weight)}]"
                ]
                is_disadvantaged = float(disadvantaged_community_score > threshold)
            else:
                # When the user does not select DAC as a priority factor,
                # all wells are assumed to not be located in a disadvantaged community.
                is_disadvantaged = float(False)
            self.well_return_df.loc[
                self.well_return_df["API Well Number"] == well_id, ["In DAC"]
            ] = is_disadvantaged
        dac_percent = (
            self.well_return_df["In DAC"].sum() / len(self.well_return_df) * 100
        )
        if dac_percent < self.dac_budget_fraction:
            self.violate_dac = dac_percent
        else:
            self.violate_dac = False
        return self.violate_dac

    def operator_assess(self) -> Dict:
        """
        Assess whether the owner well count constraint is violated or not
        """
        self.violate_operator = {}
        for operator, groups in self.well_return_df.groupby("Operator Name"):
            n_wells = len(groups)
            if n_wells > self.max_wells_per_owner:
                self.violate_operator[operator] = [
                    n_wells,
                    groups["API Well Number"].to_list(),
                ]

        return self.violate_operator

    def distance_asses(self) -> Dict:
        """
        Assess whether the maximum distance between two wells constraint is violated or not
        """
        self.violate_distance = {}
        for cluster_id, groups in self.well_return_df.groupby("Project"):
            well_id = list(groups["API Well Number"])
            num_well = len(well_id)
            groups["coor"] = list(zip(groups.Latitude, groups.Longitude))
            distance_metric_distance = haversine_vector(
                groups["coor"].to_list(),
                groups["coor"].to_list(),
                unit=Unit.MILES,
                comb=True,
            )
            well_distance = {}
            for well_1 in range(num_well - 1):
                for well_2 in range(well_1 + 1, num_well):
                    well_distance = distance_metric_distance[well_1][well_2]
                    if well_distance > self.max_distance:
                        self.violate_distance[
                            (cluster_id, well_id[well_1], well_id[well_2])
                        ] = [
                            cluster_id,
                            well_id[well_1],
                            well_id[well_2],
                            well_distance,
                        ]

        return self.violate_distance


def back_fill(
    selected_well,
    well_add_list_well,
    well_remove_list_well,
    well_gdf,
    opt_inputs_well,
    well,
):
    violation = Recalculate(
        selected_well,
        well_add_list_well,
        well_remove_list_well,
        well_gdf,
        opt_inputs_well,
    )
    violate_cost = violation.budget_assess()
    violate_operator = violation.operator_assess()
    violate_distance = violation.distance_asses()
    if (
        violate_cost is False
        and bool(violate_operator) is False
        and bool(violate_distance) is False
    ):
        return [well]


def well_candidates_list(
    well_violate,
    well_gdf,
    selected_well,
    well_add_list_well,
    well_remove_list_well,
    opt_inputs_well,
):
    well_candidates_dict = {}
    for well_id in well_violate:
        well_backfill = []
        cluster = int(well_gdf[well_gdf["API Well Number"] == well_id]["Project"])
        well_candidates = well_gdf[
            (well_gdf["Project"] == str(cluster))
            & (well_gdf["API Well Number"] != well_id)
        ]["API Well Number"]
        for well in well_candidates:
            well_add_list_well_drop = [
                int(id) for id in well_add_list_well if int(id) not in well_violate
            ]
            well_add_list_well_drop += [well]
            well_candidate = back_fill(
                selected_well,
                well_add_list_well_drop,
                well_remove_list_well,
                well_gdf,
                opt_inputs_well,
                well,
            )
            if well_candidate is not None:
                well_backfill += well_candidate
        well_candidates_dict[well_id] = well_backfill
    return well_candidates_dict
