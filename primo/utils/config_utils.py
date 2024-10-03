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
import json
import logging
import os
import typing
from typing import Any, List, Tuple

# Installed libs
import ipywidgets as widgets
from fast_autocomplete import AutoComplete
from IPython.display import display

# User-defined libs
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def read_config(path: str) -> Tuple[bool, dict]:
    """
    Reads a config file, if provided.

    Parameters
    -----------
    path : str
        The path to the config file; may or may not exist

    Returns
    --------
    Tuple[bool, dict]
        Returns tuple if config file exists, with first element True
        and the second element being the input configuration as a dict;
        returns tuple with first element as False if config file does
        not exist
    """
    if not os.path.exists(path):
        return False, {}

    with open(path, "r") as read_file:
        config = json.load(read_file)

    return True, config


def update_defaults(config_dict: dict, input_dict: dict) -> dict:
    """
    Updates the default value in input_dict based on config provided.

    Parameters
    -----------
    config_dict : dict
        Configuration provided for this scenario run

    input_dict : dict
        User-required inputs

    Returns
    --------
    dict
        Updated input_dict with default values per config_dict
    """

    for key, value in input_dict.items():
        if key in config_dict:
            input_dict[key]["default"] = config_dict[key]["default"]

            sub_dict = config_dict[key].get("sub_weights", {})
            for sub_key in value.get("sub_weights", {}):
                default_value = sub_dict[sub_key].get("default", 0)
                if config_dict[key]["default"] == 0:
                    default_value = 0
                input_dict[key]["sub_weights"][sub_key]["default"] = default_value
        else:
            input_dict[key]["default"] = 0
            for sub_key in value.get("sub_weights", {}):
                input_dict[key]["sub_weights"][sub_key]["default"] = 0

    return input_dict


def read_defaults(input_dict: dict) -> Tuple[dict, dict]:
    """
    Create input dictionaries with default initial values for main and sub-priorities.

    Parameters
    -----------
    input_dict : dict
        Input dictionary of user-provided options

    Returns
    --------
    Tuple[dict, dict]
        A tuple containing dictionaries with main and sub-weights
    """
    priority_weight = {}
    sub_priority_weight = {}
    for key, value in input_dict.items():
        priority_weight[key] = value["default"]
        sub_dict = value.get("sub_weights", {})
        if sub_dict and value["default"] > 0:
            sub_priority_weight[key] = {}
            for sub_key, sub_value in sub_dict.items():
                sub_priority_weight[key][sub_key] = sub_value["default"]

    return priority_weight, sub_priority_weight


def is_valid(input_dict: dict, reference_dict: dict) -> bool:
    """
    Utility validates whether the input config provided by a user follows an expected structure.
    NOTE: It does not validate whether the values in the config are of the right type or have
    acceptable values.

    Parameters
    -----------
    input_dict : dict
        Input dictionary of user-provided options

    reference_dict : dict
        Reference dictionary of user-provided options

    Returns
    --------
    bool
        True if the dictionary provided is valid. False otherwise
    """

    try:
        copy_dict(input_dict, reference_dict)
        return True
    except ValueError:
        return False


def copy_dict(input_dict: dict, output_dict: dict) -> dict:
    """
    Utility accepts two dictionaries with an identical structure of keys and values.
    The "non-default" values provided in input_dict are copied over into the output_dict.
    This makes it easier to validate user-defined inputs since the structure of the input_dict
    is validated against an output_dict populated with default values.

    Parameters
    -----------
    input_dict : dict
        Input dictionary of non-default user-provided options; can be nested but must have
        same structure as output_dict

    output_dict : dict
        Output dictionary of default user-provided options

    Returns
    --------
    dict
        Uses the same structure as output_dict with non-default values provided in input_dict
        copied over
    """
    for key in input_dict.keys():
        if key not in output_dict:
            raise_exception("Found key not expected in input dict", ValueError)
        output_dict = copy_values(input_dict, output_dict, key)
    return output_dict


def copy_values(sub_input_dict: dict, sub_output_dict: dict, key: Any) -> dict:
    """
    Helper function to copy_dict that takes two dictionaries and copies the sub-structure
    associated with a "key" in sub_input_dict to sub_output_dict.

    Parameters
    -----------
    sub_input_dict : dict
        Sub-input dictionary of non-default user-provided options; sub_input_dict[key]
        can be nested but must have same structure as sub_output_dict[key]

    sub_output_dict : dict
        Output dictionary of default user-provided options

    key : Any
        A dictionary key that must be present in both input dictionaries

    Returns
    --------
    dict
        Uses the same structure as output_dict[key] with non-default values provided in
        input_dict[key] copied over
    """
    if key not in sub_input_dict:
        raise_exception(f"Unknown key: {key} not found in input dict", ValueError)

    if key not in sub_output_dict:
        raise_exception(f"Unknown key: {key} not found in output dict", ValueError)

    val = sub_input_dict[key]
    if not isinstance(val, dict):
        sub_output_dict[key] = val
        return sub_output_dict

    for sub_key in val.keys():
        sub_dict = copy_values(sub_input_dict[key], sub_output_dict[key], sub_key)
        sub_output_dict[key] = sub_dict
    return sub_output_dict


def _get_checkbox_params(param_dict: dict) -> Tuple[int, int, int, int]:
    """
    Returns the parameters required to initialize a CheckBoxWidget object
    """
    default = param_dict["default"]
    min_val = param_dict.get("min_val", 0)
    max_val = param_dict.get("max_val", 100)
    incr = param_dict.get("incr", 5)
    return (default, min_val, max_val, incr)


class CheckBoxWidget:
    """
    A simple wrapper for a combination of two widgets---Checkbox and
    IntSlider---that are to be used together.

    Parameters
    ----------

    description : str
        The name of the input parameter

    default : int
        The default value used for the CheckBox

    min_val : int, default = 0
        The minimum valid value for the input parameter

    max_val : int, default = 100
        The maximum valid value for the input parameter

    incr : int, default = 5
        The increment step value used in the IntSlider

    indent : bool, default = False
        Whether to indent display for this CheckBoxWidget

    Attributes
    ----------
    checkbox_ : widgets.Checkbox
        The checkbox associated with the object

    slider_ : widgets.IntSlider
        The slider associated with the object

    h_box_ : widgets.HBox
        The horizontal box that contains the checkbox and slider appended together
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        description: str,
        default: int,
        min_val: int = 0,
        max_val: int = 100,
        incr: int = 5,
        indent: bool = False,
    ):
        checkbox_value = default != 0

        if indent:
            self.checkbox_ = widgets.Checkbox(
                value=checkbox_value,
                description=description,
                layout=widgets.Layout(margin="0 0 0 8.5em"),
            )
        else:
            self.checkbox_ = widgets.Checkbox(
                value=checkbox_value, description=description
            )
        self.slider_ = widgets.IntSlider(
            value=default, min=min_val, max=max_val, step=incr, description="Weights"
        )
        self.h_box_ = widgets.HBox([self.checkbox_, self.slider_])
        self.checkbox_.observe(self._observe_change, "value")

    def _observe_change(self, change: dict):
        """
        Dynamically updates the status of the object. When unselected, this disables
        the associated IntSlider. Another toggle enables the IntSlider to be enabled.
        """
        if change["owner"] == self.checkbox_:
            self.slider_.disabled = not self.checkbox_.value

    def display(self) -> widgets.VBox:
        """
        Returns an object that can be displayed in the Jupyter Notebook
        """
        return widgets.VBox([self.h_box_])

    def is_active(self) -> bool:
        """
        Returns True if the checkbox is selected; False otherwise
        """
        return (not self.checkbox_.disabled) and self.checkbox_.value

    def return_value(self) -> Tuple[str, int]:
        """
        Returns the description and the value of the checkbox
        """
        return (self.checkbox_.description, self.slider_.value)


class SubCheckBoxWidget:
    """
    Parameters
    ----------
    weight_dict : dict
        A dictionary that determines the parameters for the widget to be displayed.
        An example structure includes:
        weight_dict = {"main": {"default": 30, "min_val": 0, "max_val": 100, "incr": 5,
        "sub_weights": {"Sub": {"default": 20}}}}
        Note that the keys "min_val," "max_val," "incr" default to 0, 100, and 5, respectively
        for both the main checkbox and its sub-boxes

    Attributes
    ----------
    checkbox_widget_ : CheckBoxWidget
        The main checkbox widget associated with this option

    sub_widgets_ : List[CheckBoxWidget]
        A list of sub-widgets associated with the main widget
    """

    def __init__(self, weight_dict):
        if len(weight_dict.keys()) != 1:
            raise_exception("Expect only one key in dictionary", ValueError)
        description = list(weight_dict.keys())[0]
        info_dict = weight_dict[description]
        default, min_val, max_val, incr = _get_checkbox_params(info_dict)

        self.checkbox_widget_ = CheckBoxWidget(
            description, default, min_val, max_val, incr
        )
        self.sub_widgets_ = []
        sub_weights_dict = info_dict.get("sub_weights", {})
        for description, sub_dict in sub_weights_dict.items():
            default, min_val, max_val, incr = _get_checkbox_params(sub_dict)

            sub_widget = CheckBoxWidget(
                description, default, min_val, max_val, incr, indent=True
            )
            self.sub_widgets_.append(sub_widget)

        self.checkbox_widget_.checkbox_.observe(self._observe_change, "value")

    def _observe_change(self, change: dict):
        """
        Dynamically updates the status of the object. When unselected, this disables
        the associated IntSlider and all sub-widgets. Another toggle enables the IntSlider
        to be enabled and all associated sub-widgets
        """
        main_widget = self.checkbox_widget_
        if change["owner"] == main_widget.checkbox_:
            for sub_widget in self.sub_widgets_:
                sub_widget.checkbox_.disabled = not main_widget.checkbox_.value
                sub_widget.slider_.disabled = not (
                    main_widget.checkbox_.value and sub_widget.checkbox_.value
                )
            main_widget.slider_.disabled = not main_widget.checkbox_.value

    def display(self) -> widgets.VBox:
        """
        Returns an object that can be displayed in the Jupyter Notebook
        """
        return widgets.VBox(
            [self.checkbox_widget_.h_box_]
            + [sub_widget.h_box_ for sub_widget in self.sub_widgets_]
        )

    def validate(self) -> bool:
        """
        Checks whether the sub_sliders, if configured, have values that sum to 100
        """
        if not self.sub_widgets_:
            return True

        if self.checkbox_widget_.checkbox_.disabled:
            return True

        count = 0
        for sub_widget in self.sub_widgets_:
            if (not sub_widget.checkbox_.disabled) and sub_widget.checkbox_.value:
                count += sub_widget.slider_.value
        return count == 100

    def is_active(self) -> bool:
        """
        Returns True if the main checkbox is selected; False otherwise
        """
        return self.checkbox_widget_.is_active()

    def return_value(self) -> Tuple[dict, dict]:
        """
        Returns the description and the value of the checkbox
        """
        main_description, value = self.checkbox_widget_.return_value()
        priority_dict = {main_description: value}
        sub_priority_dict = {}
        for sub_widget in self.sub_widgets_:
            if sub_widget.is_active():
                sub_priority, sub_value = sub_widget.return_value()
                if sub_value:
                    sub_priority_dict[sub_priority] = sub_value
        if sub_priority_dict:
            sub_dict = {main_description: sub_priority_dict}
        else:
            sub_dict = {}
        return priority_dict, sub_dict


class UserPriorities:
    """
    Class to seek user priorities as a collection of ipywidgets in a Jupyter Notebook

    Parameters
    ----------
    config_dict : dict
        A dictionary that determines the parameters for the widget to be displayed.
        An example structure includes:
        config_dict = {"1. Methane Emissions (Proxies)": {"default": 20,
        "sub_weights":
        {"1.1 Leak [Yes/No]": {"default": 40}},
        "1.2 Compliance [Yes/No]" : {"default": 60}}
        }

    validate : bool, default = False
        A bool to determine whether user inputs should be validated by the class

    Attributes
    ----------

    sub_check_box_widgets_ : List[SubCheckBoxWidget]
        List of all user-defined inputs to seek

    confirm_button_ : widgets.Button
        A button to confirm and validate all weights

    priorities_ : dict
        The user-defined values provided for main priorities

    sub_priorities_ : dict
        The user-defined values provided for sub-priorities

    validate_ : bool
        Bool to indicate whether validation checks should be run on user-defined inputs
    """

    def __init__(self, config_dict, validate=True):
        self.sub_check_box_widgets_ = []
        for priority, priority_dict in config_dict.items():
            weight_dict = {priority: priority_dict}
            self.sub_check_box_widgets_.append(SubCheckBoxWidget(weight_dict))

        self.confirm_button_ = widgets.Button(description="Confirm Weights")
        self.priorities_ = {}
        self.sub_priorities_ = {}
        self.validate_ = validate
        self.confirm_button_.on_click(self.confirm_weights)

        # A mapping to keep track of arbitrary objects that are mapped to
        # and from widget labels
        self._to_widget_labels = {}
        self._from_widget_labels = {}

    def get_widget_label_maps(self) -> Tuple[dict, dict]:
        """
        Gets the widget label maps that are used to keep track of widget labels with arbitrary
        objects
        """
        return (self._to_widget_labels, self._from_widget_labels)

    def set_widget_label_maps(self, to_widget_labels: dict, from_widget_labels: dict):
        """
        Sets the widget label maps that are used to keep track of widget labels with
        arbitrary objects
        """
        self._to_widget_labels = to_widget_labels
        self._from_widget_labels = from_widget_labels

    def validate(self) -> bool:
        """
        Checks whether the sub_sliders, if configured, have values that sum to 100;
        also checks whether all the main sliders have values that sum to 100
        """
        count = 0
        for sub_check_box_widget in self.sub_check_box_widgets_:
            if sub_check_box_widget.is_active():
                if not sub_check_box_widget.validate():
                    return False
                priority_dict, _ = sub_check_box_widget.return_value()
                count += list(priority_dict.values())[0]

        return count == 100

    def display(self) -> widgets.VBox:
        """
        Returns an object that can be displayed in the Jupyter Notebook
        """
        boxes = [
            sub_check_box_widget.display()
            for sub_check_box_widget in self.sub_check_box_widgets_
        ]
        return widgets.VBox(boxes + [self.confirm_button_])

    def confirm_weights(self, _):
        """
        Confirms that the weights provided by the user are valid
        """
        if self.validate_ and not self.validate():
            raise_exception(
                "Priority weights or sub_priority weights do not add up to 100",
                ValueError,
            )
        self.priorities_ = {}
        self.sub_priorities_ = {}

        for sub_check_box_widget in self.sub_check_box_widgets_:
            if sub_check_box_widget.is_active():
                priority_dict, sub_priority_dict = sub_check_box_widget.return_value()
                self.priorities_.update(priority_dict)
                self.sub_priorities_.update(sub_priority_dict)

        print("Weights confirmed and saved successfully!")

    def return_value(self) -> Tuple[dict, dict]:
        """
        Returns the description and the value of the checkbox
        """
        return self.priorities_, self.sub_priorities_


class BaseSelectWidget:
    """
    Class for displaying an autofill widget in Jupyter Notebook to select multiple choices from a
    list of choices provided. The widget comes configured with an "Undo" button that exclude the
    designated well from the selections

    Parameters
    ----------
    choices: typing.Iterable[str]
        Full collection of choices

    button_description: str
        The description to be displayed on the widget

    type_description: str
        The type of object to be selected, displayed on the widget

    Attributes
    ----------

    button : widgets.Button
        A button to confirm and add the selected option to the list of choices

    selected_list : List[str]
        A list containing all options selected by a user

    widget : widgets.Combobox
        A text widget with autofill feature for selecting options from a list
    """

    def __init__(
        self,
        choices: typing.Iterable[str],
        button_description: str,
        type_description: str,
    ):
        self.choices = choices

        # Initialize text
        self._text = ""
        self.widget = widgets.Combobox(
            value="",
            placeholder=f"Select {type_description}",
            description=type_description,
            disabled=False,
        )

        self.widget.observe(self._on_change, names="value")

        layout = widgets.Layout(width="auto", height="auto")

        # Add button
        self.button_add = widgets.Button(description=button_description, layout=layout)
        self.button_add.on_click(self._add)

        # Remove button
        self.button_remove = widgets.Button(description="Undo", layout=layout)
        self.button_remove.on_click(self._remove)

        self.selected_list = []

    def _on_change(self, data) -> None:
        """
        Dynamically update the list of choices available in the drop down widget
        based on what is already selected
        """
        # AutoComplete box requires a dictionary

        words_dict = {word: {} for word in self.choices}

        self._autocomplete = AutoComplete(words=words_dict)

        self._text = data["new"]

        values = self._autocomplete.search(self._text, max_cost=3, size=3)

        # convert nested list to flat list
        values = list(sorted(set(str(item) for sublist in values for item in sublist)))

        self.widget.options = values

    def _add(self, _) -> None:
        """
        Adds a selected choice and prints confirmation message in Jupyter notebook
        """

        if self._text == "":
            raise_exception("Nothing selected, cannot add to list", ValueError)
        if self._text in self.selected_list:
            msg = f"Choice: {self._text} already included in list of selections"
            LOGGER.info(msg)
            print(msg)
        else:
            self.selected_list.append(self._text)
            msg = f"Choice {self._text} has been added to the list of selections"
            LOGGER.info(msg)
            print(msg)

    def _remove(self, _) -> None:
        """
        Remove a selected choice and prints confirmation message in Jupyter Notebook
        """

        if self._text == "":
            raise_exception("Nothing selected, cannot remove from list", ValueError)
        if self._text not in self.selected_list:
            raise_exception(
                f"Choice {self._text} is not in the list",
                ValueError,
            )
        else:
            self.selected_list.remove(self._text)
            msg = f"Choice {self._text} has been removed from the list."
            LOGGER.info(msg)
            print(msg)

    def display(self):
        """
        Display the widget and button in the Jupyter Notebook
        """
        buttons = widgets.HBox([self.button_add, self.button_remove])
        vbox = widgets.VBox([self.widget, buttons])
        vbox.layout.align_items = "flex-end"
        return vbox

    def return_selections(self) -> List[int]:
        """
        Return the list of selections made by a user
        """
        return [int(item) for item in self.selected_list]

    def _pass_current_selection(self):
        return self._text


class SelectWidget(BaseSelectWidget):
    """SelectWidget for direct selection of choices."""


class SubSelectWidget(BaseSelectWidget):
    """
    SubSelectWidget for displaying an autofill widget that depends on selections from another widget.

    Parameters
    ----------
    choices: typing.Iterable[str]
        Full collection of choices

    button_description: str
        Description displayed on the widget

    type_description: str
        Type of object to be selected, displayed on the widget

    well_data: DataFrame
        Data containing well information
    """

    def __init__(
        self,
        cluster_choices: typing.Iterable[str],
        button_description_cluster: str,
        button_description_well: str,
        well_data,
    ):
        super().__init__(cluster_choices, button_description_well, "Well")
        self.cluster_widget = SelectWidget(
            cluster_choices, button_description_cluster, "Cluster"
        )
        # self.cluster_widget.display()
        self.wd = well_data

    def _on_change(self, data) -> None:
        """
        Dynamically update the list of choices available in the drop down widget
        based on what is already selected
        """

        cluster = self.cluster_widget._pass_current_selection()
        well_candidate = self.wd.data[
            self.wd.data[self.wd._col_names.cluster] == int(cluster)
        ][self.wd._col_names.well_id]

        words_dict = {word: {} for word in well_candidate}

        self._autocomplete = AutoComplete(words=words_dict)

        self._text = data["new"]

        values = self._autocomplete.search(self._text, max_cost=3, size=3)

        # convert nested list to flat list
        values = list(sorted(set(str(item) for sublist in values for item in sublist)))

        self.widget.options = values


class SelectWidgetAdd(SelectWidget):
    def __init__(
        self,
        well_choices: typing.Iterable[str],
        button_description: str,
        type_description: str,
    ):
        self.wd = well_choices
        super().__init__(
            self.wd.data[self.wd._col_names.well_id],
            button_description,
            type_description,
        )

        self.re_cluster = widgets.BoundedIntText(
            # value=current_cluster,
            min=min(self.wd.data[self.wd._col_names.cluster]),
            max=max(self.wd.data[self.wd._col_names.cluster]),
            step=1,
            description="To Cluster:",
            disabled=False,
        )

        self.re_cluster_dict = {}

    # def _on_change(self, _) -> None:
    #     super()._on_change(_)
    #     self.current_cluster = self.cluster_widget._pass_current_selection()

    def _add(self, _) -> None:
        super()._add(_)
        well_index = self.wd.data[
            self.wd.data[self.wd._col_names.well_id] == self._text
        ].index.item()
        self.re_cluster_dict.setdefault(self.re_cluster.value, []).append(well_index)

    def _remove(self, _) -> None:
        super()._remove(_)
        well_index = self.well_choices[
            self.wd.data[self.wd._col_names.well_id] == self._text
        ].index.item()
        self.re_cluster_dict[self.re_cluster.value].remove(well_index)

    def display(self):
        """
        Display the widget and button in the Jupyter Notebook
        """
        widget_box = widgets.HBox([self.widget, self.re_cluster])
        buttons = widgets.HBox([self.button_add, self.button_remove])
        vbox = widgets.VBox([widget_box, buttons])
        # vbox.layout.align_items = "flex-end"
        return vbox

    def return_selections(self) -> List[int]:
        """
        Return the list of selections made by a user
        """
        return [item for item in self.selected_list], self.re_cluster_dict


class UserSelection:
    def __init__(self, well_selected: tuple, model_inputs):

        self.wd = model_inputs.config.well_data
        self.cluster_selected = list(well_selected.keys())
        well_selected_list = [
            well for wells in well_selected.values() for well in wells
        ]
        self.well_selected = self.wd._construct_sub_data(well_selected_list)

        all_cluster = self.wd.data[self.wd._col_names.cluster].astype(str)
        cluster_selected_choice = [str(cluster) for cluster in self.cluster_selected]

        all_wells = self.wd.data.index
        well_add_candidate_list = [
            well for well in all_wells if well not in well_selected_list
        ]
        well_add_candidate = self.wd._construct_sub_data(well_add_candidate_list)
        # well_add_candidate = self.wd.data[~self.wd.data.index.isin(well_selected_list)]
        # well_add_candidate = self.wd.data[~self.wd.data[self.wd._col_names.well].isin(well_selected_list)]

        self.add_widget = SelectWidgetAdd(
            well_add_candidate, "Select wells to manually add", "Well"
        )

        self.remove_widget = SubSelectWidget(
            cluster_selected_choice,
            "Select clusters to manually remove",
            "Select wells to manually remove",
            self.well_selected,
        )

        self.unlock_widget = SubSelectWidget(
            cluster_selected_choice,
            "Select clusters to manually unlock",
            "Select wells to manually unlock",
            self.well_selected,
        )

        self.widgets_dict = {
            "Add": self.add_widget,
            "Remove": self.remove_widget,
            "Unlock": self.unlock_widget,
        }

    def display(self) -> None:
        for action, widget in self.widgets_dict.items():
            if action == "Add":
                well_vbox = widget.display()
                display(f"{action} wells", well_vbox)
            else:
                cluster_vbox = widget.cluster_widget.display()
                well_vbox = widget.display()
                widget = widgets.HBox([cluster_vbox, well_vbox])
                display(f"{action} clusters/wells", widget)

    def return_value(self):
        # def create_dict(selections, value):
        #     return {item: value for item in selections}

        def return_well_index_cluster(selections):
            selections_dict = {}
            for well in selections:
                well_index = self.wd.data[
                    self.wd.data[self.wd._col_names.well_id] == str(well)
                ].index.item()
                cluster = self.wd.data[
                    self.wd.data[self.wd._col_names.well_id] == str(well)
                ][self.wd._col_names.cluster].item()
                selections_dict.setdefault(cluster, []).append(well_index)
            return selections_dict

        add_widget_return = (
            return_well_index_cluster(self.add_widget.return_selections()[0]),
            self.add_widget.return_selections()[1],
        )

        remove_widget_return = (
            self.remove_widget.cluster_widget.return_selections(),
            return_well_index_cluster(self.remove_widget.return_selections()),
        )

        cluster_unlock_list = self.unlock_widget.cluster_widget.return_selections()
        well_unlock_list = self.unlock_widget.return_selections()
        cluster_lock_list = [
            cluster
            for cluster in self.cluster_selected
            if cluster not in cluster_unlock_list
        ]
        well_lock_list = [
            well
            for well in self.well_selected[self.wd._col_names.well_id]
            if well not in well_unlock_list
        ]

        unlock_widget_return = (
            cluster_lock_list,
            return_well_index_cluster(well_lock_list),
        )

        return [add_widget_return, remove_widget_return, unlock_widget_return]
