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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Installed libs
import ipywidgets as widgets
from IPython.display import clear_output, display

# User-defined libs
# from primo.data_parser.well_data import WellData
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-lines


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
    # pylint: disable=unspecified-encoding
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
    A base class for displaying an autofill widget in Jupyter Notebook to select multiple
    choices from a list of choices provided. The widget comes configured with an "Undo"
    button that exclude the designated well from the selections

    Parameters
    ----------
    choices: List[int]
        A list of collection of choices

    button_description: str
        The description to be displayed on the button_add

    type_description: str
        The type of object (project or well) to be selected, displayed on the widget

    Attributes
    ----------

    button_add : widgets.Button
        Button to confirm and add the selected option.

    button_remove : widgets.Button
        Button to remove the selected option.

    widget : widgets.Combobox
        Text widget with autofill feature for selecting options from a list.

    vBox : widget.Vbox
        A vertical box which includes the button_add, button_remove, and the
        autofill feature widget

    selected_list : List[str]
        List containing all projects or wells selected by the user.

    _text : str
        The current selection of the widget
    """

    def __init__(
        self,
        choices: List[int],
        button_description: str,
        type_description: str,
    ):
        self.choices = [str(choice) for choice in choices]

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

        # Output widget for displaying messages
        self.output = widgets.Output()

        self.selected_list = []

    def _on_change(self, data) -> None:
        """
        Dynamically update the currently selection
        """
        self._text = data["new"]

        self.widget.options = self.choices

    def _add(self, _) -> None:
        """
        Adds a selected choice and prints confirmation message in Jupyter notebook
        """
        with self.output:
            if self._text == "":
                raise_exception("Nothing selected, cannot add to list", ValueError)
            if self._text in self.selected_list:
                raise_exception(
                    f"Choice {self._text} already included in list of selections",
                    ValueError,
                )
            else:
                self.selected_list.append(self._text)
                msg = f"Choice {self._text} has been added to the list of selections"
                LOGGER.info(msg)
                print(msg)

    def _remove(self, _) -> None:
        """
        Remove a selected choice and prints confirmation message in Jupyter Notebook
        """
        with self.output:
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
        return vbox, self.output

    def return_selections(self) -> List[int]:
        """
        Return the list of selections made by the user
        """
        return [int(item) for item in self.selected_list]

    def _pass_current_selection(self):
        """
        Return the currently selection text
        """
        return self._text


class SelectWidget(BaseSelectWidget):
    """SelectWidget for choosing the projects."""


class SubSelectWidget(BaseSelectWidget):
    """
    SubSelectWidget for displaying an autofill widget, where the well choices for the widget
    depend on the cluster selection from the SelectWidget.

    Parameters
    ----------
    cluster_choices : List[int]
         list of collection of cluster choices for the SelectWidget

    button_description_cluster : str
        Description for the cluster selection button

    button_description_well : str
        Description for the well selection button

    well_data : pd.DataFrame
        Data containing well information
    """

    def __init__(
        self,
        cluster_choices: List[int],
        button_description_cluster: str,
        button_description_well: str,
        well_data,
    ):
        super().__init__(cluster_choices, button_description_well, "Well")
        # set the widget for selecting clusters
        self.cluster_widget = SelectWidget(
            cluster_choices, button_description_cluster, "Project"
        )
        self.wd = well_data

    def _on_change(self, data) -> None:
        """
        Dynamically update the list of well choices available in the drop down widget
        based on the cluster selected
        """
        # pylint: disable=protected-access

        # obtain the current cluster selection
        self._text = data["new"]

        cluster = self.cluster_widget._pass_current_selection()
        # obtain the well candidates under the selected cluster

        well_candidate = self.wd.data[
            self.wd.data[self.wd._col_names.cluster] == int(cluster)
        ][self.wd._col_names.well_id]

        values = list(well_candidate)

        self.widget.options = values


class SelectWidgetAdd(SelectWidget):
    """
    SelectWidgetAdd for adding wells with associated cluster selection.

    Parameters
    ----------
    well_choices : WellData
        The WellData object for wells that are candidates for the add widget

    button_description : str
        Description for the add button

    type_description : str
        Type of object to be selected

    Attributes
    ----------

    button_add : widgets.Button
        Button to confirm and add the selected option

    button_remove : widgets.Button
        Button to remove the selected option

    widget : widgets.Combobox
        Text widget with autofill feature for selecting options from a list

    re_cluster : widgets.BoundedIntText
        Text widget which allows the user to type in the project number that
        they would like to move the selected well to. The default value for the
        project is the cluster of the well

    vBox : widget.Vbox
        A vertical box which includes the button_add, button_remove, the re-cluster
        widget and the autofill feature widget

    selected_list : List[str]
        List containing all projects or wells selected by the user
    """

    # pylint: disable=protected-access

    def __init__(
        self,
        well_choices,
        button_description: str,
        type_description: str,
    ):
        self.wd = well_choices
        col_names = self.wd._col_names
        super().__init__(
            self.wd.data[self.wd._col_names.well_id],
            button_description,
            type_description,
        )

        self.re_cluster = widgets.BoundedIntText(
            value=0,
            min=min(self.wd[col_names.cluster]),
            max=max(self.wd[col_names.cluster]),
            step=1,
            description="To Project:",
            disabled=False,
        )

        # Attach observer to update cluster when selection changes
        self.widget.observe(self._update_re_cluster, names="value")

        self.re_cluster_dict = {}

    def _update_re_cluster(self, change):
        """Update value show in the re_cluster widget
        based on cluster of the selected well."""

        selected_well = change["new"]
        if selected_well:
            cluster_value = self.wd.data.loc[
                self.wd.data[self.wd._col_names.well_id] == selected_well,
                self.wd._col_names.cluster,
            ].values
            if cluster_value.size > 0:
                self.re_cluster.value = cluster_value[
                    0
                ]  # Update to the cluster of the selected well

    def _add(self, _) -> None:
        if self._text not in self.wd.data[self.wd._col_names.well_id].values:
            raise_exception(
                "The well is already assigned to a project. It must be removed "
                "from the current project before it can be assigned to another one.",
                ValueError,
            )
        super()._add(_)
        well_index = self.wd.data[
            self.wd.data[self.wd._col_names.well_id] == self._text
        ].index.item()
        self.re_cluster_dict.setdefault(self.re_cluster.value, []).append(well_index)

    def _remove(self, _) -> None:
        super()._remove(_)
        well_index = self.wd[
            self.wd.data[self.wd._col_names.well_id] == self._text
        ].index.item()
        self.re_cluster_dict[self.re_cluster.value].remove(well_index)

    def display(self):
        """
        Display the widget and button in the Jupyter Notebook
        """
        # widget_box = widgets.HBox([self.widget, self.re_cluster])
        buttons = widgets.HBox([self.button_add, self.button_remove])
        vbox_left = widgets.VBox([self.widget])
        vbox_right = widgets.VBox([self.re_cluster, buttons])
        vbox_right.layout.align_items = "flex-end"
        vbox = widgets.HBox([vbox_left, vbox_right])
        return vbox, self.output

    def return_selections(self) -> List[int]:
        """
        Return the list of selections made by a user
        """
        return list(self.selected_list), self.re_cluster_dict


class UserSelection:
    """
    Class for managing user selections of override projects and wells

    Parameters
    ----------
    opt_campaign: Dict[int,List[int]]
        Dictionary of the optimal campaigns obtained from the optimization problem

    model_inputs: OptModelInputs
        Object containing the necessary inputs for the optimization model
    """

    # pylint: disable=too-many-instance-attributes,protected-access

    def __init__(self, opt_campaign: dict, model_inputs: object):
        self.wd = model_inputs.config.well_data
        self.opt_campaign = opt_campaign
        self.well_selected_list = [
            well for wells in opt_campaign.values() for well in wells
        ]
        self.well_selected = self.wd._construct_sub_data(self.well_selected_list)

        self.cluster_remove_choice = list(opt_campaign.keys())

        self.all_wells = self.wd.data.index

        self.remove_widget = SubSelectWidget(
            self.cluster_remove_choice,
            "Select projects to manually remove",
            "Select wells to manually remove",
            self.well_selected,
        )

        self.add_widget = SelectWidgetAdd(self.wd, "", "")
        self.lock_widget = SubSelectWidget([], "", "", self.wd)

        # Confirm button
        layout = widgets.Layout(width="auto", height="50%")
        self.button_remove_confirm = widgets.Button(
            description="Confirm Removal", layout=layout
        )
        self.button_remove_confirm.on_click(self._process_remove_input)
        self.output = widgets.Output()

        self.cluster_lock_choice = []
        self.well_lock_choice = []

    def display(self) -> None:
        """Display the remove_widget"""
        self._display_remove_widget()

    def _display_remove_widget(self):
        """Construct and display the remove_widget"""
        cluster_vbox, cluster_output = self.remove_widget.cluster_widget.display()
        well_vbox, well_output = self.remove_widget.display()
        widget = widgets.HBox([cluster_vbox, well_vbox, self.button_remove_confirm])
        output = widgets.VBox([cluster_output, well_output, self.output])
        remove_widget_container = widgets.VBox([widget, output])

        widget_placeholder = widgets.Label(
            "Add and lock widgets will display after clicking Confirm Removal button."
        )

        # Combine everything into a container for display
        container = widgets.VBox([remove_widget_container, widget_placeholder])

        display("Remove projects/wells", container)

    def _process_remove_input(self, _):
        """Construct the cluster and well candidates for the add_widget and lock_widget
        based on the removed projects and wells"""

        with self.output:
            clear_output(wait=True)  # Clear previous output

            # obtain projects that have been removed
            remove_selections_cluster = (
                self.remove_widget.cluster_widget.return_selections()
            )

            # obtain wells that have been removed both from the cluster
            # and well selection widgets
            remove_well = []
            for remove_cluster in remove_selections_cluster:
                remove_well += self.opt_campaign[remove_cluster]
            remove_selections_well = (
                self._return_well_index_cluster(self.remove_widget.return_selections())[
                    1
                ]
                + remove_well
            )

            # prepare add_widget and unlock_widget data
            self._set_add_widget(remove_selections_well)
            self._set_lock_widget(remove_selections_cluster, remove_selections_well)

            # display both the add_widget and lock_widget at the same time
            self._display_all_widgets()

    def _set_add_widget(self, remove_selections_well):
        """Set up the widget for adding wells"""
        well_add_candidate_list = [
            well for well in self.all_wells if well not in self.well_selected_list
        ] + remove_selections_well
        well_add_candidate = self.wd._construct_sub_data(well_add_candidate_list)

        self.add_widget = SelectWidgetAdd(
            well_add_candidate, "Select wells to manually add", "Add Well"
        )

    def _set_lock_widget(self, remove_selections_cluster, remove_selections_well):
        """Set up the widget for locking wells and projects"""
        self.cluster_lock_choice = [
            cluster
            for cluster in self.cluster_remove_choice
            if cluster not in remove_selections_cluster
        ]

        well_lock_list = [
            well
            for well in self.well_selected_list
            if well not in remove_selections_well
        ]
        self.well_lock_choice = self.wd._construct_sub_data(well_lock_list)

        self.lock_widget = SubSelectWidget(
            self.cluster_lock_choice,
            "Select projects to manually lock",
            "Select wells to manually lock",
            self.well_lock_choice,
        )

    def _display_all_widgets(self):
        """Display both the add_widget and lock_widget"""

        add_widget, add_output = self.add_widget.display()
        add_container = widgets.VBox([add_widget, add_output])

        cluster_vbox, cluster_output = self.lock_widget.cluster_widget.display()
        well_vbox, well_output = self.lock_widget.display()
        lock_widget = widgets.HBox([cluster_vbox, well_vbox])
        lock_output = widgets.VBox([cluster_output, well_output])
        lock_container = widgets.VBox([lock_widget, lock_output])

        display("Add wells", add_container)
        display("Lock projects/wells", lock_container)

    def _return_well_index_cluster(self, selections):
        """Obtain index of selected wells according to their API Well Number"""
        selections_dict = {}
        selections_list = []
        for well in selections:
            well_index = self.wd.data[
                self.wd.data[self.wd._col_names.well_id] == str(well)
            ].index.item()
            cluster = self.wd.data[
                self.wd.data[self.wd._col_names.well_id] == str(well)
            ][self.wd._col_names.cluster].item()
            selections_dict.setdefault(cluster, []).append(well_index)
            selections_list.append(well_index)
        return selections_dict, selections_list

    def return_value(self):
        """A wrapper for returning selections of all widgets"""

        # Projects chose to be removed through the remove project button
        cluster_remove_list = self.remove_widget.cluster_widget.return_selections()

        # Wells chose to be removed through the remove well button
        well_remove_dict, well_remove_list = self._return_well_index_cluster(
            self.remove_widget.return_selections()
        )

        # Add wells belongs to the removed project to the well_remove_dict
        for cluster in cluster_remove_list:
            well_remove_dict[cluster] = list(self.opt_campaign[cluster])

        remove_widget_return = OverrideRemoveLockInfo(
            cluster_remove_list,
            well_remove_dict,
        )

        existing_clusters, new_clusters = self.add_widget.return_selections()
        well_add_dict, _ = self._return_well_index_cluster(existing_clusters)
        add_widget_return = OverrideAddInfo(well_add_dict, new_clusters)

        cluster_lock_list = self.lock_widget.cluster_widget.return_selections()
        well_lock_dict, _ = self._return_well_index_cluster(
            self.lock_widget.return_selections()
        )
        for cluster in cluster_lock_list:
            well_lock_dict[cluster] = [
                well
                for well in self.opt_campaign[cluster]
                if well not in well_remove_list
            ]

        lock_widget_return = OverrideRemoveLockInfo(cluster_lock_list, well_lock_dict)

        return OverrideSelections(
            remove_widget_return=remove_widget_return,
            add_widget_return=add_widget_return,
            lock_widget_return=lock_widget_return,
        )


@dataclass
class OverrideRemoveLockInfo:
    """
    Class for storing the information returns from the remove and lock widget

    Parameters
    ----------
    cluster : List[int]
        List of projects that are removed or locked

    well : Dict[int, List[int]]
        Dictionary of list of wells that are removed or locked in a cluster
    """

    cluster: List[int]
    well: Dict[int, List[int]]

    def __str__(self):
        return (
            f"OverrideRemoveLockInfo(project={self.cluster}, " f"well_dict={self.well})"
        )


@dataclass
class OverrideAddInfo:
    """
    Class for storing the add widget return

    Parameters
    ----------
    existing_clusters : Dict[int, List[int]]
        Dictionary of list of wells being added and their original cluster;
        key=> cluster, value=> well list

    new_clusters : Dict[int, List[int]]
        Dictionary of list of wells being added and their new cluster;
        key=> cluster, value=> well list
    """

    existing_clusters: Dict[int, List[int]]
    new_clusters: Dict[int, List[int]]

    def __str__(self):
        return (
            f"OverrideAddInfo(existing_cluster_dict={self.existing_clusters}, "
            f"new_cluster_dict={self.new_clusters})"
        )


@dataclass
class OverrideSelections:
    """
    Class for storing the return from the remove_widget, add_widget, and lock_widget

    Parameters
    ----------
    remove_widget_return : OverrideRemoveLockInfo
        Object returned by the remove_widget

    add_widget_return : OverrideAddInfo
        Object returned by the add_widget

    lock_widget_return : OverrideRemoveLockInfo
        Object returned by the lock_widget
    """

    remove_widget_return: OverrideRemoveLockInfo
    add_widget_return: OverrideAddInfo
    lock_widget_return: OverrideRemoveLockInfo

    def __str__(self):
        return (
            f"OverrideSelections("
            f"remove_widget_return={self.remove_widget_return}, "
            f"add_widget_return={self.add_widget_return}, "
            f"lock_widget_return={self.lock_widget_return})"
        )
