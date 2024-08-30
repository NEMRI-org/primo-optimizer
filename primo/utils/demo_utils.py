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
import json
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Union

# Installed libs
import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from ipyfilechooser import FileChooser
from IPython.display import display

# User-defined libs
from primo.utils import CONVERSION_FACTOR
from primo.utils.census_utils import get_census_key
from primo.utils.geo_utils import get_nearest_neighbors

LOGGER = logging.getLogger(__name__)


COLS_TO_KEEP = [
    "API Well Number",
    "Census Tract ID",
    "State Code",
    "County Code",
    "Tract Code",
    "Land Area",
    "Gas [Mcf/Year]",
    "Oil [bbl/Year]",
    "Age [Years]",
    "Latitude",
    "Longitude",
    "Incident [Yes/No]",
    "Violation [Yes/No]",
    "Compliance [Yes/No]",
    "Leak [Yes/No]",
    "Depth [ft]",
    "Operator Name",
    "H2S Leak [Yes/No]",
    "State Wetlands (0 - 300 ft)",
    "State Wetlands (300 - 1,320 ft)",
    "Federal Wetlands (0 - 300 ft)",
    "Federal Wetlands (300 - 1,320 ft)",
    "Buildings (0 - 300 ft)",
    "Buildings (300 - 1,320 ft)",
    "5-Year Oil Production [bbl]",
    "5-Year Gas Production [Mcf]",
    "Well Name",
    "Lifelong Oil Production [bbl]",
    "Lifelong Gas Production [Mcf]",
    "Elevation Delta [m]",
    "Distance to Road [miles]",
]

COLS_TO_KEEP_ALT = [
    "Well_Permit_Num_MCWLOC_",
    "Geoid",
    "State_Code",
    "County_Code",
    "Tract_Code",
    "Land_Area",
    "GAS_QUANTITY_1",
    "OIL_QUANTITY_1",
    "Age",
    "Well_Type",
    "Latitude",
    "Longitude",
    "Incident [Yes/No]",
    "Violation [Yes/No]",
    "Compliance [Yes/No]",
    "Leak [Yes/No]",
    "Depth",
    "Company_Name",
    "Wells_Within_1_mile",
    "Closest Road Point",
    "elevation delta",
    "Distance to Road (miles)",
    "contamination",
    "methane leak",
]


def file_path_widget(
    description: str, file_type: str, on_upload_change_str: str, default: str = None
):
    """
    Displays a widget on a Jupyter Notebook to provide a path for a file.

    Parameters
    ----------
    description : str
        The description to be displayed on the FileChooser widget

    file_type : str
        The file types that will be allowable by the widget

    on_upload_change_str : str
        The string to be displayed when the file is uploaded

    default : str, default=None
        The default path suggested by the widget

    Returns
    -------
    A FileChooser widget
    """
    kwargs = {"filter_pattern": f"*{file_type}", "title": f"<b> {description}</b>"}

    default_exists = False
    if default:
        if os.path.exists(default):
            default_exists = True
            default_path = default
        else:
            # check if the issue is using Windows style path on Linux or vice versa
            # 1: Test as windows path
            win_path = Path(PureWindowsPath(default))
            linux_path = Path(PurePosixPath(default))
            if os.path.exists(win_path):
                default_path = str(win_path)
                default_exists = True
            elif os.path.exists(linux_path):
                default_path = str(linux_path)
                default_exists = True
            else:
                LOGGER.warning(f"Default file path: {default} doesn't exist. Ignoring")

    if default_exists:
        file_dir, file_path = os.path.split(default_path)
        kwargs["path"] = file_dir
        kwargs["filename"] = file_path
        kwargs["select_default"] = True
        kwargs["title"] = f"<b> Default {description} Selected </b>"

    def change_title(chooser):
        chooser.title = f"<b> {on_upload_change_str} </b>"

    file_widget = FileChooser(**kwargs)
    file_widget.register_callback(change_title)
    display(file_widget)
    return file_widget


def file_upload_widget(
    description: str, file_type: str, on_upload_change_str: str, multiple: bool = False
):
    """
    Displays a widget on a Jupyter Notebook to upload a file_type,

    Parameters
    ----------
    description : str
        The description to be displayed on the file upload widget

    file_type : str
        The file types that will be allowable by the widget


    on_upload_change_str : str
        The string to be displayed when the file is uploaded

    multiple : bool
        Whether multiple files are allowed to be uploaded

    Returns
    -------
    A FileUpload widget
    """

    input_file = widgets.FileUpload(
        accept=file_type, multiple=multiple, description=description
    )
    label = widgets.Label("")

    def on_upload_change(change):
        label.value = on_upload_change_str

    input_file.observe(on_upload_change, names="value")

    display(input_file, label)
    return input_file


def weight_display(value: float) -> Union[int, float]:
    """
    Determines the best way to display a priority weight. If the value is an integer,
    displays it as an integer; otherwise uses decimal points and displays it as a float.

    Parameters
    ----------
    value : float
        The value to be displayed


    Returns
    -------
    Union[int, float]
        The value to be displayed in the Jupyter Notebook
    """
    nearest_int = round(value)
    if np.isclose(value, nearest_int):
        return nearest_int
    return value


def priority_by_value(column):
    """
    Prioritize values in a column, with priority given to "Yes", followed by nulls, and then "No".

    Parameters
    ----------
    column : pd.Series
        The input column to prioritize

    Returns
    -------
    pd.Series
        The prioritized column
    """
    # Prioritize Yes, followed by Nulls, followed by No
    if column.dtype == "int64" or column.dtype == "float64":
        return column

    val_map = {"Yes": 1, "NULL": 2, "No": 3}
    return column.map(val_map)


def sort_columns(df: pd.DataFrame, columns: Dict[str, bool]) -> pd.DataFrame:
    """
    Sort columns in a DataFrame after mapping categorical values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    columns : Dict[str, bool]
        A dictionary with column names as keys and sorting order (ascending) as values

    Returns
    -------
    pd.DataFrame
        The DataFrame with sorted columns
    """
    # Sort columns after mapping categorical values
    sort_order = []
    column_names = []
    for col_name, ascending in columns.items():
        df[col_name] = df[col_name].fillna("NULL")
        column_names.append(col_name)
        sort_order.append(ascending)

    return df.sort_values(
        by=column_names, ascending=sort_order, key=priority_by_value
    ).reset_index(drop=True)


def sort_by_disadvantaged_community_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a DataFrame by the impact on disadvantaged communities.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame

    Returns
    -------
    pd.DataFrame
        The DataFrame sorted by disadvantaged community impact
    """
    screening_file = "disadvantaged_community_data_2010_census_public.csv"
    if not os.path.exists(screening_file):
        screening_file = os.path.join("primo", "demo", screening_file)
    screening_data = pd.read_csv(
        screening_file,
        usecols=[
            "Census tract 2010 ID",
            "Percentage of tract that is disadvantaged by area",
            "Share of neighbors that are identified as disadvantaged",
        ],
    )
    screening_data.columns = [
        "Geoid",
        "Disadvantaged area %",
        "Disadvantaged pop %",
    ]
    screening_data["Disadvantaged Community"] = (
        screening_data["Disadvantaged area %"] + screening_data["Disadvantaged pop %"]
    ) / 2
    screening_data = screening_data.drop(
        ["Disadvantaged area %", "Disadvantaged pop %"], axis=1
    )
    df = df.merge(screening_data, how="left", on="Geoid")
    df = df.sort_values(by="Disadvantaged Community", ascending=False).reset_index(
        drop=True
    )
    return df


def sort_by_nearest_wells(df: pd.DataFrame, distance: float) -> pd.DataFrame:
    """
    Sort a DataFrame by the distance to nearest wells.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    distance : float
        The distance threshold

    Returns
    -------
    pd.DataFrame
        The DataFrame sorted by nearest wells
    """
    well_locations = list(zip(df["latitude"], df["longitude"]))
    new_col = f"Wells within {distance} miles"
    df[new_col] = get_nearest_neighbors(well_locations, distance, "MILES")
    df = df.sort_values(by=new_col, ascending=True).reset_index(drop=True)
    return df


def get_well_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare gas and oil production levels and add a "Well Type" column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with "Gas [Mcf/Year]" and "Oil [bbl/Year]" columns

    Returns
    -------
    pd.DataFrame
        Output DataFrame with an additional "Well Type" column
    """

    # Ensure the required columns are present in the DataFrame
    if "Gas [Mcf/Year]" not in df.columns or "Oil [bbl/Year]" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Gas [Mcf/Year]' and 'Oil [bbl/Year]' columns."
        )

    # Convert oil production from bbl/Year to Mcf/Year using a conversion factor

    df["Oil [Mcf/Year]"] = df["Oil [bbl/Year]"] * CONVERSION_FACTOR

    # Add 'Well Type' column based on the comparison of gas and converted oil production
    df["Well Type"] = df.apply(
        lambda row: "Gas" if row["Gas [Mcf/Year]"] > row["Oil [Mcf/Year]"] else "Oil",
        axis=1,
    )

    # Drop the intermediate 'Oil [Mcf/year]' column
    df = df.drop(columns=["Oil [Mcf/Year]"])

    return df


def get_population_by_state(state_code: int) -> pd.DataFrame:
    """
    Retrieves population data by state from the U.S. Census API.

    Parameters
    ----------
    state_code : int
        The state code

    Returns
    -------
    pd.DataFrame
        Population data by census tract in the specified state
    """
    CENSUS_KEY = get_census_key()
    url = "https://api.census.gov/data/2020/dec/dhc"
    params = {
        "get": "NAME,P1_001N",
        "for": "tract:*",
        "in": f"state:{state_code} county:*",
        "key": CENSUS_KEY,
    }
    session = requests.session()
    resp = session.get(url, params=params)
    assert resp.status_code == 200
    data = resp.json()
    pop = pd.DataFrame(data[1:], columns=data[0])
    pop = pop.rename(columns={"P1_001N": "Total Population"})
    for col in ["state", "county", "tract", "Total Population"]:
        pop[col] = pop[col].astype("int")
    return pop


def get_well_depth(df: pd.DataFrame, depth_threshold: float = 2000) -> pd.DataFrame:
    """
    Separate deep wells and shallow wells and add a "Well Depth Type" column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with "Depth [ft]" columns

    depth_threshold: float
        The depth threshold to classify a well as a deep well

    Returns
    -------
    pd.DataFrame
        Output DataFrame with an additional "Well Depth Type" column
    """

    # Ensure the required columns are present in the DataFrame
    if "Depth [ft]" not in df.columns:
        raise ValueError("DataFrame must contain 'Depth [ft]' column.")

    if depth_threshold < 0:
        raise ValueError("The threshold must be a positive value.")

    # Add 'Well Type' column based on the comparison of gas and converted oil production
    df["Well Depth Type"] = df.apply(
        lambda row: (
            "NULL"
            if np.isnan(row["Depth [ft]"])
            else ("Deep" if row["Depth [ft]"] > depth_threshold else "Shallow")
        ),
        axis=1,
    )

    return df


def generate_configurations(
    weights_file_path: str,
    config_path: str,
    output_folder_path: str,
    life_prod_values: List,
    owner_well_counts: List,
):
    """
    Generate configuration files based on scenarios defined in an Excel file.

    Parameters
    ----------
    weights_file_path : str
        Path to the Excel file containing weight scenarios
    config_path : str
        Path to the base configuration JSON file for reference
    output_folder_path : str
        Path to the output folder where the generated configuration files will be saved
    life_prod_values: List
        List of values to be used for the lifelong production filters
    owner_well_counts: List
        List of values to be used for the owner well-count constraint in the opt model

    Returns
    -------
    None
    """

    # Load the Excel file with two sheets
    try:
        df_scenarios = pd.read_excel(weights_file_path, sheet_name="default")
    except ValueError as e:
        raise ValueError(f"Error loading sheet 'default': {e}")

    try:
        df_scenarios_owc = pd.read_excel(weights_file_path, sheet_name="owc constraint")
    except ValueError as e:
        raise ValueError(f"Error loading sheet 'owc constraint': {e}")

    num_scenarios = (
        df_scenarios.shape[1] - 3
    )  # Scenarios start after first three columns in weights excel file
    num_scenarios_owc = (
        df_scenarios_owc.shape[1] - 3
    )  # Scenarios start after first three columns in weights excel file

    # Load the base configuration
    with open(config_path, "r") as file:
        base_config = json.load(file)

    # Function to update the configuration with default values from a scenario
    def update_config_with_scenario(
        config: dict, df: pd.DataFrame, scenario_index: int
    ) -> dict:
        """
        Update the configuration dictionary with default values from a given scenario in the DataFrame.

        Parameters
        ----------
        config : dict
            The base configuration dictionary to be updated.
        df : pd.DataFrame
            The DataFrame containing scenario values. It should have columns 'Major Priority', 'Sub priority',
            and 'Scenario {scenario_index}'.
        scenario_index : int
            The index of the scenario to use for updating the configuration.

        Returns
        -------
        dict
            The updated configuration dictionary with values from the specified scenario.
        """
        current_major_priority = None

        for _, row in df.iterrows():
            major_priority = row["Major Priority"]
            sub_priority = row["Sub Priority"]
            default_value = row[f"Priority Weight for Scenario {scenario_index}"]

            if pd.notna(major_priority):
                current_major_priority = major_priority.strip()
                if current_major_priority not in config["impact_weights"]:
                    config["impact_weights"][current_major_priority] = {
                        "default": default_value,
                        "min_val": 0,
                        "max_val": 100,
                        "incr": 5,
                        "sub_weights": {},
                    }
                else:
                    config["impact_weights"][current_major_priority][
                        "default"
                    ] = default_value

            if pd.notna(sub_priority):
                sub_key = sub_priority.strip()
                if (
                    "sub_weights"
                    not in config["impact_weights"][current_major_priority]
                ):
                    config["impact_weights"][current_major_priority]["sub_weights"] = {}
                config["impact_weights"][current_major_priority]["sub_weights"][
                    sub_key
                ] = {
                    "default": default_value,
                    "min_val": 0,
                    "max_val": 100,
                    "incr": 5,
                }

        return config

    # Ensure output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate over 9 scenarios for default scenarios
    for scenario_index in range(1, num_scenarios + 1):
        config = copy.deepcopy(base_config)
        config = update_config_with_scenario(config, df_scenarios, scenario_index)

        del config["program_constraints"]["Owner Well Count Constraint"]
        del config["program_constraints"]["Lifelong Production Constraint"]

        output_path = os.path.join(
            output_folder_path, f"config_scenario_{scenario_index}.json"
        )
        with open(output_path, "w") as outfile:
            json.dump(config, outfile, indent=4)

    # Generate configurations for the second sheet with owc constraints

    for scenario_index in range(1, num_scenarios_owc + 1):
        for owc in owner_well_counts:
            config = copy.deepcopy(base_config)
            config = update_config_with_scenario(
                config, df_scenarios_owc, scenario_index
            )

            # Update the Owner Well Count Constraint
            config["program_constraints"]["Owner Well Count Constraint"][
                "default"
            ] = owc

            output_path = os.path.join(
                output_folder_path, f"config_scenario_{scenario_index}_owc_{owc}.json"
            )
            with open(output_path, "w") as outfile:
                json.dump(config, outfile, indent=4)

    # Generate configurations for the prod cases

    for scenario_index in range(1, num_scenarios_owc + 1):
        for prod in life_prod_values:
            config = copy.deepcopy(base_config)
            config = update_config_with_scenario(
                config, df_scenarios_owc, scenario_index
            )

            # Update the 5-year Production Constraint
            config["program_constraints"]["Lifelong Production Constraint"][
                "default"
            ] = prod

            output_path = os.path.join(
                output_folder_path, f"config_scenario_{scenario_index}_prod_{prod}.json"
            )
            with open(output_path, "w") as outfile:
                json.dump(config, outfile, indent=4)
