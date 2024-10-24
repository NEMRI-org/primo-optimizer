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

# Standard lib

# Installed libs
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# User-defined libs
from primo.utils.raise_exception import raise_exception


def _is_numeric_valid_column(
    group: pd.DataFrame, column_name: str, estimation_method="no"
):
    """
    Checks if a column in a DataFrame is numeric and has all non-null values

    Parameters
    ----------
    group : pd.DataFrame
        The group DataFrame.
    column_name : str
        The column name for column under consideration
    estimation_method: str
        "yes" if this being applied to a method that's estimating values,
        or "no" if it's anything else.

    Raises
    ------
    ValueError
        if column_name is not a valid column in the DataFrame or if any
        of the values in the column are null, or if the column is non_numeric
    """
    if column_name not in group.columns:
        msg = f"column: {column_name} not found in DataFrame"
        raise_exception(msg, ValueError)

    if estimation_method == "no":
        if any(pd.isnull(group[column_name])):
            msg = f"Empty values were detected for column: {column_name}."
            raise_exception(msg, ValueError)

    if not is_numeric_dtype(group[column_name]):
        msg = f"Column: {column_name} is not numeric"
        raise_exception(msg, ValueError)


def calculate_range(group: pd.DataFrame, column_name: str) -> float:
    """
    Calculate the range value of the specified metric within a group.

    Parameters
    ----------
    group : pd.DataFrame
        The group DataFrame
    column_name : str
        The column name of the specified metric for calculation

    Returns
    -------
    float
        The range within the group for the specified metric

    Raises
    ------
    ValueError
        If column_name is not a valid column in the DataFrame, or if any
        of the values in the column are null, or if the column is non_numeric
    """

    _is_numeric_valid_column(group, column_name)
    return group[column_name].max() - group[column_name].min()


def calculate_well_number(group: pd.DataFrame) -> int:
    """
    Calculate the number of wells within a group.

    Parameters
    ----------
    group : pd.DataFrame
        The group DataFrame

    Returns
    -------
    int
        The number of wells within the group
    """
    return len(group)


def calculate_average(
    group: pd.DataFrame, column_name: str, estimation_method="no"
) -> float:
    """
    Calculate the average value of the specified metric within a group.

    Parameters
    ----------
    group : pd.DataFrame
        The group DataFrame
    column_name: str
        The column name of the specified metric for calculation
    estimation_method: str
        "yes" if this being applied to a method that's estimating values,
        or "no" if it's anything else

    Returns
    -------
    float
        The average value within the group for that specified metric

    Raises
    ------
    ValueError
        If column_name is not a valid column in the DataFrame, or if any
        of the values in the column are null, or if the column is non_numeric
    """
    _is_numeric_valid_column(group, column_name, estimation_method)

    # If all values in the column are NaN, a run time warning is raised by np.nanmean
    col = np.abs(group[column_name])
    if pd.isna(col).all():
        return np.nan

    return np.nanmean(col)


def calculate_number_of_owners(group: pd.DataFrame) -> int:
    """
    Calculate the number of unique owners within a group.

    Parameters
    ----------
    group : pd.DataFrame
        The group DataFrame

    Returns
    -------
    int
        The number of unique owners within the group
    """

    if any(group["Operator Name"].eq("")):
        msg = "Empty values were detected for column: Operator Name."
        raise_exception(msg, ValueError)

    return len(set(group["Operator Name"]))


# pylint: disable=too-many-locals
def process_data(merged_df: pd.DataFrame, centroids: dict) -> pd.DataFrame:
    """
    Process the merged DataFrame to compute various statistics for each project.

    Parameters
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame containing project data
    centroids : Dict
        A dictionary mapping project names to centroid coordinates

    Returns
    -------
    pd.DataFrame
        A DataFrame containing computed statistics for each project
    """
    cluster_groups = merged_df.groupby("Project")

    average_age = cluster_groups.apply(
        calculate_average, column_name="Age [Years]", include_groups=False
    )

    average_age_df = pd.DataFrame(
        {"Project": average_age.index, "Average Age [Years]": average_age.values}
    )

    average_depth = cluster_groups.apply(
        calculate_average, column_name="Depth [ft]", include_groups=False
    )
    average_depth_df = pd.DataFrame(
        {"Project": average_depth.index, "Average Depth [ft]": average_depth.values}
    )

    age_ranges = cluster_groups.apply(
        calculate_range, column_name="Age [Years]", include_groups=False
    )
    age_ranges_df = pd.DataFrame(
        {"Project": age_ranges.index, "Age Range [Years]": age_ranges.values}
    )

    depth_ranges = cluster_groups.apply(
        calculate_range, column_name="Depth [ft]", include_groups=False
    )
    depth_ranges_df = pd.DataFrame(
        {"Project": depth_ranges.index, "Depth Range [ft]": depth_ranges.values}
    )

    well_number = cluster_groups.apply(calculate_well_number, include_groups=False)
    well_number_df = pd.DataFrame(
        {
            "Project": well_number.index,
            "Average Well Score": well_number.values,
        }
    )

    elevation_average = cluster_groups.apply(
        calculate_average,
        column_name="Elevation Delta [m]",
        estimation_method="yes",
        include_groups=False,
    )
    elevation_average_df = pd.DataFrame(
        {
            "Project": elevation_average.index,
            "Average Elevation Delta [m]": elevation_average.values,
        }
    )

    road_distance_average = cluster_groups.apply(
        calculate_average, column_name="Distance to Road [miles]", include_groups=False
    )
    road_distance_average_df = pd.DataFrame(
        {
            "Project": road_distance_average.index,
            "Average Distance to Road [miles]": road_distance_average.values,
        }
    )

    centroid_distance_average = cluster_groups.apply(
        calculate_average,
        column_name="Distance to Centroid [miles]",
        include_groups=False,
    )
    centroid_distance_average_df = pd.DataFrame(
        {
            "Project": centroid_distance_average.index,
            "Distance to Centroid [miles]": centroid_distance_average.values,
        }
    )

    number_of_owners = cluster_groups.apply(
        calculate_number_of_owners, include_groups=False
    )
    number_of_owners_df = pd.DataFrame(
        {"Project": number_of_owners.index, "Operator Name": number_of_owners.values}
    )

    average_priority_score = cluster_groups.apply(
        calculate_average, column_name="Priority Score [0-100]", include_groups=False
    )
    average_priority_score_df = pd.DataFrame(
        {
            "Project": average_priority_score.index,
            "Average Priority Score [0-100]": average_priority_score.values,
        }
    )

    new_data = {
        "Project": list(centroids.keys()),
        "Project Centroid": list(centroids.values()),
        "Number of Wells": well_number_df["Average Well Score"].tolist(),
        "Number of Unique Owners": number_of_owners_df["Operator Name"].tolist(),
        "Average Elevation Delta [m]": elevation_average_df[
            "Average Elevation Delta [m]"
        ].tolist(),
        "Average Distance to Road [miles]": road_distance_average_df[
            "Average Distance to Road [miles]"
        ].tolist(),
        "Distance to Centroid [miles]": centroid_distance_average_df[
            "Distance to Centroid [miles]"
        ].tolist(),
        "Age Range [Years]": age_ranges_df["Age Range [Years]"].tolist(),
        "Average Age [Years]": average_age_df["Average Age [Years]"].tolist(),
        "Depth Range [ft]": depth_ranges_df["Depth Range [ft]"].tolist(),
        "Average Depth [ft]": average_depth_df["Average Depth [ft]"].tolist(),
        "Impact Score [0-100]": average_priority_score_df[
            "Average Priority Score [0-100]"
        ].tolist(),
    }

    new_df = pd.DataFrame(new_data)
    return new_df
