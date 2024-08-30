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
from math import radians
from typing import List

# Installed libs
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# User-defined libs
from primo.utils import EARTH_RADIUS
from primo.utils.elevation_utils import get_elevation
from primo.utils.geo_utils import (
    get_distance,
    is_in_bounds,
    is_valid_lat,
    is_valid_long,
)
from primo.utils.kpi_utils import _is_numeric_valid_column
from primo.utils.raise_exception import raise_exception


def is_valid_age(age: float):
    """
    Check if an age value specified is valid (between 0 and 200).
    200 was chosen as an upper value since the first commercial
    well was drilled in 1859, therefore no well can be older than
    200.
    Parameters
    ----------
    age : float
        The age value in years

    Raises
    ------
    ValueError
        If the argument is not a valid value for age
    """
    if not is_in_bounds(age, "well age", 0, 200, False):
        msg = f"Valid values for age is 0<=age<=200. Provided {age}"
        raise_exception(msg, ValueError)


def _is_valid_lat_and_long_set(
    df: pd.DataFrame, lat_column_name: str, long_column_name: str
):
    """
    Check whether a DataFrame containing floats indicating latitude and longitude
    values respectively is valid.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the latitude and longitude data
    lat_column_name : str
        The column name of for the latitude values
    long_column_name : str
        The column name for the longitude values

    Raises
    ------
    ValueError
        If the argument contains none valid latitude and longitude values
    """

    df[lat_column_name].apply(is_valid_lat, axis=1)
    df[long_column_name].apply(is_valid_long, axis=1)


def age_estimation(
    dataframe: pd.DataFrame, api_col_name="API number", age_col_name="Age", num_values=5
):
    """
    Estimate missing age values in a DataFrame based on the closest non-zero values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing well data
    api_col_name : str, optional
        The name of the column containing API numbers. Default is "API number"
    age_col_name : str, optional
        The name of the column containing age values. Default is 'Age'
    num_values : int, optional
        The number of values to consider above and below for age estimation. Default is five

    Returns
    -------
    pd.DataFrame
        DataFrame with missing age values filled based on the closest non-zero values;
        if closest non-zero values do not exist, returns 0
    """
    # Make sure age is within valid bounds and has valid values

    _is_numeric_valid_column(dataframe, age_col_name, estimation_method="yes")
    dataframe[age_col_name].apply(is_valid_age)

    # Sort the DataFrame by 'API number'
    df = dataframe.sort_values(by=api_col_name)

    # Function to fill missing age values
    def age_estimation_row(row):
        if row[age_col_name] == 0 or np.isnan(row[age_col_name]):
            # Find the closest non-zero values above and below
            above = (
                df[df[api_col_name] > row[api_col_name]][age_col_name]
                .astype(float)
                .replace(to_replace=[0], value=pd.NaT)
                .dropna()
                .head(num_values)
                .tolist()
            )
            below = (
                df[df[api_col_name] < row[api_col_name]][age_col_name]
                .astype(float)
                .replace(to_replace=[0], value=pd.NaT)
                .dropna()
                .tail(num_values)
                .tolist()
            )

            # If both above and below values are available, calculate the average
            if len(above) > 0 and len(below) > 0:
                return np.average(above + below)
            # If only above or below value is available, use that
            elif len(above) > 0:
                return np.average(above)
            elif len(below) > 0:
                return np.average(below)
            # If no valid above or below values are found, return 0
            else:
                return 0
        else:
            return row[age_col_name]

    # Apply the function to fill missing age values
    df[age_col_name] = df.apply(age_estimation_row, axis=1)

    return df


def depth_estimation(
    df,
    tif_file_path,
    lat_col_name="Latitude",
    lon_col_name="Longitude",
    depth_col_name="Depth [ft]",
    distance_miles=0.5,
):
    """
    Fill missing depth values in a DataFrame by estimating values based on nearby wells.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing well data
    lat_col_name : str, optional
        The name of the column containing latitude values. Default is "Latitude"
    lon_col_name : str, optional
        The name of the column containing longitude values. Default is "Longitude"
    depth_col_name : str, optional
        The name of the column containing depth values. Default is "Depth [ft]"
    distance_miles : float
        The distance in miles to consider for nearby wells; default is 0.5 miles
    tif_file_path : str
        The file path for elevation raster files

    Returns
    -------
    pd.DataFrame
        DataFrame with missing depth values filled based on estimates from nearby wells
    """
    # Make sure well depth column has valid values
    _is_numeric_valid_column(df, depth_col_name, estimation_method="yes")

    # Make sure the latitude and longitude columns have valid values.
    _is_valid_lat_and_long_set(df, lat_col_name, lon_col_name)

    df[depth_col_name] = df[depth_col_name].fillna(0)
    df = df.dropna(subset=[lat_col_name, lon_col_name])

    df["lat_rad"] = df[lat_col_name].apply(radians)
    df["long_rad"] = df[lon_col_name].apply(radians)

    coordinates = df[["lat_rad", "long_rad"]].to_numpy()

    ball_tree = BallTree(coordinates, metric="haversine")

    df["neighbors_within_distance"] = df.apply(
        lambda row: ball_tree.query_radius(
            [[row["lat_rad"], row["long_rad"]]], r=distance_miles / EARTH_RADIUS
        )[0],
        axis=1,
    )

    df["num_neighbors_within_distance"] = df["neighbors_within_distance"].apply(len)

    for index, row in df.iterrows():
        if row[depth_col_name] == 0:
            num_neighbors = row["num_neighbors_within_distance"]

            if num_neighbors > 0:
                neighbors_indices = df.iloc[row["neighbors_within_distance"]].index
                neighbors_with_depth = df.loc[neighbors_indices][df[depth_col_name] > 0]

                if not neighbors_with_depth.empty:
                    closest_wells_depths = neighbors_with_depth.apply(
                        lambda x: get_distance(
                            (row[lat_col_name], row[lon_col_name]),
                            (
                                x[lat_col_name],
                                x[lon_col_name],
                            ),
                        ),
                        axis=1,
                    ).nsmallest(10)

                    elevation_differences = get_elevation(
                        df.loc[closest_wells_depths.index, lat_col_name],
                        df.loc[closest_wells_depths.index, lon_col_name],
                        tif_file_path,
                    ) - get_elevation(
                        row[lat_col_name], row[lon_col_name], tif_file_path
                    )
                    # Update the estimated depth with elevation differences before taking the mean
                    estimated_depths = (
                        df.loc[closest_wells_depths.index, depth_col_name].values
                        - elevation_differences
                    )

                    # Take the mean of the updated depths
                    estimated_depth = estimated_depths.mean()
                    df.at[index, depth_col_name] = estimated_depth

    df = df.drop(
        [
            "lat_rad",
            "long_rad",
            "neighbors_within_distance",
            "num_neighbors_within_distance",
        ],
        axis=1,
    )

    return df


def get_record_completeness(
    df: pd.DataFrame,
    criteria_columns: List[str],
) -> pd.DataFrame:
    """
    Calculate the record completeness score for each point
    by summing up the total amount of user defined columns
    that have missing values for each row and subtracting that from the
    total number of user defined columns that must be filled
    in order for a well/point to have a complete record.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame of the input data
    criteria_columns: List
         A list of strings where each entry
         is the name of the column or attribute
         being considered for record completeness

    Return
    ------
    df: pd.DataFrame
        A new DataFrame containing the record completeness score

    """
    col_in_dataframe = df.columns.values.tolist()

    # Making sure all columns in the DataFrame and all str values in
    # criteria_columns list are unique.
    indices_dataframe_columns = [
        i for i, x in enumerate(col_in_dataframe) if col_in_dataframe.count(x) > 1
    ]
    indices_input_columns = [
        i for i, x in enumerate(criteria_columns) if criteria_columns.count(x) > 1
    ]

    # Raise an exception if there are duplicates
    if indices_dataframe_columns != []:
        msg = (
            "There are duplicate columns in the input dataframe"
            "All columns in input datafile must be unique"
        )
        raise_exception(msg, ValueError)

    if indices_input_columns != []:
        msg = (
            "There're duplicate columns in the criteria column list "
            f"at indices: {indices_input_columns}"
        )

        raise_exception(msg, ValueError)

    # converting the lists to dictionaries.
    criteria_columns_dic = {col for col in criteria_columns}
    col_in_dataframe_dic = {col for col in col_in_dataframe}

    # Raise an exception if criteria_columns is not an exact subset of the
    # total list of columns in the DataFrame
    if criteria_columns_dic.issubset(col_in_dataframe_dic) is False:
        msg = (
            "All string entries in the criteria columns must exist in the DataFrame columns"
            "Please make sure entries in the criteria column match the columns in the DataFrame"
            "to proceed with this method"
        )
        raise_exception(msg, ValueError)

    # Constructing record completeness value
    df_filter_data = df.filter(items=criteria_columns)
    # Ensure all empty entires become pd.NA entries.
    df_filter_data = df_filter_data.replace(
        to_replace=["NULL", pd.NaT, pd.NA], value=(pd.NA)
    )

    # Here we calculate the number of columns that
    # have missing entries("isna().sum(axis=1)") for each row.
    # Then .multiply(-1) changes the sign of the value obtained for each row to negative.
    # We then add the number of entries in criteria_columns to those values for each row.
    # (- number of columns that have missing entries[row]) + (number of columns in criteria_columns)
    #  = (the number of columns that have non-missing entries[row]) for all rows in the DataFrame

    df_record_completeness = (
        df_filter_data.isna().sum(axis=1).multiply(-1).add(len(criteria_columns))
    )

    df["record completeness"] = df_record_completeness

    return df
