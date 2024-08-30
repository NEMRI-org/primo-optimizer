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

# Installed libs
import pandas as pd
from sklearn.neighbors import BallTree

# User-defined libs
from primo.utils import EARTH_RADIUS


def calculate_nearby_count(
    well_df,
    target_df,
    well_coordinates,
    target_coordinates,
    distance_column_name,
    distance_miles=1,
):
    """
    Calculate the number of nearby targets within a specified distance using the Haversine formula.

    Parameters
    ----------
    well_df : pd.DataFrame
        The input DataFrame containing well coordinates
    target_df : pd.DataFrame
        The DataFrame containing target coordinates
    well_coordinates : tuple
        The tuple containing column names for well latitude and longitude values
    target_coordinates : tuple
        The tuple containing column names for target latitude and longitude values
    distance_column_name : str
        The name of the column to be added to the DataFrame indicating the number
        of nearby targets within the specified distance for each well
    distance_miles : float, optional
        The distance in miles to consider for nearby targets; default is one mile

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column indicating the number of nearby targets
        within the specified distance for each well
    """

    # Unpack column tuples
    well_lat_col, well_lon_col = well_coordinates
    target_lat_col, target_lon_col = target_coordinates

    # Convert 0 values to NaN
    well_df[[well_lat_col, well_lon_col]] = well_df[
        [well_lat_col, well_lon_col]
    ].replace(0, pd.NaT)
    well_df = well_df.dropna(subset=[well_lat_col, well_lon_col])

    well_df["well_lat_rad"] = well_df[well_lat_col].apply(radians)
    well_df["well_lon_rad"] = well_df[well_lon_col].apply(radians)

    target_df[[target_lat_col, target_lon_col]] = target_df[
        [target_lat_col, target_lon_col]
    ].replace(0, pd.NaT)
    target_df = target_df.dropna(subset=[target_lat_col, target_lon_col])

    target_df["target_lat_rad"] = target_df[target_lat_col].apply(radians)
    target_df["target_lon_rad"] = target_df[target_lon_col].apply(radians)

    well_coordinates = well_df[["well_lat_rad", "well_lon_rad"]].to_numpy()
    target_coordinates = target_df[["target_lat_rad", "target_lon_rad"]].to_numpy()

    well_ball_tree = BallTree(well_coordinates, metric="haversine")

    neighbors_within_distance = well_ball_tree.query_radius(
        target_coordinates, r=distance_miles / EARTH_RADIUS
    )

    well_df[distance_column_name] = 0  # Initialize with zeros

    for neighbors in neighbors_within_distance:
        well_df.loc[well_df.index.isin(neighbors), distance_column_name] += 1

    well_df = well_df.drop(["well_lat_rad", "well_lon_rad"], axis=1)

    return well_df


def nearby_total_school_count(
    well_df,
    school_df,
    well_coordinates=("Well_Latitude", "Well_Longitude"),
    school_coordinates=("School_Latitude", "School_Longitude"),
    distance_miles=1,
):
    """
    Calculate the number of nearby schools (both public and private)
    within a specified distance using Haversine formula.

    Parameters
    ----------
    well_df : pd.DataFrame
        The input DataFrame containing well coordinates
    school_df : pd.DataFrame
        The DataFrame containing school coordinates
    well_coordinates : tuple, optional
        The tuple containing column names for well latitude and longitude values;
        default is ("Well_Latitude", "Well_Longitude")
    school_coordinates : tuple, optional
        The tuple containing column names for school latitude and longitude values;
        default is ("School_Latitude", "School_Longitude")
    distance_miles : float, optional
        The distance in miles to consider for nearby schools; default is one mile

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column 'num_total_schools_within_distance' indicating the number
        of nearby schools (both public and private) within the specified distance for each well
    """

    return calculate_nearby_count(
        well_df,
        school_df,
        well_coordinates,
        school_coordinates,
        "Schools Within Distance",
        distance_miles,
    )


def nearby_hospital_count(
    well_df,
    hospital_df,
    well_coordinates=("Well_Latitude", "Well_Longitude"),
    hospital_coordinates=("Hospital_Latitude", "Hospital_Longitude"),
    distance_miles=1,
):
    """
    Calculate the number of nearby hospitals within a specified distance using the
    Haversine formula.

    Parameters
    ----------
    well_df : pd.DataFrame
        The input DataFrame containing well coordinates
    hospital_df : pd.DataFrame
        The DataFrame containing hospital coordinates
    well_coordinates : tuple, optional
        The tuple containing column names for well latitude and longitude values;
        default is ("Well_Latitude", "Well_Longitude")
    hospital_coordinates : tuple, optional
        The tuple containing column names for hospital latitude and longitude values;
        default is ("Hospital_Latitude", "Hospital_Longitude")
    distance_miles : float, optional
        The distance in miles to consider for nearby hospitals; default is one mile

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column 'num_hospitals_within_distance' indicating the number
        of nearby hospitals within the specified distance for each well
    """

    return calculate_nearby_count(
        well_df,
        hospital_df,
        well_coordinates,
        hospital_coordinates,
        "Hospitals Within Distance",
        distance_miles,
    )
