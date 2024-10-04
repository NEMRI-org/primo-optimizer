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
import os
from typing import Dict, Tuple

# Installed libs
import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import BeautifyIcon

# User-defined libs
from primo.utils.download_utils import download_file, unzip_file
from primo.utils.raise_exception import raise_exception


def add_shapefile_to_map(
    map_obj: folium.Map, state_shapefile: gpd.GeoDataFrame
) -> None:
    """
    Add a GeoDataFrame as a GeoJson layer to a folium map.

    Parameters
    ----------
    map_obj : folium.Map
        Folium map object

    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data

    Returns
    -------
    None
    """

    folium.GeoJson(state_shapefile).add_to(map_obj)


def add_county_names_to_map(
    map_obj: folium.Map, state_shapefile: gpd.GeoDataFrame
) -> None:
    """
    Add county names as markers to a folium map.

    Parameters
    ----------
    map_obj : folium.Map
        Folium map object

    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data with county names

    Returns
    -------
    None
    """
    for county in state_shapefile.itertuples():
        # TODO: Need to a more general method for taking the county name when the column is called with names
        # different from "NAME" and "COUNTY_NAME".
        try:
            county_name = county.NAME  # First case
        except AttributeError:
            county_name = county.COUNTY_NAM  # Second case
        # Project county to a flat project before taking centroid
        centroid = [county.geometry.centroid.y, county.geometry.centroid.x]
        folium.map.Marker(
            location=centroid,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 11pt; color: black; text-align: center; font-weight: bold;">{county_name}</div>'
            ),
        ).add_to(map_obj)


def common_visualization(
    map_obj: folium.Map, state_shapefile: gpd.GeoDataFrame
) -> None:
    """
    Add common visualization elements to the map.

    Parameters
    ----------
    map_obj : folium.Map
        Folium map object

    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data

    Returns
    -------
    None
    """

    add_shapefile_to_map(map_obj, state_shapefile)
    add_county_names_to_map(map_obj, state_shapefile)


def download_and_unzip_shapefile(
    shpfile_name: str, shpfile_url: str, scratch_dir: str, shp_name: str
) -> str:
    """
    Download and unzip a shapefile.

    Parameters
    ----------
    shpfile_name : str
        Name of the compressed shapefile to download

    shpfile_url : str
        URL of the shapefile to download

    scratch_dir : str
        Directory to save and extract the shapefile

    shp_name : str
        Name of the shapefile directory

    Returns
    -------
    str
        Directory where the shapefile is extracted
    """

    # Create scratch directory if does not already exist
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    shapefile = os.path.join(scratch_dir, shpfile_name)
    download_file(shapefile, shpfile_url)
    extract_dir = os.path.join(scratch_dir, shp_name)
    unzip_file(shapefile, extract_dir)
    return extract_dir


def prepare_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Prepare a GeoDataFrame from a DataFrame with latitude and longitude.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with point geometries
    """

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4269",
    )
    gdf.index = gdf.index + 1
    return gdf


def get_mean_centroid(data_frame: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Returns the mean centroid of a Geopandas DataFrame

    Parameters
    ----------
    data_frame: gpd.GeoDataFrame
        GeoDataFrame

    Returns
    -------
    tuple[float, float]
        Returns the mean y and x coordinates of the centroid
    """
    if data_frame.crs is None or not data_frame.crs.is_projected:
        # If a geographic CRS is used, the results from centroid are likely incorrect
        # Hence convert to a projected CRS (We use cea here) before calculating centroid
        centroid = data_frame.to_crs("+proj=cea").centroid.to_crs(data_frame.crs)
    else:
        centroid = data_frame.centroid
    return [centroid.y.mean(), centroid.x.mean()]


def create_map_with_legend(state_shapefile: gpd.GeoDataFrame) -> folium.Map:
    """
    Create a folium map centered around the region with a legend.

    Parameters
    ----------
    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data

    Returns
    -------
    folium.Map
        Folium map object
    """

    # map_center = [state_shapefile.centroid.y.mean(), state_shapefile.centroid.x.mean()]
    map_center = get_mean_centroid(state_shapefile)
    map_obj = folium.Map(location=map_center, zoom_start=8.2)

    gas_legend = '<i style="color:red">o - Gas Well</i>'
    oil_legend = '<i style="color:blue">x - Oil Well</i>'
    legend_html = f"""
    <div style="position: fixed;
                 top: 10px; right: 10px; width: 120px; height: 80px;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color: white;
                 ">
      <center>
      <br>
      {gas_legend}<br>
      {oil_legend}<br>
      </center>
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(legend_html))
    return map_obj


def add_well_markers_to_map(
    gdf: gpd.GeoDataFrame, map_obj: folium.Map, well_type_to_plot: str = None
) -> None:
    """
    Add well markers to a folium map based on well type.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing well data

    map_obj : folium.Map
        Folium map object

    well_type_to_plot : str
        The specific well type to plot ("Oil" or "Gas"). If None, both well types
        are plotted

    Returns
    -------
    None

    Raises
    ------
    ValueError if well_type_to_plot takes value other than "Oil", "Gas" or None
    """
    if well_type_to_plot not in ("Oil", "Gas") and well_type_to_plot is not None:
        raise_exception(f"Unknown well type: {well_type_to_plot}", ValueError)

    age_column_index = gdf.columns.get_loc("Age [Years]") + 1
    depth_column_index = gdf.columns.get_loc("Total Depth [ft]") + 1
    try:
        well_id_index = gdf.columns.get_loc("Well Identifier") + 1
    except KeyError:
        well_id_index = gdf.columns.get_loc("API") + 1

    total_score_index = None  # Set default value to None

    if "Priority Score [0-100]" in gdf.columns:
        total_score_index = gdf.columns.get_loc("Priority Score [0-100]") + 1
    for row in gdf.itertuples():
        if row.geometry.is_empty:
            continue

        well_id = row[well_id_index]
        age = row[age_column_index]
        depth = row[depth_column_index]

        if total_score_index is not None:
            total_score = round(row[total_score_index], 2)
            popup_text = f"API: {well_id}<br>Age: {age}<br>Depth: {depth}<br>Well Type: {well_type_to_plot}<br>Impact score: {total_score}"
        else:
            popup_text = f"API: {well_id}<br>Age: {age}<br>Depth: {depth}<br>Well Type: {well_type_to_plot}"

        if well_type_to_plot == "Gas":
            icon = folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=popup_text,
                fill=True,
                color="red",
            )
            icon.add_to(map_obj)
        elif well_type_to_plot == "Oil":
            icon_cross = BeautifyIcon(
                icon="times",
                inner_icon_style="color:blue;font-size:18px;",  # Adjust size here
                background_color="transparent",
                border_color="transparent",
            )
            icon = folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=popup_text,
                icon=icon_cross,
            )
            icon.add_to(map_obj)
        else:
            continue  # Skip points with other well types


def visualize_data(
    df: pd.DataFrame,
    shpfile_name: str,
    shpfile_url: str,
    shp_name: str,
    well_type: str = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, folium.Map]:
    """
    Visualize data by adding well markers to a folium map.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the well data

    shpfile_name : str
        Name of the compressed shapefile

    shpfile_url : str
        URL to download the shapefile

    shp_name : str
        Name of the shapefile

    well_type : str, defaults to None
        The well_type to be plotted ("Oil" or "Gas"). If None,
        both well types are plotted

    Returns
    -------
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, folium.Map]
        Tuple containing the state shapefile GeoDataFrame, the well GeoDataFrame, and the folium map
    """

    scratch_dir = os.path.join(os.getcwd(), "temp")
    extract_dir = download_and_unzip_shapefile(
        shpfile_name, shpfile_url, scratch_dir, shp_name
    )
    state_shapefile = gpd.read_file(os.path.join(extract_dir, shp_name))
    state_shapefile = state_shapefile.to_crs("EPSG:4269")
    gdf = prepare_gdf(df)
    map_obj = create_map_with_legend(state_shapefile)
    common_visualization(map_obj, state_shapefile)
    add_well_markers_to_map(gdf, map_obj, well_type)
    return state_shapefile, gdf, map_obj


def get_cluster_colors(num_cluster: int, cluster_list: list) -> Dict[int, str]:
    """
    Get a color scheme for clusters.

    Parameters
    ----------
    num_cluster : int
        Number of clusters

    Returns
    -------
    Dict[int, str]
        Dictionary mapping cluster numbers to colors

    Raises
    ------
    ValueError if value of num_cluster is invalid
    """
    if num_cluster <= 0:
        raise_exception(
            f"Invalid value: {num_cluster} for number of clusters", ValueError
        )
    if num_cluster != len(cluster_list):
        raise_exception(
            f"The specified number of clusters does not match the actual number of clusters present in the provided data",
            ValueError,
        )
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "yellow",
        "cyan",
        "magenta",
        "pink",
        "brown",
        "black",
    ]
    return {cluster_list[i]: colors[i % len(colors)] for i in range(num_cluster)}


def add_cluster_markers_to_map(
    full_data_points: gpd.GeoDataFrame,
    map_obj: folium.Map,
    cluster_colors: Dict[int, str],
) -> None:
    """
    Add cluster markers to a folium map.

    Parameters
    ----------
    full_data_points : gpd.GeoDataFrame
        GeoDataFrame containing well data with clusters.

    map_obj : folium.Map
        Folium map object.

    cluster_colors : Dict[int, str]
        Dictionary mapping clusters to colors.

    Returns
    -------
    None
    """

    age_index = full_data_points.columns.get_loc("Age [Years]") + 1
    depth_index = full_data_points.columns.get_loc("Total Depth [ft]") + 1
    try:
        well_id_index = full_data_points.columns.get_loc("Well Identifier") + 1
    except KeyError:
        well_id_index = full_data_points.columns.get_loc("API") + 1

    for row in full_data_points.itertuples():
        cluster = getattr(row, "Clusters", None)  # Use 'Clusters' instead of 'cluster'
        if pd.isna(cluster):
            continue
        latitude = row.geometry.y
        longitude = row.geometry.x
        age = row[age_index]
        depth = row[depth_index]
        well_id = row[well_id_index]
        popup_text = (
            f"Project: {cluster}<br>Well ID: {well_id}<br>Latitude: {latitude}"
            f"<br>Longitude: {longitude}<br>Depth: {depth}<br>Age: {age}"
        )
        color = cluster_colors.get(
            cluster, "gray"
        )  # Use 'gray' if cluster not in color scheme

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=8,
            popup=popup_text,
            fill=True,
            color=color,
        ).add_to(map_obj)


def visualize_data_with_clusters(
    num_cluster: int, full_data: pd.DataFrame, state_shapefile: gpd.GeoDataFrame
) -> folium.Map:
    """
    Visualize data with clusters by adding cluster markers to a folium map.

    Parameters
    ----------
    num_cluster : int
        Number of clusters already existing in the full_data
    full_data : pd.DataFrame
        DataFrame containing the well data
    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data

    Returns
    -------
    folium.Map
        Folium map object.
    """

    geometry = gpd.points_from_xy(full_data["Longitude"], full_data["Latitude"])
    full_data_points = gpd.GeoDataFrame(full_data, geometry=geometry, crs="EPSG:4326")

    map_obj = folium.Map(
        location=get_mean_centroid(state_shapefile),
        zoom_start=8.2,
    )
    common_visualization(map_obj, state_shapefile)
    cluster_list = pd.unique(full_data["Clusters"])
    cluster_colors = get_cluster_colors(num_cluster, cluster_list)
    add_cluster_markers_to_map(full_data_points, map_obj, cluster_colors)
    return map_obj


def add_project_markers_to_map(
    selected: pd.DataFrame, map_obj: folium.Map, cluster_colors: Dict[str, str]
) -> None:
    """
    Add project markers to a folium map.

    Parameters
    ----------
    selected : pd.DataFrame
        DataFrame containing selected project data
    map_obj : folium.Map
        Folium map object
    cluster_colors : Dict[str, str]
        Dictionary mapping projects to colors

    Returns
    -------
    None
    """
    gas_index = selected.columns.get_loc("Gas [Mcf/Year]") + 1
    oil_index = selected.columns.get_loc("Oil [bbl/Year]") + 1
    score_index = selected.columns.get_loc("Priority Score [0-100]") + 1
    for row in selected.itertuples():
        if pd.isna(row.Project):
            continue
        gas = row[gas_index]
        oil = row[oil_index]
        cluster = "Project: " + str(row.Project)
        score = row[score_index]
        popup_text = (
            f"Oil [bbl/day]: {oil}<br>Gas [Mcf/day]: {gas}<br>"
            f"Candidate Project: {cluster}<br>Score: {score}<br>"
        )
        color = cluster_colors.get(
            row.Project, "gray"
        )  # Use 'gray' if cluster not in color scheme
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=5,
            popup=popup_text,
            fill=True,
            color=color,
        ).add_to(map_obj)


def visualize_data_with_projects(
    selected: pd.DataFrame, state_shapefile: gpd.GeoDataFrame
) -> folium.Map:
    """
    Visualize data with projects by adding project markers to a folium map.

    Parameters
    ----------
    selected : pd.DataFrame
        DataFrame containing selected project data
    state_shapefile : gpd.GeoDataFrame
        GeoDataFrame containing shapefile data

    Returns
    -------
    folium.Map
        Folium map object
    """
    map_obj = folium.Map(
        location=get_mean_centroid(state_shapefile),
        zoom_start=7,
    )
    common_visualization(map_obj, state_shapefile)
    cluster_list = pd.unique(selected["Project"])
    num_cluster = len(cluster_list)
    cluster_colors = get_cluster_colors(num_cluster, cluster_list)
    add_project_markers_to_map(selected, map_obj, cluster_colors)
    return map_obj
