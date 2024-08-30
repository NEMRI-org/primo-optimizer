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

# Installed libs
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

# User-defined libs
from primo.utils.map_utils import (
    download_and_unzip_shapefile,
    get_cluster_colors,
    prepare_gdf,
    visualize_data,
    visualize_data_with_clusters,
    visualize_data_with_projects,
)


@pytest.mark.parametrize(
    "input_dict, status",
    [  # Case 1: pass case
        (
            [
                {"Latitude": 40.589335, "Longitude": -79.92741},
                {"Latitude": 40.642804, "Longitude": -79.715295},
            ],
            True,
        ),
        # Case 2: missing data
        (
            [
                {"Latitude": np.nan, "Longitude": -79.92741},
                {"Latitude": 40.642804, "Longitude": -79.715295},
            ],
            True,
        ),
        # Case 3: missing 'Latitude'
        (
            [
                {"Longitude": -79.92741},
                {"Longitude": -79.715295},
            ],
            False,
        ),
        # Add more test cases as needed
    ],
)
def test_prepare_gdf(input_dict, status):
    input_df = pd.DataFrame(input_dict)
    if status:
        assert "geometry" in prepare_gdf(input_df)
    else:
        with pytest.raises(KeyError):
            prepare_gdf(input_df)


@pytest.mark.parametrize(
    "number_of_clusters, cluster_list, return_color_list, status",
    [  # Case 1: pass case
        (
            5,
            [1, 2, 3, 4, 5],
            {1: "red", 2: "blue", 3: "green", 4: "orange", 5: "purple"},
            True,
        ),
        # Case 2: number of cluster is not valid
        (
            -1,
            [1, 2, 3, 4, 5],
            "Error",
            False,
        ),
        # Case 3: str for cluster name
        (
            5,
            ["cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"],
            {
                "cluster 1": "red",
                "cluster 2": "blue",
                "cluster 3": "green",
                "cluster 4": "orange",
                "cluster 5": "purple",
            },
            True,
        ),
        # Case 4: number of cluster given is greater than the number of cluster in the list
        (
            10,
            ["cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"],
            {
                "cluster 1": "red",
                "cluster 2": "blue",
                "cluster 3": "green",
                "cluster 4": "orange",
                "cluster 5": "purple",
            },
            False,
        ),
        # Case 5: number of cluster given is less than the number of cluster in the list
        (
            2,
            ["cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"],
            {
                "cluster 1": "red",
                "cluster 2": "blue",
                "cluster 3": "green",
                "cluster 4": "orange",
                "cluster 5": "purple",
            },
            False,
        ),
    ],
)
def test_get_cluster_colors(
    number_of_clusters, cluster_list, return_color_list, status
):
    if status:
        assert get_cluster_colors(number_of_clusters, cluster_list) == return_color_list
    else:
        with pytest.raises(ValueError):
            get_cluster_colors(number_of_clusters, cluster_list)


DF_BASE = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "cluster": 1,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "cluster": 2,
        },
    ],
)
DF_PRIORITY_SCORE = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Priority Score [0-100]": 50,
            "Latitude": 42.07661,
            "Longitude": -77.88081,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Priority Score [0-100]": 0,
            "Latitude": 42.07983,
            "Longitude": -77.76817,
        },
    ],
)
DF_API = pd.DataFrame(
    [
        {
            "API": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "cluster": "cluster 1",
        },
        {
            "API": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "cluster": "cluster 1",
        },
    ],
)
SHP_FILE_NAME = "NYS_Civil_Boundaries.shp.zip"
SHP_FILE_URL = (
    "https://gisdata.ny.gov/GISData/State/Civil_Boundaries/NYS_Civil_Boundaries.shp.zip"
)
SHP_NAME = "Counties_Shoreline.shp"

SHP_FILE_NAME_WRONG = "Civil_Boundaries.zip"
SHP_FILE_URL_WRONG = (
    "https://gisdata.ny.gov/GISData/State/Civil_Boundaries/Civil_Boundaries.shp.zip"
)
SHP_NAME_WRONG = "Counties_Shoreli.shp"


SHP_FILE_NAME_COUNTY_NAME = "PaCounty2024_05.zip"
SHP_FILE_URL_COUNTY_NAME = (
    "https://www.pasda.psu.edu/download/padot/boundary_layers/PaCounty2024_05.zip"
)
SHP_NAME_COUNTY_NAME = "PaCounty2024_05.shp"


@pytest.mark.parametrize(
    "df, shpfile_name, shpfile_url, shp_name, well_type, status",
    [
        (
            DF_BASE,
            SHP_FILE_NAME,
            SHP_FILE_URL,
            SHP_NAME,
            "Gas",
            True,
        ),  # Case 1: pass case with well type specified where the column name for
        # county_name is NAME
        (
            DF_BASE,
            SHP_FILE_NAME,
            SHP_FILE_URL,
            SHP_NAME,
            None,
            True,
        ),  # Case 2: pass case without well type specified
        (
            DF_BASE,
            SHP_FILE_NAME,
            SHP_FILE_URL,
            SHP_NAME,
            "Deep",
            False,
        ),  # Case 3: wrong well type called
        (
            DF_PRIORITY_SCORE,
            SHP_FILE_NAME,
            SHP_FILE_URL,
            SHP_NAME,
            None,
            True,
        ),  # Case 4: pass case with priority score
        (
            DF_API,
            SHP_FILE_NAME,
            SHP_FILE_URL,
            SHP_NAME,
            None,
            True,
        ),  # Case 5: pass case with well number being called as API
        (
            DF_BASE,
            SHP_FILE_NAME_WRONG,
            SHP_FILE_URL_WRONG,
            SHP_NAME_WRONG,
            "Gas",
            False,
        ),  # Case 6: fail case with wrong shape file name and url
        (
            DF_BASE,
            SHP_FILE_NAME_COUNTY_NAME,
            SHP_FILE_URL_COUNTY_NAME,
            SHP_NAME_COUNTY_NAME,
            "Gas",
            True,
        ),  # Case 7: pass case where the column name for county_name is COUNTY_NAME
    ],
)
def test_visualize_data(df, shpfile_name, shpfile_url, shp_name, well_type, status):
    if status:
        state_shapefile, gdf, map_object = visualize_data(
            df, shpfile_name, shpfile_url, shp_name, well_type
        )
        assert isinstance(state_shapefile, gpd.GeoDataFrame)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert isinstance(map_object, folium.Map)
        assert isinstance(state_shapefile["geometry"], gpd.GeoSeries)
        assert isinstance(gdf["geometry"], gpd.GeoSeries)
    else:
        with pytest.raises(Exception):
            visualize_data(df, shpfile_name, shpfile_url, shp_name, well_type)


@pytest.fixture
def state_shapefile():
    scratch_dir = os.path.join(os.getcwd(), "temp")
    extract_dir = download_and_unzip_shapefile(
        SHP_FILE_NAME, SHP_FILE_URL, scratch_dir, SHP_NAME
    )
    state_shapefile = gpd.read_file(os.path.join(extract_dir, SHP_NAME))
    state_shapefile = state_shapefile.to_crs("EPSG:4269")

    return state_shapefile


STATE_SHAPEFILE_WRONG = gpd.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])


@pytest.mark.parametrize(
    "num_cluster, full_data, status",
    [
        (
            2,
            DF_BASE,
            True,
        ),  # Case 1: pass case
        (
            1,
            DF_BASE,
            False,
        ),  # Case 2: fail case with wrong number of cluster given
        (
            1,
            DF_PRIORITY_SCORE,
            None,
        ),  # Case 3: fail case where cluster information is missing in the data input
        (
            1,
            DF_API,
            True,
        ),  # Case 4: pass case with well number being called as API
    ],
)
@pytest.mark.parametrize(
    "num_cluster_wrong_shapefile, full_data_wrong_shapefile, state_shapefile_wrong_shapefile",
    [
        (
            2,
            DF_BASE,
            STATE_SHAPEFILE_WRONG,
        ),  # Case 5: fail case with wrong state_shapefile
    ],
)
def test_visualize_data_with_clusters(
    num_cluster,
    full_data,
    status,
    state_shapefile,
    num_cluster_wrong_shapefile,
    full_data_wrong_shapefile,
    state_shapefile_wrong_shapefile,
):
    if status:
        map_object = visualize_data_with_clusters(
            num_cluster,
            full_data,
            state_shapefile,
        )
        assert isinstance(map_object, folium.Map)
    elif status is None:
        with pytest.raises(KeyError):
            visualize_data_with_clusters(num_cluster, full_data, state_shapefile)
    else:
        with pytest.raises(ValueError):
            visualize_data_with_clusters(num_cluster, full_data, state_shapefile)

    with pytest.raises(ValueError):
        visualize_data_with_clusters(
            num_cluster_wrong_shapefile,
            full_data_wrong_shapefile,
            state_shapefile_wrong_shapefile,
        )


SELECTED_BASE = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Priority Score [0-100]": 50,
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": 1,
            "Gas [Mcf/Year]": 0,
            "Oil [bbl/Year]": 1000,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Priority Score [0-100]": 0,
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": 1,
            "Gas [Mcf/Year]": 1000,
            "Oil [bbl/Year]": 0,
        },
    ],
)
SELECTED_MISSING_PRIORITY = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": 1,
            "Gas [Mcf/Year]": 0,
            "Oil [bbl/Year]": 1000,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": 2,
            "Gas [Mcf/Year]": 1000,
            "Oil [bbl/Year]": 0,
        },
    ],
)
SELECTED_API = pd.DataFrame(
    [
        {
            "API": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Priority Score [0-100]": 50,
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": "Project 1",
            "Gas [Mcf/Year]": 0,
            "Oil [bbl/Year]": 1000,
        },
        {
            "API": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Priority Score [0-100]": 0,
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": "Project 1",
            "Gas [Mcf/Year]": 1000,
            "Oil [bbl/Year]": 0,
        },
    ],
)
SELECTED_MISSING_DATA = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Priority Score [0-100]": 50,
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": 1,
            "Gas [Mcf/Year]": 0,
            "Oil [bbl/Year]": 1000,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Priority Score [0-100]": 0,
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": 2,
            "Gas [Mcf/Year]": np.nan,
            "Oil [bbl/Year]": 0,
        },
    ],
)
SELECTED_MISSING_PROJECT = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Priority Score [0-100]": 50,
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": 1,
            "Gas [Mcf/Year]": 0,
            "Oil [bbl/Year]": 1000,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Priority Score [0-100]": 0,
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": np.nan,
            "Gas [Mcf/Year]": np.nan,
            "Oil [bbl/Year]": 0,
        },
    ],
)


@pytest.mark.parametrize(
    "selected, status",
    [
        (
            SELECTED_BASE,
            True,
        ),  # Case 1: pass case
        (
            SELECTED_MISSING_PRIORITY,
            False,
        ),  # Case 2: fail case where the 'priority score' column is missing
        (
            SELECTED_API,
            True,
        ),  # Case 3: pass case with well number being called as API and str as project name
        (
            SELECTED_MISSING_DATA,
            True,
        ),  # Case 4: pass case with missing data
        (
            SELECTED_MISSING_PROJECT,
            True,
        ),  # Case 5: fail case with missing data in 'project'
    ],
)
@pytest.mark.parametrize(
    "selected_wrong_shapefile, state_shapefile_wong_shapefile",
    [
        (
            SELECTED_BASE,
            STATE_SHAPEFILE_WRONG,
        ),  # Case 6: fail case with wrong state_shapefile
    ],
)
def test_visualize_data_with_projects(
    selected,
    state_shapefile,
    status,
    selected_wrong_shapefile,
    state_shapefile_wong_shapefile,
):
    if status:
        map_object = visualize_data_with_projects(selected, state_shapefile)
        assert isinstance(map_object, folium.Map)
    else:
        with pytest.raises(KeyError):
            visualize_data_with_projects(selected, state_shapefile)

    with pytest.raises(ValueError):
        visualize_data_with_projects(
            selected_wrong_shapefile, state_shapefile_wong_shapefile
        )
