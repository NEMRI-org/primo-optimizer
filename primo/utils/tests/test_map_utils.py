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
from unittest.mock import MagicMock, patch

# Installed libs
import folium
import geopandas as gpd
import pytest
from shapely.geometry import Point

# User-defined libs
from primo.utils.map_utils import VisualizeData, get_cluster_colors


def test_get_cluster_colors():
    """
    Test the get_cluster_colors function to ensure it returns the correct color mapping
    for the provided clusters.

    It tests the function with 3 clusters and validates if the colors are assigned
    correctly based on the expected output.
    """
    cluster_list = [1, 2, 3]

    expected_output = {1: "red", 2: "blue", 3: "green"}

    result = get_cluster_colors(cluster_list)
    assert result == expected_output


def test_get_state_shapefile():
    """
    Test the _get_state_shapefile method by mocking file handling and shapefile operations
    to avoid local file creation.
    """
    # pylint: disable=protected-access
    # Mock dependencies
    mock_download_file = MagicMock()
    mock_unzip_file = MagicMock()

    # Create a mocked GeoDataFrame with a geometry column and CRS
    mock_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4269")

    with (
        patch("primo.utils.map_utils.download_file", mock_download_file),
        patch("primo.utils.map_utils.unzip_file", mock_unzip_file),
        patch("geopandas.read_file", MagicMock(return_value=mock_gdf)),
    ):

        shpfile_name = "test_shapefile.zip"
        shpfile_url = "http://example.com/shapefile.zip"
        shp_name = "shapefile.shp"

        well_data = MagicMock()
        visualize_data = VisualizeData(well_data, "", "", "")

        result = visualize_data._get_state_shapefile(
            shpfile_name, shpfile_url, shp_name
        )

        # Assertions
        mock_download_file.assert_called_with(
            os.path.join(os.getcwd(), "temp", shpfile_name), shpfile_url
        )
        mock_unzip_file.assert_called()
        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty
        assert (
            result.crs.to_string() == "EPSG:4269"
        )  # Check that the CRS is set correctly


def test_create_map_with_legend():
    """
    Test the _create_map_with_legend method to verify it creates a folium map with
    a title and optional legend, without requiring a shapefile.
    """
    # pylint: disable=protected-access
    # Mock dependencies
    mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
    mock_crs = MagicMock()
    mock_crs.is_projected = False
    mock_gdf.crs = mock_crs
    mock_centroid = MagicMock()
    mock_centroid.y = 10.0
    mock_centroid.x = 20.0
    mock_gdf.centroid = MagicMock(return_value=mock_centroid)

    with (
        patch("primo.utils.map_utils.get_data_as_geodataframe", return_value=mock_gdf),
        patch("os.makedirs"),
        patch("os.path.exists", return_value=True),
    ):
        well_data = MagicMock()
        visualize_data = VisualizeData(
            well_data,
            state_shapefile_url="mock_url",
            state_shapefile_name="mock_name",
            shp_name="mock_shp",
        )

        map_obj = visualize_data._create_map_with_legend(
            legend=True, map_title="Test Map", shapefile=False
        )

        assert isinstance(map_obj, folium.Map)  # Ensure it returns a folium map
        rendered_html = map_obj.get_root().html.render()

        # Check that the map title is correctly included
        assert "<h1" in rendered_html
        assert "Test Map" in rendered_html

        # Check that the legend is correctly included
        assert "o - Gas Well" in rendered_html
        assert "x - Oil Well" in rendered_html


def test_add_well_markers():
    """
    Test the _add_well_markers method to ensure it adds markers for wells of the specified
    type (e.g., Gas) to a folium map.
    """
    # pylint: disable=protected-access
    well_data = MagicMock()
    visualize_data = VisualizeData(well_data, "", "", "")

    map_obj = folium.Map(location=[0, 0], zoom_start=8)
    visualize_data.well_data.data = MagicMock()
    visualize_data.well_data.data.itertuples.return_value = [
        MagicMock(
            **{
                "geometry.y": 1.0,
                "geometry.x": 1.0,
            }
        )
    ]
    visualize_data._add_well_markers(map_obj, well_type_to_plot="Gas")
    assert len(map_obj._children) > 0  # Check if markers are added to the map


def test_add_campaign_markers():
    """
    Test the _add_campaign_markers method to verify it adds markers for wells belonging to
    specific campaigns to a folium map.
    """
    # pylint: disable=protected-access
    well_data = MagicMock()
    visualize_data = VisualizeData(well_data, "", "", "")
    map_obj = folium.Map(location=[0, 0], zoom_start=8)

    visualize_data.well_data.data = MagicMock()
    visualize_data.well_data.data.itertuples.return_value = [
        MagicMock(
            **{
                "geometry.y": 1.0,
                "geometry.x": 1.0,
            }
        )
    ]

    campaign = MagicMock()
    campaign.get_project_id_by_well_id.return_value = 1

    visualize_data._add_campaign_markers(map_obj, campaign)

    assert len(map_obj._children) > 0  # Check if markers are added to the map


def test_visualize_wells():
    """
    Test the visualize_wells method to verify it generates a folium map with well markers
    for a specified well type (e.g., Gas).
    """
    well_data = MagicMock()
    visualize_data = VisualizeData(well_data, "", "", "")

    visualize_data.well_data.data = MagicMock()
    visualize_data.well_data.data.itertuples.return_value = [
        MagicMock(
            **{
                "geometry.y": 1.0,
                "geometry.x": 1.0,
            }
        )
    ]

    map_obj = visualize_data.visualize_wells(well_type_to_plot="Gas", shapefile=False)
    assert isinstance(map_obj, folium.Map)  # Check if a folium map is returned


def test_visualize_campaign():
    """
    Test the visualize_campaign method to ensure it generates a folium map displaying well
    markers and campaign data.
    """
    well_data = MagicMock()
    campaign = MagicMock()
    visualize_data = VisualizeData(well_data, "", "", "")

    visualize_data.well_data.data = MagicMock()
    visualize_data.well_data.data.itertuples.return_value = [
        MagicMock(
            **{
                "geometry.y": 1.0,
                "geometry.x": 1.0,
            }
        )
    ]

    with pytest.raises(ValueError):
        visualize_data.visualize_campaign(campaign=None, shapefile=False)

    map_obj = visualize_data.visualize_campaign(campaign=campaign, shapefile=False)
    assert isinstance(map_obj, folium.Map)  # Check if a folium map is returned
