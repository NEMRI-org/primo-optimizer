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

# Installed libs
import folium
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.map_utils import VisualizeData

# Sample DataFrame for testing
DF_BASE = pd.DataFrame(
    [
        {
            "API Well Number": "31003007660000",
            "Age [Years]": 0,
            "Depth [ft]": 0,
            "Well Type": "Oil",
            "Latitude": 42.07661,
            "Longitude": -77.88081,
            "Project": 1,
        },
        {
            "API Well Number": "31003043620000",
            "Age [Years]": 61,
            "Depth [ft]": 2483,
            "Well Type": "Gas",
            "Latitude": 42.07983,
            "Longitude": -77.76817,
            "Project": 2,
        },
    ],
)


@pytest.fixture
def visualize_data_fixture():
    """Fixture to initialize the VisualizeData class."""
    # Mock data setup for WellData
    data = [
        {"lat": 40.7749, "lon": -74.4194, "id": "Well1", "cluster": "Cluster1"},
        {"lat": 40.6428, "lon": -74.2437, "id": "Well2", "cluster": "Cluster2"},
        {"lat": 40.7128, "lon": -74.0060, "id": "Well3", "cluster": "Cluster3"},
    ]

    column_names = {
        "latitude": "lat",
        "longitude": "lon",
        "well_id": "id",
        "cluster": "cluster",
    }
    well_data_instance = WellData(data, column_names)

    return VisualizeData(
        well_data=well_data_instance,
        state_shapefile_url=(
            "https://gisdata.ny.gov/GISData/State/Civil_Boundaries/"
            "NYS_Civil_Boundaries.shp.zip"
        ),
        state_shapefile_name="NYS_Civil_Boundaries.shp.zip",
        shp_name="Counties_Shoreline.shp",
    )


@pytest.mark.integration
def test_integration_visualize_data(visualize_fixture):
    """Integration test for the end-to-end functionality of VisualizeData methods."""
    # Prepare and visualize data
    shapefile, gdf, map_object = visualize_fixture.visualize_data(
        DF_BASE, well_type="Gas"
    )

    # Check that shapefile and well data were properly integrated
    assert isinstance(
        shapefile, gpd.GeoDataFrame
    ), "Shapefile data should be a GeoDataFrame."
    assert isinstance(
        gdf, gpd.GeoDataFrame
    ), "Well data should be transformed into a GeoDataFrame."
    assert isinstance(
        map_object, folium.Map
    ), "The final output should be a folium Map instance."


@pytest.mark.integration
def test_integration_create_map_with_legend_and_markers(visualize_data):
    """Integration test to check map creation with legends and markers."""
    clusters = {"Cluster1": "blue", "Cluster2": "green", "Cluster3": "red"}

    # Create map with legend and verify that legend appears with expected clusters
    map_obj = visualize_data.create_map_with_legend(clusters=clusters)
    # pylint: disable=protected-access
    legend_html = map_obj.get_root().html._children["legend"].render()
    for cluster, color in clusters.items():
        assert cluster in legend_html, f"Legend for {cluster} is missing."
        assert (
            color in legend_html
        ), f"Color {color} is missing for cluster {cluster} in legend."

    # Add markers and verify marker data
    well_data = [
        {"lat": 40.7749, "lon": -74.4194, "id": "Well1", "color": "blue"},
        {"lat": 40.6428, "lon": -74.2437, "id": "Well2", "color": "green"},
    ]
    map_obj = visualize_data.add_markers_to_map(well_data=well_data)
    # pylint: disable=protected-access
    marker_html = map_obj.get_root().html._children["marker"].render()
    for well in well_data:
        assert well["id"] in marker_html, f"Marker for {well['id']} is missing."
        assert (
            well["color"] in marker_html
        ), f"Color {well['color']} for {well['id']} is missing."


@pytest.mark.integration
def test_integration_get_state_shapefile(visualize_data, mocker):
    """Integration test for loading and handling of the state shapefile."""
    # Mocking shapefile download and reading
    mock_shapefile = gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 1), (1, 0)])]})
    mocker.patch("geopandas.read_file", return_value=mock_shapefile)

    shapefile = visualize_data.get_state_shapefile()
    assert isinstance(
        shapefile, gpd.GeoDataFrame
    ), "Shapefile should load as a GeoDataFrame."
    assert not shapefile.empty, "Shapefile should contain data."
    assert "geometry" in shapefile.columns, "Shapefile must include a geometry column."


@pytest.mark.integration
def test_filter_well_data_by_type(visualize_fixture):
    """Test that well data can be filtered correctly by well type."""
    well_data = DF_BASE.copy()
    filtered_data = visualize_fixture.filter_data_by_type(well_data, well_type="Gas")
    assert not filtered_data.empty, "Filtered data should not be empty."
    assert (
        filtered_data["Well Type"] == "Gas"
    ).all(), "All rows must have the specified well type."


@pytest.mark.integration
def test_initialize_map(visualize_fixture):
    """Test that the map initializes with correct bounds and layers."""
    map_obj = visualize_fixture.initialize_map(
        center=[40.7128, -74.0060], zoom_start=10
    )
    assert isinstance(
        map_obj, folium.Map
    ), "Map object should be a folium.Map instance."
    assert map_obj.location == [40.7128, -74.0060], "Map center location is incorrect."
    assert map_obj.options["zoom"] == 10, "Map zoom level is incorrect."


@pytest.mark.integration
def test_add_clusters_to_map(visualize_fixture):
    """Test adding clusters to the map."""
    clusters = {
        "Cluster1": {"lat": 40.7749, "lon": -74.4194, "color": "blue"},
        "Cluster2": {"lat": 40.6428, "lon": -74.2437, "color": "green"},
    }
    map_obj = visualize_fixture.add_clusters_to_map(clusters)
    assert isinstance(map_obj, folium.Map), "Map should be a folium Map instance."
    # pylint: disable=protected-access
    assert (
        len(map_obj._children) > 0
    ), "Map should contain child elements (e.g., clusters)."


@pytest.mark.integration
def test_add_project_data(visualize_fixture):
    """Test that project data can be added to the visualization."""
    project_data = [
        {"id": 1, "project": "Project A"},
        {"id": 2, "project": "Project B"},
    ]
    updated_data = visualize_fixture.add_project_data(DF_BASE, project_data)
    assert (
        "project" in updated_data.columns
    ), "Project column should be added to the data."
    assert len(updated_data) == len(DF_BASE), "Number of rows should remain consistent."


# pylint: disable=R0903
@pytest.mark.integration
def test_download_shapefile(visualize_fixture, monkeypatch):
    """Test downloading a shapefile."""

    def mock_requests_get():
        class MockResponse:
            """Mock class to test shapefile function"""

            content = b"Fake shapefile content"

        return MockResponse()

    monkeypatch.setattr("requests.get", mock_requests_get)
    shapefile_path = visualize_fixture.download_shapefile()
    assert shapefile_path.endswith(
        ".zip"
    ), "Downloaded shapefile should be a .zip file."


@pytest.mark.integration
def test_extract_shapefile(visualize_fixture, tmpdir):
    """Test extracting a shapefile."""
    zip_path = tmpdir.join("shapefile.zip")
    extracted_path = visualize_fixture.extract_shapefile(zip_path)
    assert extracted_path.exists(), "Extracted shapefile directory should exist."


@pytest.mark.integration
def test_handle_missing_coordinates(visualize_fixture):
    """Test that rows with missing or invalid coordinates are removed."""
    data = DF_BASE.copy()
    data.loc[0, "Latitude"] = None
    cleaned_data = visualize_fixture.handle_missing_coordinates(data)
    assert (
        cleaned_data["Latitude"].notnull().all()
    ), "All rows must have valid latitude values."
    assert (
        cleaned_data["Longitude"].notnull().all()
    ), "All rows must have valid longitude values."


@pytest.mark.integration
def test_generate_legend_html(visualize_fixture):
    """Test that the legend HTML is generated correctly."""
    clusters = {"Cluster1": "blue", "Cluster2": "green"}
    legend_html = visualize_fixture.generate_legend_html(clusters)
    assert "Cluster1" in legend_html, "Legend should include 'Cluster1'."
    assert "blue" in legend_html, "Legend should include the color 'blue'."


@pytest.mark.integration
def test_handle_empty_data(visualize_fixture):
    """Test that methods handle empty well data gracefully."""
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="Input data is empty."):
        visualize_fixture.visualize_data(empty_data)


@pytest.mark.integration
def test_state_shapefile_integration(visualize_fixture, monkeypatch):
    """Test integration of state shapefile with well data."""

    def mock_read_file():
        return gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 1), (1, 0)])]})

    monkeypatch.setattr("geopandas.read_file", mock_read_file)

    shapefile = visualize_fixture.get_state_shapefile()
    integrated_map = visualize_fixture.integrate_shapefile_and_data(shapefile, DF_BASE)
    assert isinstance(
        integrated_map, folium.Map
    ), "Integrated map should be a folium.Map instance."
