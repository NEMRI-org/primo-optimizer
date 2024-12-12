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
from typing import Dict

# Installed libs
import folium
import geopandas as gpd
from folium.plugins import BeautifyIcon

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.opt_model.result_parser import Campaign
from primo.utils.census_utils import get_data_as_geodataframe
from primo.utils.download_utils import download_file, unzip_file
from primo.utils.raise_exception import raise_exception


def get_cluster_colors(cluster_list: list) -> Dict[int, str]:
    """Generate a color scheme for clusters."""
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
    return {cluster_list[i]: colors[i % len(colors)] for i in range(len(cluster_list))}


class VisualizeData:
    """
    Class to visualize well data using folium and geopandas.
    """

    def __init__(
        self,
        well_data: WellData,
        state_shapefile_url: str,
        state_shapefile_name: str,
        shp_name: str,
    ):
        """
        Initialize the VisualizeData class.

        Parameters
        ----------
        well_data : WellData
            Instance of the WellData class containing well-related information.

        state_shapefile_url : str
            URL of the state shapefile to download.

        state_shapefile_name : str
            Name of the compressed shapefile to download.

        shp_name : str
            Name of the shapefile to extract.
        """

        self.well_data = well_data
        self.state_shapefile_url = state_shapefile_url
        self.state_shapefile_name = state_shapefile_name
        self.shp_name = shp_name
        self.state_shapefile = None
        self.df = get_data_as_geodataframe(self.well_data)

    def _get_state_shapefile(
        self, shpfile_name: str, shpfile_url: str, shp_name: str
    ) -> gpd.GeoDataFrame:
        """
        Download, unzip, and load the state shapefile into a GeoDataFrame.

        Parameters
        ----------
        shpfile_name : str
            Name of the shapefile archive.

        shpfile_url : str
            URL of the shapefile archive.

        shp_name : str
            Name of the extracted shapefile.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame of the state shapefile.
        """

        scratch_dir = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(scratch_dir):
            os.mkdir(scratch_dir)

        shapefile = os.path.join(scratch_dir, shpfile_name)
        download_file(shapefile, shpfile_url)
        extract_dir = os.path.join(scratch_dir, shp_name)
        unzip_file(shapefile, extract_dir)

        state_shapefile = gpd.read_file(os.path.join(extract_dir, shp_name))
        return state_shapefile.to_crs("EPSG:4269")

    def _create_map_with_legend(
        self,
        legend=False,
        map_title: str = None,
        shapefile=True,
    ) -> folium.Map:
        """
        Create a folium map centered around the region with an optional legend and map title.

        Parameters
        ----------
        legend : bool, optional
            Whether to add a legend to the map. Default is False.

        map_title : str, optional
            Title of the map. Default is None.

        shapefile: bool, optional
            Whether to include a shapefile on the map or not. Default is True

        Returns
        -------
        folium.Map
            Folium map object centered around the region.
        """
        if shapefile and self.state_shapefile is None:
            self.state_shapefile = self._get_state_shapefile(
                self.state_shapefile_name, self.state_shapefile_url, self.shp_name
            )

        if self.state_shapefile is not None:
            if (
                self.state_shapefile.crs is None
                or not self.state_shapefile.crs.is_projected
            ):
                centroid = self.state_shapefile.to_crs("+proj=cea").centroid.to_crs(
                    self.state_shapefile.crs
                )
            else:
                centroid = self.state_shapefile.centroid
        else:
            centroid = None

        if centroid is not None:
            map_center = (centroid.y.mean(), centroid.x.mean())
        else:
            first_well = self.well_data.data.iloc[0]
            map_center = (
                first_well[self.well_data.col_names.latitude],
                first_well[self.well_data.col_names.longitude],
            )

        map_obj = folium.Map(location=map_center, zoom_start=8.2)

        if shapefile and self.state_shapefile is not None:
            folium.GeoJson(self.state_shapefile).add_to(map_obj)

            # Add county names as markers
            for county in self.state_shapefile.itertuples():
                county_name = (
                    getattr(county, "NAME", None)
                    or getattr(county, "County_Nam", None)
                    or getattr(county, "COUNTY_NAM", None)
                )

                if county_name is None:
                    raise_exception(
                        "None of the county name attributes are found.", AttributeError
                    )

                centroid = [county.geometry.centroid.y, county.geometry.centroid.x]
                folium.map.Marker(
                    location=centroid,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size: 11pt; color: black; text-align: center; '
                            f'font-weight: bold;">{county_name}</div>'
                        )
                    ),
                ).add_to(map_obj)

        # Create legend
        if legend is True:
            gas_legend = '<i style="color:red">o - Gas Well</i>'
            oil_legend = '<i style="color:blue">x - Oil Well</i>'
            legend_html = f"""
            <div style="position: fixed;
                        top: 10px; right: 10px; width: 120px; height: 80px;
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color: white;">
            <center><br>{gas_legend}<br>{oil_legend}<br></center>
            </div>
            """
            map_obj.get_root().html.add_child(folium.Element(legend_html))

        if map_title is not None:
            title_html = f'<h1 style="position:absolute;z-index:100000;left:35vw" >{map_title}</h1>'
            map_obj.get_root().html.add_child(folium.Element(title_html))

        return map_obj

    def _add_well_markers(
        self,
        map_obj: folium.Map,
        well_type_to_plot: str = None,
    ) -> None:
        """
        Add markers to a folium map based on the visualization type and well type.

        Parameters
        ----------
        map_obj : folium.Map
            Folium map object to add markers to.

        well_type_to_plot : str, optional
            Type of well to plot ('Gas' or 'Oil'). Default is None.

        Raises
        ------
        ValueError
            If visualize_type is 'project' but campaign is not provided.
        """

        for row in self.df.itertuples():
            well_id = row[self.df.columns.get_loc(self.well_data.col_names.well_id) + 1]
            age = row[self.df.columns.get_loc(self.well_data.col_names.age) + 1]
            depth = row[self.df.columns.get_loc(self.well_data.col_names.depth) + 1]

            popup_text = f"Well ID: {well_id}<br>Age: {age}<br>Depth: {depth}"

            if well_type_to_plot == "Gas":
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=popup_text,
                    fill=True,
                    color="red",
                ).add_to(map_obj)
            elif well_type_to_plot == "Oil":
                icon_cross = BeautifyIcon(
                    icon="times",
                    inner_icon_style="color:blue;font-size:18px;",
                    background_color="transparent",
                    border_color="transparent",
                )
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=popup_text,
                    icon=icon_cross,
                ).add_to(map_obj)
            else:
                # Default marker style for other well types
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=popup_text,
                    fill=True,
                    color="green",
                ).add_to(map_obj)

    def _add_campaign_markers(
        self,
        map_obj: folium.Map,
        campaign: Campaign = None,
    ) -> None:
        """
        Add markers to a folium map to visualize campaigns.

        Parameters
        ----------
        map_obj : folium.Map
            Folium map object to add markers to.

        campaign : Campaign, optional
            Campaign instance to map wells to projects when visualize_type is 'project'.
            Default is None.

        Raises
        ------
        ValueError
            If visualize_type is 'project' but campaign is not provided.
        """

        if campaign is None:
            raise ValueError(
                "A Campaign instance must be provided when visualize_type is 'project'."
            )
        project_ids = set()  # Use a set to avoid duplicates
        for row in self.df.itertuples():
            well_id = row[self.df.columns.get_loc(self.well_data.col_names.well_id) + 1]
            project_id = campaign.get_project_id_by_well_id(well_id)
            if project_id is not None:
                project_ids.add(project_id)

        project_colors = get_cluster_colors(list(project_ids))

        for row in self.df.itertuples():
            well_id = row[self.df.columns.get_loc(self.well_data.col_names.well_id) + 1]
            project_id = campaign.get_project_id_by_well_id(well_id)

            if project_id is not None:
                popup_text = f"Candidate Project: Project {project_id}"
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=popup_text,
                    fill=True,
                    color=project_colors.get(project_id, "gray"),
                ).add_to(map_obj)

    def visualize_wells(
        self,
        well_type_to_plot: str = None,
        legend: bool = True,
        map_title: str = "All MCWs",
        shapefile: bool = True,
    ) -> folium.Map:
        """
        Visualize well data on a folium map.

        Parameters
        ----------
        well_type_to_plot : str, optional
            Type of well to plot ('Gas' or 'Oil'). Default is None.

        legend : bool, optional
            Whether to include a legend on the map. Default is False.

        map_title : str, optional
            Title of the map. Default is None.

        shapefile: bool, optional
            Whether to include a shapefile on the map or not. Default is True

        Returns
        -------
        folium.Map
            Folium map object with visualized well data.
        """
        map_obj = self._create_map_with_legend(
            legend=legend, map_title=map_title, shapefile=shapefile
        )
        self._add_well_markers(map_obj, well_type_to_plot)

        return map_obj

    def visualize_campaign(
        self,
        legend: bool = False,
        map_title: str = "Plugging Campaign",
        campaign: Campaign = None,
        shapefile: bool = True,
    ) -> folium.Map:
        """
        Visualize campaigns on a folium map.

        Parameters
        ----------
        legend : bool, optional
            Whether to include a legend on the map. Default is False.

        map_title : str, optional
            Title of the map. Default is None.

        campaign : Campaign, optional
            Campaign instance to map wells to projects when visualize_type is 'project'.
            Default is None.

        shapefile: bool, optional
            Whether to include a shapefile on the map or not. Default is True

        Returns
        -------
        folium.Map
            Folium map object with visualized campaigns.
        """
        map_obj = self._create_map_with_legend(
            legend=legend, map_title=map_title, shapefile=shapefile
        )
        self._add_campaign_markers(map_obj, campaign)

        return map_obj
