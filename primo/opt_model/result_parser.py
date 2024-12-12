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
import logging
from typing import List, Optional, Union

# Installed libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User-defined libs
from primo.data_parser import EfficiencyMetrics
from primo.data_parser.well_data import WellData
from primo.utils.kpi_utils import calculate_average

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class Project:
    """
    Class for storing optimal projects
    """

    def __init__(
        self, wd: WellData, index: list, plugging_cost: float, project_id: int
    ):
        """
        Constructs an object for storing optimal project results
        Parameters
        ----------
        wd : WellData
            A WellData object

        index : list
            List of indices/rows/wells belonging to the cluster/group

        plugging_cost : float
            Cost of plugging

        project_id : int
            Project id
        """
        # Motivation for storing the entire DataFrame: After the problem is solved
        # it is desired to display/highlight flagged wells for which the information
        # was not available, and the missing data was filled. Having the entire
        # DataFrame allows ease of access to flagged wells (columns containing _flag)
        self.well_data = wd._construct_sub_data(index)
        self._col_names = self.well_data.col_names
        col_names = self._col_names
        self.project_id = project_id

        # Must display essential columns while printing a DataFrame
        self._essential_cols = [
            self.col_names.well_id,
            self.col_names.latitude,
            self.col_names.longitude,
            self.col_names.age,
            self.col_names.depth,
            self.col_names.priority_score,
        ]
        self._priority_score_cols = wd.get_priority_score_columns
        self._flag_cols = wd.get_flag_columns
        self.num_wells = len(index)
        # Optimization problem uses million USD. Convert it to USD
        self.plugging_cost = plugging_cost * 1e6
        self.efficiency_score = 0
        accessibility_column_attr = ["elevation_delta", "dist_to_road"]
        self.accessibility_attr = [
            attribute
            for attribute in accessibility_column_attr
            if hasattr(col_names, attribute)
        ]

    def __iter__(self):
        return iter(self.well_data.data.index)

    def __str__(self) -> str:
        msg = (
            f"Number of wells in project {self.project_id}\t\t: {self.num_wells}\n"
            f"Estimated Project Cost\t\t\t: ${round(self.plugging_cost)}\n"
            f"Impact Score [0-100]\t\t\t: {self.impact_score:.2f}\n"
            f"Efficiency Score [0-100]\t\t: {self.efficiency_score:.2f}\n"
        )
        return msg

    def _check_column_exists(self, col_name):
        """
        Checks if a column exists
        """
        if col_name is None:
            raise ValueError("The column is not in the welldatacolumns class")

    @property
    def num_wells_near_hospitals(self):
        """Returns number of wells that are near hospitals"""
        col_name = self._col_names.hospitals
        self._check_column_exists(col_name)
        return len(self.well_data[self.well_data[col_name] > 0].index)

    @property
    def num_wells_near_schools(self):
        """Returns number of wells that are near schools"""
        col_name = self._col_names.schools
        self._check_column_exists(col_name)
        return len(self.well_data[self.well_data[col_name] > 0].index)

    @property
    def average_age(self):
        """
        Returns average age of the wells in the project
        """
        return self.well_data[self._col_names.age].mean()

    @property
    def age_range(self):
        """
        Returns the range of the age of the project
        """
        return (
            self.well_data[self._col_names.age].max()
            - self.well_data[self._col_names.age].min()
        )

    @property
    def average_depth(self):
        """
        Returns the average depth of the project
        """
        return self.well_data[self._col_names.depth].mean()

    @property
    def depth_range(self):
        """
        Returns the range of the depth of the project
        """
        return (
            self.well_data[self._col_names.depth].max()
            - self.well_data[self._col_names.depth].min()
        )

    @property
    def avg_elevation_delta(self):
        """
        Returns the average elevation delta of the project
        """
        col_name = self._col_names.elevation_delta
        self._check_column_exists(col_name)
        return calculate_average(
            self.well_data.data,
            self._col_names.elevation_delta,
            estimation_method="yes",
        )

    @property
    def centroid(self):
        """
        Returns the centroid of the project
        """
        return tuple(
            np.round(
                np.mean(
                    self.well_data.data[
                        [self._col_names.latitude, self._col_names.longitude]
                    ].values,
                    axis=0,
                ),
                6,
            )
        )

    @property
    def avg_dist_to_road(self):
        """
        Returns the average distance to road for a project
        """
        col_name = self._col_names.dist_to_road
        self._check_column_exists(col_name)
        return calculate_average(self.well_data.data, self._col_names.dist_to_road)

    @property
    def col_names(self):
        """
        Returns column names associated with project data
        """
        return self._col_names

    @property
    def essential_cols(self):
        """
        Returns the minimal essential/necessary cols associated with a project
        """
        return self._essential_cols

    @property
    def num_unique_owners(self):
        """
        Returns the number of well owners for a project
        """
        col_name = self._col_names.operator_name
        self._check_column_exists(col_name)
        return len(set(self.well_data[col_name].values))

    @property
    def impact_score(self):
        """
        Returns the average priority score for the project
        """
        if not hasattr(self._col_names, "priority_score"):
            raise AttributeError(
                "The priority score has not been computed for the Well Data"
            )
        return self.well_data[self._col_names.priority_score].mean()

    @property
    def accessibility_score(self):
        """
        Returns the accessibility score and the total weight of the accessibility
        score for a project
        """
        names_attributes = [
            attribute_name
            for attribute_name in dir(self)
            if "eff_score" in attribute_name
        ]
        names_attributes_accessibility = [
            efficiency_score_name
            for efficiency_score_name in names_attributes
            if any(
                accessibility_score_name in efficiency_score_name
                for accessibility_score_name in self.accessibility_attr
            )
        ]
        if len(names_attributes_accessibility) == 0:
            return None
        total_weight = sum(
            int(name.split("_0_")[-1]) for name in names_attributes_accessibility
        )
        return total_weight, sum(
            getattr(self, attribute) for attribute in names_attributes_accessibility
        )

    def get_max_val_col(self, col_name) -> float:
        """
        Returns the maximum value of a column

        Parameters
        ----------
        col_name : str
            column name

        Returns
        -------
        float
            the maximum value of the column

        """
        return self.well_data[col_name].max()

    def update_efficiency_score(self, value: Union[int, float]):
        """
        Updates the efficiency score of a project

        Parameters
        ----------
        value : Union[int, float]
            The value to update the efficiency score with
        """
        self.efficiency_score += value

    def get_well_info_dataframe(self):
        """
        Returns the data frame to display in the notebook
        """
        if self.well_data.config.verify_operator_name:
            self._essential_cols.insert(1, self._col_names.operator_name)
        return self.well_data[self._essential_cols]


class Campaign:
    """
    Represents an optimal campaign that consists of multiple projects.
    """

    def __init__(self, wd: WellData, clusters_dict: dict, plugging_cost: dict):
        """
        Represents an optimal campaign that consists of multiple projects.

        Parameters
        ----------
        wd : WellData
            WellData object

        clusters_dict : dict
            A dictionary where keys are cluster numbers and values
            are list of wells for each cluster.

        plugging_cost : dict
            A dictionary where keys are cluster numbers and values
            are plugging cost for that cluster
        """
        # for now include a pointer to well data, so that I have column names
        self.wd = wd
        self.projects = {}
        self.clusters_dict = clusters_dict

        index = 1
        for cluster, wells in self.clusters_dict.items():
            self.projects[cluster] = Project(
                wd=wd,
                index=wells,
                plugging_cost=plugging_cost[cluster],
                project_id=cluster,
            )
            index += 1

        self.num_projects = len(self.projects)
        self.efficiency_calculator = EfficiencyCalculator(self)

    def get_project_id_by_well_id(self, well_id: str) -> Optional[int]:
        """
        Returns the project_id associated with the given well_id.

        Parameters
        ----------
        well_id : str
            The ID of the well.

        Returns
        -------
        Optional[int]
            The project ID if the well exists in any project; otherwise, None.
        """
        for project_id, project in self.projects.items():
            if well_id in project.well_data.data[self.wd.col_names.well_id].values:
                return project_id
        return None

    def __str__(self) -> str:
        msg = (
            f"The optimal campaign has {self.num_projects} projects.\n"
            f"The total cost of the campaign is ${round(self.total_plugging_cost)}\n\n"
        )
        for _, project in self.projects.items():
            msg += str(project)
            msg += "\n"

        return msg

    @property
    def total_plugging_cost(self):
        """
        Returns the total plugging cost of the campaign
        """
        return sum(project.plugging_cost for _, project in self.projects.items())

    def get_max_value_across_all_projects(self, attribute: str) -> Union[float, int]:
        """
        Returns the max value for an attribute across projects

        Parameters
        ----------
        attribute : str
            name of the attribute of interest
        """
        if not hasattr(next(iter(self.projects.values())), attribute):
            raise AttributeError(
                "The project does not have the requested attribute: " + attribute
            )
        return max(getattr(project, attribute) for _, project in self.projects.items())

    def get_min_value_across_all_projects(self, attribute: str) -> Union[float, int]:
        """
        Returns the min value for an attribute across projects

        Parameters
        ----------
        attribute : str
            name of the attribute of interest

        """
        if not hasattr(next(iter(self.projects.values())), attribute):
            raise AttributeError(
                "The project does not have the requested attribute: " + attribute
            )
        return min(getattr(project, attribute) for _, project in self.projects.items())

    def get_max_value_across_all_wells(self, col_name: str) -> Union[float, int]:
        """
        Returns the max value for all wells in the data set

        Parameters
        ----------
        col_name : str
            name of the column containing the values of interest

        """
        return max(self.wd[col_name].values)

    def get_min_value_across_all_wells(self, col_name: str) -> Union[float, int]:
        """
        Returns the max value for all wells in the data set

        Parameters
        ----------
        col_name : str
            name of the column containing the values of interest
        """
        return min(self.wd[col_name].values)

    def plot_campaign(self, title: str):
        """
        Plots the projects of the campaign

        Parameters
        ---------
        title : str
            Title for the plot
        """
        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=[
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
        )
        plt.figure()
        ax = plt.gca()
        for _, project in self.projects.items():
            ax.scatter(
                project.well_data[project.col_names.longitude],
                project.well_data[project.col_names.latitude],
            )
        plt.title(title)
        plt.xlabel("x-coordinate of wells")
        plt.ylabel("y-coordinate of wells")
        plt.show()

    def get_project_well_information(self):
        """
        Returns a dict of DataFrames corresponding to each project containing essential data
        """
        return {
            project.project_id: project.get_well_info_dataframe()
            for _, project in self.projects.items()
        }

    def get_efficiency_score_project(self, project_id: int) -> float:
        """
        Returns the efficiency score of a project in the campaign given its id
        Parameters
        ----------
        project_id : int
            Project id

        Returns
        -------
        float
            The efficiency score of the project
        """
        return self.projects[project_id].efficiency_score

    def get_impact_score_project(self, project_id: int) -> float:
        """
        Returns the impact score of a project in the campaign given its id
        Parameters
        ----------
        project_id : int
            Project id

        Returns
        -------
        float
            The impact score of the project
        """
        return self.projects[project_id].impact_score

    def _extract_column_header_for_efficiency_metrics(self, attribute_name: str):
        """
        Returns a string with the appropriate column header
        """
        # the format of the string is name_eff_score_0_10
        upper_range = attribute_name.split("0_")[-1]
        name = attribute_name.split("eff_score")[0][:-1]
        name = [word.capitalize() for word in name.split("_")]
        name = " ".join(name)
        return f"{name} Score [0-{upper_range}]"

    def get_efficiency_metrics(self):
        """
        Returns a data frame with the different efficiency scores for the projects
        """
        # TODO What to do with single well projects

        project_column = [project.project_id for _, project in self.projects.items()]
        first_key = list(self.projects.keys())[0]
        names_attributes = [
            attribute_name
            for attribute_name in dir(self.projects[first_key])
            if "eff_score" in attribute_name
        ]

        attribute_data = [
            [getattr(project, attribute) for _, project in self.projects.items()]
            for attribute in names_attributes
        ]

        header = ["Project ID"] + [
            self._extract_column_header_for_efficiency_metrics(attr)
            for attr in names_attributes
        ]

        if self.projects[first_key].accessibility_score is not None:
            # accessibility score
            total_weights, accessibility_data = map(
                list,
                zip(
                    *[
                        project.accessibility_score
                        for _, project in self.projects.items()
                    ]
                ),
            )
            efficiency_scores = [
                project.efficiency_score for _, project in self.projects.items()
            ]

            data = list(
                zip(
                    project_column,
                    *attribute_data,
                    accessibility_data,
                    efficiency_scores,
                )
            )
            header.append(f"Accessibility Score [0-{total_weights[0]}]")
            header.append("Efficiency Score [0-100]")
        # if there is data for the accessibility score
        else:
            efficiency_scores = [
                project.efficiency_score for _, project in self.projects.items()
            ]
            header.append("Efficiency Score [0-100]")
            data = list(zip(project_column, *attribute_data, efficiency_scores))

        return pd.DataFrame(data, columns=header)

    def get_campaign_summary(self):
        """
        Returns a pandas data frame of the project summary for demo printing
        """
        rows = [
            [
                project.project_id,
                project.num_wells,
                project.impact_score,
                project.efficiency_score,
            ]
            for _, project in self.projects.items()
        ]
        header = [
            "Project ID",
            "Number of Wells",
            "Impact Score [0-100]",
            "Efficiency Score [0-100]",
        ]
        return pd.DataFrame(rows, columns=header)

    def export_data(
        self,
        excel_writer: pd.ExcelWriter,
        campaign_category: str,
        columns_to_export: list = None,
    ):
        """
        Exports campaign data to an excel file

        Parameters
        ----------
        excel_writer : pd.ExcelWriter
            The excel writer
        campaign_category : str
            The label for the category of the campaign (e.g., "oil", "gas")
        """
        first_key = list(self.projects.keys())[0]
        col_names = self.projects[first_key].col_names
        # the priority score must have been previously computed
        assert hasattr(col_names, "priority_score")
        if columns_to_export is None:
            columns_to_export = self.wd.col_names.values()

        # add the project data
        start_row = 0
        for _, project in self.projects.items():
            wells_df = project.well_data.data[columns_to_export].copy()
            wells_df["Project ID"] = pd.Series(
                [project.project_id] * len(wells_df), index=wells_df.index
            )
            cols = list(wells_df.columns)
            cols.insert(0, cols.pop(cols.index("Project ID")))
            wells_df = wells_df.loc[:, cols]
            wells_df.rename(
                columns={col_names.priority_score: "Well Priority Score [0-100]"}
            )
            wells_df.to_excel(
                excel_writer,
                sheet_name=campaign_category + " Well Projects",
                startrow=start_row,
                startcol=0,
                index=False,
            )
            start_row += len(wells_df) + 2  # Add one line spacing after each table

        # add the campaign summary
        self.get_campaign_summary().to_excel(
            excel_writer, sheet_name=campaign_category + "Project Scores", index=False
        )

    def set_efficiency_weights(self, eff_metrics: EfficiencyMetrics):
        """
        Wrapper for the EfficiencyCalculator set_efficiency_weights method
        """
        self.efficiency_calculator.set_efficiency_weights(eff_metrics)

    def compute_efficiency_scores(self):
        """
        Wrapper for the EfficiencyCalculator compute efficiency scores method
        """
        self.efficiency_calculator.compute_efficiency_scores()


class EfficiencyCalculator:
    """
    A class to compute efficiency scores for projects.
    """

    def __init__(self, campaign: Campaign):
        """
        Constructs the object for all of the efficiency computations for a campaign

        Parameters
        ----------

        campaign : Campaign
            The final campaign for efficiencies to be computed

        """
        self.campaign = campaign
        self.efficiency_weights = None

    def set_efficiency_weights(self, eff_metrics: EfficiencyMetrics):
        """
        Sets the attribute containing the efficiency weights
        """
        self.efficiency_weights = eff_metrics

    def compute_efficiency_attributes_for_all_projects(self):
        """
        Computes efficiency attributes for all the projects in the campaign
        """
        for _, project in self.campaign.projects.items():
            LOGGER.info(f"Computing efficiency scores for project {project.project_id}")
            self.compute_efficiency_attributes_for_project(project)

    def compute_efficiency_attributes_for_project(self, project: Project):
        """
        Adds attributes to each project object with the metric efficiency score

        Parameters
        ----------
        project : Project
            project in an Campaign

        """
        assert self.efficiency_weights is not None
        for metric in self.efficiency_weights:
            if metric.weight == 0 or hasattr(metric, "submetrics"):
                # Metric/submetric is not chosen, or
                # This is a parent metric, so no data assessment is required
                continue

            LOGGER.info(
                f"Computing scores for metric/submetric {metric.name}/{metric.full_name}."
            )

            if metric.name == "num_unique_owners":
                metric.data_col_name = metric.name
                max_value = self.campaign.get_max_value_across_all_projects(metric.name)
                min_value = 1
            elif metric.name == "num_wells":
                metric.data_col_name = metric.name
                max_value = self.campaign.get_max_value_across_all_projects(metric.name)
                min_value = 1
            else:
                max_value = self.campaign.get_max_value_across_all_wells(
                    metric.data_col_name
                )
                min_value = self.campaign.get_min_value_across_all_wells(
                    metric.data_col_name
                )

            assert getattr(project, metric.score_attribute, None) is None
            assert hasattr(project, metric.name)

            # Check if division by a zero is likely
            if np.isclose(max_value, min_value, rtol=0.001):
                # All cells in this column have equal value.
                # To avoid division by zero, set min_value = 0
                min_value = 0

            if np.isclose(max_value, 0, rtol=0.001):
                # All values in this column are likely zeros.
                # To avoid division by zero, set max_value = 1
                max_value = 1.0

            if metric.has_inverse_priority:
                setattr(
                    project,
                    metric.score_attribute,
                    max(
                        0,
                        min(
                            1,
                            (
                                (max_value - getattr(project, metric.name))
                                / (max_value - min_value)
                            ),
                        ),
                    )
                    * metric.effective_weight,
                )

            else:
                setattr(
                    project,
                    metric.score_attribute,
                    max(
                        0,
                        min(
                            1,
                            (getattr(project, metric.name) - min_value)
                            / (max_value - min_value),
                        ),
                    )
                    * metric.effective_weight,
                )

    def compute_overall_efficiency_scores_project(self, project: Project):
        """
        Computes the overall efficiency score for a project

        Parameters
        ----------
        project : Project
            project in an Campaign
        """
        names_attributes = [
            attribute_name
            for attribute_name in dir(project)
            if "eff_score" in attribute_name
        ]
        assert len(names_attributes) == len(
            [
                metric.name
                for metric in self.efficiency_weights
                if metric.weight != 0 and not hasattr(metric, "submetric")
            ]
        )
        LOGGER.info(
            f"Computing overall efficiency score for project {project.project_id}"
        )
        for attr in names_attributes:
            project.update_efficiency_score(getattr(project, attr))

    def compute_overall_efficiency_scores_campaign(self):
        """
        Computes the overall efficiency score for all projects in a campaign
        """
        for _, project in self.campaign.projects.items():
            self.compute_overall_efficiency_scores_project(project)

    def compute_efficiency_scores(self):
        """
        Function that wraps all the methods needed to compute efficiency scores for the campaign
        """
        self.compute_efficiency_attributes_for_all_projects()
        self.compute_overall_efficiency_scores_campaign()


def export_data_to_excel(
    output_file_path: str,
    campaigns: List[Campaign],
    campaign_categories: List[str],
):
    """
    Exports the data from campaigns to an excel file

    Parameters
    ----------
    output_file_path : str
        The path to the output file
    campaigns : List[Campaign]
        A list of campaigns to output data for
    campaign_categories : List[str]
        A list of labels corresponding to the campaigns in the campaigns argument
    """
    excel_writer = pd.ExcelWriter(output_file_path, engine="xlsxwriter")
    for idx, campaign in enumerate(campaigns):
        campaign.export_data(excel_writer, campaign_categories[idx])
    excel_writer.close()
