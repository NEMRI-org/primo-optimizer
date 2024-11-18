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

# Installed libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# User-defined libs
from primo.data_parser import WellDataColumnNames
from primo.data_parser.metric_data import EfficiencyMetrics, ImpactMetrics
from primo.data_parser.well_data import WellData
from primo.opt_model.result_parser import Campaign, export_data_to_excel


# pylint: disable=missing-function-docstring
@pytest.fixture(name="get_campaign", scope="function")
def get_campaign_fixture():
    im_metrics = ImpactMetrics()

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "ch4_emissions": 35,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 15,
            "well_count": 10,
        },
        submetrics={
            "ch4_emissions": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    im_metrics.check_validity()

    im_metrics.delete_metric(
        "dac_impact"
    )  # Deletes the metric as well submetrics "fed_dac" and "state_dac"
    im_metrics.delete_metric("other_emissions")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        hospitals="Number of Nearby Hospitals",
        schools="Number of Nearby Schools",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        # These are user-specific columns
        elevation_delta="Elevation Delta [m]",
        dist_to_road="Distance to Road [miles]",
    )

    data = {
        "API Well Number": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Leak [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Violation [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Incident [Yes/No]": {0: "Yes", 1: "Yes", 2: "No", 3: "No", 4: "Yes", 5: "Yes"},
        "Compliance [Yes/No]": {0: "No", 1: "Yes", 2: "No", 3: "Yes", 4: "No", 5: "No"},
        "Oil [bbl/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Gas [Mcf/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Age [Years]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Depth [ft]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Elevation Delta [m]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Distance to Road [miles]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Operator Name": {
            0: "Owner 56",
            1: "Owner 136",
            2: "Owner 137",
            3: "Owner 190",
            4: "Owner 196",
            5: "Owner 196",
        },
        "x": {0: 0.99982, 1: 0.99995, 2: 1.51754, 3: 1.51776, 4: 1.51964, 5: 1.51931},
        "y": {0: 1.95117, 1: 1.9572, 2: 1.9584, 3: 1.95746, 4: 1.95678, 5: 1.95674},
        "Number of Nearby Hospitals": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        "Number of Nearby Schools": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
    }

    well_data = WellData(pd.DataFrame(data), col_names, impact_metrics=im_metrics)

    well_data.compute_priority_scores()

    return Campaign(well_data, {2: [0, 1], 3: [2, 3], 4: [4, 5]}, {2: 10, 3: 15, 4: 20})


@pytest.fixture(name="get_minimal_campaign", scope="function")
def get_minimal_campaign_fixture():
    im_metrics = ImpactMetrics()

    # Specify weights
    im_metrics.set_weight(
        primary_metrics={
            "ch4_emissions": 35,
            "sensitive_receptors": 20,
            "ann_production_volume": 20,
            "well_age": 15,
            "well_count": 10,
        },
        submetrics={
            "ch4_emissions": {
                "leak": 40,
                "compliance": 30,
                "violation": 20,
                "incident": 10,
            },
            "sensitive_receptors": {
                "schools": 50,
                "hospitals": 50,
            },
            "ann_production_volume": {
                "ann_gas_production": 50,
                "ann_oil_production": 50,
            },
        },
    )

    im_metrics.check_validity()

    im_metrics.delete_metric(
        "dac_impact"
    )  # Deletes the metric as well submetrics "fed_dac" and "state_dac"
    im_metrics.delete_metric("other_emissions")
    im_metrics.delete_metric("five_year_production_volume")
    im_metrics.delete_metric("well_integrity")
    im_metrics.delete_metric("environment")

    # Submetrics can also be deleted in a similar manner
    im_metrics.delete_submetric("buildings_near")
    im_metrics.delete_submetric("buildings_far")

    col_names = WellDataColumnNames(
        well_id="API Well Number",
        latitude="x",
        longitude="y",
        operator_name="Operator Name",
        age="Age [Years]",
        depth="Depth [ft]",
        leak="Leak [Yes/No]",
        compliance="Compliance [Yes/No]",
        violation="Violation [Yes/No]",
        incident="Incident [Yes/No]",
        hospitals="Number of Nearby Hospitals",
        schools="Number of Nearby Schools",
        ann_gas_production="Gas [Mcf/Year]",
        ann_oil_production="Oil [bbl/Year]",
        # These are user-specific columns
    )

    data = {
        "API Well Number": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Leak [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Violation [Yes/No]": {0: "No", 1: "No", 2: "No", 3: "No", 4: "No", 5: "No"},
        "Incident [Yes/No]": {0: "Yes", 1: "Yes", 2: "No", 3: "No", 4: "Yes", 5: "Yes"},
        "Compliance [Yes/No]": {0: "No", 1: "Yes", 2: "No", 3: "Yes", 4: "No", 5: "No"},
        "Oil [bbl/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Gas [Mcf/Year]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Age [Years]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Depth [ft]": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
        "Operator Name": {
            0: "Owner 56",
            1: "Owner 136",
            2: "Owner 137",
            3: "Owner 190",
            4: "Owner 196",
            5: "Owner 196",
        },
        "x": {0: 0.99982, 1: 0.99995, 2: 1.51754, 3: 1.51776, 4: 1.51964, 5: 1.51931},
        "y": {0: 1.95117, 1: 1.9572, 2: 1.9584, 3: 1.95746, 4: 1.95678, 5: 1.95674},
        "Number of Nearby Hospitals": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
        "Number of Nearby Schools": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3},
    }

    well_data = WellData(pd.DataFrame(data), col_names, impact_metrics=im_metrics)

    well_data.compute_priority_scores()

    return Campaign(well_data, {1: [0, 1], 2: [2, 3], 3: [4]}, {1: 10, 2: 15, 3: 20})


@pytest.fixture(name="get_project", scope="function")
def get_project_fixture(get_campaign):
    return get_campaign.projects[2]


@pytest.fixture(name="get_eff_metrics", scope="function")
def get_eff_metrics_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 20,
            "num_unique_owners": 30,
            "avg_elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


@pytest.fixture(name="get_eff_metrics_accessibility", scope="function")
def get_eff_metrics_accessibility_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 10,
            "num_unique_owners": 30,
            "avg_elevation_delta": 20,
            "age_range": 10,
            "depth_range": 20,
            "avg_dist_to_road": 10,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


def test_check_column_exists(get_project):
    get_project.col_names.hospitals = None
    with pytest.raises(ValueError):
        print(get_project.num_wells_near_hospitals)


# test Project Class
def test_project_attributes(get_project):
    project = get_project
    for index in project:
        assert index in project.well_data.data.index
    assert len(project.well_data.data) == 2
    assert project.project_id == 2
    assert project.num_wells == 2
    assert project.plugging_cost == 10e6
    assert project.efficiency_score == 0
    assert project.num_wells_near_hospitals == 2
    assert project.num_wells_near_schools == 2
    assert project.average_age == 1.5
    assert project.age_range == 1
    assert project.average_depth == 1.5
    assert project.depth_range == 1
    assert project.avg_elevation_delta == 1.5
    assert project.centroid == (0.999885, 1.954185)
    assert project.avg_dist_to_road == 1.5
    assert project.num_unique_owners == 2
    assert project.impact_score == 38.25
    delattr(project.col_names, "priority_score")
    with pytest.raises(AttributeError):
        project.impact_score += 2.0


def test_project_attributes_minimal(get_minimal_campaign):
    project = get_minimal_campaign.projects[1]

    # checking for missing attributes
    with pytest.raises(ValueError):
        print(project.avg_dist_to_road)

    with pytest.raises(ValueError):
        print(project.avg_elevation_delta)


def test_max_val_col(get_project):
    project = get_project
    assert project.get_max_val_col(project.col_names.age) == 2


def test_update_efficiency_score(get_project):
    project = get_project
    assert project.efficiency_score == 0
    project.update_efficiency_score(0.5)
    assert project.efficiency_score == 0.5
    project.update_efficiency_score(1)
    assert project.efficiency_score == 1.5


def test_get_well_info_data_frame(get_project):
    project = get_project
    well_data = project.get_well_info_dataframe()
    for col in project.essential_cols:
        assert col in well_data.columns
    assert (
        "Violation [Yes/No]" == project.col_names.violation
        and "Violation [Yes/No]" not in well_data.columns
    )
    assert len(well_data) == 2
    assert all(i in [1, 2] for i in well_data["Age [Years]"].values)


def test_compute_accessibility_score(get_campaign, get_eff_metrics_accessibility):
    # this can only be called after the efficiency scores have been assigned
    get_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_eff_metrics_accessibility
    )
    get_campaign.set_efficiency_weights(get_eff_metrics_accessibility)
    get_campaign.efficiency_calculator.compute_efficiency_scores()
    project = get_campaign.projects[2]
    assert project.accessibility_score == (
        30,
        pytest.approx((6 - 1.5) / 5 * 20 + (6 - 1.5) / 5 * 10),
    )
    delattr(project, "avg_elevation_delta_eff_score_0_20")
    delattr(project, "avg_dist_to_road_eff_score_0_10")
    assert project.accessibility_score is None


def test_compute_accessibility_score_2(get_campaign, get_eff_metrics):
    # this can only be called after the efficiency scores have been assigned
    get_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_eff_metrics
    )
    get_campaign.set_efficiency_weights(get_eff_metrics)
    get_campaign.efficiency_calculator.compute_efficiency_scores()
    project = get_campaign.projects[2]
    assert project.accessibility_score == (20, pytest.approx((6 - 1.5) / 5 * 20))


def test_project_str(get_project):
    project = get_project
    assert (
        str(project)
        == "Number of wells in project 2\t\t: 2\n"
        + "Estimated Project Cost\t\t\t: $10000000\n"
        + "Impact Score [0-100]\t\t\t: 38.25\n"
        + "Efficiency Score [0-100]\t\t: 0.00\n"
    )


# test Campaign Class
def test_campaign_attributes(get_campaign):
    campaign = get_campaign
    assert len(campaign.projects) == 3
    assert campaign.num_projects == 3
    assert campaign.total_plugging_cost == 45e6
    assert len(campaign.wd) == 6
    # already tested the project string function
    msg = (
        "The optimal campaign has 3 projects.\n"
        "The total cost of the campaign is $45000000\n\n"
    )
    for project in campaign.projects.values():
        msg += str(project)
        msg += "\n"
    assert str(campaign) == msg
    assert campaign.efficiency_calculator.efficiency_weights is None


def test_get_max_value_across_all_projects(get_campaign):
    assert get_campaign.get_max_value_across_all_projects("average_depth") == 5.5
    assert get_campaign.get_max_value_across_all_projects("num_wells_near_schools") == 2
    with pytest.raises(AttributeError):
        get_campaign.get_max_value_across_all_projects("Dev is a good dev")


def test_get_min_value_across_all_projects(get_campaign):
    assert get_campaign.get_min_value_across_all_projects("average_depth") == 1.5
    assert get_campaign.get_min_value_across_all_projects("num_wells_near_schools") == 2
    with pytest.raises(AttributeError):
        get_campaign.get_min_value_across_all_projects("Dev is a good dev")


# errors checked in test_check_col_in_data
def test_get_max_value_across_all_wells(get_campaign):
    assert get_campaign.get_max_value_across_all_wells("Depth [ft]") == 6
    assert get_campaign.get_max_value_across_all_wells("Age [Years]") == 6


def test_get_min_value_across_all_wells(get_campaign):
    assert get_campaign.get_min_value_across_all_wells("Depth [ft]") == 1
    assert get_campaign.get_min_value_across_all_wells("Age [Years]") == 1


# for now leaving the plotting out of the tests
def test_get_project_well_information(get_campaign):
    info = get_campaign.get_project_well_information()
    assert all(i in [2, 3, 4] for i in info.keys())
    # already tested well_info_dataframe


def test_get_efficiency_score_project(get_campaign):
    project = get_campaign.projects[2]
    assert get_campaign.get_efficiency_score_project(2) == 0
    project.update_efficiency_score(5)
    assert get_campaign.get_efficiency_score_project(2) == 5


def test_get_impact_score_project(get_campaign):
    assert get_campaign.get_impact_score_project(2) == 38.25


def test_extract_column_header_for_efficiency_metrics(get_campaign):
    # pylint: disable=protected-access
    assert (
        get_campaign._extract_column_header_for_efficiency_metrics(
            "this_is_a_name_eff_score_0_100"
        )
        == "This Is A Name Score [0-100]"
    )
    assert (
        get_campaign._extract_column_header_for_efficiency_metrics(
            "this_is_a_name_eff_score_0_5"
        )
        == "This Is A Name Score [0-5]"
    )


def test_campaign_summary(get_campaign):
    summary = get_campaign.get_campaign_summary()
    assert all(
        i
        in [
            "Project ID",
            "Number of Wells",
            "Impact Score [0-100]",
            "Efficiency Score [0-100]",
        ]
        for i in summary.columns
    )
    assert list(summary["Project ID"].values) == [2, 3, 4]
    assert summary["Impact Score [0-100]"].values[0] == 38.25
    assert len(summary) == 3


# test the Efficiency Calculator
def test_set_efficiency_weights(get_campaign, get_eff_metrics):
    campaign = get_campaign
    eff_metrics = get_eff_metrics

    assert campaign.efficiency_calculator.efficiency_weights is None
    campaign.set_efficiency_weights(eff_metrics)
    # based on default equality comparison
    assert campaign.efficiency_calculator.efficiency_weights == get_eff_metrics


@pytest.fixture(name="get_efficiency_calculator", scope="function")
def get_efficiency_calculator_fixture(get_campaign, get_eff_metrics):
    get_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_eff_metrics
    )
    get_campaign.set_efficiency_weights(get_eff_metrics)
    return get_campaign


@pytest.fixture(name="get_efficiency_metrics_minimal", scope="function")
def get_efficiency_metrics_minimal_fixture():
    eff_metrics = EfficiencyMetrics()
    eff_metrics.set_weight(
        primary_metrics={
            "num_wells": 0,
            "age_range": 30,
            "depth_range": 30,
            "num_unique_owners": 40,
        }
    )

    # Check validity of the metrics
    eff_metrics.check_validity()
    return eff_metrics


def test_compute_efficiency_score_edge_cases(
    get_minimal_campaign, get_efficiency_metrics_minimal
):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_efficiency_metrics_minimal
    )
    get_minimal_campaign.set_efficiency_weights(get_efficiency_metrics_minimal)
    get_minimal_campaign.efficiency_calculator.compute_efficiency_scores()
    assert all(
        "num_wells_eff_score" not in entry
        for entry in dir(get_minimal_campaign.projects[1])
    )
    with pytest.raises(ValueError):
        print(get_minimal_campaign.projects[1].avg_elevation_delta)


def test_single_well(get_minimal_campaign, get_efficiency_metrics_minimal):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_efficiency_metrics_minimal
    )
    get_minimal_campaign.set_efficiency_weights(get_efficiency_metrics_minimal)
    get_minimal_campaign.efficiency_calculator.compute_efficiency_scores()
    assert get_minimal_campaign.projects[3].efficiency_score == 100


def test_zeros(get_minimal_campaign, get_efficiency_metrics_minimal):
    get_minimal_campaign.wd.set_impact_and_efficiency_metrics(
        efficiency_metrics=get_efficiency_metrics_minimal
    )
    get_minimal_campaign.set_efficiency_weights(get_efficiency_metrics_minimal)
    get_minimal_campaign.projects[1].well_data.data["Age [Years]"] = [0, 0]
    get_minimal_campaign.projects[2].well_data.data["Age [Years]"] = [0, 0]
    get_minimal_campaign.projects[3].well_data.data["Age [Years]"] = [0]
    get_minimal_campaign.wd.data["Age [Years]"] = np.zeros(
        len(get_minimal_campaign.wd.data["Age [Years]"])
    )
    get_minimal_campaign.efficiency_calculator.compute_efficiency_scores()
    assert get_minimal_campaign.projects[1].age_range_eff_score_0_30 == 30.0


def test_compute_efficiency_attributes_for_project(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    project = campaign.projects[2]
    campaign.efficiency_calculator.compute_efficiency_attributes_for_project(project)
    assert project.num_wells_eff_score_0_20 == 20.0
    assert project.num_unique_owners_eff_score_0_30 == pytest.approx(0.0)
    assert project.avg_elevation_delta_eff_score_0_20 == pytest.approx(
        (6 - 1.5) / 5 * 20
    )
    assert project.age_range_eff_score_0_10 == pytest.approx(10)
    assert project.depth_range_eff_score_0_20 == pytest.approx(20)


def test_compute_overall_efficiency_scores_project(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    project = campaign.projects[2]
    campaign.efficiency_calculator.compute_efficiency_attributes_for_project(project)
    campaign.efficiency_calculator.compute_overall_efficiency_scores_project(project)
    assert project.efficiency_score == pytest.approx(
        20 + 0 + (6 - 1.5) / 5 * 20 + 10 + 20
    )


def test_compute_efficiency_attributes_for_all_projects(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    campaign.efficiency_calculator.compute_efficiency_attributes_for_all_projects()
    for _, project in campaign.projects.items():
        assert project.num_wells_eff_score_0_20 >= 0.0
        assert project.num_unique_owners_eff_score_0_30 >= 0.0
        assert project.avg_elevation_delta_eff_score_0_20 >= 0.0
        assert project.age_range_eff_score_0_10 >= 0.0
        assert project.depth_range_eff_score_0_20 >= 0.0


def test_compute_overall_efficiency_scores_campaign(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    campaign.efficiency_calculator.compute_efficiency_attributes_for_all_projects()
    campaign.efficiency_calculator.compute_overall_efficiency_scores_campaign()
    for _, project in campaign.projects.items():
        assert project.efficiency_score >= 0


def test_compute_efficiency_scores(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    campaign.efficiency_calculator.compute_efficiency_scores()
    for _, project in campaign.projects.items():
        assert project.efficiency_score >= 0


def test_compute_efficiency_scores_wrapper(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    campaign.compute_efficiency_scores()
    for _, project in campaign.projects.items():
        assert project.efficiency_score >= 0


# last test for the campaign class
def test_get_efficiency_metrics(get_efficiency_calculator):
    campaign = get_efficiency_calculator
    campaign.efficiency_calculator.compute_efficiency_scores()
    efficiency_metric_output = campaign.get_efficiency_metrics()
    assert all(
        i
        in [
            "Project ID",
            "Num Wells Score [0-20]",
            "Num Unique Owners Score [0-30]",
            "Avg Elevation Delta Score [0-20]",
            "Age Range Score [0-10]",
            "Depth Range Score [0-20]",
            "Accessibility Score [0-20]",
            "Efficiency Score [0-100]",
        ]
        for i in efficiency_metric_output.columns
    )
    assert len(efficiency_metric_output) == 3

    assert all(
        list(efficiency_metric_output.iloc[0, :].values)[i]
        == pytest.approx([2, 10.0, 18.0, 20.0, 0.0, 20][i])
        for i in range(6)
    )
    for _, project in campaign.projects.items():
        delattr(project, "avg_elevation_delta_eff_score_0_20")
    efficiency_metric_output = campaign.get_efficiency_metrics()
    assert all(
        i
        in [
            "Project ID",
            "Num Wells Score [0-20]",
            "Num Unique Owners Score [0-30]",
            "Age Range Score [0-10]",
            "Depth Range Score [0-20]",
            "Efficiency Score [0-100]",
        ]
        for i in efficiency_metric_output.columns
    )
    assert all(
        list(efficiency_metric_output.iloc[0, :].values)[i]
        == pytest.approx([2, 10.0, 20.0, 0.0, 20][i])
        for i in range(5)
    )


def test_export_data_to_excel(get_campaign):
    output_file_path = "export_data_test.xlsx"
    campaigns = [get_campaign]
    campaign_labels = ["export data test"]
    export_data_to_excel(output_file_path, campaigns, campaign_labels)
    assert True


def test_plot_campaign(get_campaign):
    get_campaign.plot_campaign("Just some toy data")
    plt.close()
