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
from dataclasses import dataclass
from typing import Optional, Union

# Installed libs
from pyomo.common.config import Bool, NonNegativeFloat, NonNegativeInt

# This file should contain all the constants
FEASIBILITY_TOLERANCE = 1e-6  # Optimization Feasibility tolerance
EARTH_RADIUS = 3959.0  # Earth's radius in Miles
CENSUS_YEAR = 2020
CONVERSION_FACTOR = 5.614583  # convert Bbl of oil to Mcf of gas
BING_MAPS_BASE_URL = "http://dev.virtualearth.net/REST/V1/Routes/Driving"
# Fixed start coordinates in elevation utility to get nearest road point
START_COORDINATES = (40.44, -79.94)

# CEJST data available on https://screeningtool.geoplatform.gov
# relies on Census Tracts from 2010
DAC_TRACT_YEAR = 2010


# Set of supported impact metrics along with
# the required data for the analysis.
# pylint: disable = too-many-instance-attributes
@dataclass()
class _SupportedContent:
    name: str
    full_name: str
    has_submetrics: bool = False
    is_submetric: bool = False
    parent_metric: Optional[str] = None
    # required_data: keys in WellDataColumnNames class
    # This will be used to check if the input data has
    # required columns or not.
    required_data: Optional[Union[str, list]] = None
    # Is the value of this metric inversely proportional to plugging priority?
    # E.g., Compliance, production volume, etc.
    has_inverse_priority: bool = False
    fill_missing_value: Optional[dict] = None


SUPP_IMPACT_METRICS = {
    # Primary metrics
    "ch4_emissions": _SupportedContent(
        name="ch4_emissions",
        full_name="Methane Emissions (Proxies)",
        has_submetrics=True,
    ),
    "dac_impact": _SupportedContent(
        name="dac_impact",
        full_name="Disadvantaged Community Impact",
        has_submetrics=True,
    ),
    "sensitive_receptors": _SupportedContent(
        name="sensitive_receptors",
        full_name="Sensitive Receptors",
        has_submetrics=True,
    ),
    "ann_production_volume": _SupportedContent(
        name="ann_production_volume",
        full_name="Annual Production Volume",
        has_submetrics=True,
    ),
    "five_year_production_volume": _SupportedContent(
        name="five_year_production_volume",
        full_name="Five-year Production Volume",
        has_submetrics=True,
    ),
    "well_age": _SupportedContent(
        name="well_age",
        full_name="Well Age",
        required_data="age",
    ),
    "well_count": _SupportedContent(
        name="well_count",
        full_name="Owner Well Count",
        required_data="operator_name",
    ),
    "other_emissions": _SupportedContent(
        name="other_emissions",
        full_name="Other Emissions",
        has_submetrics=True,
    ),
    "well_integrity": _SupportedContent(
        name="well_integrity",
        full_name="Well Integrity Issues [Yes/No]",
        required_data="well_integrity",
        # If it is not specified, assume no well integrity issues
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "environment": _SupportedContent(
        name="environment",
        full_name="Environment",
        has_submetrics=True,
    ),
    # Submetrics of ch4_emissions
    "leak": _SupportedContent(
        name="leak",
        full_name="Leak [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="leak",
        # Assume that the well is not leaking if it not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "compliance": _SupportedContent(
        name="compliance",
        full_name="Compliance [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="compliance",
        # Priority should be higher if the well is not compliant
        has_inverse_priority=True,
        # Assuming well is compliant if it is not specified
        fill_missing_value={"domain": Bool, "default": True},
    ),
    "violation": _SupportedContent(
        name="violation",
        full_name="Violation [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="violation",
        # Assuming that the well is not in violation
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "incident": _SupportedContent(
        name="incident",
        full_name="Incident [Yes/No]",
        is_submetric=True,
        parent_metric="ch4_emissions",
        required_data="incident",
        # Incident is assumed to be False if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    # Submetrics of dac_impact
    "fed_dac": _SupportedContent(
        name="fed_dac",
        full_name="Justice40 Federal DAC",
        is_submetric=True,
        parent_metric="dac_impact",
        # WellData class computes this information
        required_data=None,
    ),
    "state_dac": _SupportedContent(
        name="state_dac",
        full_name="State DAC",
        is_submetric=True,
        parent_metric="dac_impact",
        required_data="state_dac",
        # If it is not specified, set the state DAC priority to zero
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    # Submetrics of sensitive_receptors
    "hospitals": _SupportedContent(
        name="hospitals",
        full_name="Hospitals",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="hospitals",
        fill_missing_value={"domain": NonNegativeInt, "default": 0},
    ),
    "schools": _SupportedContent(
        name="schools",
        full_name="Schools",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="schools",
        fill_missing_value={"domain": NonNegativeInt, "default": 0},
    ),
    "buildings_near": _SupportedContent(
        name="buildings_near",
        full_name="Buildings (Close Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_near",
        # Assuming that no buildings are nearby if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "buildings_far": _SupportedContent(
        name="buildings_far",
        full_name="Buildings (Distant Range)",
        is_submetric=True,
        parent_metric="sensitive_receptors",
        required_data="buildings_far",
        # Assuming no distant buildings
        fill_missing_value={"domain": Bool, "default": False},
    ),
    # Submetrics of environment
    "fed_wetlands_near": _SupportedContent(
        name="fed_wetlands_near",
        full_name="Federal Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_near",
        # Assuming no nearby federal wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "fed_wetlands_far": _SupportedContent(
        name="fed_wetlands_far",
        full_name="Federal Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="fed_wetlands_far",
        # Assuming no distant federal wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "state_wetlands_near": _SupportedContent(
        name="state_wetlands_near",
        full_name="State Wetlands (Close Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_near",
        # Assuming no nearby state wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "state_wetlands_far": _SupportedContent(
        name="state_wetlands_far",
        full_name="State Wetlands (Distant Range)",
        is_submetric=True,
        parent_metric="environment",
        required_data="state_wetlands_far",
        # Assuming no distant state wetlands
        fill_missing_value={"domain": Bool, "default": False},
    ),
    # Submetrics of production_volume
    "ann_gas_production": _SupportedContent(
        name="ann_gas_production",
        full_name="Annual Gas Production [in Mcf/Year]",
        is_submetric=True,
        parent_metric="ann_production_volume",
        required_data="ann_gas_production",
        # Higher gas production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing gas, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "ann_oil_production": _SupportedContent(
        name="ann_oil_production",
        full_name="Annual Oil Production [in bbl/Year]",
        is_submetric=True,
        parent_metric="ann_production_volume",
        required_data="ann_oil_production",
        # Higher oil production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing oil, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "five_year_gas_production": _SupportedContent(
        name="five_year_gas_production",
        full_name="Five-year Gas Production [in Mcf]",
        is_submetric=True,
        parent_metric="five_year_production_volume",
        required_data="five_year_gas_production",
        # Higher gas production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing gas, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    "five_year_oil_production": _SupportedContent(
        name="five_year_oil_production",
        full_name="Five-year Oil Production [in bbl]",
        is_submetric=True,
        parent_metric="five_year_production_volume",
        required_data="five_year_oil_production",
        # Higher oil production => lower priority for plugging
        has_inverse_priority=True,
        # Assuming that well is not producing oil, if it is not specified
        fill_missing_value={"domain": NonNegativeFloat, "default": 0.0},
    ),
    # Submetrics of other_emissions
    "h2s_leak": _SupportedContent(
        name="h2s_leak",
        full_name="H2S Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_emissions",
        required_data="h2s_leak",
        # Assuming no H2S leak if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
    "brine_leak": _SupportedContent(
        name="brine_leak",
        full_name="Brine Leak [Yes/No]",
        is_submetric=True,
        parent_metric="other_emissions",
        required_data="brine_leak",
        # Assuming no brine leak if it is not specified
        fill_missing_value={"domain": Bool, "default": False},
    ),
}

# note the names of these metrics must be identical to
# a property of the Optimal Project class for the
# efficiency calculation
SUPP_EFF_METRICS = {
    "num_wells": _SupportedContent(
        name="num_wells", full_name="Number of Wells", required_data="well_id"
    ),
    "num_unique_owners": _SupportedContent(
        name="num_unique_owners",
        full_name="Number of Unique Owners",
        required_data="operator_name",
        has_inverse_priority=True,
    ),
    "avg_elevation_delta": _SupportedContent(
        name="avg_elevation_delta",
        full_name="Average Elevation Delta [m]",
        required_data="elevation_delta",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
    "age_range": _SupportedContent(
        name="age_range",
        full_name="Age Range [Years]",
        required_data="age",
        has_inverse_priority=True,
    ),
    "depth_range": _SupportedContent(
        name="depth_range",
        full_name="Depth Range [ft]",
        required_data="depth",
        has_inverse_priority=True,
    ),
    "record_completeness": _SupportedContent(
        name="record_completeness",
        full_name="Record Completeness",
    ),
    "avg_dist_to_road": _SupportedContent(
        name="avg_dist_to_road",
        full_name="Distance to Road [miles]",
        required_data="dist_to_road",
        has_inverse_priority=True,
        fill_missing_value={"domain": NonNegativeFloat, "default": 0},
    ),
}
