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
from dataclasses import InitVar, dataclass
from typing import Dict, Optional

# User-defined libs
from primo.data_parser import ImpactMetrics
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


@dataclass
class WellDataColumnNames:  # pylint: disable=too-many-instance-attributes
    """
    Dataclass for storing column names
    """

    # Essential columns for preliminary data check
    well_id: str
    latitude: str
    longitude: str
    age: str
    depth: str
    operator_name: Optional[str] = None

    # Columns for ch4_emissions metric
    leak: Optional[str] = None
    compliance: Optional[str] = None
    violation: Optional[str] = None
    incident: Optional[str] = None

    # Columns for dac_impact metric. fed_dac will be calculated
    state_dac: Optional[str] = None

    # Columns for sensitive_receptors metric
    hospitals: Optional[str] = None
    schools: Optional[str] = None
    buildings_near: Optional[str] = None
    buildings_far: Optional[str] = None

    # Columns for environment metric
    fed_wetlands_near: Optional[str] = None
    fed_wetlands_far: Optional[str] = None
    state_wetlands_near: Optional[str] = None
    state_wetlands_far: Optional[str] = None

    # Columns for production_volume metric
    well_type: Optional[str] = None  # Oil/Gas Type
    well_type_by_depth: Optional[str] = None  # Shallow/Deep Type
    ann_gas_production: Optional[str] = None
    ann_oil_production: Optional[str] = None
    five_year_gas_production: Optional[str] = None
    five_year_oil_production: Optional[str] = None
    life_gas_production: Optional[str] = None
    life_oil_production: Optional[str] = None

    # Columns for other_emissions metric
    h2s_leak: Optional[str] = None
    brine_leak: Optional[str] = None

    # Columns for well_integrity metric
    well_integrity: Optional[str] = None

    # Columns for efficiency metrics
    elevation_delta: Optional[str] = None
    dist_to_road: Optional[str] = None

    # Additional user-specific columns
    additional_columns: InitVar[dict] = {}

    def __post_init__(self, additional_columns):
        self.register_new_columns(additional_columns)

    def __contains__(self, key):
        # Checks if an attribute is present
        return key in self.__dict__

    def __iter__(self):
        # Iterate over all attributes/columns
        return iter(self.__dict__)

    def register_new_columns(self, col_names: Dict[str, str]):
        """Registers an attribute for a new column"""
        for key, val in col_names.items():
            if key in self:
                raise_exception(
                    f"Attribute {key} is already defined. Use a different name.",
                    AttributeError,
                )

            if not key.isidentifier():
                raise_exception(
                    f"Key {key} is not a valid python variable name!", ValueError
                )
            setattr(self, key, val)

    def keys(self):
        """Returns defined internal names of the columns"""
        keys = [key for key, val in self.__dict__.items() if val is not None]
        return keys

    def values(self):
        """Returns user names of the columns that are not None"""
        val = [val for val in self.__dict__.values() if val is not None]
        return val

    def items(self):
        """Returns internal-user name pairs"""
        data = {key: val for key, val in self.__dict__.items() if val is not None}
        return data.items()

    def check_columns_available(
        self,
        impact_metrics: ImpactMetrics,
    ):
        """
        Checks if the required columns are provided based on the
        weights of impact metrics.

        Parameters
        ----------
        impact_metrics : ImpactMetrics
            Impact Metrics object
        """
        # pylint: disable=protected-access
        im_wt = impact_metrics

        for obj in im_wt:
            # the centroid can only be computed after optimization
            if not hasattr(obj, "_required_data"):
                # This is not a supported metric, so continue
                LOGGER.warning(
                    f"Metric/submetric {obj.name}/{obj.full_name} is not supported. "
                    f"Users are required to process the data for this metric, and "
                    f"assign the name of the column to the attribute `data_col_name` "
                    f"in the metric/submetric {obj.name} object."
                )
                continue

            # For supported metrics, skip if there is no essential data requirement,
            # or if the weight of the metric is zero.
            if obj._required_data is None or obj.weight == 0:
                continue

            # This is a supported metric with a nonzero weight. Raise an error if the
            # data is not provided
            if isinstance(obj._required_data, str):
                col_name = getattr(self, obj._required_data)
                if col_name is None:
                    msg = (
                        f"Weight of the metric {obj.name} is nonzero, so attribute "
                        f"{obj._required_data} is an essential input in the "
                        f"WellDataColumnNames object."
                    )
                    raise_exception(msg, AttributeError)

                # Column name is specified, so continue to the next metric
                # Register the column name in Metric object
                obj.data_col_name = col_name
