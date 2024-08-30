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
from dataclasses import dataclass
from typing import Union

# User-defined libs
from primo.data_parser import ImpactMetrics
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


# pylint: disable = too-many-instance-attributes
# pylint: disable = trailing-whitespace, protected-access
# pylint: disable = logging-fstring-interpolation
@dataclass
class WellDataColumnNames:
    """
    Dataclass for storing column names
    """

    # Essential columns for preliminary data check
    well_id: str
    latitude: str
    longitude: str
    age: str
    depth: str
    operator_name: Union[str, None] = None

    # Columns for ch4_emissions metric
    leak: Union[str, None] = None
    compliance: Union[str, None] = None
    violation: Union[str, None] = None
    incident: Union[str, None] = None

    # Columns for dac_impact metric. fed_dac will be calculated
    state_dac: Union[str, None] = None

    # Columns for sensitive_receptors metric
    hospitals: Union[str, None] = None
    schools: Union[str, None] = None
    buildings_near: Union[str, None] = None
    buildings_far: Union[str, None] = None

    # Columns for environment metric
    fed_wetlands_near: Union[str, None] = None
    fed_wetlands_far: Union[str, None] = None
    state_wetlands_near: Union[str, None] = None
    state_wetlands_far: Union[str, None] = None

    # Columns for production_volume metric
    well_type: Union[str, None] = None
    ann_gas_production: Union[str, None] = None
    ann_oil_production: Union[str, None] = None
    five_year_gas_production: Union[str, None] = None
    five_year_oil_production: Union[str, None] = None
    life_gas_production: Union[str, None] = None
    life_oil_production: Union[str, None] = None

    # Columns for other_emissions metric
    h2s_leak: Union[str, None] = None
    brine_leak: Union[str, None] = None

    # Columns for well_integrity metric
    well_integrity: Union[str, None] = None

    def __contains__(self, key):
        # Checks if an attribute is present
        return key in self.__dict__

    def __iter__(self):
        # Iterate over all attributes/columns
        return iter(self.__dict__)

    def register_new_columns(self, col_names: dict):
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
        """Returns internal names of the columns"""
        return self.__dict__.keys()

    def values(self):
        """Returns user names of the columns"""
        return self.__dict__.values()

    def items(self):
        """Returns internal-user name pairs"""
        return self.__dict__.items()

    # pylint: disable = logging-fstring-interpolation
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
        im_wt = impact_metrics

        for obj in im_wt:
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
                continue

            # If the code reaches here, then _required_data must be a list.
            # Currently, this happens only for production_volume metrics
            # The `data_col_name` attribute will be assigned after processing
            # the data production volume data
            for col in obj._required_data:
                if getattr(self, col) is None:
                    msg = (
                        f"Weight of the metric {obj.name} is nonzero, so attribute "
                        f"{col} is an essential input in the WellDataColumnNames object."
                    )
                    raise_exception(msg, AttributeError)
