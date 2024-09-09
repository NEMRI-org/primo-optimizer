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
import copy
import logging
import math
from typing import Union

# Installed libs
import numpy as np
import pandas as pd
from pyomo.common.config import (
    Bool,
    ConfigDict,
    ConfigValue,
    In,
    NonNegativeFloat,
    document_kwargs_from_configdict,
)

# User-defined libs
from primo.data_parser import ImpactMetrics
from primo.data_parser.default_data import CONVERSION_FACTOR
from primo.data_parser.well_data_column_names import WellDataColumnNames
from primo.utils.domain_validators import InRange
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)

# Input options for well data class
CONFIG = ConfigDict()
CONFIG.declare(
    "census_year",
    ConfigValue(
        default=2020,
        domain=In(list(range(2020, 2101, 10))),
        doc="Year for collecting census data",
    ),
)
CONFIG.declare(
    "preliminary_data_check",
    ConfigValue(
        default=True, domain=Bool, doc="If True, performs preliminary data checks."
    ),
)
CONFIG.declare(
    "ignore_operator_name",
    ConfigValue(
        default=True,
        domain=Bool,
        doc="Remove well if operator name is not provided",
    ),
)
CONFIG.declare(
    "missing_age",
    ConfigValue(
        default="fill",
        domain=In(["fill", "estimate", "remove"]),
        doc="Method for processing missing age information",
    ),
)
CONFIG.declare(
    "missing_depth",
    ConfigValue(
        default="fill",
        domain=In(["fill", "estimate", "remove"]),
        doc="Method for processing missing depth information",
    ),
)
CONFIG.declare(
    "fill_age",
    ConfigValue(
        default=100,
        # Assuming that no well is older than 350 years
        domain=InRange(0, 350),
        doc="Value to fill with, if the age is missing",
    ),
)
CONFIG.declare(
    "fill_depth",
    ConfigValue(
        default=1000,
        # Assuming that no well is deeper than 40,000 ft
        domain=InRange(0, 40000),
        doc="Value to fill with, if the depth is missing",
    ),
)
CONFIG.declare(
    "fill_well_type",
    ConfigValue(
        default="Oil",
        domain=In(["Oil", "Gas"]),
        doc="Well-type assumption if it is not specified",
    ),
)
CONFIG.declare(
    "fill_well_type_depth",
    ConfigValue(
        default="Deep",
        domain=In(["Deep", "Shallow"]),
        doc="Well-type (by depth) assumption if it is not specified",
    ),
)
CONFIG.declare(
    "fill_ann_gas_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the annual gas production "
            "[in Mcf/Year] is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_ann_oil_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the annual oil production "
            "[in bbl/Year] is not specified."
        ),
    ),
)
CONFIG.declare(
    "fill_life_gas_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the lifelong gas production [in Mcf] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "fill_life_oil_production",
    ConfigValue(
        default=0.0,
        domain=NonNegativeFloat,
        doc=(
            "Value to fill with, if the lifelong oil production [in bbl] "
            "is not specified"
        ),
    ),
)
CONFIG.declare(
    "threshold_gas_production",
    ConfigValue(
        domain=NonNegativeFloat,
        doc=(
            "If specified, wells whose lifelong gas production volume [in Mcf] is "
            "above the threshold production volume will be removed from the dataset."
        ),
    ),
)
CONFIG.declare(
    "threshold_oil_production",
    ConfigValue(
        domain=NonNegativeFloat,
        doc=(
            "If specified, wells whose lifelong oil production volume [in bbl] is "
            "above the threshold production volume will be removed from the dataset."
        ),
    ),
)
CONFIG.declare(
    "threshold_depth",
    ConfigValue(
        domain=NonNegativeFloat,
        doc="Threshold depth [in ft] for classifying a well as shallow or deep",
    ),
)


# pylint: disable = too-many-instance-attributes
# pylint: disable = trailing-whitespace, protected-access
# pylint: disable = logging-fstring-interpolation
class WellData:
    """
    Reads, processes, and anlyzes well data.
    """

    __slots__ = (
        "config",  # ConfigDict containing input options
        "data",  # DataFrame containing well data
        # Private variables
        "_col_names",  # Pointer to WellDataColumnNames object
        "_removed_rows",  # dict containing list of rows removed
        "_well_types",  # dict containing well types: oil, gas, shallow, deep, etc.
    )

    # Adds documentation for all the keyword arguments
    @document_kwargs_from_configdict(CONFIG)
    def __init__(
        self,
        filename: str,
        column_names: WellDataColumnNames,
        **kwargs,
    ) -> None:
        """
        Reads well data and performs initial high-level data processing.

        Parameters
        ----------
        filename : str
            Name of the file containing the well data. Currently, only
            .xlsx, .xls, and .csv formats are supported.

        column_names : WellDataColumnNames
            A WellDataColumnNames object containing the names of various columns
        """
        # Import columns whose column names have been provided.
        LOGGER.info("Reading the well data from the input file.")
        if isinstance(filename, pd.DataFrame):
            # Allows passing the input data as a DataFrame. This feature would
            # be useful for creating new objects with existing DataFrame objects.
            # Otherwise, the data needs to be saved to a file and then read.
            # Use case: Partition the wells as gas and oil wells and return them as
            # two new WellData objects
            self.data = filename
            # Check if the columns are present in the data
            for col in column_names.values():
                assert col in self.data.columns

        elif filename.split(".")[-1] in ["xlsx", "xls"]:
            self.data = pd.read_excel(
                filename,
                sheet_name=kwargs.pop("sheet", 0),
                usecols=column_names.values(),
                # Store well ids as strings
                dtype={column_names.well_id: str},
            )

            # Updating the `index` to keep it consistent with the row numbers in file
            self.data.index += 2

        elif filename.split(".")[-1] == "csv":
            self.data = pd.read_csv(
                filename,
                usecols=column_names.values(),
                # Store well ids as strings
                dtype={column_names.well_id: str},
            )

            # Updating the `index` to keep it consistent with the row numbers in file
            self.data.index += 2

        else:
            raise_exception(
                "Unsupported input file format. Only .xlsx, .xls, and .csv are supported.",
                TypeError,
            )

        # Read input options
        self.config = CONFIG(kwargs)
        self._removed_rows = {}
        self._col_names = column_names
        self._well_types = {}

        # Store number of wells in the input data
        num_wells_input = self.data.shape[0]

        LOGGER.info("Finished reading the well data.")
        if not self.config.preliminary_data_check:
            return

        LOGGER.info("Beginning to perform preliminary data checks.")
        self._process_input_data()

        num_wells_processed = self.data.shape[0]
        LOGGER.info("Finished preliminary processing.")

        perc_wells_removed = round(
            100 * (num_wells_input - num_wells_processed) / num_wells_input, 2
        )
        msg = (
            f"Preliminary processing removed {num_wells_input - num_wells_processed} "
            f"wells ({perc_wells_removed}% wells in the input data set) because of missing "
            f"information. To include these wells in the analysis, please provide the missing "
            f"information. List of wells removed can be queried using the get_removed_wells "
            f"and get_removed_wells_with_reason properties."
        )
        if num_wells_processed < num_wells_input:
            LOGGER.warning(msg)

    def __contains__(self, val):
        """Checks if a column is available in the well data"""
        return val in self.data.columns

    def __iter__(self):
        """Iterate over all rows of the well data"""
        return iter(self.data.index)

    @property
    def get_removed_wells(self):
        """Returns the list of wells removed from the data set"""
        row_list = []
        for val in self._removed_rows.values():
            row_list += val

        row_list.sort()

        return row_list

    @property
    def get_removed_wells_with_reason(self):
        """
        Returns the list of wells removed from the data set.
        Keys contain the reason, and the values contain the list of rows removed.
        """
        return self._removed_rows

    @property
    def get_flag_columns(self):
        """Returns all the columns containing flagged wells"""
        return [col for col in self.data.columns if "_flag" in col]

    @property
    def get_priority_score_columns(self):
        """Returns all the columns containing metric scores"""
        return [col for col in self.data.columns if " Score " in col]

    @property
    def get_gas_oil_wells(self):
        """
        Partitions the set of wells as gas wells and oil wells
        """
        well_type = self._well_types  # Well Types
        if well_type["gas"] is not None and well_type["oil"] is not None:
            return {
                "gas": self._construct_sub_data(well_type["gas"]),
                "oil": self._construct_sub_data(well_type["oil"]),
            }

        LOGGER.warning(
            "Insufficient information for well categorization. Either specify "
            "well_type in the input data, or provide both ann_gas_production "
            "and ann_oil_production"
        )

        return None

    @property
    def get_shallow_deep_wells(self):
        """
        Partitions the set of wells as shallow or deep
        """
        well_type = self._well_types  # Well Types
        if well_type["shallow"] is not None and well_type["deep"] is not None:
            return {
                "deep": self._construct_sub_data(well_type["deep"]),
                "shallow": self._construct_sub_data(well_type["shallow"]),
            }

        LOGGER.warning(
            "Insufficient information for well categorization. Either specify "
            "well_type_by_depth in the input data, or specify threshold_depth "
            "while instantiating the WellData object."
        )

        return None

    @property
    def get_fully_partitioned_data(self):
        """
        Partitions the wells as Oil-Deep, Oil-Shallow, Gas-Deep, Gas-Shallow
        wells, if all the required information is available
        """
        well_type = self._well_types
        if None not in well_type.values():
            return {
                "deep_oil": self._construct_sub_data(
                    well_type["deep"].intersection(well_type["oil"])
                ),
                "shallow_oil": self._construct_sub_data(
                    well_type["shallow"].intersection(well_type["oil"])
                ),
                "deep_gas": self._construct_sub_data(
                    well_type["deep"].intersection(well_type["gas"])
                ),
                "shallow_gas": self._construct_sub_data(
                    well_type["shallow"].intersection(well_type["gas"])
                ),
            }

        LOGGER.warning("Insufficient information for well categorization.")

        return None

    def has_incomplete_data(self, col_name: str):
        """
        Checks if a column contains empty cells.

        Parameters
        ----------
        col_name : str
            Name of the column

        Returns
        -------
        flag : bool
            True, if the column has empty cells; False, otherwise

        empty_cells : list
            A list of rows containing empty cells
        """
        flag = False

        empty_cells = list(self.data[self.data[col_name].isna()].index)
        if len(empty_cells) > 0:
            LOGGER.warning(f"Found empty cells in column {col_name}")
            flag = True

        return flag, empty_cells

    def drop_incomplete_data(self, col_name: str, dict_key: str):
        """
        Removes rows(wells) if the cell in a specific column is empty.

        Parameters
        ----------
        col_name : str
            Name of the column

        dict_key : str
            A key/identifier to store the reason for removal

        Returns
        -------
        None
        """
        # NOTE: Can avoid reassignment by passing the argument inplace=True.
        # But this is not recommended.
        new_well_data = self.data.dropna(subset=col_name)

        if new_well_data.shape[0] == self.data.shape[0]:
            # There is no incomplete data in col_list, so return
            return

        LOGGER.warning(
            f"Removed a few wells because {col_name} information is not available for them."
        )
        removed_rows = [i for i in self.data.index if i not in new_well_data.index]
        if dict_key in self._removed_rows:
            self._removed_rows[dict_key] += removed_rows
        else:
            self._removed_rows[dict_key] = removed_rows

        # Replace the pointer to the new DataFrame
        self.data = new_well_data

    def fill_incomplete_data(
        self,
        col_name: str,
        value: Union[float, int, str],
        flag_col_name: Union[None, str] = None,
    ):
        """
        Fill empty cells in a column with a constant value

        Parameters
        ----------
        col_name : str
            Name of the column

        value : float, int, str
            Empty cells in the column will be filled with `value`

        flag_col_name : str
            If specified, creates a new column called `flag_col_name`,
            and flags those wells with empty cells in column `col_name`.
        """

        # This loop adds a flag if an empty cell if filled with a default value
        if flag_col_name is not None:
            flag, empty_cells = self.has_incomplete_data(col_name)
            self.flag_wells(rows=empty_cells, col_name=flag_col_name)
            # Return if there are no empty cells
            if not flag:
                return

        LOGGER.warning(
            f"Filling any empty cells in column {col_name} with value {value}."
        )

        self.data[col_name] = self.data[col_name].fillna(value)

    def replace_data(self, col_name: str, from_to_map: dict):
        """
        Method to replace specific data with new values.

        Parameters
        ----------
        col_name : str
            Name of the column

        from_to_map : dict
            {key = current data, and value = new data}
        """
        with pd.option_context("future.no_silent_downcasting", True):
            self.data[col_name] = (
                self.data[col_name].replace(from_to_map).infer_objects()
            )

    def convert_data_to_binary(self, col_name: str):
        """
        Converts 1/True/"Yes"/"Y" -> 1 and 0/False/"No"/"N" -> 0.
        The strings "Yes", "Y", "No", and "N" are case-insensitive i.e.,
        "yes", "y", "no", "n" are also acceptable.

        Parameters
        ----------
        col_name : str
            Name of the column

        Raises
        ------
        ValueError
            If the column contains non-boolean-type data
        """
        # First convert the data to True or False
        try:
            self.data[col_name] = self.data[col_name].apply(Bool)
        except ValueError as excp:
            raise ValueError(
                f"Column {col_name} is expected to contain boolean-type "
                f"data. Received a non-boolean value for some/all rows."
            ) from excp

        # Now convert the data to binary
        self.data[col_name] = self.data[col_name].apply(int)

    def check_data_in_range(self, col_name: str, lb: float, ub: float):
        """
        Utility to check if all the values in a column are within
        a valid interval.

        Parameters
        ----------
        col_name : str
            Name of the column
        lb : float
            Lower bound for the value
        ub : float
            Upper bound for the value

        Raises
        ------
        ValueError
            If one or more values in the column lie outside the valid range
        """
        # Check the validity of latitude and longitude
        valid_data = self.data[col_name].apply(lambda val: lb <= val <= ub)
        invalid_data = list(self.data[~valid_data].index)

        if len(invalid_data) > 0:
            msg = (
                f"Values in column {col_name} are expected to be in the interval "
                f"{[lb, ub]}. However, the value in one or more rows "
                f"lies outside the interval."
            )
            raise_exception(msg, ValueError)

    def is_data_numeric(self, col_name: str):
        """
        Checks if all the cells in a column are numeric.

        Parameters
        ----------
        col_name : str
            Name of the column
        """
        # There are two ways to do this:
        # The current approach is a safe option since it also works for mixed
        # data types, but it is a bit messy.
        # Alternatively, one can use pd.api.types.is_numeric_dtype() method.
        # This is quite compact, but this has risks. In the future, pandas will not
        # allow modification of cells in a column to a different dtype. Then, it will
        # be safe to use the pd.api.types.is_numeric_dtype() method.
        flag = True
        non_numeric_rows = []
        for i in self:
            # Record the row number if it is an empty cell, or
            # if it does not contain a numeric value
            if pd.isnull(self.data.loc[i, col_name]) or not pd.api.types.is_number(
                self.data.loc[i, col_name]
            ):
                non_numeric_rows.append(i)

        if len(non_numeric_rows) > 0:
            flag = False

        return flag, non_numeric_rows

    def flag_wells(self, rows: list, col_name: str):
        """
        Utility to flag a specific set of wells. Useful to record
        wells for which a specific data/metric is estimated.

        rows: list
            List of rows(wells) in the DataFrame

        col_name : str
            Name of the new column that contains the flag information
        """
        if len(rows) == 0:
            # Nothing to flag, so return.
            return

        self.data[col_name] = 0
        for row in rows:
            self.data.loc[row, col_name] = 1

    def add_new_columns(self):
        """
        Adds new columns to the DataFrame. Read -> Remove deleted rows -> Join columns
        """
        # NOTE: Can avoid using this method by importing all the required columns
        # at the beginning. Just have to register unsupported columns in the
        # WellDataColumnNames object using obj.register_new_columns() method before
        # creating the WellData object.
        # NOTE: Not supporting it currently, since it is not crucial. This can be
        # supported in the future.
        msg = (
            "This method is not supported currently. Ensure that all the required columns "
            "are specified in the WellDataColumnNames object. To read unsupported data, "
            "register the corresponding column using the `register_new_columns` method."
        )
        raise_exception(msg, NotImplementedError)

    def _check_age_depth_availability(self, column):
        """
        Checks if the age/depth of the well is available or not. If not, then
        this method fills the missing information.

        column : str
            Must be either "age" or "depth"
        """
        col_name = getattr(self._col_names, column)
        missing_method = getattr(self.config, "missing_" + column)
        fill_value = getattr(self.config, "fill_" + column)
        flag_col_name = column + "_flag"

        # Uncomment this when estimation is supported.
        # estimiation_function = {
        #     "age": None,  # Replace None with function name later
        #     "depth": None,  # Replace None with function name later
        # }

        if missing_method == "fill":
            self.fill_incomplete_data(col_name, fill_value, flag_col_name)

        elif missing_method == "remove":
            self.drop_incomplete_data(col_name=col_name, dict_key=column)

        elif missing_method == "estimate":
            LOGGER.warning(f"Estimating the {column} of a well if it is missing.")
            # Uncomment the line below when it is supported
            # estimiation_function[column](self)
            # NOTE: Make sure to flag wells for which info is estimated
            raise_exception(
                f"{column} estimation feature is not supported currently.",
                NotImplementedError,
            )

    def _filter_production_volume(self):
        """
        Removes wells whose lifelong production volume is greater
        than a threshold value. The threshold value for gas and oil can be specified
        via threshold_gas_production and threshold_oil_production arguments
        while instantiating the WellData object
        """
        wcn = self._col_names
        if wcn.life_gas_production is None and wcn.life_oil_production is None:
            LOGGER.warning(
                "Unable to filter wells based on production volume because both lifelong gas "
                "production [in Mcf] and lifelong oil production [in bbl] are missing in the "
                "input data. Please provide at least one of the columns via attributes "
                "life_gas_production and life_oil_production in the WellDataColumnNames object."
            )
            return

        self._removed_rows["production_volume"] = []
        if wcn.life_gas_production in self:
            self.fill_incomplete_data(
                col_name=wcn.life_gas_production,
                value=self.config.fill_life_gas_production,
                flag_col_name="life_gas_production_flag",
            )
            # Remove wells if their production volume is greater than the threshold
            production_volume = self.config.threshold_gas_production
            remove_rows = self.data[
                self.data[wcn.life_gas_production] >= production_volume
            ].index
            self._removed_rows["production_volume"] += list(remove_rows)
            self.data = self.data.drop(remove_rows)

        if wcn.life_oil_production in self:
            self.fill_incomplete_data(
                col_name=wcn.life_oil_production,
                value=self.config.fill_life_oil_production,
                flag_col_name="life_oil_production_flag",
            )
            # Remove wells if their production volume is greater than the threshold
            production_volume = self.config.threshold_oil_production
            remove_rows = self.data[
                self.data[wcn.life_oil_production] >= production_volume
            ].index
            self._removed_rows["production_volume"] += list(remove_rows)
            self.data = self.data.drop(remove_rows)

        if len(self._removed_rows["production_volume"]) > 0:
            LOGGER.warning(
                "Some wells have been removed based on the lifelong production volume."
            )

    def _categorize_oil_gas(self):
        """
        Compare gas and oil production levels and add a 'Well Type' column.
        """
        # NOTE: If possible, consolidate _categorize_oil_gas and
        #   _categorize_shallow_deep methods into one.
        wcn = self._col_names
        wt_col_name = wcn.well_type
        if wt_col_name in self:
            # Well Type is already present in the input data
            self.fill_incomplete_data(
                col_name=wt_col_name,
                value=self.config.fill_well_type,
                flag_col_name="well_type_flag",
            )

            oil_wells, gas_wells = set(), set()
            for r in self:
                if self.data.loc[r, wt_col_name].lower() in "oil":
                    # Writing it in this manner to support case-insensitive entries
                    oil_wells.add(r)
                elif self.data.loc[r, wt_col_name].lower() in "gas":
                    gas_wells.add(r)
                else:
                    msg = (
                        f"Well-type must be either oil or gas. Received "
                        f"{self.data.loc[r, wt_col_name]} in row {r}."
                    )
                    raise_exception(msg, ValueError)

            self._well_types["oil"] = oil_wells
            self._well_types["gas"] = gas_wells
            return

        # Ensure the required columns are present in the DataFrame
        # FIXME: This value is being accepted as input at two different locations.
        #   Value-check is added in the assess_supported_metrics method for now.
        if wcn.ann_gas_production in self and wcn.ann_oil_production in self:
            self.fill_incomplete_data(
                wcn.ann_gas_production,
                self.config.fill_ann_gas_production,
                "ann_gas_production_flag",
            )
            self.fill_incomplete_data(
                wcn.ann_oil_production,
                self.config.fill_ann_oil_production,
                "ann_oil_production_flag",
            )

            # Uncomment the code to add the column to self.data (Not recommended)
            # # Convert oil production from bbl/Year to Mcf/Year using the conversion factor
            # self.data["Oil [Mcf/Year]"] = (
            #     self.data[wcn.ann_oil_production] * CONVERSION_FACTOR
            # )
            # wcn.well_type = "Well Type"
            # self.data[wcn.well_type] = self.data.apply(
            #     lambda row: (
            #         "Gas" if row[wcn.ann_gas_production] > row["Oil [Mcf/Year]"] else "Oil"
            #     ),
            #     axis=1,
            # )
            # # Drop the intermediate 'Oil [Mcf/year]' column
            # self.data = self.data.drop(columns=["Oil [Mcf/Year]"])

            self._well_types["oil"] = set(
                self.data[
                    self.data[wcn.ann_oil_production] * CONVERSION_FACTOR
                    >= self.data[wcn.ann_gas_production]
                ].index
            )
            self._well_types["gas"] = set(self.data.index) - self._well_types["oil"]
            return

        # Required information is not provided: cannot categorize wells
        self._well_types["oil"] = None
        self._well_types["gas"] = None

    def _categorize_shallow_deep(self):
        """
        Categorizes each well as either deep or shallow.
        """
        depth_col_name = self._col_names.depth
        wt_col_name = self._col_names.well_type_by_depth
        threshold_depth = self.config.threshold_depth

        if wt_col_name in self:
            # Information is already provided. Fill empty cells.
            self.fill_incomplete_data(
                col_name=wt_col_name,
                value=self.config.fill_well_type_depth,
                flag_col_name="depth_type_flag",
            )

            deep_wells, shallow_wells = set(), set()
            for r in self:
                if self.data.loc[r, wt_col_name].lower() in "shallow":
                    # Writing it in this manner to support case-insensitive entries
                    shallow_wells.add(r)
                elif self.data.loc[r, wt_col_name].lower() in "deep":
                    deep_wells.add(r)
                else:
                    msg = (
                        f"Well-type by depth must be either shallow or deep. Received "
                        f"{self.data.loc[r, wt_col_name]} in row {r}."
                    )
                    raise_exception(msg, ValueError)

            self._well_types["deep"] = deep_wells
            self._well_types["shallow"] = shallow_wells
            return

        if threshold_depth is not None:
            # Well-type is not specified, but the threshold depth is specified.
            # Classifying wells based on this information.
            # Depth information has already been processed, so there will not be
            # any incomplete cells in this column

            # Uncomment the following lines to add the column to self.data (Not recommended)
            # wt_col_name = "Well Type by Depth"
            # self._col_names.well_type_by_depth = wt_col_name
            # self.data[wt_col_name] = self.data.apply(
            #     lambda row: (
            #         "Deep" if row[depth_col_name] > threshold_depth else "Shallow"
            #     ),
            #     axis=1,
            # )
            self._well_types["deep"] = set(
                self.data[self.data[depth_col_name] >= threshold_depth].index
            )
            self._well_types["shallow"] = (
                set(self.data.index) - self._well_types["deep"]
            )
            return

        # Required information is not given, so cannot categorize wells
        self._well_types["deep"] = None
        self._well_types["shallow"] = None

    def _construct_sub_data(self, rows):
        """
        Constructs a new WellData object with a subset of rows.

        rows: iterable object
            list/set/any iterable object containing the subset of row indices
            to include in the new object
        """
        row_list = list(rows)  # Convert the object to a list
        row_list.sort()

        config_options = dict(self.config)
        # Preliminary data check is not needed, since the data is already processed
        config_options["preliminary_data_check"] = False

        # This helps keep the new object independent from the original object
        column_names = copy.deepcopy(self._col_names)

        return WellData(
            filename=self.data.loc[row_list],
            column_names=column_names,
            **config_options,
        )

    def _process_input_data(self):
        """
        Performs data checks i.e., if they are within valid interval, fills/removes
        missing information, and classifies wells as oil/gas/deep/shallow.
        """
        wcn = self._col_names

        LOGGER.info("Checking availability and uniqueness of well ids.")
        # Drop wells for which well id is not provided.
        self.drop_incomplete_data(col_name=wcn.well_id, dict_key="well_id")

        # Check uniqueness
        well_list = set(self.data[wcn.well_id])
        if len(well_list) < self.data.shape[0]:
            raise_exception(
                "Well ID must be unique. Found multiple wells with the same Well ID.",
                ValueError,
            )

        # Drop wells for which longitude and latitude information is not available
        LOGGER.info(
            "Checking if latitude and longitude information is available for all wells."
        )
        self.drop_incomplete_data(col_name=wcn.latitude, dict_key="latitude")
        self.drop_incomplete_data(col_name=wcn.longitude, dict_key="longitude")

        # Drop wells if the operator name is not available
        if self.config.ignore_operator_name:
            LOGGER.info("Checking if the operator name is available for all wells.")
            self.drop_incomplete_data(
                col_name=wcn.operator_name, dict_key="operator_name"
            )

        # Sometimes owner_name is listed as unknown. Remove those as well
        unknown_owner = []
        for row in self:
            if self.data.loc[row, wcn.operator_name].lower() == "unknown":
                unknown_owner.append(row)

        if self.config.ignore_operator_name and len(unknown_owner) > 0:
            LOGGER.warning(
                "Owner name for some wells is listed as unknown."
                "Treating these wells as if the owner name is not provided, "
                "so removing them from the dataset."
            )
            self.data = self.data.drop(unknown_owner)
            self._removed_rows["unknown_owner"] = unknown_owner

        # Check if age data is available, and calculate it if it is missing
        LOGGER.info("Checking if age of all wells is available.")
        self._check_age_depth_availability(column="age")

        # Check if depth data is available, and calculate it if it is missing
        LOGGER.info("Checking if depth of all wells is available.")
        self._check_age_depth_availability(column="depth")

        # Filter wells based on production volume
        if (
            self.config.threshold_gas_production is not None
            or self.config.threshold_oil_production is not None
        ):
            self._filter_production_volume()

        # Categorize wells: Shallow/Deep, Oil/Gas
        self._categorize_oil_gas()
        self._categorize_shallow_deep()

        # Check the validity of values of latitude, longitude, age, depth
        # Assuming that no well is older than 350 year old
        # Assuming that no well is deeper than 40000 ft
        self.check_data_in_range(wcn.latitude, -90, 90)
        self.check_data_in_range(wcn.longitude, -90, 90)
        self.check_data_in_range(wcn.age, 0.0, 350.0)
        self.check_data_in_range(wcn.depth, 0.0, 40000.0)

        LOGGER.info("Completed processing the essential inputs.")

    def _append_fed_dac_data(self):
        """Appends federal DAC data"""
        census_year = self.config.census_year
        # TODO:
        # Append Tract ID/GEOID, population density, Total population, land area,
        # federal DAC score to the DataFrame

    def compute_priority_scores(self, impact_metrics: ImpactMetrics):
        """
        Computes scores for all metrics/submetrics (supported and custom metrics) and
        the total priority score. This method must be called after processing the
        data for custom metrics (if any).

        Parameters
        ----------
        impact_metrics : ImpactMetrics
            Object containing impact metrics and their weights
        """
        # Check the validity of impact metrics before calculating priority score
        im_mt = impact_metrics
        im_mt.check_validity()

        # Check if all the required columns for supported metrics are specified
        # If yes, register the name of the column containing the data in the
        # data_col_name attribute
        # TODO: Combine the check_columns_available method with this method.
        # TODO: Check fill data consistency for ann_gas_production and
        # ann_oil_production. See _categorize_gas_oil_wells method for details.
        self._col_names.check_columns_available(im_mt)

        for metric in im_mt:
            if metric.weight == 0 or hasattr(metric, "submetrics"):
                # Metric/submetric is not chosen, or
                # This is a parent metric, so no data assessment is required
                continue

            LOGGER.info(
                f"Computing scores for metric/submetric {metric.name}/{metric.full_name}."
            )

            if metric.name == "fed_dac":
                self._append_fed_dac_data()
                continue

            if metric.name == "well_count":
                operator_name = self._col_names.operator_name
                weight = im_mt.well_count.effective_weight
                metric.data_col_name = "Owner Well-Count"
                self.data[metric.data_col_name] = self.data.groupby(operator_name)[
                    operator_name
                ].transform("size")
                self.data[metric.score_col_name] = round(
                    weight
                    * self.data[metric.data_col_name].apply(
                        lambda x: math.e ** ((1 - x) / 10)
                    ),
                    4,
                )
                continue

            # For all other metrics/submetrics with a nonzero weight.
            # Step 1: Ensure that the data for this metric is provided.
            if metric.data_col_name is None:
                # This will not happen for supported metrics, because this check
                # is performed before in col_names.check_columns_available()
                # method. However, this may happen for non-supported/user-defined
                # custom metrics.
                msg = (
                    f"data_col_name attribute for metric {metric.name}/{metric.full_name} "
                    f"is not provided."
                )
                raise_exception(msg, ValueError)

            # Step 2: Fill incomplete data
            self.fill_incomplete_data(
                col_name=metric.data_col_name,
                value=metric.fill_missing_value,
                flag_col_name=metric.name + "_flag",
            )

            # Step 3: If it is a binary-type metric, then convert the data to 0-1
            if metric.is_binary_type:
                self.convert_data_to_binary(metric.data_col_name)

            # Step 4: Ensure that the column in numeric
            flag, non_numeric_rows = self.is_data_numeric(col_name=metric.data_col_name)
            if not flag:
                msg = (
                    f"Unable to compute scores for metric {metric.name}/"
                    f"{metric.full_name},  because the column {metric.data_col_name} "
                    f"contains non-numeric values in rows {non_numeric_rows}."
                )
                raise_exception(msg, ValueError)

            # Step 5: Compute the priority score
            max_value = self.data[metric.data_col_name].max()
            min_value = self.data[metric.data_col_name].min()

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
                self.data[metric.score_col_name] = (
                    (max_value - self.data[metric.data_col_name])
                    / (max_value - min_value)
                ) * metric.effective_weight

            else:
                self.data[metric.score_col_name] = (
                    (self.data[metric.data_col_name] - min_value)
                    / (max_value - min_value)
                ) * metric.effective_weight

        LOGGER.info("Computing the total priority score.")
        self.data["Priority Score [0-100]"] = self.data[
            self.get_priority_score_columns
        ].sum(axis=1)

        self.check_data_in_range("Priority Score [0-100]", 0.0, 100.0)
        LOGGER.info("Completed the calculation of priority scores.")

    def save_to_file(self, filename: str):
        """
        Writes the data to a file.

        Parameters
        ----------
        filename: str
            Name of the file
        """
        if filename.split(".")[-1] in ["xlsx", "xls"]:
            self.data.to_excel(filename)

        elif filename.split(".")[-1] == "csv":
            self.data.to_csv(filename)

        else:
            file_format = "." + filename.split(".")[-1]
            raise_exception(
                f"Format {file_format} is not supported.",
                ValueError,
            )