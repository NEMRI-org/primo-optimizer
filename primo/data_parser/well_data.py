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
# pylint: disable=too-many-lines

# Standard libs
import copy
import logging
import math
import pathlib
from typing import Optional, Union

# Installed libs
import numpy as np
import pandas as pd
from pyomo.common.config import Bool, document_kwargs_from_configdict

# User-defined libs
from primo.data_parser import EfficiencyMetrics, ImpactMetrics, SetOfMetrics
from primo.data_parser.default_data import CONVERSION_FACTOR, DAC_TRACT_YEAR
from primo.data_parser.input_config import data_config
from primo.data_parser.well_data_columns import WellDataColumnNames
from primo.utils.census_utils import (
    get_cejst_data,
    get_data_as_geodataframe,
    get_state_census_tracts,
    identify_state,
)
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)

CONFIG = data_config()

OWNER_WELL_COLUMN_NAME = "Owner Well-Count"


# pylint: disable=too-many-public-methods
class WellData:
    """
    Reads, processes, and analyzes well data.
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
        data: Union[str, pd.DataFrame],
        column_names: WellDataColumnNames,
        **kwargs,
    ) -> None:
        """
        Reads well data and performs initial high-level data processing.

        Parameters
        ----------
        data : Union[str, pd.DataFrame]
            If a string is provided, the argument is interpreted as a file path
            containing the well data. Currently, only .xlsx, .xls, and .csv
            formats are supported.
            If a DataFrame is provided, it is directly utilized

        column_names : WellDataColumnNames
            A WellDataColumnNames object containing the names of various columns
        """
        # Import columns whose column names have been provided.
        if isinstance(data, pd.DataFrame):
            self.data = data
            # Check if the columns are present in the data
            for col in column_names.values():
                assert col in self.data.columns

        elif isinstance(data, str):
            LOGGER.info("Reading the well data from the input file.")
            extension = pathlib.Path(data).suffix
            if extension in [".xlsx", ".xls"]:
                self.data = pd.read_excel(
                    data,
                    sheet_name=kwargs.pop("sheet", 0),
                    usecols=column_names.values(),
                    # Store well ids as strings
                    dtype={column_names.well_id: str},
                )
            elif extension == ".csv":
                self.data = pd.read_csv(
                    data,
                    usecols=column_names.values(),
                    # Store well ids as strings
                    dtype={column_names.well_id: str},
                )

            else:
                raise_exception(
                    "Unsupported input file format. Only .xlsx, .xls, and .csv are supported.",
                    TypeError,
                )
            # Updating the `index` to keep it consistent with
            # the row numbers in file
            self.data.index += 2
            LOGGER.info("Finished reading the well data.")

        else:
            raise_exception("Unknown variable type for input data", TypeError)

        # Read input options
        self.config = CONFIG(kwargs)
        self._removed_rows = {}
        self._col_names = column_names
        self._well_types = {}

        # Store number of wells in the input data
        num_wells_input = self.data.shape[0]

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

        self.set_impact_and_efficiency_metrics(
            self.config.impact_metrics, self.config.efficiency_metrics
        )

    def __contains__(self, val: str):
        """Checks if a column is available in the well data"""
        return val in self.data.columns

    def __getitem__(self, col_name: str):
        """Returns the col_name column in the DataFrame"""
        return self.data[col_name]

    def __iter__(self):
        """Iterate over all rows of the well data"""
        return iter(self.data.index)

    def __len__(self):
        """Returns number of wells in the dataset"""
        return len(self.data)

    @property
    def col_names(self) -> WellDataColumnNames:
        """
        Returns the WellDataColumnNames object associated with the current object
        """
        return self._col_names

    @property
    def get_removed_wells(self):
        """Returns the list of wells removed from the data set"""
        row_list = []
        for val in self._removed_rows.values():
            row_list += val

        return sorted(row_list)

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

    def get_high_priority_wells(self, num_wells: int):
        """Returns the top n wells by priority"""

        if not hasattr(self._col_names, "priority_score"):
            LOGGER.warning("Returning None, since priority scores are not available!")
            return None

        well_list = (
            self.data.sort_values("Priority Score [0-100]", ascending=False)
            .head(num_wells)
            .index.to_list()
        )
        return self._construct_sub_data(well_list)

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
        new_well_data = self.data.dropna(subset=col_name)

        if new_well_data.shape[0] == self.data.shape[0]:
            # There is no incomplete data in col_list, so return
            return

        LOGGER.warning(
            f"Removed wells because {col_name} information is not available for them."
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
        flag_col_name: Optional[str] = None,
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

        # This loop adds a flag if an empty cell is filled with a default value
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
        self.data.loc[rows, col_name] = 1

    def add_new_column_ordered(
        self,
        column_var_name: str,
        column_header_name: str,
        values: Union[np.array, pd.DataFrame, list],
    ):
        """
        Adds a single column to the WellData and other related data structures
        The column must be in the correct order corresponding to the existing data

        Parameters
        ----------
        col_name : list(str)
            List of 2 strings: the column variable name, the column header for the data
        values : np.array, pd.DataFrame, list
            The values for the column
        """

        self._col_names.register_new_columns({column_var_name: column_header_name})

        if len(values) != len(self.data):
            raise_exception(
                "The length of the added column must match the length of the current data.",
                AttributeError,
            )

        self.data[column_header_name] = values

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
        # estimation_function = {
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
            # estimation_function[column](self)
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
                elif self.data.loc[r, wt_col_name].lower() in "both":
                    row = self.data.iloc[r]
                    oil_prod = row[wcn.ann_oil_production]
                    gas_prod = row[wcn.ann_gas_production]

                    if oil_prod * CONVERSION_FACTOR > gas_prod:
                        oil_wells.add(r)
                    else:
                        gas_wells.add(r)
                else:
                    msg = (
                        f"Well-type must be either oil or gas or both. Received "
                        f"{self.data.loc[r, wt_col_name]} in row {r}."
                    )
                    raise_exception(msg, ValueError)

            self._well_types["oil"] = oil_wells
            self._well_types["gas"] = gas_wells
            return

        # Ensure the required columns are present in the DataFrame
        # FIXME: This value is being accepted as input at two different locations. # pylint: disable=fixme
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
            data=self.data.loc[row_list],
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
        unknown_owner = []
        if self.config.verify_operator_name:
            LOGGER.info("Checking if the operator name is available for all wells.")
            self.drop_incomplete_data(
                col_name=wcn.operator_name, dict_key="operator_name"
            )
            # Sometimes owner_name is listed as unknown. Remove those as well
            for row in self:
                if self.data.loc[row, wcn.operator_name].lower() == "unknown":
                    unknown_owner.append(row)

            if len(unknown_owner) > 0:
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
        self.check_data_in_range(wcn.longitude, -180, 180)
        self.check_data_in_range(wcn.age, 0.0, 350.0)
        self.check_data_in_range(wcn.depth, 0.0, 40000.0)

        LOGGER.info("Completed processing the essential inputs.")

    def _append_fed_dac_data(self):
        """Appends federal DAC data"""
        if len(self.data) == 0:
            # Nothing to do if no data
            return

        state_code = identify_state(self)
        census_tracts = get_state_census_tracts(state_code, DAC_TRACT_YEAR)
        gdf = get_data_as_geodataframe(self)

        # Spatial join to identify tract id associated with every well
        # Per 2010 data
        gdf = gdf.sjoin(census_tracts, how="left", predicate="within")
        self.add_new_column_ordered(
            "geoid", "Census Tract ID [2010]", pd.to_numeric(gdf["GEOID10"])
        )

        cejst_data = get_cejst_data()
        joined_data = self.data.merge(
            cejst_data,
            left_on="Census Tract ID [2010]",
            right_on="Census tract 2010 ID",
            how="left",
        )

        fed_dac_data = (
            joined_data["Percentage of tract that is disadvantaged by area"]
            + joined_data["Percentage of tract that is disadvantaged by area"]
        ) / 2

        self.add_new_column_ordered("fed_dac", "Federal DAC Data", fed_dac_data)

        # Assume disadvantaged score of 0 for rows where value was not found
        self.fill_incomplete_data(self.col_names.fed_dac, 0, "fed_dac_flag")

    def _set_metric(self, metrics: SetOfMetrics):
        """
        Validates and processes data for a set of metrics
        """
        # validate metric
        metrics.check_validity()
        # check columns
        self._col_names.check_columns_available(metrics)
        # process columns
        self._process_data(metrics)

    def _process_data(self, metrics: SetOfMetrics):
        """
        Process the data
        """
        for metric in metrics:
            if metric.weight == 0 or hasattr(metric, "submetrics"):
                # Metric/submetric is not chosen, or
                # This is a parent metric, so no data assessment is required
                continue

            if metric.name in (
                "fed_dac",
                "well_count",
                "num_wells",
                "num_unique_owners",
            ):
                # these have their own processing functions (or none at all)
                continue

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

            # check incomplete data
            self.fill_incomplete_data(
                col_name=metric.data_col_name,
                value=metric.fill_missing_value,
                flag_col_name=metric.name + "_flag",
            )

            # Step 3: If it is a binary-type metric, then convert the data to 0-1
            if metric.is_binary_type:
                self.convert_data_to_binary(metric.data_col_name)

            # Step 4: Ensure that the column is numeric
            flag, non_numeric_rows = self.is_data_numeric(col_name=metric.data_col_name)
            if not flag:
                msg = (
                    f"Unable to compute scores for metric {metric.name}/"
                    f"{metric.full_name},  because the column {metric.data_col_name} "
                    f"contains non-numeric values in rows {non_numeric_rows}."
                )
                raise_exception(msg, ValueError)

    def _process_dac_data(self):
        """
        processes the DAC data
        """
        self._append_fed_dac_data()
        metric = self.config.impact_metrics.fed_dac
        weight = metric.effective_weight
        metric.data_col_name = self.col_names.fed_dac
        self.data[metric.score_col_name] = (
            self.data[metric.data_col_name] * weight / 100
        )

    def _compute_well_count_score(self):
        """
        process well count data
        """
        metrics = self.config.impact_metrics
        metric = [obj for obj in metrics if obj.name == "well_count"][0]

        operator_name = self._col_names.operator_name
        weight = metrics.well_count.effective_weight
        metric.data_col_name = OWNER_WELL_COLUMN_NAME
        self.data[metric.data_col_name] = self.data.groupby(operator_name)[
            operator_name
        ].transform("size")
        self.data[metric.score_col_name] = round(
            weight
            * self.data[metric.data_col_name].apply(lambda x: math.e ** ((1 - x) / 10)),
            4,
        )

    def set_impact_and_efficiency_metrics(
        self,
        impact_metrics: Optional[ImpactMetrics] = None,
        efficiency_metrics: Optional[EfficiencyMetrics] = None,
    ):
        """
        Validates and processes data for the impact and efficiency metrics.

        Parameters
        ----------
        impact_metrics : ImpactMetrics
            impact metrics object to be used for prioritization
        efficiency_metrics : EfficiencyMetrics
            efficiency metrics object to be used for efficiency score computation
        """
        if impact_metrics is not None:
            self.config.impact_metrics = impact_metrics
            self._set_metric(self.config.impact_metrics)
        if efficiency_metrics is not None:
            self.config.efficiency_metrics = efficiency_metrics
            self._set_metric(self.config.efficiency_metrics)

    def compute_priority_scores(self):
        """
        Computes scores for all metrics/submetrics (supported and custom metrics) and
        the total priority score. This method must be called after processing the
        data for custom metrics (if any).

        """
        # Check if all the required columns for supported metrics are specified
        # If yes, register the name of the column containing the data in the
        # data_col_name attribute
        # ann_oil_production. See _categorize_gas_oil_wells method for details.

        for metric in self.config.impact_metrics:
            if metric.weight == 0 or hasattr(metric, "submetrics"):
                # Metric/submetric is not chosen, or
                # This is a parent metric, so no data assessment is required
                continue

            LOGGER.info(
                f"Computing scores for metric/submetric {metric.name}/{metric.full_name}."
            )

            if metric.name == "fed_dac":
                self._process_dac_data()
                continue

            if metric.name == "well_count":
                self._compute_well_count_score()
                continue

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
        self.add_new_column_ordered(
            "priority_score",
            "Priority Score [0-100]",
            self.data[self.get_priority_score_columns].sum(axis=1),
        )

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
        extension = pathlib.Path(filename).suffix
        if extension in [".xlsx", ".xls"]:
            self.data.to_excel(filename)

        elif extension == ".csv":
            self.data.to_csv(filename)

        else:
            raise_exception(
                f"Format {extension} is not supported.",
                ValueError,
            )
