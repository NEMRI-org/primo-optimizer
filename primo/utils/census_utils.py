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

"""
This file contains utilities to query and use census data.
"""

# Standard libs
import logging
import os
import tempfile
from typing import List, Tuple, Union

# Installed libs
import censusgeocode as cg
import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv

# User-defined libs
from primo.utils import CENSUS_YEAR
from primo.utils.download_utils import download_file, unzip_file
from primo.utils.geo_utils import is_acceptable, is_valid_lat, is_valid_long
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)

CODE_LENGTH = {"STATE": 2, "COUNTY": 3, "TRACT": 6, "BLOCK_GROUP": 1, "BLOCK": 3}

CODE_ORDER = ["STATE", "COUNTY", "TRACT", "BLOCK_GROUP", "BLOCK"]


def get_state_census_tracts(state_code: str, census_year: int) -> gpd.GeoDataFrame:
    """
    Retrieves a GeoDataFrame based on a shapefile available via the US Census
    that makes it easy to identify census tract ids for a list of lat/longs

    Parameters
    ----------
    state_code : str
        The two-digit string identifier for a state

    census_year : int
        The census year for which the census tract designations is to be downloaded

    Returns
    -------
    gpd.GeoDataFrame
        The census tract ids associated with the state
    """
    url = None
    if census_year == 2020:
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER2020/"
            f"TRACT/tl_2020_{state_code}_tract.zip"
        )
    elif census_year == 2010:
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER2010/"
            f"TRACT/2010/tl_2010_{state_code}_tract10.zip"
        )
    else:
        raise_exception(
            f"Getting census tracts for census_year: {census_year} is not implemented",
            ValueError,
        )
    # pylint: disable=consider-using-with
    # Disabling this is necessary as unzip_file seems to run into issues with temp
    # paths
    temp_path = tempfile.NamedTemporaryFile().name
    extract_path = tempfile.NamedTemporaryFile().name
    download_file(temp_path, url)
    unzip_file(temp_path, extract_path)
    return gpd.read_file(extract_path)


def get_cejst_data() -> pd.DataFrame:
    """
    Downloads and returns the CEJST data from
    https://screeningtool.geoplatform.gov as a DataFrame

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the Climate and Economic Justice Screening Tool Data
    """
    # pylint: disable=consider-using-with
    # Disabling this is necessary as pd.read_csv seems to run into issues with temp path
    temp_path = tempfile.NamedTemporaryFile().name
    url = (
        "https://static-data-screeningtool.geoplatform.gov/data-versions/"
        "1.0/data/score/downloadable/1.0-communities.csv"
    )
    download_file(temp_path, url)
    return pd.read_csv(temp_path)


def get_census_key() -> str:
    """
    Retrieve the US Census API key from the environment file stored under .env.

    Returns
    -------
    str
        The Census API key.

    Raises
    ------
    KeyError if CENSUS_KEY is not found in .env file
    """
    load_dotenv()
    return os.environ["CENSUS_KEY"]


def identify_state(data) -> str:
    """
    Attempts to infer the state for which the dataset belong.
    Assumes that all data points belong to the same state and
    we have at least one row of data

    Parameters
    ----------
    data : WellData
        A well data object

    Returns
    -------
    Two-digit code identifying the state for which the data belongs
    """
    wcn = data.col_names
    lat = data.data[wcn.latitude].iloc[0]
    long = data.data[wcn.longitude].iloc[0]
    fips_code = get_fips_code(lat, long)
    state = get_fips_part(fips_code, "STATE")
    return state


def get_data_as_geodataframe(data) -> gpd.GeoDataFrame:
    """
    Returns a Geopandas GeoDataFrame object from WellData Object

    Parameters
    ----------
    data : WellData
        Object containing relevant well data

    Returns
    -------
    gpd.GeoDataFrame
        A Geopandas GeoDataFrame object
    """
    wcn = data.col_names
    gdf = gpd.GeoDataFrame(
        data.data,
        geometry=gpd.points_from_xy(data[wcn.longitude], data[wcn.latitude]),
        crs="EPSG:4326",
    )
    return gdf


def make_fips_code(
    state: str,
    county: Union[str, None] = None,
    tract: Union[str, None] = None,
    block_group: Union[str, None] = None,
    block: Union[str, None] = None,
) -> str:
    """
    Returns a valid FIPS code based on the union of the individual components.
    The optional components are arranged in order of priority; for example,
    providing a value for tract without providing a value for county will lead to
    an Exception.

    Parameters
    ----------
    state : str
        two-digit code for U.S. state
    county : Union[str, None], optional
        three-digit code for county, defaults to None
    tract : Union[str, None], optional
        six-digit code for County Tract, defaults to None
    block_group : Union[str, None], optional
        one-digit code for block group, defaults to None
    block : Union[str, None], optional
        three-digit code for block, defaults to None

    Returns
    -------
    str
        Combined FIPS code for the appropriate county tract

    Raises
    ------
    ValueError
        If lower priority arguments are provided without higher priority arguments.
    """

    # Using strings to avoid having to pad 0s

    if len(state) != CODE_LENGTH["STATE"]:
        raise_exception(
            f"STATE FIPS Code is expected to be of "
            f"size {CODE_LENGTH['STATE']}, received: {state} ",
            ValueError,
        )

    fips_code = state

    skip_next = False

    for partial_code in (county, tract, block_group, block):
        if partial_code is None:
            skip_next = True
        else:
            if skip_next:
                raise_exception(
                    "Lower priority code provided without higher priority code",
                    ValueError,
                )

            fips_code += partial_code

    return fips_code


def get_identifier(fips_code: str, identifier: str) -> str:
    """
    Returns the appropriate part of the FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest
    identifier : str
        Identifier for the part of the FIPS code to extract

    Returns
    -------
    str
        Partial identifier within the FIPS code requested

    Raises
    ------
    ValueError
        If an invalid identifier is provided
    """
    is_acceptable(identifier, CODE_ORDER, "identifier", True)
    start = 0
    end = 0

    for keyword in CODE_ORDER:
        if keyword != identifier:
            start += CODE_LENGTH[keyword]
        else:
            end = start + CODE_LENGTH[keyword]
            break

    if len(fips_code) < end:
        raise_exception(
            f"FIPS code is of insufficient length to " f"extract {identifier}",
            ValueError,
        )
    return fips_code[start:end]


def get_state(fips_code: str) -> str:
    """
    Returns the two-digit state identifier from FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest

    Returns
    -------
    str
        The two-digit state identifier
    """
    return get_identifier(fips_code, "STATE")


def get_county(fips_code: str) -> str:
    """
    Returns a county identifier from FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest

    Returns
    -------
    str
        The county identifier
    """
    return get_identifier(fips_code, "COUNTY")


def get_tract(fips_code: str) -> str:
    """
    Returns the tract identifier from FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest

    Returns
    -------
    str
        The tract identifier
    """
    return get_identifier(fips_code, "TRACT")


def get_block_group(fips_code: str) -> str:
    """
    Returns a block group identifier from FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest

    Returns
    -------
    str
        The block identifier
    """
    return get_identifier(fips_code, "BLOCK_GROUP")


def get_block(fips_code: str) -> str:
    """
    Returns a block identifier from FIPS code.

    Parameters
    ----------
    fips_code : str
        FIPS code of interest

    Returns
    -------
    str
        The block identifier
    """
    return get_identifier(fips_code, "BLOCK")


def get_fips_code(latitude: float, longitude: float) -> str:
    """
    Returns the full FIPS code from latitude and longitude information.

    Parameters
    ----------
    latitude : float
        latitude
    longitude : float
        longitude

    Returns
    -------
    str
        The full 15-digit FIPS code
    """
    is_valid_lat(latitude)
    is_valid_long(longitude)
    try:
        geocoded_info = cg.coordinates(x=longitude, y=latitude)
    except ValueError:
        LOGGER.warning(f"FIPS Code not found for lat: {latitude}, long: {longitude}")
        return ""

    if not geocoded_info:
        LOGGER.warning(
            "No FIPS code found. Check if the latitude and longitude values "
            "are valid for a point within the US."
        )
        return ""

    key = f"{CENSUS_YEAR} Census Blocks"
    if key not in geocoded_info:
        LOGGER.warning("Requested census block is not available.")
        LOGGER.warning("FIPS Code returned will not have block or block group info.")
        key = "Census Tracts"

    return geocoded_info[key][0]["GEOID"]


def get_fips_part(fips_code: str, identifier: str) -> str:
    """
    Return part of the FIPS code corresponding to the identifier.
    Acceptable values for the identifier are:
    "STATE," "COUNTY," "TRACT," "BLOCK_GROUP," "BLOCK".

    Parameters
    ----------
    fips_code : str
        The full FIPS code string
    identifier : str
        One of the five possible values for a part of the FIPS code

    Returns
    -------
    str
        The part of the FIPS code corresponding to the identifier
    """
    is_acceptable(identifier, CODE_ORDER, "identifier", True)

    if identifier == "STATE":
        return get_state(fips_code)

    if identifier == "COUNTY":
        return get_county(fips_code)

    if identifier == "TRACT":
        return get_tract(fips_code)

    if identifier == "BLOCK_GROUP":
        return get_block_group(fips_code)

    return get_block(fips_code)


class CensusAPIException(Exception):
    """
    Custom Exception class to capture and log any exceptions
    arising from querying the CensusAPI
    """


class CensusAPIKeyError(CensusAPIException):
    """
    Custom Exception class to capture and log exceptions arising out of
    invalid/unavailable Census API Key
    """


class CensusClient:
    """
    Sets up methods to interact with and extract data from US Census API
    """

    def __init__(self, key: str):
        """
        Initialize the class.

        Parameters
        ----------
        key : str
            API Key registered with US Census
        """
        self._key = key
        self.session = requests.session()

    def _generate_geo_identifiers(self, fips_code: str) -> Tuple[str, Union[str, None]]:
        """
        Generate the string needed to query the Census API based on
        the FIPS code.

        Parameters
        ----------
        fips_code : str
            The FIPS code.

        Returns
        -------
        Tuple[str, Union[str, None]]
            Tuple containing 'for_string' and 'in_string'.
        """
        running_length = 0
        for_string = ""
        in_string = ""
        for keyword in CODE_ORDER:
            running_length += CODE_LENGTH[keyword]
            if len(fips_code) >= running_length:
                # The part of the FIPS code is available!
                part = get_fips_part(fips_code, keyword)
                part_string = f"{keyword.lower()}:{part}"
                if not for_string:
                    for_string = part_string
                else:
                    if not in_string:
                        in_string = for_string
                    else:
                        in_string += f" {for_string}"

                    for_string = part_string
            if keyword == "TRACT":
                # Census does not make data available at more granular level
                LOGGER.debug(
                    f"FIPS Code: {fips_code} provided at more granular level than "
                    "census tract"
                )
                LOGGER.debug(
                    "Census data is only available at the tract level. "
                    "Ignoring additional granularity"
                )
                break
        return for_string, in_string

    def get(
        self, fields: List[str], collection: str, dataset: str, fips_code: str
    ) -> pd.DataFrame:
        """
        Query the US Census API to retrieve fields of interest.

        Parameters
        ----------
        fields : List[str]
            The fields of interest in the table.
        collection : str
            The collection of interest (e.g., "dec", "acs5").
        dataset : str
            The table of interest in the census.
        fips_code : str
            The geography of interest.

        Returns
        -------
        pd.DataFrame
            The values for the fields requested.
        """

        url = f"https://api.census.gov/data/{CENSUS_YEAR}/{collection}/{dataset}"
        for_string, in_string = self._generate_geo_identifiers(fips_code)
        params = {"get": ",".join(fields), "key": self._key, "for": for_string}

        if in_string:
            params["in"] = in_string

        resp = self.session.get(url, params=params)

        if resp.status_code == 200:
            if "<title>Invalid Key</title>" in resp.text:
                raise_exception(resp.text, CensusAPIKeyError)

            try:
                response = resp.json()
            except (
                requests.exceptions.RequestException,
                requests.exceptions.InvalidJSONError,
                TypeError,
            ) as e:
                msg = f"Exception details: {str(e)}"
                msg += f"Response received is: {resp.text}"
                raise_exception(msg, CensusAPIException)

            return pd.DataFrame([response[1]], columns=response[0])

        if resp.status_code == 204:
            LOGGER.warning(
                "No data found from Census API. "
                "Please ensure all fields, FIPS Code, collection, and "
                "dataset info are correct."
            )
            return pd.DataFrame([], columns=fields)

        # All other status codes untreated for now
        LOGGER.debug(f"Untreated status code is: {resp.status_code}")
        LOGGER.debug(f"Untreated response text is: {resp.text}")
        raise_exception("Untreated response code", CensusAPIException)
        return pd.DataFrame([], columns=fields)

    def get_total_population(self, latitude: float, longitude: float) -> float:
        """
        Get total population in a census tract available from the US Census.

        Parameters
        ----------
        latitude : float
            The latitude associated with the geopoint.
        longitude : float
            The longitude associated with the geopoint.

        Returns
        -------
        float
            The total population associated with the census tract for the given lat/long.
        """
        fips_code = get_fips_code(latitude, longitude)
        if not fips_code:
            return 0

        data = self.get(["NAME", "P1_001N"], "dec", "dhc", fips_code)
        return data.iloc[0]["P1_001N"]
