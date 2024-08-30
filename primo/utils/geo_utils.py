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
from math import radians
from typing import Any, List, Set, Tuple, Union

# Installed libs
import folium
from haversine import Unit, haversine
from sklearn.neighbors import BallTree

# User-defined libs
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def is_in_bounds(
    arg: Union[float, int],
    arg_name: str,
    lower: Union[float, int, None] = None,
    upper: Union[float, int, None] = None,
    raise_except: bool = False,
) -> bool:
    """
    Checks if an argument is within lower and upper bounds. if specified.

    Parameters
    ----------
    arg : Union[float, int]
        The argument to be validated; should be a float or an int.
    arg_name : str
        The name of the argument in the function whose argument is being validated
    lower : Union[float, int, None], optional
        The lower bound on the arg; by default None
    upper : Union[float, int, None], optional
        The upper bound on the arg; by default None
    raise_except : bool, optional
        If True, raises a ValueError if the argument is not within specified
        lower and upper bounds; by default False

    Returns
    -------
    bool
        True if the argument is considered valid; False otherwise

    Raises
    ------
    ValueError
        If the argument is not within specified lower and upper bounds and if
        `raise_except` is True.
    """
    is_valid = True

    if lower is not None and arg < lower:
        is_valid = False

    if upper is not None and arg > upper:
        is_valid = False

    if raise_except and not is_valid:
        msg = (
            f"Value {arg} is not within the specified "
            "lower bound: {lower} and upper bound: {upper}"
        )
        msg += f"\n. Please provide a valid value for {arg_name}"
        raise_exception(msg, ValueError)

    return is_valid


def is_acceptable(
    arg: Any, valid_args: Set[Any], arg_name: str, raise_except: bool = False
) -> bool:
    """
    Check if an argument is among a set that is considered valid.

    Parameters
    ----------
    arg : Any
        The argument to be validated.; could be any type
    valid_args : Set[Any]
        The set of values considered valid
    arg_name : str
        The name of the argument in the function whose argument is being validated
    raise_except : bool, optional
        If True, in the case of an invalid argument, a ValueError is raised;
        by default False

    Returns
    -------
    bool
        True if the argument is considered valid; False otherwise

    Raises
    ------
    ValueError
        If the argument is not in the set of `valid_args` and if
        `raise_except` is True.
    """
    if arg not in valid_args:
        if raise_except:
            msg = f"Value {arg} is not a valid argument for {arg_name}\n"
            msg += f"Allowable values are: {','.join(valid_args)}"
            raise_exception(msg, ValueError)
        return False
    return True


def is_valid_lat(lat: float):
    """
    Check if a latitude value specified is valid (between -90 and 90).

    Parameters
    ----------
    lat : float
        The latitude value in degrees

    Raises
    ------
    ValueError
        If the argument is not a valid value for latitude
    """
    if not is_in_bounds(lat, "Latitude", -90, 90, False):
        msg = f"Valid values for latitude is -90<=lat<=90. Provided {lat}"
        raise_exception(msg, ValueError)


def is_valid_long(long: float):
    """
    Check if a longitude value specified is valid (between -180 and 180).

    Parameters
    ----------
    long : float
        The longitude value in degrees.

    Raises
    ------
    ValueError
        If the argument is not a valid value for longitude.
    """
    if not is_in_bounds(long, "Longitude", -180, 180, False):
        msg = f"Valid values for latitude is -180<=long<=180. Provided {long}"
        raise_exception(msg, ValueError)


def is_valid_geopoint(point: Tuple[float, float]):
    """
    Check if a tuple of floats indicating latitude and longitude values, respectively,
    is valid.

    Parameters
    ----------
    point : Tuple[float, float]
        The latitude/longitude pair specifying the location of a point

    Raises
    ------
    ValueError
        If the argument is not a valid geopoint
    """
    lat, long = point
    is_valid_lat(lat)
    is_valid_long(long)


def get_distance(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    distance_type: str = "haversine",
    distance_unit: str = "MILES",
) -> float:
    """
    Compute the distance between two tuples specifying the latitude and longitude,
    respectively, for two points; currently only supports computation of the haversine
    distances but can be extended to other distance measures.

    Parameters
    ----------
    origin : Tuple[float, float]
        Tuple specifying the latitude and longitude of the origin
    dest : Tuple[float, float]
        Tuple specifying the latitude and longitude of the destination
    distance_type : str, optional
        Formula to use for distance calculation; currently,
        only supports `haversine` formula, by default, "haversine"
    distance_unit : str, optional
        The unit for distance calculation; one of `MILES` or `KMS`---by default, "MILES"

    Returns
    -------
    float
        Distance between origin and destination

    Raises
    ------
    ValueError
        In case of invalid input arguments
    """

    is_acceptable(
        arg=distance_type,
        valid_args={"haversine"},
        arg_name="distance_type",
        raise_except=True,
    )

    is_acceptable(
        arg=distance_unit,
        valid_args={"MILES", "KMS"},
        arg_name="distance_unit",
        raise_except=True,
    )

    is_valid_geopoint(origin)
    is_valid_geopoint(dest)

    unit = Unit.MILES if distance_unit == "MILES" else Unit.KILOMETERS
    return haversine(origin, dest, unit)


def get_nearest_neighbors(
    points: List[Tuple[float, float]], cutoff: float, distance_unit: str = "MILES"
) -> List[int]:
    """
    Given a list of points denoted by their latitudes and longitudes,
    this function returns the number of points that are within a distance specified by
    `cutoff` with the appropriate `distance_unit`.

    Parameters
    ----------
    points : List[Tuple[float, float]]
        List of points specified with their latitudes and longitudes
    cutoff : float
        The distance to be used as a cutoff
    distance_unit : str, optional
        The unit for distance; one of `MILES` or `KMS`---by default "MILES"

    Returns
    -------
    List[int]
        List specifying the number of points available within the threshold
        specified by `cutoff` and `distance_unit`.

    Raises
    ------
    ValueError
        In the case of invalid input arguments.
    """

    is_acceptable(
        arg=distance_unit,
        valid_args={"MILES", "KMS"},
        arg_name="distance_unit",
        raise_except=True,
    )

    is_in_bounds(cutoff, "cutoff", 0, None, True)

    for point in points:
        is_valid_geopoint(point)

    # Convert latitudes and longitudes to radians
    points_radians = [(radians(lat), radians(lon)) for lat, lon in points]

    # Build a BallTree for efficient nearest neighbor search
    ball_tree = BallTree(points_radians, metric="haversine")

    # Query the BallTree to find neighbors within the specified cutoff
    neighbors = [
        len(ball_tree.query_radius([point], r=cutoff, count_only=True)[0]) - 1
        for point in points_radians
    ]

    return neighbors


def visualize_bing_maps_route(route, waypoints):
    """
    Given a list of coordinates, this function returns the route between these points
    using the Bing Maps API.

    Parameters
    ----------
    route : object
        The route between these points using the Bing Maps API obtained by the
        get_route function
    waypoints : List[Tuple[float, float]]
        A list of all coordinates to generate a route in that order

    Returns
    -------
    Tuple
        Map showing the route between the waypoints, total distance between all points;
        total duration of travel in seconds

    """

    map_ = folium.Map(location=waypoints[0], zoom_start=10)

    for i, waypoint in enumerate(waypoints):
        folium.Marker(
            location=waypoint, popup=f"Waypoint {i + 1}", icon=folium.Icon(color="blue")
        ).add_to(map_)

    total_distance = 0
    total_duration = 0

    for leg in route["routeLegs"]:
        leg_distance = leg["travelDistance"]
        leg_duration = leg["travelDuration"]

        total_distance += leg_distance
        total_duration += leg_duration

        if "line" in leg:
            route_path = leg["line"]["coordinates"]
            folium.PolyLine(
                locations=route_path, color="red", weight=2.5, opacity=1
            ).add_to(map_)

    total_distance = round(total_distance, 3)

    return map_, total_distance, total_duration
