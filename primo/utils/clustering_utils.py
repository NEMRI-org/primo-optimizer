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

# Installed libs
import numpy as np
from haversine import Unit, haversine_vector
from sklearn.cluster import AgglomerativeClustering

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def distance_matrix(wd: WellData, weights: dict) -> np.ndarray:
    """
    Generate a distance matrix based on the given features and
    associated weights for each pair of the given well candidates.

    Parameters
    ----------
    wd : WellData
        WellData object

    weights : dict
        Weights assigned to the features---distance, age, and
        depth when performing the clustering.

    Returns
    -------
    np.ndarray
        Distance matrix to be used for the agglomerative
        clustering method

    Raises
    ------
    ValueError
        1. if a spurious feature's weight is included apart from
            distance, age, and depth.
        2. if the sum of feature weights does not equal 1.
    """

    # If a feature is not provided, then set its weight to zero
    wt_dist = weights.pop("distance", 0)
    wt_age = weights.pop("age", 0)
    wt_depth = weights.pop("depth", 0)

    if len(weights) > 0:
        msg = (
            f"Received feature(s) {[*weights.keys()]} that are not "
            f"supported in the clustering step."
        )
        raise_exception(msg, ValueError)

    if not np.isclose(wt_dist + wt_depth + wt_age, 1, rtol=0.001):
        raise_exception("Feature weights do not add up to 1.", ValueError)

    # Construct the matrices only if the weights are non-zero
    cn = wd.col_names  # Column names
    coordinates = list(zip(wd[cn.latitude], wd[cn.longitude]))
    dist_matrix = (
        haversine_vector(coordinates, coordinates, unit=Unit.MILES, comb=True)
        if wt_dist > 0
        else 0
    )

    age_range_matrix = (
        np.abs(np.subtract.outer(wd[cn.age].to_numpy(), wd[cn.age].to_numpy()))
        if wt_age > 0
        else 0
    )

    depth_range_matrix = (
        np.abs(np.subtract.outer(wd[cn.depth].to_numpy(), wd[cn.depth].to_numpy()))
        if wt_depth > 0
        else 0
    )

    return (
        wt_dist * dist_matrix
        + wt_age * age_range_matrix
        + wt_depth * depth_range_matrix
    )


def perform_clustering(wd: WellData, distance_threshold: float = 10.0):
    """
    Partitions the data into smaller clusters.

    Parameters
    ----------
    distance_threshold : float, default = 10.0
        Threshold distance for breaking clusters

    Returns
    -------
    n_clusters : int
        Returns number of clusters
    """
    if hasattr(wd.col_names, "cluster"):
        # Clustering has already been performed, so return.
        # Return number of cluster.
        LOGGER.warning(
            "Found cluster attribute in the WellDataColumnNames object."
            "Assuming that the data is already clustered. If the corresponding "
            "column does not correspond to clustering information, please use a "
            "different name for the attribute cluster while instantiating the "
            "WellDataColumnNames object."
        )
        return len(set(wd[wd.col_names.cluster]))

    # Hard-coding the weights data since this should not be a tunable parameter
    # for users. Move to arguments if it is desired to make it tunable.
    # TODO: Need to scale each metric appropriately. Since good scaling
    # factors are not available right now, setting the weights of age and depth
    # as zero.
    weights = {"distance": 1, "age": 0, "depth": 0}

    distance_metric = distance_matrix(wd, weights)
    clustered_data = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=distance_threshold,
    ).fit(distance_metric)

    wd.data["Clusters"] = clustered_data.labels_
    # Uncomment the line below to convert labels to strings. Keeping them as
    # integers for convenience.
    # wd.data["Clusters"] = "Cluster " + wd.data["Clusters"].astype(str)
    wd.col_names.register_new_columns({"cluster": "Clusters"})

    return clustered_data.n_clusters_
