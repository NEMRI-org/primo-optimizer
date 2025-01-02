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
from itertools import combinations
from typing import Optional

# Installed libs
import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector
from sklearn.cluster import AgglomerativeClustering

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils.raise_exception import raise_exception

LOGGER = logging.getLogger(__name__)


def distance_matrix(
    wd: WellData, weights: dict, list_wells: Optional[list] = None
) -> pd.DataFrame:
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

    list_wells : list, default = None
        If specified, returns the distance matrix only for the
        specified subset of wells

    Returns
    -------
    pd.DataFrame
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
    # Converting list_wells to a list to handle non-list instances,
    # such as Pyomo Set, tuple, etc.
    data = wd.data if list_wells is None else wd.data.loc[list(list_wells)]
    cn = wd.column_names  # Column names

    coordinates = list(zip(data[cn.latitude], data[cn.longitude]))
    dist_matrix = wt_dist * (
        haversine_vector(coordinates, coordinates, unit=Unit.MILES, comb=True)
        if wt_dist > 0
        else 0
    )

    # Modifying the object in-place to save memory for large datasets
    dist_matrix += wt_age * (
        np.abs(np.subtract.outer(data[cn.age].to_numpy(), data[cn.age].to_numpy()))
        if wt_age > 0
        else 0
    )

    dist_matrix += wt_depth * (
        np.abs(np.subtract.outer(data[cn.depth].to_numpy(), data[cn.depth].to_numpy()))
        if wt_depth > 0
        else 0
    )

    return pd.DataFrame(dist_matrix, columns=data.index, index=data.index)


def perform_clustering(wd: WellData, distance_threshold: float = 10.0):
    """
    Partitions the data into smaller clusters.

    Parameters
    ----------
    wd : WellData
        Object containing the information on all wells

    distance_threshold : float, default = 10.0
        Threshold distance for breaking clusters

    Returns
    -------
    n_clusters : int
        Returns number of clusters
    """
    if hasattr(wd.column_names, "cluster"):
        # Clustering has already been performed, so return.
        # Return number of cluster.
        LOGGER.warning(
            "Found cluster attribute in the WellDataColumnNames object."
            "Assuming that the data is already clustered. If the corresponding "
            "column does not correspond to clustering information, please use a "
            "different name for the attribute cluster while instantiating the "
            "WellDataColumnNames object."
        )
        return len(set(wd[wd.column_names.cluster]))

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

    wd.add_new_column_ordered("cluster", "Clusters", clustered_data.labels_)

    return clustered_data.n_clusters_


def get_pairwise_metrics(wd: WellData, list_wells: list) -> pd.DataFrame:
    """
    Returns pairwise metric values for all possible pairs of wells in
    `list_wells`.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    list_wells : list
        List of wells for which pairwise metrics are needed to
        be calculated

    Returns
    -------
    pairwise_metrics : DataFrame
        DataFrame containing the pairwise metric values
    """
    well_pairs = list(combinations(list_wells, 2))
    pairwise_metrics = pd.DataFrame()

    # Compute pairwise distances
    pairwise_metrics["distance"] = distance_matrix(
        wd, {"distance": 1}, list_wells
    ).stack()[well_pairs]

    # Compute pairwise age range
    if wd.column_names.age is not None:
        pairwise_metrics["age_range"] = distance_matrix(
            wd, {"age": 1}, list_wells
        ).stack()[well_pairs]

    # Compute pairwise depth range
    if wd.column_names.depth is not None:
        pairwise_metrics["depth_range"] = distance_matrix(
            wd, {"depth": 1}, list_wells
        ).stack()[well_pairs]

    return pairwise_metrics
