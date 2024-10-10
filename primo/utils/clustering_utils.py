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
from types import SimpleNamespace
from typing import Dict, Optional

# Installed libs
import numpy as np
from haversine import Unit, haversine_vector
from sklearn.cluster import AgglomerativeClustering

# User-defined libs
from primo.data_parser import WellData
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
    cn = wd.column_names
    coordinates = list(zip(wd[cn.latitude], wd[cn.longitude]))
    dist_matrix = wt_dist * (
        haversine_vector(coordinates, coordinates, unit=Unit.MILES, comb=True)
        if wt_dist > 0
        else 0
    )

    # Modifying the object in-place to save memory for large datasets
    dist_matrix += wt_age * (
        np.abs(np.subtract.outer(wd[cn.age].to_numpy(), wd[cn.age].to_numpy()))
        if wt_age > 0
        else 0
    )

    dist_matrix += wt_depth * (
        np.abs(np.subtract.outer(wd[cn.depth].to_numpy(), wd[cn.depth].to_numpy()))
        if wt_depth > 0
        else 0
    )

    return dist_matrix


def _well_clusters(wd: WellData) -> dict:
    """Returns well clusters"""
    col_names = wd.column_names
    set_clusters = set(wd[col_names.cluster])
    return {
        cluster: wd.data[wd[col_names.cluster] == cluster].index.to_list()
        for cluster in set_clusters
    }


def perform_clustering(wd: WellData, distance_threshold: float) -> dict:
    """
    Partitions the data into smaller clusters.

    Parameters
    ----------
    wd: WellData
        Object containing well data

    distance_threshold : float
        Threshold distance for breaking clusters

    Returns
    -------
    well_clusters : dict
        Dictionary of list of wells contained in each cluster
    """
    col_names = wd.column_names
    if hasattr(col_names, "cluster"):
        # Clustering has already been performed, so return.
        LOGGER.warning("Input well data is already clustered.")
        return _well_clusters(wd)

    # Hard-coding the weights data since this should not be a tunable parameter
    # for users. Move to arguments if it is desired to make it tunable.
    weights = {"distance": 1, "age": 0, "depth": 0}

    distance_metric = distance_matrix(wd, weights)
    clustered_data = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=distance_threshold,
    ).fit(distance_metric)

    wd.add_new_column_ordered("cluster", "Clusters", clustered_data.labels_)
    return _well_clusters(wd)


def get_pairwise_metrics(wd: WellData, well_clusters: Dict[int, list]):
    """
    Returns pairwise metric values for all well pairs in each clusters

    Parameters
    ----------
    wd : WellData
        Object containing well data

    well_clusters : dict
        Dictionary containing well clusters
    """
    # DataFrame index -> metric_array index map
    df_to_array = {
        df_index: array_index for array_index, df_index in enumerate(wd.data.index)
    }
    pairwise_metrics = SimpleNamespace()

    for metric in ["distance", "age", "depth"]:
        metric_array = distance_matrix(wd, {metric: 1})
        # {cluster: {(w1, w2): metric, (w1, w3): metric,...}, ...}
        data = {
            cluster: {
                (w1, w2): metric_array[df_to_array[w1], df_to_array[w2]]
                for w1, w2 in combinations(well_list, 2)
            }
            for cluster, well_list in well_clusters.items()
        }
        setattr(pairwise_metrics, metric, data)

    return pairwise_metrics


def get_admissible_well_pairs(
    pairwise_metrics: SimpleNamespace,
    max_distance: float,
    max_age_range: Optional[float] = None,
    max_depth_range: Optional[float] = None,
):
    """
    Returns the set of well pairs that are admissible and not admissible
    for  each cluster.

    Parameters
    ----------
    pairwise_metrics : SimpleNamespace
        Object containing pairwise metric data

    max_distance : float
        Maximum distance [in miles] for filtering out well pairs

    max_age_range : float
        Maximum age range [in years] for filtering out well pairs

    max_depth_range : float
        Maximum depth range [in ft] for filtering out well pairs

    Returns
    -------
    well_pairs_keep : Admissible well pairs for each cluster
    well_pairs_remove : Non admissible well pairs for each cluster
    """
    distance_data = pairwise_metrics.distance
    age_data = pairwise_metrics.age  # age_range_data
    depth_data = pairwise_metrics.depth  # depth_range_data

    if max_age_range is None:
        max_age_range = 1e6  # Set a large number to avoid filtering pairs

    if max_depth_range is None:
        max_depth_range = 1e6  # Set a large number to avoid filtering pairs

    well_pairs = {
        cluster: list(pairs.keys()) for cluster, pairs in distance_data.items()
    }
    well_pairs_keep = {cluster: [] for cluster in well_pairs}
    well_pairs_remove = {cluster: [] for cluster in well_pairs}

    for cluster, pairs in well_pairs.items():
        for w1, w2 in pairs:
            if (
                distance_data[cluster][w1, w2] > max_distance
                or age_data[cluster][w1, w2] > max_age_range
                or depth_data[cluster][w1, w2] > max_depth_range
            ):
                well_pairs_remove[cluster].append((w1, w2))
                # Delete the data from the matrices. Removing these
                # numbers to accuaratelt calculate the scaling factors
                # for efficiency calculations.
                del distance_data[cluster][w1, w2]
                del age_data[cluster][w1, w2]
                del depth_data[cluster][w1, w2]

            else:
                well_pairs_keep[cluster].append((w1, w2))

    return well_pairs_keep, well_pairs_remove


def get_wells_in_dac(wd: WellData, well_clusters: Dict[int, list]):
    """
    Returns the list of well in disadvantaged communities in
    each cluster

    Parameters
    ----------
    wd : WellData
        Object containing well data

    well_clusters : dict
        Dictionary containing well clusters
    """
    wells_in_dac = {cluster: [] for cluster in well_clusters}

    # Update the attribute name when federal DAC data is supported
    if hasattr(wd.column_names, "is_disadvantaged"):
        for cluster, well_list in well_clusters.items():
            for well in well_list:
                if wd.data.loc[well, "is_disadvantaged"]:
                    wells_in_dac[cluster].append(well)

    return wells_in_dac
