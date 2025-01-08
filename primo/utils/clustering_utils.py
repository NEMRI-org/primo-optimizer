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
import networkx as nx
import numpy as np
from haversine import Unit, haversine_vector
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import BallTree

# User-defined libs
from primo.data_parser.well_data import WellData
from primo.utils import EARTH_RADIUS
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


def _well_clusters(wd: WellData) -> dict:
    """
    Returns well clusters

    Parameters
    ----------
    wd : WellData
        The WellData object to return well clusters.

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """
    col_names = wd.col_names
    return (
        wd.data.groupby(wd[col_names.cluster])
        .apply(lambda group: group.index.tolist())
        .to_dict()
    )


def _check_existing_cluster(wd: WellData):
    """
    Checks if clustering has already been performed on the WellData object.

    Parameters
    ----------
    wd : WellData
        The WellData object to check for existing clusters.

    Returns
    -------
    Bool
        True if clustering exists, otherwise False.
    """
    if hasattr(wd.col_names, "cluster"):
        # Clustering has already been performed, so return the number of clusters.
        LOGGER.warning(
            "Found cluster attribute in the WellDataColumnNames object. "
            "Assuming that the data is already clustered. If the corresponding "
            "column does not correspond to clustering information, please use a "
            "different name for the attribute cluster while instantiating the "
            "WellDataColumnNames object."
        )
        return True
    return False


def perform_agglomerative_clustering(wd: WellData, threshold_distance: float = 10.0):
    """
    Partitions the data into smaller clusters using agglomerative clustering.

    Parameters
    ----------
    threshold_distance : float, default = 10.0
        Threshold distance for breaking clusters

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """

    if _check_existing_cluster(wd):
        return _well_clusters(wd)

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
        distance_threshold=threshold_distance,
    ).fit(distance_metric)

    wd.data["Clusters"] = clustered_data.labels_
    # Uncomment the line below to convert labels to strings. Keeping them as
    # integers for convenience.
    # wd.data["Clusters"] = "Cluster " + wd.data["Clusters"].astype(str)
    wd.col_names.register_new_columns({"cluster": "Clusters"})

    return _well_clusters(wd)


def perform_louvain_clustering(
    wd: WellData,
    threshold_distance: float,
    threshold_cluster_size: int,
    nearest_neighbors: int,
    seed: int = 4242,
    resolution: float = None,
    max_resolution: float = 10.0,
) -> dict:
    """
    Partitions the data into smaller clusters using the Louvain community detection method,
    limiting each well to connect only with its nearest_neighbors within a threshold distance.
    Dynamically adjusts the resolution parameter from 1 to max_resolution with 0.5 increments,
    to ensure no cluster exceeds the size threshold if resolution parameter is not provided.

    Parameters
    ----------
    wd : WellData
        Object containing well data

    threshold_distance : float
        Threshold distance (in miles) for breaking clusters

    threshold_cluster_size : int
        cluster size of the largest cluster

    nearest_neighbors : int
        nearest neighbors to consider when constructing graph for Louvain clustering

    seed : int
        random seed for Louvain communities function

    resolution : float
        resolution for Louvain communities function

    max_resolution : float
        maximum resolution while dynamically setting the resolution for Louvain communities function

    Returns
    -------
    dict
        Dictionary of lists of wells contained in each cluster
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    if _check_existing_cluster(wd):
        return _well_clusters(wd)

    coordinates = list(
        zip(wd.data[wd.col_names.latitude], wd.data[wd.col_names.longitude])
    )
    ball_tree = BallTree(np.radians(coordinates), metric="haversine")

    # k=nearest neighbors + 1 to include the well itself
    distances, indices = ball_tree.query(
        np.radians(coordinates), k=nearest_neighbors + 1
    )

    well_graph = nx.Graph()
    well_ids = wd.data.index.tolist()
    well_graph.add_nodes_from(well_ids)
    # Add edges within distance threshold
    for i, neighbors in enumerate(indices):
        well_id_1 = well_ids[i]
        for j in range(1, len(neighbors)):  # Skip the point itself (index 0)
            well_id_2 = well_ids[neighbors[j]]
            if not well_graph.has_edge(well_id_1, well_id_2):
                distance = distances[i][j] * EARTH_RADIUS  # Convert to distance
                if distance <= threshold_distance:
                    well_graph.add_edge(
                        well_id_1, well_id_2, weight=threshold_distance - distance
                    )

    # Louvain clustering
    def _get_clusters(seed: int, resolution: float):
        communities = nx.community.louvain_communities(
            well_graph, seed=seed, resolution=resolution
        )
        well_cluster_map = {
            well: cluster_id
            for cluster_id, community in enumerate(communities)
            for well in community
        }
        _cluster_list = [well_cluster_map[well] for well in well_ids]
        _max_cluster_size = max(len(community) for community in communities)
        LOGGER.debug(
            f"For resolution={resolution}, the maximum cluster size = {max_cluster_size}"
        )

        return _cluster_list, _max_cluster_size

    max_cluster_size = len(well_ids)  # Set a large number
    cluster_list = None

    # If length of data is less than cluster_threshold, assign all wells to one cluster
    if max_cluster_size <= threshold_cluster_size:
        cluster_list = np.ones(len(wd.data))
        LOGGER.info(
            "The size of the data is less than the cluster threshold. "
            "Assigning all wells to the same cluster."
        )

    elif resolution is not None:
        # Using user-specified resolution for clustering
        cluster_list, max_cluster_size = _get_clusters(seed=seed, resolution=resolution)

    else:
        # Adaptively adjust resolution to control the cluster size
        resolution = 0.5  # Initial resolution
        while max_cluster_size > threshold_cluster_size:
            resolution += 0.5  # increase resolution--> starts from 1; increases in every iteration
            cluster_list, max_cluster_size = _get_clusters(
                seed=seed, resolution=resolution
            )
            if resolution == max_resolution:
                raise_exception("Could not reach desired cluster sizes", RuntimeError)

    wd.col_names.register_new_columns({"cluster": "Clusters"})
    wd.data["Clusters"] = cluster_list

    LOGGER.info(f"The resolution parameter is set to {resolution}.")
    LOGGER.info(
        f"There are {len(wd.data["Clusters"].unique())} clusters with "
        f"the largest cluster containing {max_cluster_size} wells."
    )

    return _well_clusters(wd)
