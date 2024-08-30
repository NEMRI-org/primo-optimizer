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
# Standard lib

# Installed libs
import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

# User-defined libs
from primo.utils.raise_exception import raise_exception


def distance_matrix(candidates: pd.DataFrame, weights: dict) -> np.ndarray:
    """
    Generate a distance matrix based on the given features and associated weights for each
    pair of the given well candidates.

    Parameters
    ----------
    candidates : pd.DataFrame
        Input DataFrame with "Latitude," "Longitude," "Age [Years],"
        and "Depth [ft]" columns of the well candidates.

    weights : dict
        Weights assigned to the features---distance, age, and depth when performing the clustering.

    Returns
    -------
    np.ndarray
        Distance matrix to be used for the agglomerative clustering method

    Raises
    ------
    ValueError
        1. If the latitude, longitude, age, or depth is missing in the input DataFrame;
        2. if the sum of feature weights does not equal 1;
        3. if the weight of any feature---distance, age, or depth---is not provided;
        4. if a spurious feature's weight is included apart from the distance, age, and depth.
    """

    weights_sum = sum(weights.values())

    if np.isclose(weights_sum, 1):
        pass
    else:
        raise_exception("Feature weights do not add up to 1", ValueError)

    df_name = list(candidates.columns)
    weights_key = weights.keys()
    if "Latitude" not in df_name or "Longitude" not in df_name:
        raise_exception(
            "The latitude or longitude information of well is not provided", ValueError
        )
    elif "Age [Years]" not in df_name:
        raise_exception(
            "The age information of well is not provided or the corresponding "
            "column name is not Age [Years]",
            ValueError,
        )
    elif "Depth [ft]" not in df_name:
        raise_exception(
            "The depth information of well is not provided or the corresponding "
            "column name is not Depth [ft]",
            ValueError,
        )

    criterion = {"distance", "age", "depth"}

    for key in weights_key:
        if key not in criterion:
            raise_exception(
                f"Currently, the feature {key} is not supported in the clustering step",
                ValueError,
            )

    for key in criterion:
        if key not in weights_key:
            raise_exception(f"The weight for feature {key} is not provided", ValueError)

    candidates["coor"] = list(zip(candidates.Latitude, candidates.Longitude))
    distance_matrix_distance = haversine_vector(
        candidates["coor"].to_list(),
        candidates["coor"].to_list(),
        unit=Unit.MILES,
        comb=True,
    )
    distance_matrix_age = np.abs(
        np.subtract.outer(
            (candidates["Age [Years]"]).to_numpy(),
            (candidates["Age [Years]"]).to_numpy(),
        )
    )
    distance_matrix_depth = np.abs(
        np.subtract.outer(
            (candidates["Depth [ft]"]).to_numpy(), (candidates["Depth [ft]"]).to_numpy()
        )
    )
    return (
        distance_matrix_distance * weights["distance"]
        + weights["age"] * distance_matrix_age
        + weights["depth"] * distance_matrix_depth
    )
