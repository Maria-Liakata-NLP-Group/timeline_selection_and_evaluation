import pandas as pd
import numpy as np


def create_minimum_distances_table(
    anchor_points, centroids, centroid_location="medoid"
):
    """
    Minimum distance to centroid location.
    """
    methods = list(anchor_points.keys())
    centroids = centroids.reset_index()
    centroids["centroid_location"] = centroids[centroid_location]

    column_names = ["timeline_id", "centroid_location"]
    column_names.extend(methods)
    min_distances = pd.DataFrame(columns=column_names)

    idx_counter = 0
    dict_min_distances = {}

    # Loop over each centroid
    for timeline_id in centroids["timeline_id"]:
        centroid_location = centroids[centroids["timeline_id"] == timeline_id][
            "centroid_location"
        ].values[0]
        u_id = centroids[centroids["timeline_id"] == timeline_id]["user_id"].values[0]

        dict_min_distances["timeline_id"] = timeline_id
        dict_min_distances["centroid_location"] = centroid_location

        # Minimum distance, for each method
        for method in methods:  # Skip switch/escalation
            if len(anchor_points[method][u_id]) < 1:  # No change-points
                dict_min_distances[method] = np.nan
            else:
                min_dist = abs(
                    (
                        centroid_location
                        - pd.DataFrame({"datetime": anchor_points[method][u_id]})[
                            "datetime"
                        ].apply(lambda x: np.datetime64(x))
                    )
                ).dt.days.min()
                dict_min_distances[method] = min_dist

        min_distances_for_current_centroid = pd.DataFrame(
            dict_min_distances, index=[idx_counter]
        )
        idx_counter += 1
        min_distances = min_distances.append(min_distances_for_current_centroid)

    return min_distances


def sign(x):

    discrete_value = np.sign(x)

    # Enforce sign to be either +1 or -1, can't be 0.
    if discrete_value == 0:
        discrete_value = +1.0

    return discrete_value


def compute_distance_scores(
    anchor_points, centroids, epsilon=0.001, centroid_type="medoid", score_type="binary"
):
    """
    Score type can be either:
    * binary (centroids that have less than median density score are given -1, and +1 otherwise),
    * min_max (centroids are linearly scaled such that minimum density score is -1 and maximum density score is +1)

    For simplicity, I think binary score is more suitable than min_max, as this ensures there's a relatively equally
    distribution of negative and positive centroids.
    """

    min_distances = create_minimum_distances_table(
        anchor_points, centroids, centroid_location="medoid"
    )

    signs = centroids.set_index("timeline_id")
    signs = signs[score_type].apply(lambda x: sign(x))
    signs = pd.DataFrame(signs)
    signs = signs.rename(columns={score_type: "sign"})

    min_distances = min_distances.set_index(["timeline_id"])
    min_distances = min_distances.drop("centroid_location", axis=1)

    # Add the small epsilon term to the distances
    method_scores = min_distances + epsilon

    # Multiply the sign to the epsilon distances
    method_scores = method_scores.mul(signs["sign"], axis=0)

    return method_scores
