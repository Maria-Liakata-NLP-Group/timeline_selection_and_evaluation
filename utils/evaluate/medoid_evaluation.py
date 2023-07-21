import datetime
import pickle
from turtle import distance

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.evaluate.create_centroids import output_all_centroid_styles
from utils.io.my_pickler import my_pickler
from utils.visualize.plot_results import plot_varying_thresholds


def create_minimum_distances_table(
    anchor_points,
    centroids,
    centroid_location="medoid",
    users_can_have_multiple_timelines=True,
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

            if users_can_have_multiple_timelines:
                u_id = timeline_id  # anchor points were processed to have timeline ids, rather than user ids
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
    anchor_points,
    centroids,
    epsilon=0.001,
    centroid_type="medoid",
    score_type="binary",
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

    signs = centroids.reset_index().set_index("timeline_id")
    # signs = centroids.set_index("timeline_id")
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


def assign_unnormalized_votes(
    distance_scores,
    reward_style="only_rewards",
    distance_threshold=10,
    return_sum=False,
):

    if reward_style == "penalizes_and_rewards":
        # Penalizes and rewards
        v = distance_scores.copy()
        v[(v >= 0) & (v <= distance_threshold)] = 1
        v[(v < 0) & (v >= -distance_threshold)] = -1
        v[(v != 1) & (v != -1)] = 0
        unnormalized_votes = v

        if return_sum:
            unnormalized_votes = pd.DataFrame(
                v.sum(axis=0).sort_values(ascending=False), columns=["Summed Votes"]
            )

    elif reward_style == "only_rewards":
        # Only rewards
        v = distance_scores.copy()
        v[(v >= 0) & (v <= distance_threshold)] = 1
        v[(v != 1)] = 0
        unnormalized_votes = v

        if return_sum:
            unnormalized_votes = pd.DataFrame(
                v.sum(axis=0).sort_values(ascending=False), columns=["Summed Votes"]
            )

    return unnormalized_votes


def normalize_votes(v, anchor_points, reddit_timelines=True):
    """
    Normalize the vote scores, by dividing the votes per timeline, 
    by the number of CPs declared by that method for that user.
    """

    v = v.reset_index()
    if reddit_timelines:
        v = v.set_index("timeline_id")
    else:
        v["user_id"] = v["timeline_id"].apply(lambda x: int(x.split("_")[0]))
        v = v.set_index("user_id")

    # Number of CPs declared by each method, per user
    n_cps_per_user = pd.DataFrame(anchor_points).applymap(lambda x: len(x))
    n_cps_per_user.index.rename("user_id", inplace=True)
    n_cps_per_user

    # Avoid division by zero
    n_cps_per_user = n_cps_per_user.replace(0, np.NaN)

    if not reddit_timelines:
        normalized_v = v.drop(["timeline_id"], axis=1).div(n_cps_per_user, axis=1)
    else:
        normalized_v = v.div(n_cps_per_user, axis=1)

    normalized_v = pd.DataFrame(
        normalized_v.sum(axis=0).sort_values(ascending=False),
        columns=["Normalized Votes Score"],
    )

    # my_pickler("o", "normalized_votes_reddit", normalized_v)

    return normalized_v


def full_experiment(
    cmocs=None,
    load_centroids=False,
    load_positive_annotations=True,
    annotated_timelines=None,
    only_positive_changes=None,
    save_data=False,
    reward_style="only_rewards",
    distance_thresholds=range(0, 21),
    limit_within_annotated_spans=True,
    visualize=True,
    users_can_have_multiple_timelines=False,
    user_timeline_sub_dict=None,
    reddit_timelines=False,
):

    # Load Data
    if cmocs == None:
        cmocs = my_pickler(
            "i", "cmocs_with_100_random"
        )  # alternatively: 'candidate_moments_of_change'
    if load_centroids:
        print("Loading centroids...")
        centroids = my_pickler("i", "all_centroids")
    else:
        if load_positive_annotations:
            # Load from file
            only_positive_changes = my_pickler("i", "df_positive_changes")

        # Create centroids using annotations
        centroids = output_all_centroid_styles(
            only_positive_changes,
            annotated_timelines=annotated_timelines,
            remove_duplicate_annotations=False,
            apply_clustering=None,
            reddit_timelines=reddit_timelines,
        )

    # Limit the cmocs to be within the annotated spans, if desired.
    if limit_within_annotated_spans:
        if users_can_have_multiple_timelines:
            cmocs = limit_data_to_annotated_spans(
                cmocs,
                methods=None,
                user_ids=None,
                searched_timeline_spans=None,
                annotated_timeline=None,
                users_can_have_multiple_timelines=True,
                user_timeline_sub_dict=user_timeline_sub_dict,
            )

        else:
            cmocs = limit_data_to_annotated_spans(cmocs)

    # Compute distance scores, based on distance to centroids
    distance_scores = compute_distance_scores(
        cmocs, centroids, epsilon=0.001, centroid_type="medoid", score_type="binary",
    )

    my_pickler("o", "cmocs_within_annotated_spans_reddit", cmocs)
    my_pickler("o", "centroids_reddit", centroids)

    # Return unnormalized votes
    initialized = False
    for t in distance_thresholds:
        v = assign_unnormalized_votes(
            distance_scores, reward_style=reward_style, distance_threshold=t
        )

        sorted_column_of_results = normalize_votes(v, cmocs)

        # Set column to the size of the distance threshold
        sorted_column_of_results = sorted_column_of_results.rename(
            columns={"Normalized Votes Score": t}
        )

        # Append all results to a large dataframe, where each column corresponds to the distance threshold
        if initialized:
            all_results = pd.concat([all_results, sorted_column_of_results], axis=1)
        else:
            all_results = sorted_column_of_results
        initialized = True

    # Handle random cmocs
    all_results = handle_random_cmocs(all_results)

    # Export data, if desired
    if save_data:
        # current_date = str(datetime.datetime.now(tz=None).date())
        # file_name = "{}_medoid_results_reward_style={}_thresh=range({},{})".format(
        #     current_date,
        #     reward_style,
        #     min(distance_thresholds),
        #     max(distance_thresholds) + 1,
        # )

        # if limit_within_annotated_spans:
        #     file_name += "limit_within_annotated_spans"

        file_name = "medoid_results_within_annotated_spans"

        my_pickler(
            "o", file_name, all_results,
        )

    # return all_results

    if visualize:
        plot_varying_thresholds(
            all_results,
            metric="Medoid Evaluation",
            alpha=0.5,
            figsize=(10, 6),
            save_fig=True,
            dpi=500,
            grids=True,
            ylims=None,
            xlims=None,
        )

    return all_results


def handle_random_cmocs(df_scores):
    """
    Returns a single score for many random CMoC methods, by averaging them. 
    Averages across methods for each timeline individually.

    Args:
        df_scores ([type]): [description]
        average_across_methods_first (bool, optional): Whether to . Defaults to True.

    Returns:
        [type]: [description]
    """

    df_scores = df_scores.T

    n_seeds = len(df_scores.filter(like="random single day").columns)

    # Return the mean scores for each timeline, across all (100) random CMoCs methods
    mean_random = df_scores[df_scores.filter(like="random single day").columns].mean(
        axis=1
    )

    random_names = list(df_scores.filter(like="random single day").columns)

    # Save the aggregated random data
    df_scores["random single day (mean across {} seeds)".format(n_seeds)] = mean_random

    # Remove previous random single days
    df_scores = df_scores[df_scores.columns.drop(random_names)]

    return df_scores


def return_cmocs_within_span(
    cmocs_for_m_u, u, searched_timeline_spans, annotated_timeline=None
):
    # If no CMoCs for this user, will return an empty list
    if len(cmocs_for_m_u) > 0:
        # Pre-processing
        cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(columns={0: "date"})
        cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(lambda x: np.datetime64(x))

        # Create list containing dates within span ranges
        if type(searched_timeline_spans) == type(None):
        # if searched_timeline_spans == None:
            # Estimate span from annotations
            u_start = pd.to_datetime(annotated_timeline["datetime"].dt.date).min()
            u_end = pd.to_datetime(annotated_timeline["datetime"].dt.date).max()
        else:
            u_span = searched_timeline_spans[searched_timeline_spans["user_id"] == u]
            u_start = u_span["start_of_timeline"].values[0]
            u_end = u_span["end_of_timeline"].values[0]

        # Filter CMoCs to within searched span
        dates_with_cmoc_for_user_u_within_span = list(
            cmocs_for_m_u[
                (cmocs_for_m_u["date"] >= u_start) & (cmocs_for_m_u["date"] <= u_end)
            ]["date"]
        )
    else:
        dates_with_cmoc_for_user_u_within_span = []

    return dates_with_cmoc_for_user_u_within_span


def limit_data_to_annotated_spans(
    data_cmocs,
    methods=None,
    user_ids=None,
    load_searched_spans=True,
    searched_timeline_spans=None,
    annotated_timeline=None,
    users_can_have_multiple_timelines=False,
    user_timeline_sub_dict=None,
):
    """
    Used to restrict the evaluation to be performed on input data that is only
    within the annotated (observed) timeline spans.
    
    The CMoCs are only retained within spans that were observed by annotators.
    """

    # Load the annotated spans regions
    if load_searched_spans:
        searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")

        # Pre-process data
        searched_timeline_spans["start_of_timeline"] = searched_timeline_spans[
            "start_of_timeline"
        ].apply(lambda x: np.datetime64(x))
        searched_timeline_spans["end_of_timeline"] = searched_timeline_spans[
            "end_of_timeline"
        ].apply(lambda x: np.datetime64(x))
        # data_gtmoc["date"] = data_gtmoc["date"].apply(lambda x: np.datetime64(x))

    # Return CMoCs within annototated spans
    if methods == None:
        methods = list(data_cmocs.keys())
    if user_ids == None:
        user_ids = list(data_cmocs["every day"].keys())  # All annotated users

    cmocs_within_span = {}
    for m in methods:
        cmocs_within_span[m] = {}
        for u in user_ids:  # Iterate over each user
            cmocs_for_m_u = data_cmocs[m][u]

            if users_can_have_multiple_timelines:
                timeline_ids_of_user = list(user_timeline_sub_dict[u].keys())

                for tl_id in timeline_ids_of_user:
                    # The current annotated timeline to limit within
                    annotated_timeline = user_timeline_sub_dict[u][tl_id]

                    # Limit CMoCs within the span of this annotated timeline
                    cmocs_within_span[m][tl_id] = return_cmocs_within_span(
                        cmocs_for_m_u,
                        u,
                        searched_timeline_spans=None,
                        annotated_timeline=annotated_timeline,
                    )

            else:

                # Filter to CMoCs within the timeline span
                cmocs_within_span[m][u] = return_cmocs_within_span(
                    cmocs_for_m_u,
                    u,
                    searched_timeline_spans=searched_timeline_spans,
                    annotated_timeline=None,
                )

    return cmocs_within_span

