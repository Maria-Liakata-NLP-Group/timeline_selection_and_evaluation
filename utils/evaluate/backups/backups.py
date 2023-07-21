"""
12:58 14/02/22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from utils.io.my_pickler import my_pickler

# from utils.evaluate.precision_recall import precision_medoids, recall_medoids

# Load Data
data_cmocs = my_pickler("i", "candidate_moments_of_change")
searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")
centroids = my_pickler("i", "all_centroids")
data_gtmoc = my_pickler("i", "df_positive_changes")


# Data processing
searched_timeline_spans["start_of_timeline"] = searched_timeline_spans[
    "start_of_timeline"
].apply(lambda x: np.datetime64(x))
searched_timeline_spans["end_of_timeline"] = searched_timeline_spans[
    "end_of_timeline"
].apply(lambda x: np.datetime64(x))
data_gtmoc["date"] = data_gtmoc["date"].apply(lambda x: np.datetime64(x))


def precision_recall_moc(
    centroid_type="dense",
    return_sorted_mean=True,
    verbose=True,
    threshold_distance=5,
    precision_or_recall="precision",
    relative_to="GTMoC",  # Either "GTMoC" or "medoid"
    score_for_missing_target=np.NaN,  # In the case where no GTMoC
):
    methods = list(data_cmocs.keys())

    if centroid_type == "sparse":
        centroid_label = -1
    else:
        centroid_label = +1
    scores_m_u = (
        {}
    )  # Precision scores across all methods, and timelines (denoted by user ids)

    if verbose:
        print(
            "\nReturning {} scores relative to {}.".format(
                precision_or_recall, relative_to
            )
        )
    if relative_to == "medoid":
        user_ids = list(centroids[centroids["binary"] == centroid_label]["user_id"])
    elif relative_to == "GTMoC":
        user_ids = list(data_cmocs[methods[0]].keys())  # Return all 500 user ids

    # Compute score for each method, across all users, across all timelines.
    for m in methods:
        scores_m_u[m] = {}
        for u in user_ids:  # Iterate over each user
            cmocs_for_m_u = data_cmocs[m][u]

            # Specify target CPs (either medoids or raw GTMoCs)
            if relative_to == "medoid":
                target_location = centroids[centroids["user_id"] == u]["medoid"].values[
                    0
                ]
                all_target_locations = target_location
            elif relative_to == "GTMoC":
                all_target_locations = list(
                    data_gtmoc[data_gtmoc["user_id"] == u]["date"].values
                )

            # Loop over each target, to compute number of intersections with predictions
            if len(all_target_locations) > 0:
                hits_within_timeline = (
                    0  # count the numerator (number of intersections)
                )

                # See if a CMoC intersects each GMoC. Reward a maximum hit of 1, per target location.
                for target_location in all_target_locations:
                    hits_for_current_target = 0  # Initialization

                    # See if there are any CMoCs across whole user history.
                    if len(cmocs_for_m_u) > 0:
                        # Pre-processing
                        cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(
                            columns={0: "date"}
                        )
                        cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(
                            lambda x: np.datetime64(x)
                        )

                        # Create list containing dates within span ranges
                        u_span = searched_timeline_spans[
                            searched_timeline_spans["user_id"] == u
                        ]
                        u_start = u_span["start_of_timeline"].values[0]
                        u_end = u_span["end_of_timeline"].values[0]

                        # Filter CMoCs to within searched span
                        dates_with_cmoc_for_user_u_within_span = list(
                            cmocs_for_m_u[
                                (cmocs_for_m_u["date"] >= u_start)
                                & (cmocs_for_m_u["date"] <= u_end)
                            ]["date"]
                        )

                        # Loop over each CMoC within the span
                        if len(dates_with_cmoc_for_user_u_within_span) > 0:
                            for cmoc in dates_with_cmoc_for_user_u_within_span:
                                # See if the CMoC falls within the threshold distance to the centroid
                                if abs(cmoc - target_location) <= np.timedelta64(
                                    threshold_distance, "D"
                                ):
                                    hits_for_current_target += 1

                            # Limit hits to 1 per target - to avoid double-counting.
                            hits_for_current_target = (
                                1
                                if hits_for_current_target > 1
                                else hits_for_current_target
                            )
                            hits_within_timeline += hits_for_current_target  # Total number of intersections for this timeline.

                        # No predictions, despite there being a target. Thus intersection is zero.
                        else:
                            hits_within_timeline = 0

                # Limit hits to the number of CMoC in the timeline - to avoid double-counting.
                hits_within_timeline = (
                    len(dates_with_cmoc_for_user_u_within_span)
                    if hits_within_timeline
                    > len(dates_with_cmoc_for_user_u_within_span)
                    else hits_within_timeline
                )

                # Compute score across whole timeline
                if len(dates_with_cmoc_for_user_u_within_span) > 0:
                    if precision_or_recall == "precision":
                        scores_for_current_timeline = hits_within_timeline / len(
                            dates_with_cmoc_for_user_u_within_span
                        )
                    elif precision_or_recall == "recall":
                        scores_for_current_timeline = hits_within_timeline / len(
                            all_target_locations
                        )
                # No CMoCs within span, thus no intersections. Score is hence 0.
                else:
                    if precision_or_recall == "precision":
                        scores_for_current_timeline = score_for_missing_target
                    # else:
                    #     scores_for_current_timeline = 0

            # No targets for this timeline, set score to missing value.
            else:
                if precision_or_recall == "recall":
                    scores_for_current_timeline = score_for_missing_target
                # else:
                #     scores_for_current_timeline = 0

            # Save score for current timeline and method
            scores_m_u[m][u] = scores_for_current_timeline

    # Save all results as a single unified dataframe, containing scores for all timelines and methods
    df_scores = pd.DataFrame(scores_m_u)

    if return_sorted_mean:
        return df_scores.mean().sort_values(ascending=False)
    else:
        return df_scores


def visualize_varying_thresholds(
    thresholds=range(0, 16),
    alpha=0.5,
    marker="x",
    figsize=(8, 6),
    centroid_type="dense",
    score_for_missing_target=np.NaN,
    relative_to="GTMoC",
    p_or_r="precision",
    save_fig=True,
):

    threshold_scores = {}
    for t in thresholds:

        scores = precision_recall_moc(
            centroid_type=centroid_type,
            return_sorted_mean=True,
            verbose=True,
            threshold_distance=t,
            precision_or_recall=p_or_r,
            relative_to=relative_to,  # Either "GTMoC" or "medoid"
            score_for_missing_target=score_for_missing_target,  # In the case where no GTMoC
        )
        threshold_scores[t] = scores
    df = pd.DataFrame(threshold_scores)

    df.T.plot(marker=marker, figsize=figsize, alpha=alpha)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    title = "{}\nmeasured against GTMoCs, averaged across all timelines".format(
        p_or_r.title()
    )
    plt.title(title)
    plt.xlabel("Days threshold\nto count as intersection")
    plt.ylabel("Mean {}".format(p_or_r))
    if save_fig:
        save_path = "./figures/{}.png".format(title)
        plt.savefig(save_path, dpi=300)
        print("Figure saved at `{}`".format(save_path))
    plt.show()


def experiment_visualize_varying_thresholds(
    thresholds=range(0, 16),
    alpha=0.5,
    marker="x",
    figsize=(8, 6),
    centroid_type="dense",
    score_for_missing_target=np.NaN,
    relative_to="GTMoC",
    save_fig=True,
):
    for p_or_r in ["precision", "recall"]:
        visualize_varying_thresholds(
            thresholds=thresholds,
            alpha=alpha,
            marker=marker,
            figsize=figsize,
            centroid_type=centroid_type,
            score_for_missing_target=score_for_missing_target,
            relative_to=relative_to,
            p_or_r=p_or_r,
            save_fig=True,
        )

    """
23:21. 13/02/22    
    """

    #     def precision_recall_moc(
    #     centroid_type="dense",
    #     return_sorted_mean=True,
    #     verbose=True,
    #     threshold_distance=5,
    #     precision_or_recall="precision",
    #     relative_to="GTMoC",  # Either "GTMoC" or "medoid"
    # ):
    #     methods = list(data_cmocs.keys())

    #     if centroid_type == "sparse":
    #         centroid_label = +1
    #     else:
    #         centroid_label = +1
    #     scores_m_u = (
    #         {}
    #     )  # Precision scores across all methods, and timelines (denoted by user ids)

    #     # Loop over each (dense/ sparse) centroid
    #     if verbose:
    #         print(
    #             "Returning {} scores relative to {}".format(
    #                 precision_or_recall, relative_to
    #             )
    #         )
    #     if relative_to == "medoid":
    #         user_ids = list(centroids[centroids["binary"] == centroid_label]["user_id"])
    #     elif relative_to == "GTMoC":
    #         user_ids = list(data_cmocs[methods[0]].keys())  # Return all 500 user ids
    #     for m in methods:
    #         scores_m_u[m] = {}
    #         for u in user_ids:
    #             cmocs_for_m_u = data_cmocs[m][u]

    #             if relative_to == "medoid":
    #                 target_location = centroids[centroids["user_id"] == u]["medoid"].values[
    #                     0
    #                 ]
    #             elif relative_to == "GTMoC":
    #                 target_location = list(data_gtmoc[data_gtmoc["user_id"] == u]["date"].values)
    #             if len(cmocs_for_m_u) > 0:
    #                 cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(columns={0: "date"})
    #                 cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(
    #                     lambda x: np.datetime64(x)
    #                 )

    #                 # Create list containing dates within span ranges
    #                 u_span = searched_timeline_spans[
    #                     searched_timeline_spans["user_id"] == u
    #                 ]
    #                 u_start = u_span["start_of_timeline"].values[0]
    #                 u_end = u_span["end_of_timeline"].values[0]

    #                 # Filter CMoCs to within searched span
    #                 dates_with_cmoc_for_user_u_within_span = list(
    #                     cmocs_for_m_u[
    #                         (cmocs_for_m_u["date"] >= u_start)
    #                         & (cmocs_for_m_u["date"] <= u_end)
    #                     ]["date"]
    #                 )

    #                 # Loop over each CMoC within the span
    #                 hits_within_timeline = 0  # count the numerator
    #                 if len(dates_with_cmoc_for_user_u_within_span) > 0:
    #                     for cmoc in dates_with_cmoc_for_user_u_within_span:
    #                         # See if the CMoC falls within the threshold distance to the centroid
    #                         if abs(cmoc - target_location) <= np.timedelta64(
    #                             threshold_distance, "D"
    #                         ):
    #                             hits_within_timeline += 1

    #                     # Compute precision
    #                     if precision_or_recall == "precision":
    #                         scores_for_current_timeline = hits_within_timeline / len(
    #                             dates_with_cmoc_for_user_u_within_span
    #                         )
    #                     elif precision_or_recall == "recall":
    #                         scores_for_current_timeline = hits_within_timeline / len(
    #                             target_location
    #                         )
    #                 else:
    #                     precision_for_current_timeline = 0
    #             else:
    #                 scores_for_current_timeline = 0
    #             scores_m_u[m][u] = scores_for_current_timeline
    #     df_scores_scores = pd.DataFrame(scores_m_u)

    #     if return_sorted_mean:
    #         return df_scores.mean().sort_values(ascending=False)
    #     else:
    #         return df_scores

    # ====== Backups =======

    # def precision_centroids(
    #     centroid_type="dense", return_sorted_mean=True, verbose=True, threshold_distance=5
    # ):
    #     methods = list(data_cmocs.keys())
    #     user_ids = list(data_cmocs[methods[0]].keys())

    #     if centroid_type == "sparse":
    #         centroid_label = +1
    #     else:
    #         centroid_label = +1
    #     precision_scores_m_u = (
    #         {}
    #     )  # Precision scores across all methods, and timelines (denoted by user ids)

    #     # Loop over each (dense/ sparse) centroid
    #     centroid_user_ids = list(
    #         centroids[centroids["binary"] == centroid_label]["user_id"]
    #     )
    #     for m in methods:
    #         precision_scores_m_u[m] = {}
    #         for u in centroid_user_ids:
    #             cmocs_for_m_u = data_cmocs[m][u]
    #             centroid_location = centroids[centroids["user_id"] == u]["medoid"].values[0]
    #             if len(cmocs_for_m_u) > 0:
    #                 cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(columns={0: "date"})
    #                 cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(
    #                     lambda x: np.datetime64(x)
    #                 )

    #                 # Create list containing dates within span ranges
    #                 u_span = searched_timeline_spans[
    #                     searched_timeline_spans["user_id"] == u
    #                 ]
    #                 u_start = u_span["start_of_timeline"].values[0]
    #                 u_end = u_span["end_of_timeline"].values[0]

    #                 # Filter CMoCs to within searched span
    #                 dates_with_cmoc_for_user_u_within_span = list(
    #                     cmocs_for_m_u[
    #                         (cmocs_for_m_u["date"] >= u_start)
    #                         & (cmocs_for_m_u["date"] <= u_end)
    #                     ]["date"]
    #                 )

    #                 # Loop over each CMoC within the span
    #                 hits_within_timeline = 0  # count the numerator
    #                 if len(dates_with_cmoc_for_user_u_within_span) > 0:
    #                     for cmoc in dates_with_cmoc_for_user_u_within_span:
    #                         # See if the CMoC falls within the threshold distance to the centroid
    #                         if abs(cmoc - centroid_location) <= np.timedelta64(
    #                             threshold_distance, "D"
    #                         ):
    #                             hits_within_timeline += 1

    #                     # Compute precision
    #                     precision_for_current_timeline = hits_within_timeline / len(
    #                         dates_with_cmoc_for_user_u_within_span
    #                     )
    #                 else:
    #                     precision_for_current_timeline = 0
    #             else:
    #                 precision_for_current_timeline = 0
    #             precision_scores_m_u[m][u] = precision_for_current_timeline
    #     df_precision_scores = pd.DataFrame(precision_scores_m_u)

    #     if return_sorted_mean:
    #         return df_precision_scores.mean().sort_values(ascending=False)
    #     else:
    #         return df_precision_scores

    # def recall_centroids(
    #     centroid_type="dense",
    #     threshold_distance=5,
    #     return_sorted_mean=True,
    #     precision_or_recall="precision",
    # ):

    #     methods = list(data_cmocs.keys())
    #     user_ids = list(data_cmocs[methods[0]].keys())

    #     if centroid_type == "sparse":
    #         centroid_label = -1
    #     else:
    #         centroid_label = +1

    #     recall_scores_m_u = (
    #         {}
    #     )  # Precision scores across all methods, and timelines (denoted by user ids)
    #     # Loop over each (dense/ sparse) centroid
    #     centroid_user_ids = list(
    #         centroids[centroids["binary"] == centroid_label]["user_id"]
    #     )
    #     for m in methods:
    #         recall_scores_m_u[m] = {}
    #         for u in centroid_user_ids:
    #             cmocs_for_m_u = data_cmocs[m][u]
    #             centroid_location = centroids[centroids["user_id"] == u]["medoid"].values[0]
    #             if len(cmocs_for_m_u) > 0:
    #                 cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(columns={0: "date"})
    #                 cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(
    #                     lambda x: np.datetime64(x)
    #                 )

    #                 # Create list containing dates within span ranges
    #                 u_span = searched_timeline_spans[
    #                     searched_timeline_spans["user_id"] == u
    #                 ]
    #                 u_start = u_span["start_of_timeline"].values[0]
    #                 u_end = u_span["end_of_timeline"].values[0]

    #                 # Filter CMoCs to within searched span
    #                 dates_with_cmoc_for_user_u_within_span = list(
    #                     cmocs_for_m_u[
    #                         (cmocs_for_m_u["date"] >= u_start)
    #                         & (cmocs_for_m_u["date"] <= u_end)
    #                     ]["date"]
    #                 )

    #                 # Loop over each CMoC within the span
    #                 hits_within_timeline = 0  # count the numerator
    #                 if len(dates_with_cmoc_for_user_u_within_span) > 0:
    #                     for cmoc in dates_with_cmoc_for_user_u_within_span:
    #                         # See if the CMoC falls within the threshold distance to the centroid
    #                         if abs(cmoc - centroid_location) <= np.timedelta64(
    #                             threshold_distance, "D"
    #                         ):
    #                             hits_within_timeline += 1

    #                     # Compute precision or recall
    #                     recall_for_current_timeline = hits_within_timeline / 1
    #                 else:
    #                     recall_for_current_timeline = 0
    #             else:
    #                 recall_for_current_timeline = 0

    #             recall_scores_m_u[m][u] = recall_for_current_timeline
    #     df_recall_scores = pd.DataFrame(recall_scores_m_u)

    #     if return_sorted_mean:
    #         return df_recall_scores.mean().sort_values(ascending=False)
    #     else:
    #         return df_recall_scores
