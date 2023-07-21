import re
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.evaluate.medoid_evaluation import (limit_data_to_annotated_spans,
                                              return_cmocs_within_span)
from utils.evaluate.tcpdbench_metrics import f_measure
from utils.io.my_pickler import my_pickler


def precision_recall_moc(
    cmocs=None,  # These should be within the annotated spans
    gtmocs=None,
    user_timeline_sub_dict=None,
    users_can_have_multiple_timelines=False,
    centroids=None,
    methods=["all"],
    centroid_type="dense",
    return_sorted_mean=True,
    verbose=True,
    threshold_distance=5,
    precision_or_recall="precision",
    relative_to="GTMoC",  # Either "GTMoC" or "medoid"
    score_for_missing_target=np.NaN,  # In the case where no GTMoC
    tcpdbench=False,
):
    label = "label_2"
    # Retain only positive annotations for GTMoCs, for evaluation
    gtmocs = gtmocs[gtmocs[label] == 1]

    if methods == ["all"]:
        methods = list(cmocs.keys())

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
        user_ids = list(
            cmocs["bocpd_pg_h1000_a0.01_b10"].keys()
        )  # Return all 500 user ids
        # user_ids = list(data_gtmoc["user_id"].unique())\

        if verbose:
            print("len(user_ids) = {}".format(len(user_ids)))

    # Compute score for each method, across all users, across all timelines.
    for m in methods:
        scores_m_u[m] = {}
        for u in user_ids:  # Iterate over each user
            # Filter to CMoCs within the timeline span
            user_cmocs = cmocs[m][u]

            # if (m == "kde_high_activity_comments_received") & (u == "72309e4e06"):
            #     print()
            #     print("threshold_distance = {}".format(threshold_distance))
            #     print("len(user_cmocs) = {}".format(len(user_cmocs)))
            #     print("len(gtmocs) = {}".format(len(gtmocs)))
            #     print(f"min(user_cmocs) = {min(user_cmocs)}")
            #     print(f"max(user_cmocs) = {max(user_cmocs)}")

            # Specify target CPs (either medoids or raw GTMoCs)
            if relative_to == "medoid":
                target_location = centroids[centroids["user_id"] == u]["medoid"].values[
                    0
                ]
                all_target_locations = target_location
            elif relative_to == "GTMoC":
                # Create empty lists in cases where there are no values (GTMoCs) for users
                # which have no annotations
                if users_can_have_multiple_timelines:
                    all_target_locations = list(
                        pd.to_datetime(
                            gtmocs[gtmocs["timeline_id"] == u]["date"].values
                        )
                    )
                    # if (m == "kde_high_and_low_activity_posts") & (u == "72309e4e06"):
                    #     print(
                    #         f"min(all_target_locations) = {min(all_target_locations)}"
                    #     )
                    #     print(
                    #         f"max(all_target_locations) = {max(all_target_locations)}"
                    #     )
                else:

                    all_target_locations = list(
                        gtmocs[gtmocs["user_id"] == u]["date"].values
                    )

            # See if a CMoC intersects each GMoC. Reward a maximum hit of 1, per target location.
            used = []
            # Create a sorted list where each target location is unique (set).
            if verbose:
                print("len(all_target_locations) = {}".format(all_target_locations))
            all_target_locations = list(sorted(list(set(all_target_locations))))
            if verbose:
                print(
                    "Sorted set... len(all_target_locations) = {}".format(
                        len(all_target_locations)
                    )
                )
                print("all_target_locations = {}".format(all_target_locations))
            for (
                t
            ) in (
                all_target_locations
            ):  # Loop over each target (each day can only happen once, due to preprocessing)
                for (
                    c
                ) in (
                    user_cmocs
                ):  # Loop over each CMoC, making sure it intersects only once as a maximum.
                    # if (m == "kde_high_activity_comments_received") & (
                    #     u == "72309e4e06"
                    # ):
                    #     print(f"abs(c-t) = {abs(c - t)}")
                    #     print(f"c = {c}")
                    #     print(f"t = {t}")
                    #     print(f"len(used) = {len(used)}")
                    if (abs(c - t) <= np.timedelta64(threshold_distance, "D")) & (
                        c not in used
                    ):
                        # if (m == "kde_high_and_low_activity_posts") & (
                        #     u == "72309e4e06"
                        # ):
                        # print("[Interesection]")
                        # print(f"abs(c-t) = {abs(c - t)}")
                        # print(
                        #     f"np.timedelta64(threshold_distance, 'D') = {np.timedelta64(threshold_distance, 'D')}"
                        # )
                        used.append(c)
                        break  # As soon as this CMoC has been found to intersect a GTMoC, stop searching other CMoCs.
            intersections = len(set(used))
            # if (m == "kde_high_and_low_activity_posts") & (u == "72309e4e06"):
            #     print(f"intersectons = {intersections}")
            #     print(f"len(used) = {len(used)}")
            #     print(
            #         f"len(set(all_target_locations)) = {len(set(all_target_locations))}"
            #     )

            # Compute score across whole timeline
            if precision_or_recall == "precision":
                if len(user_cmocs) > 0:
                    scores_for_current_timeline = intersections / len(
                        list(set(user_cmocs))
                    )
                else:
                    scores_for_current_timeline = score_for_missing_target
            elif precision_or_recall == "recall":
                if len(all_target_locations) > 0:
                    scores_for_current_timeline = intersections / len(
                        all_target_locations
                    )
                else:
                    scores_for_current_timeline = score_for_missing_target

            scores_m_u[m][u] = scores_for_current_timeline

    # Save all results as a single unified dataframe, containing scores for all timelines and methods
    df_scores = pd.DataFrame(scores_m_u)

    # Aggregate scores for random methods
    df_scores = handle_multiple_random_cmocs(df_scores)

    if return_sorted_mean:
        return df_scores.mean().sort_values(ascending=False)
    else:
        return df_scores


def f1_moc(
    cmocs=None,
    gtmocs=None,
    user_timeline_sub_dict=None,
    users_can_have_multiple_timelines=False,
    centroid_type="dense",
    return_sorted_mean=True,
    verbose=False,
    threshold_distance=5,
    relative_to="GTMoC",  # Either "GTMoC" or "medoid"
    score_for_missing_target=np.NaN,  # In the case where no GTMoC
    # micro_or_macro="macro",  # Macro or micro f1 score.
):

    df_precision_scores = precision_recall_moc(
        cmocs=cmocs,  # These should be within the annotated spans
        gtmocs=gtmocs,
        user_timeline_sub_dict=user_timeline_sub_dict,
        users_can_have_multiple_timelines=users_can_have_multiple_timelines,
        centroid_type=centroid_type,
        return_sorted_mean=True,
        verbose=verbose,
        threshold_distance=threshold_distance,
        precision_or_recall="precision",
        relative_to=relative_to,  # Either "GTMoC" or "medoid"
        score_for_missing_target=score_for_missing_target,  # In the case where no GTMoC
    )

    df_recall_scores = precision_recall_moc(
        cmocs=cmocs,  # These should be within the annotated spans
        gtmocs=gtmocs,
        user_timeline_sub_dict=user_timeline_sub_dict,
        users_can_have_multiple_timelines=users_can_have_multiple_timelines,
        centroid_type=centroid_type,
        return_sorted_mean=True,
        verbose=verbose,
        threshold_distance=threshold_distance,
        precision_or_recall="recall",
        relative_to=relative_to,  # Either "GTMoC" or "medoid"
        score_for_missing_target=score_for_missing_target,  # In the case where no GTMoC
    )

    df_f1 = (2 * df_precision_scores * df_recall_scores) / (
        df_precision_scores + df_recall_scores
    )

    return df_f1.sort_values(ascending=False)


def evaluate_tcpdbench(
    return_sorted_mean=True, verbose=True, threshold_days=5, date_or_datetime="date"
):
    """[summary]

    Args:
        return_sorted_mean (bool, optional): [description]. Defaults to True.
        verbose (bool, optional): [description]. Defaults to True.
        threshold_days (int, optional): [description]. Defaults to 5.
        date_or_datetime (str, optional): Whether to evaluate CPs against
        GTMoCs which are either aggregated on the date level
        (date_or_datetime='date') or on the datetime level. Defaults to "date".

    Returns:
        [type]: [description]
    """
    methods = list(data_cmocs.keys())
    user_ids = list(data_cmocs[methods[0]].keys())  # Return all 500 user ids

    scores_m_u = {}
    scores_m_u["F1"] = {}
    scores_m_u["precision"] = {}
    scores_m_u["recall"] = {}

    # Compute score for each method, across all users, across all timelines.
    for m in methods:
        scores_m_u["F1"][m] = {}
        scores_m_u["precision"][m] = {}
        scores_m_u["recall"][m] = {}
        for u in user_ids:  # Iterate over each user
            # Filter to CMoCs within the timeline span
            cmocs = cmocs_within_span[m][u]

            all_target_locations = list(
                data_gtmoc[data_gtmoc["user_id"] == u][date_or_datetime].values
            )

            predictions = cmocs
            annotations = {
                0: all_target_locations
            }  # Treat as just a single annotator, id 0
            scores_for_current_timeline = f_measure(
                annotations,
                predictions,
                margin=threshold_days,
                alpha=0.5,
                return_PR=True,
            )

            f, p, r = scores_for_current_timeline

            scores_m_u["F1"][m][u] = f
            scores_m_u["precision"][m][u] = p
            scores_m_u["recall"][m][u] = r

    # Save all results as a single unified dataframe, containing scores for all timelines and methods
    df_f1 = pd.DataFrame(scores_m_u["F1"])
    df_precision = pd.DataFrame(scores_m_u["precision"])
    df_recall = pd.DataFrame(scores_m_u["recall"])

    if return_sorted_mean:
        df_f1 = df_f1.mean().sort_values(ascending=False)
        df_precision = df_precision.mean().sort_values(ascending=False)
        df_recall = df_recall.mean().sort_values(ascending=False)

    return {"F1": df_f1, "precision": df_precision, "recall": df_recall}


def experiment_visualize_varying_thresholds(
    cmocs=None,
    gtmocs=None,
    user_timeline_sub_dict=None,
    limit_within_annotated_spans=True,
    users_can_have_multiple_timelines=False,
    methods=[
        "bocpd_pg_h1000_a0.01_b10",
        "bocpd_pg_h10_a1_b1",
        "kde_high_activity_comments_received",
        "kde_low_activity_comments_received",
        "kde_high_and_low_activity_comments_received",
        "kde_high_activity_posts",
        "kde_low_activity_posts",
        "kde_high_and_low_activity_posts",
        "keywords_three_categories",
        "keywords_all",
        "every day",
        "no cps",
        "random single day (mean across 100 seeds)",
    ],
    visualize=True,
    thresholds=range(0, 16),
    alpha=0.5,
    figsize=(10, 6),
    centroid_type="dense",
    score_for_missing_target=np.NaN,
    relative_to="GTMoC",
    metrics=["F1", "precision", "recall"],
    save_fig=True,
    tcpd=False,  # Whether to use the TCPD benchmark adapted code
    verbose=False,
    return_sorted_mean=True,
    save_results=False,
):
    """
    Main function in this script. Run this and it will give you the plots for
    precision, recall, F1 as 3 subplots in a single figure. It will show the
    scores for varying window thresholds along the x-axis.
    """

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
                load_searched_spans=False,
            )

        else:
            cmocs = limit_data_to_annotated_spans(
                cmocs, users_can_have_multiple_timelines=False
            )

    threshold_scores = {}
    for metric in metrics:
        threshold_scores[metric] = {}

    for t in thresholds:
        print("threshold={}/{}".format(t, len(thresholds)))
        if tcpd:  # Use the TCPD benchmark adapted code
            results = evaluate_tcpdbench(
                return_sorted_mean=True, verbose=verbose, threshold_days=t,
            )
            for metric in metrics:
                threshold_scores[metric][t] = results[metric]

        else:
            for metric in metrics:
                if metric == "F1":
                    scores = f1_moc(
                        cmocs=cmocs,
                        gtmocs=gtmocs,
                        user_timeline_sub_dict=user_timeline_sub_dict,
                        users_can_have_multiple_timelines=users_can_have_multiple_timelines,
                        centroid_type=centroid_type,
                        return_sorted_mean=return_sorted_mean,
                        verbose=verbose,
                        threshold_distance=t,
                        relative_to=relative_to,  # Either "GTMoC" or "medoid"
                        score_for_missing_target=score_for_missing_target,  # In the case where no GTMoC
                    )
                else:
                    scores = precision_recall_moc(
                        cmocs=cmocs,
                        gtmocs=gtmocs,
                        user_timeline_sub_dict=user_timeline_sub_dict,
                        users_can_have_multiple_timelines=users_can_have_multiple_timelines,
                        centroid_type=centroid_type,
                        return_sorted_mean=return_sorted_mean,
                        verbose=verbose,
                        threshold_distance=t,
                        precision_or_recall=metric,
                        relative_to=relative_to,  # Either "GTMoC" or "medoid"
                        score_for_missing_target=score_for_missing_target,  # In the case where no GTMoC
                    )
                threshold_scores[metric][t] = scores

    # Dictionary containing dataframes. Keys are the metric type, values are the dataframes
    metrics_dict_df = {}
    for metric in metrics:
        metrics_dict_df[metric] = pd.DataFrame(threshold_scores[metric])

        # Plot the scores for varying thresholds, for all specified metrics
        df = metrics_dict_df[metric].T
        if tcpd:
            print("\nResults for TCPD benchmark adapted code:")

        if return_sorted_mean:
            if visualize:
                plot_varying_thresholds(
                    df,
                    methods=methods,
                    metric=metric,
                    alpha=alpha,
                    figsize=figsize,
                    save_fig=save_fig,
                )

        if save_results:
            my_pickler("o", "precision_recall_f1_gtmocs", metrics_dict_df)

    return metrics_dict_df  # Return the scores for varying thresholds.


def plot_varying_thresholds(
    df,
    methods=[
        "bocpd_pg_h1000_a0.01_b10",
        "bocpd_pg_h10_a1_b1",
        "kde_high_activity_comments_received",
        "kde_low_activity_comments_received",
        "kde_high_and_low_activity_comments_received",
        "kde_high_activity_posts",
        "kde_low_activity_posts",
        "kde_high_and_low_activity_posts",
        "keywords_three_categories",
        "keywords_all",
        "every day",
        "no cps",
        "random single day (mean across 100 seeds)",
    ],
    metric="precision",
    alpha=0.5,
    figsize=(10, 6),
    save_fig=True,
    dpi=500,
    grids=True,
):

    # methods = list(df.columns)

    # Define styles
    linestyles, markerstyles, colours = visualization_styles(
        bocpd_linestyles="solid",
        kde_linestyles="dashed",
        keywords_linestyles="dashdot",
        trivial_linestyles="dotted",
        bocpd_markerstyles="x",
        kde_markerstyles="o",
        keywords_markerstyles="s",
        trivial_markerstyles="^",
    )

    plt.figure(figsize=figsize, dpi=dpi)

    # Plot each method
    for m in methods:
        plt.plot(
            df[m],
            alpha=alpha,
            marker=markerstyles[m],
            linestyle=linestyles[m],
            label=m,
            color=colours[m],
        )
    # df.T.plot(marker=marker, figsize=figsize, alpha=alpha)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    title = "{}\nmeasured against GTMoCs, averaged across all timelines".format(
        metric.title()
    )
    plt.title(title)
    plt.xlabel("Days threshold\nto count as intersection")
    plt.ylabel("Mean {}".format(metric))

    if grids:
        plt.rc("grid", linestyle=":", color="black", alpha=0.1)
        plt.grid(True, which="both")

    plt.tight_layout()
    if save_fig:
        root_path = "../" * 100
        save_path = root_path
        save_path += "home/ahills/LongNLP/timeline_generation/"
        save_path += "figures/{}_measured_against_gtmocs.png".format(metric)
        plt.savefig(save_path, dpi=dpi)
        print("Figure saved at `{}`".format(save_path[len(root_path) - 1 :]))
    plt.show()


def load_data(include_100_random_cmocs=True):
    if include_100_random_cmocs:
        data_cmocs = my_pickler(
            "i", "cmocs_with_100_random"
        )  # 100 random cmocs included
    else:
        data_cmocs = my_pickler("i", "candidate_moments_of_change")
    searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")
    centroids = my_pickler("i", "all_centroids")
    data_gtmoc = my_pickler(
        "i", "df_positive_changes"
    )  # Only load where there are positive changes
    # data_gtmoc = my_pickler("i", "df_annotations")  # Here we load also where there

    return data_cmocs, searched_timeline_spans, centroids, data_gtmoc


def preprocess_data(searched_timeline_spans, data_gtmoc, data_cmocs, verbose=True):
    if verbose:
        print("Pre-processing data...")
    searched_timeline_spans["start_of_timeline"] = searched_timeline_spans[
        "start_of_timeline"
    ].apply(lambda x: np.datetime64(x))
    searched_timeline_spans["end_of_timeline"] = searched_timeline_spans[
        "end_of_timeline"
    ].apply(lambda x: np.datetime64(x))
    data_gtmoc["date"] = data_gtmoc["date"].apply(lambda x: np.datetime64(x))
    # data_gtmoc["date"] = data_gtmoc["datetime"].apply(lambda x: np.datetime64(x))

    # Return CMoCs within annototated spans
    methods = list(data_cmocs.keys())
    # user_ids = list(data_gtmoc["user_id"].unique())
    user_ids = list(
        data_cmocs["bocpd_pg_h1000_a0.01_b10"].keys()
    )  # All 500 annotated users
    # print("2. Length user ids {}".format(len(user_ids)))

    cmocs_within_span = {}
    for m in methods:
        cmocs_within_span[m] = {}
        for u in user_ids:  # Iterate over each user
            cmocs_for_m_u = data_cmocs[m][u]

            # Filter to CMoCs within the timeline span
            cmocs_within_span[m][u] = return_cmocs_within_span(cmocs_for_m_u, u)

    if verbose:
        print("Done pre-processing data")

    return searched_timeline_spans, data_gtmoc, cmocs_within_span


def visualization_styles(
    bocpd_linestyles="solid",
    kde_linestyles="dashed",
    keywords_linestyles="dashdot",
    trivial_linestyles="dotted",
    bocpd_markerstyles="x",
    kde_markerstyles="o",
    keywords_markerstyles="s",
    trivial_markerstyles="^",
):
    """
    BOCPD should be 'solid' lines, 'x'.
    kde should be 'dashed' with 'o'.
    keywords should be 'dashdot' with 's'.
    no cps, every day, random single day should be 'dotted' with '^'.
    """

    linestyles = {}
    linestyles["bocpd_pg_h1000_a0.01_b10"] = bocpd_linestyles
    linestyles["bocpd_pg_h10_a1_b1"] = bocpd_linestyles

    linestyles["kde_high_activity_comments_received"] = kde_linestyles
    linestyles["kde_low_activity_comments_received"] = kde_linestyles
    linestyles["kde_high_and_low_activity_comments_received"] = kde_linestyles
    linestyles["kde_high_activity_posts"] = kde_linestyles
    linestyles["kde_low_activity_posts"] = kde_linestyles
    linestyles["kde_high_and_low_activity_posts"] = kde_linestyles

    linestyles["keywords_three_categories"] = keywords_linestyles
    linestyles["keywords_all"] = keywords_linestyles

    linestyles["every day"] = trivial_linestyles
    linestyles["no cps"] = trivial_linestyles
    linestyles["random single day (mean across 100 seeds)"] = trivial_linestyles

    # MARKERS
    markerstyles = {}
    markerstyles["bocpd_pg_h1000_a0.01_b10"] = bocpd_markerstyles
    markerstyles["bocpd_pg_h10_a1_b1"] = bocpd_markerstyles

    markerstyles["kde_high_activity_comments_received"] = kde_markerstyles
    markerstyles["kde_low_activity_comments_received"] = kde_markerstyles
    markerstyles["kde_high_and_low_activity_comments_received"] = kde_markerstyles
    markerstyles["kde_high_activity_posts"] = kde_markerstyles
    markerstyles["kde_low_activity_posts"] = kde_markerstyles
    markerstyles["kde_high_and_low_activity_posts"] = kde_markerstyles

    markerstyles["keywords_three_categories"] = keywords_markerstyles
    markerstyles["keywords_all"] = keywords_markerstyles

    markerstyles["every day"] = trivial_markerstyles
    markerstyles["no cps"] = trivial_markerstyles
    markerstyles["random single day (mean across 100 seeds)"] = trivial_markerstyles

    # COLOURS
    colours = {}
    colours["bocpd_pg_h1000_a0.01_b10"] = "b"
    colours["bocpd_pg_h10_a1_b1"] = "red"

    colours["kde_high_activity_comments_received"] = "blue"
    colours["kde_low_activity_comments_received"] = "red"
    colours["kde_high_and_low_activity_comments_received"] = "green"
    colours["kde_high_activity_posts"] = "olive"
    colours["kde_low_activity_posts"] = "orange"
    colours["kde_high_and_low_activity_posts"] = "purple"

    colours["keywords_three_categories"] = "royalblue"
    colours["keywords_all"] = "tomato"

    colours["every day"] = "darkmagenta"
    colours["no cps"] = "orangered"
    colours["random single day (mean across 100 seeds)"] = "firebrick"

    return linestyles, markerstyles, colours


def handle_multiple_random_cmocs(df_scores):
    #  , average_across_methods_first=True
    #  ):
    """
    Returns a single score for many random CMoC methods, by averaging them.
    Averages across methods for each timeline individually.

    Args:
        df_scores ([type]): [description]
        average_across_methods_first (bool, optional): Whether to . Defaults to True.

    Returns:
        [type]: [description]
    """

    # if (
    #     average_across_methods_first
    # ):  # First return the mean across all methods, for each timeline
    #     mean_axis = 0
    # else:
    #     mean_axis = 1  # First return the mean across all timelines, for each method.

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


which_dataset = "Reddit_CLPsych2022"

if which_dataset == "talklife":
    # Load data to be used throughout this script
    data_cmocs, searched_timeline_spans, centroids, data_gtmoc = load_data()
    searched_timeline_spans, data_gtmoc, cmocs_within_span = preprocess_data(
        searched_timeline_spans, data_gtmoc, data_cmocs
    )
elif which_dataset == "Reddit_CLPsych2022":
    pass
    # data_cmocs =

    # Load data to be used throughout this script
    # data_cmocs, searched_timeline_spans, centroids, data_gtmoc = load_data()
    # searched_timeline_spans, data_gtmoc, cmocs_within_span = preprocess_data(
    #     searched_timeline_spans, data_gtmoc, data_cmocs
    # )

# class PrecisionRecallF1:
#     """
#     I will aim to make this all as a class, with shared variables via __init__.
#     """
#     def __init__(self, cmocs=None, annotated_timelines=None, user_timeline_sub_dict=None, dataset_name='Reddit'):
#         self.cmocs = None

#     def
