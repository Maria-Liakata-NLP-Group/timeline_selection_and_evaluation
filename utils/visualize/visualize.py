import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from matplotlib.pyplot import cm
import math
import numpy as np

from ..io.my_pickler import my_pickler


# Load data
data_daily_interactions = my_pickler(
    "i", "observed_data_daily_interactions", verbose=False, folder="datasets"
)


def get_start_and_end_of_span_of_cp(cp_to_annotate, span_radius=7):
    return [
        cp_to_annotate - timedelta(days=span_radius),
        cp_to_annotate + timedelta(days=span_radius),
    ]


def display_timelines(
    user_id,
    method,
    show_span=True,
    view="2",
    other_feature_to_visualize=None,
    show_switch=True,
    zoom_limit=30,
    metric_vote="Normalized Votes +1 if positive and less than or equal to 10 days, -1 if negative and greater or equal to 10 days, 0 otherwise",
):

    u_id = user_id
    plt.figure(figsize=(15, 5))

    feature = "posts"

    plt.plot(
        all_user_daily_data[u_id][feature],
        marker=".",
        markerfacecolor="grey",
        markersize=8,
        label=feature,
        alpha=0.5,
    )

    if other_feature_to_visualize != None:
        plt.plot(
            all_user_daily_data[u_id][other_feature_to_visualize],
            marker=".",
            markerfacecolor="orange",
            markersize=8,
            color="orange",
            label=other_feature_to_visualize,
        )

    cps = detected_cps[method][u_id]

    # Extract CPs in terms of datetime
    dt_cps = pd.Series(all_user_daily_data[u_id].index)[cps].values

    # Visualize vertical lines for CPs
    ym = (
        all_user_daily_data[u_id][feature].max()
        + all_user_daily_data[u_id][feature].max() / 4
    )

    plt.vlines(
        dt_cps, ymin=0, ymax=ym, color="red", linestyle="--", label=method, alpha=0.6
    )

    if show_switch:
        # Centroid datetimes
        dt_centroids = list(
            df_centroids[df_centroids["user_id"] == u_id]["centroid_location"]
        )

        # Collect marker annotations
        m_cps = list(
            manual_cps_positive_cases[manual_cps_positive_cases["user_id"] == u_id][
                "datetime"
            ]
        )
        x_markers = pd.DataFrame({"datetime": m_cps})
        x_markers["Switch/Escalation"] = 0.0  # Display along x-axis
        x_markers["date"] = x_markers["datetime"].dt.date  # Convert to discrete dates
        x_markers = x_markers.set_index("date")
        x_markers = x_markers.drop(["datetime"], axis=1)

        # Visualize
        plt.plot(
            x_markers["Switch/Escalation"],
            marker="x",
            linestyle="None",
            color="black",
            label="Switch/Escalation",
            alpha=0.5,
        )
        plt.vlines(
            dt_centroids,
            ymin=0,
            ymax=ym,
            color="green",
            linestyle="--",
            alpha=0.8,
            label="Mean centroid",
        )

    if show_span:
        for cp_to_annotate in cps:

            span = get_start_and_end_of_span_of_cp(cp_to_annotate, span_radius=7)
            try:
                min_dt_span = pd.Series(all_user_daily_data[u_id].index)[span[0]]
            except:
                min_dt_span = pd.Series(all_user_daily_data[u_id].index).min()
            try:
                max_dt_span = pd.Series(all_user_daily_data[u_id].index)[span[-1]]
            except:
                max_dt_span = pd.Series(all_user_daily_data[u_id].index).max()

            plt.axvspan(min_dt_span, max_dt_span, alpha=0.05, color="red")

    plt.ylabel("Frequency of daily {}".format(feature))
    plt.title("User: {}\nMethod: {}".format(u_id, method))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    # Returns 2 plots. 2nd plot is of fixed radius.
    if view == "2":
        # Loop over all the timelines, for this user
        for timeline_id in list(df_centroids[df_centroids["user_id"] == user_id].index):

            # Plot again
            plt.figure(figsize=(15, 5))

            plt.plot(
                all_user_daily_data[u_id][feature],
                marker=".",
                markerfacecolor="grey",
                markersize=8,
                label=feature,
            )

            if other_feature_to_visualize != None:
                plt.plot(
                    all_user_daily_data[u_id][other_feature_to_visualize],
                    marker=".",
                    markerfacecolor="orange",
                    markersize=8,
                    color="orange",
                    label=other_feature_to_visualize,
                )

            plt.vlines(
                dt_cps,
                ymin=0,
                ymax=ym,
                color="red",
                linestyle="--",
                label=method,
                alpha=0.6,
            )

            if show_switch:
                # Centroid datetimes
                dt_centroids = list(
                    df_centroids[df_centroids["user_id"] == u_id]["centroid_location"]
                )

                # Collect marker annotations
                m_cps = list(
                    manual_cps_positive_cases[
                        manual_cps_positive_cases["user_id"] == u_id
                    ]["datetime"]
                )
                x_markers = pd.DataFrame({"datetime": m_cps})
                x_markers["Switch/Escalation"] = 0.0  # Display along x-axis
                x_markers["date"] = x_markers[
                    "datetime"
                ].dt.date  # Convert to discrete dates
                x_markers = x_markers.set_index("date")
                x_markers = x_markers.drop(["datetime"], axis=1)

                # Visualize
                plt.plot(
                    x_markers["Switch/Escalation"],
                    marker="x",
                    linestyle="None",
                    color="black",
                    label="Switch/Escalation",
                    alpha=0.5,
                )
                plt.vlines(
                    dt_centroids,
                    ymin=0,
                    ymax=ym,
                    color="green",
                    linestyle="--",
                    alpha=0.8,
                    label="Mean centroid",
                )

            if show_span:
                for cp_to_annotate in cps:

                    span = get_start_and_end_of_span_of_cp(
                        cp_to_annotate, span_radius=7
                    )
                    try:
                        min_dt_span = pd.Series(all_user_daily_data[u_id].index)[
                            span[0]
                        ]
                    except:
                        min_dt_span = pd.Series(all_user_daily_data[u_id].index).min()
                    try:
                        max_dt_span = pd.Series(all_user_daily_data[u_id].index)[
                            span[-1]
                        ]
                    except:
                        max_dt_span = pd.Series(all_user_daily_data[u_id].index).max()

                    plt.axvspan(min_dt_span, max_dt_span, alpha=0.05, color="grey")

            plt.ylabel("Frequency of daily {}".format(feature))
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            #             # Note, this code currently only works for a single CP
            #             fixed_display_radius = 7 * 4 * 2  # Span will take up ~ 1/4 of screen
            #             week_display = fixed_display_radius * 2 / 7
            #             day_conversion = 86400000000000

            anchor = df_centroids.loc[timeline_id]["centroid_location"]

            xmin = anchor - timedelta(days=zoom_limit)
            xmax = anchor + timedelta(days=zoom_limit)
            plt.xlim(xmin, xmax)

            method_score = method_scores.loc[timeline_id][method]

            normalized_vote = all_metric_votes[metric_vote].loc[timeline_id][method]

            summed_vote = (
                all_metric_votes["Sorted {}".format(metric_vote)].loc[method].values[0]
            )

            plt.title(
                "Timeline: {}\nMethod = {} | Sum Vote Score Across All Timelines = {:.2f}\nMethod Timeline Score (i.e. min days) = {:.3f} | Normalized Vote = {:.3f}\nMetric = {}".format(
                    timeline_id,
                    method,
                    summed_vote,
                    method_score,
                    normalized_vote,
                    metric_vote,
                )
            )
            plt.show()


def plot_change_markers(
    user_id=None,
    timestamps=None,
    process_to=None,
    style="Switch/Escalation",
    label="Switch/Escalation",
    color="black",
    alpha=0.5,
    deduplicate_annotations=True,
):
    """
    Plots x markers where positive changes exist.

    Can optionally choose to post-process the changes to be rounded to the nearest 'date'.
    """

    # Load positive changes for this user
    if user_id != None:
        df_positive_changes = my_pickler("i", "df_positive_changes", verbose=False)
        postive_changes_for_user = df_positive_changes[
            df_positive_changes["user_id"] == user_id
        ]

        # Collect marker annotations
        marker_locs = list(postive_changes_for_user["datetime"])

    # Use the passed in timestamps, if desired
    if timestamps != None:
        marker_locs = list(timestamps)

    # Remove duplicate annotations (i.e. if multiple annotator annotate the same timestamp)
    if deduplicate_annotations:
        marker_locs = list(np.unique(marker_locs))

    x_markers = pd.DataFrame({"datetime": marker_locs})
    x_markers["Switch/Escalation"] = 0.0  # Display along x-axis

    if process_to == "date":
        x_markers["date"] = x_markers["datetime"].dt.date  # Convert to discrete dates
        x_markers = x_markers.set_index("date")
        x_markers = x_markers.drop(["datetime"], axis=1)
    elif process_to == None:
        x_markers = x_markers.set_index("datetime")

    # Visualize
    plt.plot(
        x_markers["Switch/Escalation"],
        marker="x",
        linestyle="None",
        color=color,
        label=label,
        alpha=alpha,
    )


def zoom(timeline_id=None, file="df_annotations", padding=7):
    """
    Zoom the visualization to the centroid/ boundaries of the switches/escalations (i.e. observed timeline)

    Inputs:
    =======
    padding = days to pad around the zoom limit
    """

    if timeline_id != None:
        # Load data. Could potentially put this near the top.
        df_changes = my_pickler("i", file, verbose=False)
        changes_for_user = df_changes[df_changes["timeline_id"] == timeline_id]

        xmin = changes_for_user["datetime"].min()
        xmax = changes_for_user["datetime"].max()

        # Apply padding
        xmin -= timedelta(days=padding)
        xmax += timedelta(days=padding)

        plt.xlim(xmin, xmax)


def show_observed_span(user_id):
    # Load data. Could potentially put this near the top.
    df_changes = my_pickler("i", "df_annotations", verbose=False)
    changes_for_user = df_changes[df_changes["user_id"] == user_id]
    timeline_ids = changes_for_user["timeline_id"].unique()

    # Plot each observed timeline span, for the user
    shown_label = False
    for timeline_id in timeline_ids:
        timeline_data = changes_for_user[changes_for_user["timeline_id"] == timeline_id]

        # Identify start and end of spans
        span_xmin = timeline_data["datetime"].min()
        span_xmax = timeline_data["datetime"].max()

        # Visualize
        if shown_label:
            plt.axvspan(span_xmin, span_xmax, alpha=0.05, color="black")
        else:
            plt.axvspan(
                span_xmin,
                span_xmax,
                alpha=0.05,
                color="black",
                label="Observed Timeline",
            )
            shown_label = True


def plot_frequency_of_daily_posts(user_id, alpha=0.8):
    plt.plot(data_daily_interactions[user_id]["posts"], label="posts", alpha=alpha)
    plt.title("User {}".format(user_id))


def plot_lines(
    line_data, color=None, linestyle="--", label=None, alpha=0.85, linewidth=None
):
    label_shown = False
    for line in line_data:
        if label_shown:
            plt.axvline(
                line, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth
            )
        else:
            plt.axvline(
                line,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
                linewidth=linewidth,
            )
            label_shown = True


def plot_cluster_colored_markers(cluster_results):
    """
    Plots the markers from clustering results. Each cluster given a unique colour.
    Noise points (-1) are given colours of black.
    """

    cluster_labels = cluster_results["cluster"].unique()
    n = cluster_labels.shape[0]
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for cluster_label in cluster_labels:
        timestamps = list(
            cluster_results[cluster_results["cluster"] == cluster_label]["X"]
        )
        c = next(color)
        if cluster_label == -1:
            c = "black"
        plot_change_markers(
            timestamps=timestamps, color=c, label="Cluster {}".format(cluster_label)
        )


def full_diagnostic_plot(user_id):
    plt.figure(figsize=(15, 8))
    plot_frequency_of_daily_posts(user_id)
    plot_change_markers(user_id)
    #     plot_cluster_colored_markers()
    show_observed_span(user_id)
    plt.ylabel("Frequency of Daily Posts")
    plt.legend()
    plt.show()


def visualize_history_of_gotja():
    # plt.figure(figsize=(30,15))
    histories["gotja"].plot(figsize=(10, 6))
    plt.axvspan(
        pd.to_datetime("15-05-2017, 01:46:06", format="%d-%m-%Y, %X"),
        pd.to_datetime("10-07-2017, 16:48:21", format="%d-%m-%Y, %X"),
        alpha=0.3,
        color="red",
        label="timeline = `72309e4e06`",
    )

    plt.axvspan(
        timelines["gotja_c15449b99f"]["datetime"].min(),
        timelines["gotja_c15449b99f"]["datetime"].max(),
        alpha=0.3,
        color="magenta",
        label="timeline = `c15449b99f`",
    )

    plt.axvspan(
        timelines["gotja_5c466a8d0b"]["datetime"].min(),
        timelines["gotja_5c466a8d0b"]["datetime"].max(),
        alpha=0.3,
        color="green",
        label="timeline = `5c466a8d0b`",
    )

    plt.axvspan(
        timelines["gotja_076d614ac3"]["datetime"].min(),
        timelines["gotja_076d614ac3"]["datetime"].max(),
        alpha=0.3,
        color="purple",
        label="timeline = `076d614ac3`",
    )

    plt.legend()
    plt.title("History of user = `gotja`")
    plt.tight_layout()
    plt.savefig(
        "../../figures/gotja_history.png", dpi=500, facecolor="white", transparent=False
    )
    plt.show()


class VisualizeFullPipeline:
    """
    This produces the extra visualization requested by the meta-reviewer, which
    should include:
    - [x]  gtmoc
    - [x]  cmoc
    - [x]  medoid
    - [x]  maybe dense/ non-dense
    """

    def __init__(
        self,
        histories,
        apply_noise=False,
        figsize=(9, 6),
        plot_history=True,
        title="",
        show_title=False,
    ):
        """
        Load the data, select the user to visualize. Adds noise to history
        to preserve data privacy, if desired.

        Provide the histories to this class.
        """

        plt.figure(figsize=figsize)

        if apply_noise:
            random_seed = 1
            self.apply_noise = True

        # if plot_history:
        #     self.history(histories)

        self.show_title = show_title
        self.histories = histories
        # self.user_id =   # The user being visualized.

    def plot_history(
        self,
        user_id=1220119,
        feature="posts",
        just_snippet=True,
        alpha=1.0,
        marker="None",
        markerfacecolor="grey",
        markersize=8,
        colour="grey",
        rotatation_of_x_axis=25,
        ylabel="Frequency of Daily Posts",
        xlabel="Time",
    ):
        """
        Visualize the full posting history of the selected user.
        """
        self.user_id = user_id

        plt.plot(
            self.histories[user_id][feature],
            label=feature,
            alpha=alpha,
            marker=marker,
            markerfacecolor=markerfacecolor,
            markersize=markersize,
            color=colour,
        )

        if self.show_title:
            plt.title("User {}".format(user_id))

        plt.xticks(rotation=rotatation_of_x_axis)
        # plt.yticks(rotation=rotatation_of_y_axis)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # # Make the y ticks integers, not floats
        yint = []
        locs, _ = plt.yticks()
        yint = range(math.floor(min(locs)), math.ceil(max(locs)) + 1)
        yint = yint[0::2]  # Take odd elements in list
        plt.yticks(yint)
        plt.ylim(-0.8)

    def snippet_of_history(self, x_lower, x_upper):
        """
        Visualizes just a snippet of the history.
        """

        plt.xlim(x_lower, x_upper)

        return None

    def overlay_annotated_timelines(
        self,
        annotated_spans=[[]],
        colour="black",
        alpha=0.10,
        label="annotated timelines",
    ):
        """
        Overlays a highlighted box over the history, which shows which
        time-periods were annotated.

        Inputs:
        - annotated_spans: is a list of lists. The bottom lists are the
        start and ends of the annotated spans.
        - colour: the colour of the highlighted spans to be visualized
        - alpha: opacity of the highlight


        """
        # Plot each observed timeline span, for the user
        shown_label = False
        for span in annotated_spans:

            # Identify start and end of spans
            span_xmin = min(span)
            span_xmax = max(span)

            # Visualize
            if shown_label:
                plt.axvspan(span_xmin, span_xmax, alpha=alpha, color=colour)
            else:
                plt.axvspan(
                    span_xmin,
                    span_xmax,
                    alpha=alpha,
                    color=colour,
                    label=label,
                )
                shown_label = True

    def overlay_gtmocs(
        self,
        locations=[],
        label="GTMoCs",
        marker_style="o",
        colour="purple",
        size=1,
        alpha=0.85,
        deduplicate_annotations=True,
        process_to=None,
        linestyle="None",
    ):
        """
        Overlays the GTMoCs on the history.

        Inputs:
        - locations: list containing datetimes for where to overlay the
        gtmoc annotations. Can repeat this several times, using different
        labels (e.g. switch, escalation, gtmoc)
        """
        # Remove duplicate annotations (i.e. if multiple annotator annotate the same timestamp)
        if deduplicate_annotations:
            locations = list(np.unique(locations))

        x_markers = pd.DataFrame({"datetime": locations})
        x_markers[label] = 0.0  # Display along x-axis

        if process_to == "date":
            x_markers["date"] = x_markers[
                "datetime"
            ].dt.date  # Convert to discrete dates
            x_markers = x_markers.set_index("date")
            x_markers = x_markers.drop(["datetime"], axis=1)
        elif process_to == None:
            x_markers = x_markers.set_index("datetime")

        # Visualize
        plt.plot(
            x_markers[label],
            marker=marker_style,
            linestyle=linestyle,
            color=colour,
            label=label,
            alpha=alpha,
        )

    def overlay_medoids(
        self,
        locations=[],
        label="medoid",
        linestyle="--",
        colour="green",
        alpha=0.8,
        linewidth=2.0,
    ):
        """
        Overlays the
        """
        label_shown = False
        for line in locations:
            if label_shown:
                plt.axvline(
                    line,
                    color=colour,
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                )
            else:
                plt.axvline(
                    line,
                    label=label,
                    color=colour,
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                )
                label_shown = True

        return None

    def overlay_cmocs(
        self,
        locations,
        label="change-point detection algorithm",
        linestyle="-",
        colour="blue",
        alpha=0.85,
        linewidth=2.0,
    ):
        """
        Overlays CMoCs as lines over the history.

        Inputs:
        - locations: list of datetimes.
        """

        label_shown = False
        for line in locations:
            if label_shown:
                plt.axvline(
                    line,
                    color=colour,
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                )
            else:
                plt.axvline(
                    line,
                    label=label,
                    color=colour,
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=linewidth,
                )
                label_shown = True

    def show_legends(self):
        """
        Displays the legends, and does some processing.
        """

        plt.legend(loc="upper right", bbox_to_anchor=(1, 0.99))

        return None

    def post_processing_of_figure(self, tight_layout=True, background_colour="white"):
        """
        Does some post-processing, such as making it a tight_layout, and
        also ensuring there is a background to the figure.
        """

        return None

    def save_figure(
        self,
        title="",
        dpi=500,
        file_format="pdf",
        path="home/ahills/LongNLP/timeline_generation/figures/full_pipeline/",
        facecolor="white",
        transparent=False,
    ):
        """
        Saves the figure
        """
        path = "../" * 100 + path  # Ensure is root, and then go to path
        save_path = path + title + "." + file_format

        plt.tight_layout()

        plt.savefig(save_path, dpi=dpi, facecolor=facecolor, transparent=transparent)
        return None

    def pad_visualization(self, padding=7):

        xmin = self.history_to_visualize["datetime"].min()
        xmax = self.history_to_visualize["datetime"].max()

        # Apply padding
        xmin -= timedelta(days=padding)
        xmax += timedelta(days=padding)

        plt.xlim(xmin, xmax)

    def full_visualization(self, user_id):

        return None

    def visualize_example_full_pipeline(
        self,
        user_id=1220119,
        annotated_spans=[
            [
                pd.to_datetime("05-05-2020", format="%d-%m-%Y"),
                pd.to_datetime("19-05-2020", format="%d-%m-%Y"),
            ],
            [
                pd.to_datetime("04-06-2020", format="%d-%m-%Y"),
                pd.to_datetime("18-06-2020", format="%d-%m-%Y"),
            ],
        ],
    ):

        self.full_visualization(user_id)
