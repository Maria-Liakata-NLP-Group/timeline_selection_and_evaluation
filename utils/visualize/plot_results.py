import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def visualize_timeline_basis_histograms(
    df,
    metric="precision",
    alpha=0.5,
    figsize=(10, 6),
    save_fig=True,
    dpi=500,
):
    # methods = list(df.columns)
    methods = [
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
    ]
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

    # Plot for each time threshold independently.
    # df = df[]

    # Plot each method
    for m in methods:
        plt.hist(
            df[m],
            kind="hist",
            alpha=alpha,
            marker=markerstyles[m],
            linestyle=linestyles[m],
            label=m,
            color=colours[m],
        )
    # df.T.plot(marker=marker, figsize=figsize, alpha=alpha)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if metric != "Medoid Evaluation":
        title = "{}\nmeasured against GTMoCs, for each timelines".format(metric.title())
    else:
        title = "{}".format(metric.title())
    plt.title(title)
    plt.xlabel("Days threshold\nto count as intersection")
    if metric != "Medoid Evaluation":
        plt.ylabel("Mean {}".format(metric))
    else:
        plt.ylabel("Normalized Summed Votes")
    plt.tight_layout()
    if save_fig:
        save_path = "./figures/{}.png".format(title)
        plt.savefig(save_path, dpi=dpi)
        print("Figure saved at `{}`".format(save_path))
    plt.show()


def method_name_mapper(method_names):

    mapped = []
    mapper = {
        # "bocpd_pg_h1000_a0.01_b10": "BOCPD PG ($\alpha_{0}$: $.01$; $\beta_{0}$ : $10$; $h_{0}$: $10^3$)",
        # "bocpd_pg_h10_a1_b1": "BOCPD PG ($\alpha_{0}$: $1$; $\beta_{0}$: $1$; $h_{0}$: $10$)",
        "bocpd_pg_h1000_a0.01_b10": "BOCPD PG (1)",
        "bocpd_pg_h10_a1_b1": "BOCPD PG (2)",
        # "bocpd_pg_h1000_a0.01_b10": "BOCPD PG ($\alpha_{0}: .01; \beta_{0} : 10; h_{0}: 10^3$)",
        # "bocpd_pg_h10_a1_b1": "BOCPD PG ($\alpha_{0}: 1; \beta_{0}: 1; h_{0}: 10$)",
        "kde_high_activity_comments_received": "AD (high activity: comments)",
        "kde_low_activity_comments_received": "AD (low activity: comments)",
        "kde_high_and_low_activity_comments_received": "AD (high & low activity: comments)",
        "kde_high_activity_posts": "AD (high activity: posts)",
        "kde_low_activity_posts": "AD (low activity: posts)",
        "kde_high_and_low_activity_posts": "AD (high & low activity: posts)",
        "keywords_all": "Keywords",
        "every day": "Every day",
        "random single day (mean across 100 seeds)": "Random single day",
    }

    for m in method_names:
        mapped.append(mapper[m])

    return mapper


def plot_varying_thresholds(
    df,
    metric="precision",
    alpha=0.5,
    figsize=(10, 6),
    save_fig=True,
    dpi=500,
    grids=True,
    ylims=None,
    xlims=None,
):

    # methods = list(df.columns)
    methods = [
        "bocpd_pg_h1000_a0.01_b10",
        "bocpd_pg_h10_a1_b1",
        "kde_high_activity_comments_received",
        "kde_low_activity_comments_received",
        "kde_high_and_low_activity_comments_received",
        "kde_high_activity_posts",
        "kde_low_activity_posts",
        "kde_high_and_low_activity_posts",
        # "keywords_three_categories",
        "keywords_all",
        "every day",
        # "no cps",
        "random single day (mean across 100 seeds)",
    ]

    method_mapper = method_name_mapper(methods)

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
            label=method_mapper[m],
            color=colours[m],
        )
    # df.T.plot(marker=marker, figsize=figsize, alpha=alpha)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if metric != "Medoid Evaluation":
        title = "{}\nmeasured against GTMoCs, for each timelines".format(metric.title())
    else:
        title = "{}".format(metric.title())
    plt.title(title)
    plt.xlabel("Days threshold\nto count as intersection")
    if metric != "Medoid Evaluation":
        plt.ylabel("Mean {}".format(metric))
    else:
        plt.ylabel("Normalized Summed Votes")

    if grids:
        plt.rc("grid", linestyle=":", color="black", alpha=0.1)
        plt.grid(True, which="both")

    if ylims != None:
        ymin = min(ylims)
        ymax = max(ylims)
        plt.ylim(ymin, ymax)
    if xlims != None:
        xmin = min(xlims)
        xmax = max(xlims)
        plt.xlim(xmin, xmax)

    plt.tight_layout()
    if save_fig:
        root_path = "../" * 100
        save_path = root_path
        save_path += "home/ahills/LongNLP/timeline_generation/"
        save_path += "figures/{}_measured_against_gtmocs.png".format(metric)
        plt.savefig(save_path, dpi=dpi)
        print("Figure saved at `{}`".format(save_path[len(root_path) - 1 :]))
    plt.show()


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


def plot_single(results, alpha=0.5, grids=True):
    methods = [
        "bocpd_pg_h1000_a0.01_b10",
        "bocpd_pg_h10_a1_b1",
        "kde_high_activity_comments_received",
        "kde_low_activity_comments_received",
        "kde_high_and_low_activity_comments_received",
        "kde_high_activity_posts",
        "kde_low_activity_posts",
        "kde_high_and_low_activity_posts",
        # "keywords_three_categories",
        "keywords_all",
        "every day",
        # "no cps",
        "random single day (mean across 100 seeds)",
    ]

    method_mapper = method_name_mapper(methods)

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

    # Plot each method
    for m in methods:
        plt.plot(
            results[m],
            alpha=alpha,
            marker=markerstyles[m],
            linestyle=linestyles[m],
            label=method_mapper[m],
            color=colours[m],
        )

    if grids:
        plt.rc("grid", linestyle=":", color="black", alpha=0.1)
        plt.grid(True, which="both")


def plot_4_metrics_subplots(
    p_r_f1, scaled_medoid_results, savefig=True, dpi=500, fontsize=15, xlim_upper=6, dataset_name=''
):
    xlim_upper = xlim_upper + 0.2
    plt.rcParams.update({"font.size": fontsize})

    fig, axs = plt.subplots(1, 4, sharex=False, sharey=True, figsize=(20, 5), dpi=dpi)

    plt.subplot(1, 4, 1)
    plot_single(p_r_f1["precision"])
    plt.xlim([-0.2, xlim_upper])
    plt.title("Precision")

    plt.subplot(1, 4, 2)
    plot_single(p_r_f1["recall"])
    plt.xlim([-0.2, xlim_upper])
    plt.title("Recall")

    plt.subplot(1, 4, 3)
    plot_single(p_r_f1["F1"])
    plt.xlim([-0.2, xlim_upper])
    plt.title("F1")

    plt.subplot(1, 4, 4)
    plot_single(scaled_medoid_results)
    plt.title("Medoid Votes")
    plt.xlim([-0.2, xlim_upper])

    fig.supxlabel(r"Margin of error, $\tau$ (days)")
    # fig.supylabel("Score")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(wspace=0.025, hspace=0)

    # axplts.set_xlabel("Margin of error, $\tau$ (days)")
    # axs.set_ylabel("Score")

    plt.tight_layout()

    if savefig:
        # root_path = "../" * 100
        # save_path = root_path
        # save_path += "home/ahills/LongNLP/timeline_generation/"
        # save_path += "figures/4_metrics_subplots"
        save_path='{}_4_metrics_subplots.png'.format(dataset_name)
        plt.savefig(save_path, dpi=dpi)
        # print("Figure saved at `{}`".format(save_path[len(root_path) - 1 :]))
        print("Figure saved at `{}`".format(save_path))
    plt.show()
