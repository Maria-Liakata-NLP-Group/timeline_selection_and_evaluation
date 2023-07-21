"""
This script contains lots of interactive widgets for Jupyter notebooks,
for the project with Andy the REG.
"""
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime
import numpy as np


import sys

sys.path.insert(
    0, "../../timeline_generation"
)  # Adds higher directory to python modules path

from utils.io.my_pickler import my_pickler
from utils.visualize.visualize import get_start_and_end_of_span_of_cp


class InteractiveTimelines:
    def __init__(self):
        self.data_daily_interactions = my_pickler(
            "i", "observed_data_daily_interactions", folder="datasets"
        )
        self.all_cmocs = my_pickler(
            "i", "candidate_moments_of_change", folder="datasets"
        )
        self.user_ids = list(self.data_daily_interactions.keys())
        self.cmoc_methods = list(self.all_cmocs.keys())

        # self.figsize = (16, 5)
        # self.font_size = 20

        # self.features_to_visualize = ["posts"]
        # self.history_marker_style = "."

    def plot_interactive_figure(
        self,
        user_id=950002,
        cmoc_method="bocpd_pg_h1000_a0.01_b10",
        timeline_radius=7,
        # figsize=(16, 5),
        font_size=20,
        features_to_visualize="posts",
        history_marker_style=".",
        xlim=(-2, -1),
        ylim=(-2, -1),
    ):
        self.user_id = user_id
        self.initialize_plot_styles(figsize=(16, 5), font_size=font_size)
        self.plot_history(
            user_id=user_id,
            feature_to_visualize=features_to_visualize,
            marker_style=history_marker_style,
            show=False,
        )
        self.plot_cmocs(cmoc_method=cmoc_method, user_id=user_id)
        self.plot_candidate_timelines(timeline_radius=timeline_radius)
        self.post_process_figure(
            xlim=xlim,
            ylim=ylim,
            save_fig=False,
            show=True,
            feature_to_visualize=features_to_visualize,
            ylabel="Frequency of daily posts",
            xabel="Dates",
            legend_location="upper left",
            xtick_rotation=45,
        )

    def initialize_plot_styles(self, figsize=(16, 5), font_size=20):
        plt.rcParams.update({"font.size": font_size})
        plt.figure(figsize=figsize)

    def plot_history(
        self, user_id, feature_to_visualize="posts", marker_style=".", show=False
    ):
        """
        The input df should be the history of the user. (data_daily_interactions)
        """
        self.user_id = user_id

        plt.plot(
            self.data_daily_interactions[user_id][feature_to_visualize],
            marker=marker_style,
        )

        if show:
            plt.show()

    def plot_cmocs(self, cmoc_method="bocpd_pg_h1000_a0.01_b10", user_id=950002):
        # Plot all CMoCs, for given method and user id
        self.cmocs_to_plot = self.all_cmocs[cmoc_method][user_id]
        init = False  # We will plot multiple CMoCs, so initialize first one
        for c in self.cmocs_to_plot:
            if not init:
                # plt.axvline(
                #     c,
                #     alpha=0.8,
                #     color="red",
                #     linestyle="--",
                #     label="CMoCs from BOCPD\n"
                #     + r"($\alpha_{0}$: $.01$; $\beta_{0}$ : $10$; $h_{0}$: $10^3$)",
                # )

                plt.axvline(
                    c,
                    alpha=0.8,
                    color="red",
                    linestyle="--",
                    label="CMoCs from `{}`".format(cmoc_method),
                )

                init = True
            else:
                plt.axvline(c, alpha=0.8, color="red", linestyle="--")

    def plot_candidate_timelines(self, timeline_radius=7, timeline_colour="red"):
        # Plot canditate timelines
        init = False
        for cp_to_annotate in self.cmocs_to_plot:
            span = get_start_and_end_of_span_of_cp(
                cp_to_annotate, span_radius=timeline_radius
            )
            min_dt_span = span[0]
            max_dt_span = span[-1]

            if not init:
                plt.axvspan(
                    min_dt_span,
                    max_dt_span,
                    alpha=0.05,
                    color=timeline_colour,
                    label="Candidate Timelines",
                )
                init = True
            else:
                plt.axvspan(min_dt_span, max_dt_span, alpha=0.05, color=timeline_colour)

    def post_process_figure(
        self,
        save_fig=False,
        show=True,
        ylabel="Frequency of daily posts",
        feature_to_visualize="posts",
        xabel="Dates",
        legend_location="upper left",
        xtick_rotation=45,
        xlim=(datetime.date(2000, 1, 1), datetime.date(2099, 1, 1)),
        ylim=(0, 100),
        show_legend=False,
    ):
        if show_legend:
            plt.legend(loc=legend_location)
        plt.xlabel(xabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=xtick_rotation)

        # x-limits
        if xlim[0] < self.data_daily_interactions[self.user_id].index[0]:
            xlim_lower = self.data_daily_interactions[self.user_id].index[0]
        else:
            xlim_lower = xlim[0]
        if xlim[-1] > self.data_daily_interactions[self.user_id].index[-1]:
            xlim_upper = self.data_daily_interactions[self.user_id].index[-1]
        else:
            xlim_upper = xlim[-1]

        # y-limits
        if (
            ylim[0]
            < self.data_daily_interactions[self.user_id][feature_to_visualize].min()
        ):
            ylim_lower = self.data_daily_interactions[feature_to_visualize].min()
        else:
            ylim_lower = ylim[0]
        if (
            ylim[-1]
            > self.data_daily_interactions[self.user_id][feature_to_visualize].max()
        ):
            ylim_upper = self.data_daily_interactions[self.user_id][
                feature_to_visualize
            ].max()
        else:
            ylim_upper = ylim[-1]

        plt.xlim(left=xlim_lower, right=xlim_upper)
        plt.ylim(bottom=ylim_lower, top=ylim_upper)

        plt.tight_layout()

        if save_fig:
            plt.savefig("fig1_cmocs_timelines_from_bocpd_alternative.png", dpi=500)

        if show:
            plt.show()
