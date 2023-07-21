import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../../")  # Adds higher directory to python modules path
from utils.evaluate.all_4_subplots_experiment import \
    full_experiment_all_4_subplots
# Medoid evaluation
from utils.evaluate.create_centroids import output_all_centroid_styles
from utils.evaluate.medoid_evaluation import \
    full_experiment as medoid_full_experiment
from utils.evaluate.precision_recall import \
    experiment_visualize_varying_thresholds as p_r_f1_full_experiment
from utils.io import data_handler
from utils.io.my_pickler import my_pickler
from utils.timeline_generation.anchor_points.keywords.identify_cmocs_from_reddit_timelines import \
    identify_cmocs_for_reddit_timelines_using_keywords
# 100 random timelines
from utils.timeline_generation.anchor_points.random_methods import \
    full_experiment_random_methods
# Identify CMoCs
from utils.timeline_generation.generate_anchor_points import \
    create_anchor_points


class RedditTimelinesExperiments:
    """ """

    def __init__(self):
        # Load datasets
        self.RedditDataset = data_handler.RedditDataset()

        # Annotated user timelines
        self.timelines, _ = self.RedditDataset.deanonymize_annotated_timelines()

        # Full user histories
        self.histories = self.RedditDataset.extract_all_annotated_users_histories()
        self.usernames = list(self.histories.keys())

        # Select which methods to use to create anchor points
        self.METHODS = [
            "bocpd_pg_h1000_a0.01_b10",
            "bocpd_pg_h10_a1_b1",
            "kde_high_activity_comments_received",
            "kde_low_activity_comments_received",
            "kde_high_and_low_activity_comments_received",
            "kde_high_activity_posts",
            "kde_low_activity_posts",
            "kde_high_and_low_activity_posts",
            # 'keywords_three_categories',
            # 'keywords_all',
            # "keywords_reddit",
            "every day",
            "no cps",
            # 'random single day'
        ]

        # # Identify CMoCs for all methods, for all users in the CLPsych dataset
        # self.cmocs = self.identify_cmocs_for_all_histories_in_clpsych_dataset(self)

    # def identify_cmocs_for_user_in_clpsych_dataset(self, user_history):
    #     """
    #     Returns CMoCs for a user history in the Reddit CLPsych dataset.
    #     """

    def identify_cmocs_for_all_histories_in_clpsych_dataset(self):

        # Identify CMoCs for all user histories
        candidate_moments_of_change = create_anchor_points(
            user_ids=self.usernames,
            methods=self.METHODS,
            data_daily_interactions=self.histories,
            verbose=False,
        )

        # Add the CMoCs from the keywords methods
        candidate_moments_of_change[
            "keywords_all"
        ] = identify_cmocs_for_reddit_timelines_using_keywords()

        # Add Additional Random Timelines, 100
        N_RANDOM_METHODS = 100  # How many random methods to create CMoCs for
        updated_data_cmocs = full_experiment_random_methods(
            data_cmocs=candidate_moments_of_change,
            user_ids=self.usernames,
            full_user_histories=self.histories,
            N_RANDOM_METHODS=100,  # How many random methods to create CMoCs for)
            require_posts=False,
            require_within_annotated_spans=False,
            verbose=False,
        )

        return updated_data_cmocs

    def evaluate_p_r_f1(self, visualize=True):

        gtmocs = self.RedditDataset.concatenate_all_annotations()
        cmocs = self.identify_cmocs_for_all_histories_in_clpsych_dataset()
        user_timeline_sub_dict = self.RedditDataset.user_timeline_sub_dictionary()

        print("Starting Precision, Recall, F1 full experiment...")

        results = p_r_f1_full_experiment(
            cmocs=cmocs,
            gtmocs=gtmocs,
            user_timeline_sub_dict=user_timeline_sub_dict,
            users_can_have_multiple_timelines=True,
            limit_within_annotated_spans=True,
            methods=[
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
                "no cps",
                "random single day (mean across 100 seeds)",
            ],
            visualize=visualize,
            thresholds=range(0, 16),
            alpha=0.5,
            figsize=(10, 6),
            score_for_missing_target=np.NaN,
            relative_to="GTMoC",
            metrics=["F1", "precision", "recall"],
            save_fig=True,
            tcpd=False,  # Whether to use the TCPD benchmark adapted code
            verbose=False,
            # date_or_datetime="date",  # Whether to assess CPs relative to GTMoCs as dates, or datetimes
            return_sorted_mean=True,
            save_results=False,
        )

        return results

    def evaluate_medoids(self, visualize=True):
        """
        Performs the full medoid evaluation pipeline, for all the CMoCs identified
        on the histories of the users in the Reddit CLPsych corpus.
        """

        positive_annotations = self.RedditDataset.return_only_positive_annotations()
        annotated_timelines = self.RedditDataset.concatenate_all_annotations()
        cmocs = self.identify_cmocs_for_all_histories_in_clpsych_dataset()
        user_timeline_sub_dict = self.RedditDataset.user_timeline_sub_dictionary()

        print("Starting medoid full experiment...")
        results = medoid_full_experiment(
            cmocs=cmocs,
            annotated_timelines=annotated_timelines,
            load_centroids=False,
            only_positive_changes=positive_annotations,
            save_data=False,
            reward_style="only_rewards",
            distance_thresholds=range(0, 21),
            limit_within_annotated_spans=True,
            visualize=visualize,
            users_can_have_multiple_timelines=True,
            user_timeline_sub_dict=user_timeline_sub_dict,
            reddit_timelines=True,
        )

        return results

    def full_experiment(self, tau=5, xlim_upper=15):
        """
        Main function for full experiment. Produces a single figure which
        contains the results for the precision, recall, f1, and medoids
        evaluation. Candidate MoCs are also identified, and evaluated based
        on the GTMoCs. No need to run anything else, other than this function.
        """

        # Identify CMoCs for all users, using all specified methods
        # cmocs = self.identify_cmocs_for_all_histories_in_clpsych_dataset()

        # Full pipeline for evaluation of medoids
        medoid_results = self.evaluate_medoids(visualize=True)
        p_r_f1_results = self.evaluate_p_r_f1(visualize=True)

        results = full_experiment_all_4_subplots(
            medoid_results=medoid_results,
            p_r_f1=p_r_f1_results,
            create_results=False,
            methods_to_remove=["no cps", "keywords_three_categories"],
            tau=tau,
            xlim_upper=xlim_upper,
        )

        return results
