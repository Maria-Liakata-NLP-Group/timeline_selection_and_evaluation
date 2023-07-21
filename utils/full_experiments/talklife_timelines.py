import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "../../")  # Adds higher directory to python modules path
from utils.evaluate.all_4_subplots_experiment import \
    full_experiment_all_4_subplots
# Medoid evaluation
from utils.evaluate.create_centroids import output_all_centroid_styles
from utils.evaluate.medoid_evaluation import full_experiment
from utils.evaluate.medoid_evaluation import \
    full_experiment as medoid_full_experiment
from utils.evaluate.precision_recall import \
    experiment_visualize_varying_thresholds
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
from utils.visualize.plot_results import (plot_4_metrics_subplots,
                                          plot_varying_thresholds)


class TalklifeTimelinesExperiments:
    """ """

    def __init__(self):
        self.x = 0  # dummy variable


    def full_experiment(self, tau=5, xlim_upper=6):
        """
        Main function for full experiment. Produces a single figure which
        contains the results for the precision, recall, f1, and medoids
        evaluation. Candidate MoCs are also identified, and evaluated based
        on the GTMoCs. No need to run anything else, other than this function.
        """

        # Identify CMoCs for all users, using all specified methods
        # cmocs = self.identify_cmocs_for_all_histories_in_clpsych_dataset()

        # Full pipeline for evaluation of medoids
        medoid_results = full_experiment(save_data=False,
    reward_style="only_rewards",
    distance_thresholds=range(0, 16),
    limit_within_annotated_spans=True,
    visualize=True, reddit_timelines=False)
        
        p_r_f1_results = experiment_visualize_varying_thresholds(thresholds=range(0, 16),
    alpha=0.5,
    figsize=(10, 6),
    centroid_type="dense",
    score_for_missing_target=np.NaN,
    relative_to="GTMoC",
    metrics=["F1", "precision", "recall"],
    save_fig=True,
    tcpd=False,  # Whether to use the TCPD benchmark adapted code
    verbose=False,
    date_or_datetime="date",  # Whether to assess CPs relative to GTMoCs as dates, or datetimes
    return_sorted_mean=True,
    save_results=True)

        results = full_experiment_all_4_subplots(
            medoid_results=medoid_results,
            p_r_f1=p_r_f1_results,
            create_results=False,
            methods_to_remove=["no cps", "keywords_three_categories"],
            tau=tau,
            xlim_upper=xlim_upper,
        )

        return results
    
        # full_experiment_all_4_subplots(
        #     medoid_results,
        #     p_r_f1,
        #     tau=5,  # The window size to return a single dataframe of results for.
        #     create_results=False,
        #     methods_to_remove=["no cps", "keywords_three_categories"],
        #     xlim_upper=15,
        # )
        
        
    
    
