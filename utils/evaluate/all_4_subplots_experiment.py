import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.evaluate.medoid_evaluation import full_experiment
from utils.evaluate.precision_recall import \
    experiment_visualize_varying_thresholds
from utils.visualize.plot_results import plot_4_metrics_subplots


def NormalizeData(data):

    # Get maximum and minimum values across the entire dataframe
    max_value = data.max().max()
    min_value = data.min().min()

    return (data - min_value) / (max_value - min_value)


def full_experiment_all_4_subplots(
    medoid_results,
    p_r_f1,
    tau=5,  # The window size to return a single dataframe of results for.
    create_results=False,
    methods_to_remove=["no cps", "keywords_three_categories"],
    xlim_upper=15,
):
    """
    The main function for this whole project. 
    
    Identifies CMoCs and then evalautes them using Medoids and Precision, Recall,
    F1. Visualizes the results across all 4 metrics in a single figure with 
    4 subplots.
    """
    # Get results from CMoCs to GTMoCs
    if create_results:
        medoid_results = full_experiment(
            save_data=False,
            reward_style="only_rewards",
            distance_thresholds=range(0, 16),
            limit_within_annotated_spans=True,
            visualize=True,
        )
        p_r_f1 = experiment_visualize_varying_thresholds(
            thresholds=range(0, 16),
            alpha=0.5,
            figsize=(10, 6),
            centroid_type="dense",
            score_for_missing_target=np.NaN,
            relative_to="GTMoC",
            metrics=["F1", "precision", "recall"],
            save_fig=False,
            tcpd=False,  # Whether to use the TCPD benchmark adapted code
            verbose=False,
            date_or_datetime="date",  # Whether to assess CPs relative to GTMoCs as dates, or datetimes
            return_sorted_mean=True,
            save_results=True,
        )

    prf = ["precision", "recall", "F1"]

    medoid_results = medoid_results.drop(
        columns=methods_to_remove, errors="ignore"
    )  # Ignore, if the methods don't exist
    for metric in prf:
        p_r_f1[metric] = p_r_f1[metric].T.drop(
            columns=methods_to_remove, errors="ignore"
        )

    medoid_results = medoid_results.iloc[: xlim_upper + 1, :]

    scaled_medoid_results = NormalizeData(medoid_results)

    plot_4_metrics_subplots(
        p_r_f1, scaled_medoid_results, savefig=True, xlim_upper=xlim_upper + 0.2
    )
    # plot_4_metrics_subplots(
    #     p_r_f1, scaled_medoid_results, savefig=True, xlim_upper=xlim_upper
    # )

    # Return table
    full_table = pd.concat(
        (
            p_r_f1["precision"].T[tau].rename("Precision"),
            p_r_f1["recall"].T[tau].rename("Recall"),
            p_r_f1["F1"].T[tau].rename("F1"),
            scaled_medoid_results.T[tau].rename("Medoid Votes"),
        ),
        axis=1,
    )

    return full_table
