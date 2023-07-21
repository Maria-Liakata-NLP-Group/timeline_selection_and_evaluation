import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids

# This value is equivalent to a single day in timestamps. Add this value to timestamps to apply day arithmetic.
np_day_conversion = 1000000000 * 60 * 60 * 24


def preprocess_input_data(data):
    """
    Preprocesses input data into a NumPy array of shape (x, 1).
    """

    # Convert to NumPy array, of suitable shape
    if type(data) != np.ndarray:
        data = np.array(data)
    data = data.reshape(-1, 1)

    return data


def detect_centroids(
    data,
    style="medoid",
    allow_multiple_modes=False,
    apply_clustering=None,
    eps=5,
    min_samples=5,
    sort_X=True,
):
    """
    Outputs centroid locations, for some given input feature data (univariate). 
    
    Can optionally cluster the data, and then detect centroids from those clusters.
    
    Inputs:
    =======
    
    X = timestamped data, which you want to detect cluster centroids. e.g. Positive cases where a change was annotated.
    """

    X = preprocess_input_data(data)

    # -------- Clustering ------
    # Optionally apply a clustering step before returning centroids
    if apply_clustering != None:
        if apply_clustering.lower() == "dbscan":
            # Can optionally return estimated centroids
            clustering_results = discover_clusters(
                data,
                style=apply_clustering,
                eps=eps,
                min_samples=min_samples,
                sort_X=sort_X,
            )

            for cluster_label in np.unique(clustering_results["cluster"]):

                # Noise data-points should not be given a centroid
                if cluster_label == -1:
                    clustering_results.loc[
                        clustering_results["cluster"] == cluster_label, "centroid"
                    ] = np.nan
                else:
                    # Extract data just within the current cluster
                    data_within_current_cluster = clustering_results[
                        clustering_results["cluster"] == cluster_label
                    ]["X"]
                    centroid = detect_centroids(
                        data_within_current_cluster, style=style
                    )

                    # Save the centroid for these cluster labels
                    clustering_results.loc[
                        clustering_results["cluster"] == cluster_label, "centroid"
                    ] = centroid

        centroids = clustering_results["centroid"].unique()
        centroids = centroids[~pd.isnull(centroids)]
        centroids.sort()

        return centroids

    # -------- Detect centroid locations ------
    if style.lower() == "medoid":
        # Extract single medoid
        kmedoids = KMedoids(n_clusters=1, random_state=0, metric="euclidean").fit(X)
        # Store medoid data, per timeline
        centroids = kmedoids.cluster_centers_[0][0]

    elif style.lower() == "mean":
        centroids = pd.DataFrame(X)[0].mean()
    #         centroids = np.mean(X)

    elif style.lower() == "median":
        centroids = pd.DataFrame(X)[0].quantile(0.5, interpolation="midpoint")
    #         centroids = X.quantile(0.5, interpolation="midpoint")
    #         centroids = np.median(X)

    elif style.lower() == "mode":
        #         X = pd.DataFrame(X)[0].dt.date

        if allow_multiple_modes:
            centroids = X.mode()
            centroids = pd.DataFrame(X)[0].mode()
        else:
            #             centroids = X.mode()[0]
            centroids = pd.DataFrame(X)[0].mode()[0]

    elif style.lower() == "dbscan":
        clusters = discover_clusters(
            X,
            style="DBSCAN",
            eps=5,
            min_samples=5,
            sort_X=True,
            estimate_centroids_style="medoid",
        )
        centroids = clusters["centroids"].unique()

    # Post-process to ensure always of format timestamp.
    centroids = pd.Timestamp(centroids)

    return centroids


def discover_clusters(
    X, style="DBSCAN", eps=5, min_samples=5, sort_X=True, estimate_centroids_style=None
):

    X = preprocess_input_data(X)
    if style.lower() == "dbscan":
        # Pre-processing: Convert input days
        eps *= np_day_conversion

        # Run DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        cluster_labels = dbscan.labels_

    # Save results
    clustering_results = pd.DataFrame(X, columns=["X"])
    clustering_results["cluster"] = cluster_labels

    # Can optionally return estimated centroids
    if estimate_centroids_style != None:
        for cluster_label in np.unique(cluster_labels):

            # Noise data-points should not be given a centroid
            if cluster_label == -1:
                centroid = np.NaN
            else:
                # Extract data just within the current cluster
                data_within_current_cluster = clustering_results[
                    clustering_results["cluster"] == cluster_label
                ]["X"]
                centroid = detect_centroids(
                    data_within_current_cluster, style=estimate_centroids_style
                )

                # Save the centroid for these cluster labels
                clustering_results.loc[
                    clustering_results["cluster"] == cluster_label, "centroid"
                ] = centroid

    # Optionally sort the data
    if sort_X:
        clustering_results = clustering_results.sort_values(by=["X"])

    return clustering_results


def create_sub_timelines(
    timeline_annotations,
    apply_clustering="DBSCAN",
    remove_duplicate_annotations=False,
    eps=5,
    min_samples=5,
):
    """
    Discovers clusters of dense changes within users, and then creates timelines for those.   
    """

    # Remove duplicates
    if remove_duplicate_annotations:
        timeline_annotations = timeline_annotations.drop_duplicates(["datetime"])

    # Get unique user ids
    user_ids = timeline_annotations["user_id"].unique()

    # Discover subtimelines, per user. Append these to a dataaframe, and eventually return these.
    for i, u_id in enumerate(user_ids):
        sub_timelines = discover_clusters(
            timeline_annotations[timeline_annotations["user_id"] == u_id]["datetime"],
            eps=eps,
            min_samples=min_samples,
        )

        # Remove noise samples
        sub_timelines = sub_timelines[sub_timelines["cluster"] != -1]

        # Add timeline name
        #         sub_timelines['timeline_id'] = str(u_id) + sub_timelines['cluster'] + '_dbscan_eps' + eps + '_ms' + min_samples
        sub_timelines["timeline_id"] = (
            str(u_id)
            + "_cluster_"
            + sub_timelines["cluster"].apply(lambda x: str(x))
            + "_dbscan_eps"
            + str(eps)
            + "_ms"
            + str(min_samples)
        )
        sub_timelines["user_id"] = u_id
        #         sub_timelines = sub_timelines.set_index('timeline_id')

        # Add to dataframe
        if i == 0:
            all_user_sub_timelines = sub_timelines
        else:
            all_user_sub_timelines = all_user_sub_timelines.append(sub_timelines)

    # Sort results by index
    all_user_sub_timelines = all_user_sub_timelines.sort_values(by=["timeline_id"])
    all_user_sub_timelines = all_user_sub_timelines.rename(columns={"X": "datetime"})
    all_user_sub_timelines = all_user_sub_timelines.drop(["cluster"], axis=1)
    all_user_sub_timelines = all_user_sub_timelines.reset_index().drop("index", axis=1)

    return all_user_sub_timelines


def return_full_sub_timelines(assigned_clusters, df_annotations):
    """
    Returns including the negative annotations.
    """

    #     if assigned_clusters=None:
    #         assigned_clusters = create_centroids.create_sub_timelines(df_positive_changes,
    #                                                                   eps=EPS,
    #                                                                   min_samples=MIN_SAMPLES,
    #                                                                   remove_duplicate_annotations=False)

    cluster_ids = assigned_clusters["cluster_id"].unique()

    initialized_df = False
    for c_id in cluster_ids:

        min_dt = assigned_clusters[assigned_clusters["cluster_id"] == c_id][
            "datetime"
        ].min()
        max_dt = assigned_clusters[assigned_clusters["cluster_id"] == c_id][
            "datetime"
        ].max()

        u_id = int(c_id.split("_")[0])

        full_sub_timeline = df_annotations[
            (df_annotations["user_id"] == u_id)
            & (df_annotations["datetime"] >= min_dt)
            & (df_annotations["datetime"] <= max_dt)
        ]

        full_sub_timeline["cluster_id"] = c_id

        if initialized_df:
            all_sub_timelines = all_sub_timelines.append(full_sub_timeline)
        else:
            all_sub_timelines = full_sub_timeline
            initialized_df = True

    return all_sub_timelines


def output_all_centroid_styles(
    df_positive_changes,
    annotated_timelines=None,
    apply_clustering=None,
    remove_duplicate_annotations=False,
    eps=5,
    min_samples=5,
    reddit_timelines=False,
):
    # -------- Pre-processing ---------
    # Optionally drop postive duplicates
    if remove_duplicate_annotations:
        df_positive_changes = df_positive_changes.drop_duplicates(["datetime"])

    # Collect the timeline ids for timelines where changes occured
    positive_timeline_ids = list(df_positive_changes["timeline_id"].unique())

    # Users in the dataset
    observed_user_ids = list(df_positive_changes["user_id"].unique())

    # Save date, from datetime
    df_positive_changes["date"] = df_positive_changes["datetime"].dt.date

    # Initialize centroids dataframe
    dict_centroids = {}
    dict_centroids["timeline_id"] = positive_timeline_ids
    centroids = pd.DataFrame(dict_centroids)
    if reddit_timelines:
        pass
    else:
        centroids["user_id"] = centroids["timeline_id"].apply(
            lambda x: int(x.split("_")[0])
        )
    centroids = centroids.set_index("timeline_id")

    # Dataframe containing measures of central tendency, per timeline (mean, median)
    dict_centroid_locs = {}
    means = []
    medians = []
    for tl_id in positive_timeline_ids:
        dict_centroid_locs[tl_id] = {}
        for style in ["medoid", "mean", "median", "mode"]:
            if style == "mode":
                X = df_positive_changes[df_positive_changes["timeline_id"] == tl_id][
                    "date"
                ]
            else:
                X = df_positive_changes[df_positive_changes["timeline_id"] == tl_id][
                    "datetime"
                ]

            # Save detected centroids data
            dict_centroid_locs[tl_id][style] = detect_centroids(data=X, style=style)

    central_tendency = pd.DataFrame(dict_centroid_locs).T

    # Join measures to centroids dataframe
    centroids = centroids.join(central_tendency)

    # First, cluster the timelines to discover sub-timelines.
    # Then, detect centroids
    if apply_clustering == "dbscan" or apply_clustering == True:
        # Use DBSCAN to discover clusters, and generate timelines from these
        sub_timelines = create_sub_timelines(
            df_positive_changes, eps=eps, min_samples=min_samples
        )
        sub_timeline_centroids = output_all_centroid_styles(
            sub_timelines, apply_clustering=None, remove_duplicate_annotations=False
        )
        centroids = centroids.append(sub_timeline_centroids)

    # Sort by row index (i.e. timeline_id)
    centroids = centroids.sort_index(axis=0)

    # Optionally join user_name to timeline_id
    if reddit_timelines:
        a = centroids.reset_index()
        b = df_positive_changes[["timeline_id", "user_id"]].drop_duplicates()
        centroids = a.join(b.set_index("timeline_id"), on=["timeline_id"])
        centroids = centroids.set_index("timeline_id")
        centroids = centroids.sort_index(axis=0)

        # Score timelines
        centroids = assign_scores_to_centroids(centroids, annotated_timelines)

    return centroids


def evaluate_density_of_mocs_for_each_timeline(
    annotated_timelines, reddit_timelines=True
):
    """
    Returns a single column DataFrame, showing the mean switches/ escalations
    per annotated timeline.
    """
    # if reddit_timelines:
    #     label = "label"
    # else:
    #     label = "Switch/Escalation"

    label = 'label_2'

    # Density (mean)
    mean_positive_annotations = (
        annotated_timelines.groupby("timeline_id")
        .mean()
        .rename(columns={label: "Mean Switch/Escalation"})[["Mean Switch/Escalation"]]
    )

    return mean_positive_annotations


def minmax_scale_centroids_scores(densities_of_timelines):
    """
    Scales the densities scores in range -1 and +1
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    densities_of_timelines["min_max"] = scaler.fit_transform(
        densities_of_timelines[["Mean Switch/Escalation"]]
    )


def assign_scores_to_centroids(centroids, annotated_timelines):
    """
    Centroids must be assigned (binary) scores depending on how dense the 
    the MoCs for that annotated timeline.  
    """

    densities_of_timelines = evaluate_density_of_mocs_for_each_timeline(
        annotated_timelines
    )

    # Retain only positive centroids (i.e. which have at least 1 MoC)
    mean_positive_annotations = densities_of_timelines[
        densities_of_timelines["Mean Switch/Escalation"] > 0
    ]

    # # Scale densities scores in range -1 to +1, for timelines that have at least 1 MoC
    # scaled_densities = minmax_scale_centroids_scores(mean_positive_annotations)

    # Join min_max scores of timelines to centroids, on the timeline_id index
    centroids = centroids.join(mean_positive_annotations["Mean Switch/Escalation"])

    # Set binary scores if density is greater/less than median density across all timelines
    centroids["binary"] = np.nan
    centroids["binary"].loc[
        centroids["Mean Switch/Escalation"]
        >= centroids["Mean Switch/Escalation"].median()
    ] = +1
    centroids["binary"].loc[
        centroids["Mean Switch/Escalation"]
        < centroids["Mean Switch/Escalation"].median()
    ] = -1

    return centroids


# def handle_zero_changes_timelines():

