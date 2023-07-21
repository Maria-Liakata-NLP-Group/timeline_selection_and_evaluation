"""
Provides functionality to compute the Levenshtein distance metric, to assess the 
alignment of the predicted CMoC with respect to the annotated ground-truth 
Moments of Change.
"""


import numpy as np
import pandas as pd
from datetime import date, timedelta
from ..io.my_pickler import my_pickler


def levenshtein_distance(s, t, print_distance_matrix=False, change_point_setting=False):
    """Dynamic programming approach to compute Levenshtein Distance.
    https://stackoverflow.com/questions/2460177/edit-distance-in-python 
    
    Args:
        s (string): source string / sequence containing predicted CPs.
        t (string): target string / true CP sequence.

    Returns:
        distance (int): 
        """

    if change_point_setting:
        if len(s) != len(t):
            raise TypeError(
                "The predicted change-points and true change-point sequences must be equal in length."
            )

    if len(s) > len(t):
        s, t = t, s

    distances = range(len(s) + 1)
    if print_distance_matrix:
        distance_matrix = [distances]
    for j, c2 in enumerate(t):
        distances_ = [j + 1]
        for i, c1 in enumerate(s):
            if c1 == c2:
                distances_.append(distances[i])
            else:
                distances_.append(
                    1 + min((distances[i], distances[i + 1], distances_[-1]))
                )
        distances = distances_
        if print_distance_matrix:
            distance_matrix.append(distances)

    if print_distance_matrix:
        distance_matrix = np.array(distance_matrix)
        distance_matrix = pd.DataFrame(
            distance_matrix, columns=[""] + list(s), index=[""] + list(t)
        )
        print(distance_matrix)

    return distances[-1]


def create_all_moc_strings():
    """
    Returns a tuple of (gtmoc_strings, cmoc_strings). 
    
    Loads the relevant datasets. Need to run this just once at the start of the 
    experiment pipeline.

    Returns:
        [type]: [description]
    """

    # Load some datasets
    df_positive_changes = my_pickler("i", "df_positive_changes")
    searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")
    data_cmocs = my_pickler("i", "candidate_moments_of_change")

    methods = list(data_cmocs.keys())
    user_ids = list(searched_timeline_spans["user_id"].unique())

    # Data processing
    searched_timeline_spans["start_of_timeline"] = searched_timeline_spans[
        "start_of_timeline"
    ].apply(lambda x: np.datetime64(x))
    searched_timeline_spans["end_of_timeline"] = searched_timeline_spans[
        "end_of_timeline"
    ].apply(lambda x: np.datetime64(x))

    # Create CMoC Strings
    cmoc_strings = {}
    for m in methods:
        print(m)
        cmoc_strings[m] = {}
        for u in user_ids:
            cmocs_for_m_u = data_cmocs[m][u]
            if len(cmocs_for_m_u) > 0:
                cmocs_for_m_u = pd.DataFrame(cmocs_for_m_u).rename(columns={0: "date"})
                cmocs_for_m_u["date"] = cmocs_for_m_u["date"].apply(
                    lambda x: np.datetime64(x)
                )

                # Create list containing dates within span ranges
                u_span = searched_timeline_spans[
                    searched_timeline_spans["user_id"] == u
                ]
                u_start = u_span["start_of_timeline"].values[0]
                u_end = u_span["end_of_timeline"].values[0]

                # Filter to within searched span
                dates_with_cmoc_for_user_u = list(
                    cmocs_for_m_u[
                        (cmocs_for_m_u["date"] >= u_start)
                        & (cmocs_for_m_u["date"] <= u_end)
                    ]["date"]
                )
            else:
                dates_with_cmoc_for_user_u = (
                    []
                )  # in the case where no CMoCs were predicted.

            empty_string = list(
                pd.date_range(u_start, u_end, freq="d")
            )  # List containing dates in the searched span

            # Create empty string
            df_string = pd.DataFrame(empty_string).rename(columns={0: "date"})
            df_string["string"] = "-"

            # Set 'x' where is within other list (e.g. GTMoCs)
            df_string.loc[
                df_string["date"].isin(dates_with_cmoc_for_user_u), "string"
            ] = "x"

            # Convert series into a connected string
            final_string = "".join(list(df_string["string"]))
            cmoc_string = final_string
            cmoc_strings[m][u] = cmoc_string

    # Create GTMoC Strings
    # Convert timelines into strings, where GTMoC are ‘x’, and rest is ‘_’.
    gtmoc_strings = {}
    for u in user_ids:
        # Get dates where there is a positive annotation
        dates_with_moc_for_user_u = list(
            df_positive_changes[df_positive_changes["user_id"] == u]
            .groupby("date")
            .sum()["Switch/Escalation"]
            .index
        )

        # Create list containing dates within span ranges
        u_span = searched_timeline_spans[searched_timeline_spans["user_id"] == u]
        u_start = u_span["start_of_timeline"].values[0]
        u_end = u_span["end_of_timeline"].values[0]
        empty_string = list(
            pd.date_range(u_start, u_end, freq="d")
        )  # List containing dates in the searched span

        # Create empty string
        df_string = pd.DataFrame(empty_string).rename(columns={0: "date"})
        df_string["string"] = "-"

        # Set 'x' where is within other list (e.g. GTMoCs)
        df_string.loc[df_string["date"].isin(dates_with_moc_for_user_u), "string"] = "x"

        # Convert series into a connected string
        final_string = "".join(list(df_string["string"]))
        gtmoc_string = final_string
        gtmoc_strings[u] = gtmoc_string

    return gtmoc_strings, cmoc_strings


def full_experiment():
    """
    Returns a dataframe containing the Levenshtein Distances (LDs), summed 
    across all users, and displayed in a dataframe for each method. The table 
    is then sorted where methods with the lowest LD rank at the top 
    (which is our goal).

    Returns:
        [type]: [description]
    """

    # Load Data
    gtmoc_strings, cmoc_strings = create_all_moc_strings()
    methods = list(cmoc_strings.keys())
    user_ids = list(gtmoc_strings.keys())

    ld_for_all_methods = {}
    for m in methods:
        user_distances = []
        for u in user_ids:
            cmoc_string = cmoc_strings[m][u]
            gtmoc_string = gtmoc_strings[u]

            ld = levenshtein_distance(
                cmoc_string, gtmoc_string, change_point_setting=True
            )
            user_distances.append(ld)
        summed_ld = np.sum(user_distances)
        ld_for_all_methods[m] = summed_ld
    ld_df = pd.DataFrame(pd.Series(ld_for_all_methods), columns=["summed_ld"])
    ld_df = ld_df.sort_values("summed_ld", ascending=True)

    return ld_df
