"""
Provides functionality to compute the Levenshtein distance metric, to assess the 
alignment of the predicted CMoC with respect to the annotated ground-truth 
Moments of Change.
"""


from re import S
import numpy as np
import pandas as pd
from datetime import date, timedelta
from ..io.my_pickler import my_pickler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Load some datasets
# df_positive_changes = my_pickler("i", "df_positive_changes")
# searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")
# data_cmocs = my_pickler("i", "candidate_moments_of_change")
# df_annotations = my_pickler("i", "df_annotations")
# data_daily_interactions = my_pickler("i", "data_daily_interactions")

# # Pre-processing
# df_annotations["date"] = df_annotations["datetime"].dt.date


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


def weighted_levenshtein_distance(
    s,
    t,
    print_distance_matrix=False,
    change_point_setting=True,
    cost_missed_cp=1.0,
    cost_declared_cp=2.0,
    verbose=True,
):
    """Dynamic programming approach to compute Levenshtein Distance.
    https://stackoverflow.com/questions/2460177/edit-distance-in-python 
    
    Args:
        s (string): source string / sequence containing predicted CPs.
        t (string): target string / true CP sequence.
        cost_missed_cp (float): False-negative cost for incorrectly missing a CP. True = 'x', pred = '-'
        cost_declared_cp (float): False-positive cost for incorrectly declaring a CP. True = '-', pred ='x'
        

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
    for j, c2 in enumerate(t):  # True string
        distances_ = [j + 1]
        for i, c1 in enumerate(s):  # Predicted string
            if c1 == c2:  # Correct prediction
                cost = distances[i]
            elif (c1 == "x") & (c2 == "-"):  # False positive cost
                cost = cost_declared_cp + min(
                    (distances[i], distances[i + 1], distances_[-1])
                )
            elif (c1 == "-") & (c2 == "x"):  # False negative cost
                cost = cost_missed_cp + min(
                    (distances[i], distances[i + 1], distances_[-1])
                )
            else:
                raise TypeError(
                    "Input string has wrong characters. Must be either 'x', or '-'."
                )
            distances_.append(cost)
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

    # Load data
    # data_cmocs = my_pickler("i", "candidate_moments_of_change")
    data_cmocs = my_pickler("i", "cmocs_with_100_random")
    searched_timeline_spans = my_pickler("i", "timeline_spans_sent_to_adam")
    df_positive_changes = my_pickler("i", "df_positive_changes")
    df_annotations = my_pickler("i", "df_annotations")
    data_daily_interactions = my_pickler("i", "data_daily_interactions")

    # Pre-processing
    df_annotations["date"] = df_annotations["datetime"].dt.date

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
    all_user_string_stats = {}
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

        # Collect associated number of GTMoCs and posts per day within this string
        d = df_string.set_index("date")
        d = d.join(data_daily_interactions[u]["posts"])
        d = d.join(
            df_annotations[df_annotations["user_id"] == u]
            .groupby("date")
            .sum()["Switch/Escalation"]
        )
        d = d.fillna(0)
        d["density_gtmoc"] = d["Switch/Escalation"] / d["posts"]
        d = d.fillna(0)  # Dates with no posts, will have a density of zero
        string_stats = d

        # Convert series into a connected string
        final_string = "".join(list(df_string["string"]))
        gtmoc_string = final_string
        gtmoc_strings[u] = gtmoc_string

        # Store number of GTMoCs and posts for all users
        all_user_string_stats[u] = string_stats

    return gtmoc_strings, cmoc_strings, all_user_string_stats


def full_experiment_unweighted(
    equal_costs=True,
    print_distance_matrix=False,
    cost_missed_cp=1.0,
    cost_declared_cp=2.0,
    verbose=True,
    n_gtmoc_weighting=None,
):
    """
    Returns a dataframe containing the Levenshtein Distances (LDs), summed 
    across all users, and displayed in a dataframe for each method. The table 
    is then sorted where methods with the lowest LD rank at the top 
    (which is our goal).

    Returns:
        [type]: [description]
    """

    # Load Data
    gtmoc_strings, cmoc_strings, _ = create_all_moc_strings()
    methods = list(cmoc_strings.keys())
    user_ids = list(gtmoc_strings.keys())

    if verbose:
        if equal_costs:
            print("LD with equal weighting...")
        else:
            print(
                "{} = cost_missed_cp = \n{} = cost_declared_cp".format(
                    cost_missed_cp, cost_declared_cp
                )
            )

    ld_for_all_methods = {}
    for m in methods:
        user_distances = []
        for u in user_ids:
            cmoc_string = cmoc_strings[m][u]
            gtmoc_string = gtmoc_strings[u]

            if equal_costs:
                ld = levenshtein_distance(
                    cmoc_string, gtmoc_string, change_point_setting=True
                )
            else:
                ld = weighted_levenshtein_distance(
                    cmoc_string,
                    gtmoc_string,
                    print_distance_matrix=print_distance_matrix,
                    change_point_setting=True,
                    cost_missed_cp=cost_missed_cp,
                    cost_declared_cp=cost_declared_cp,
                )
            user_distances.append(ld)
        summed_ld = np.sum(user_distances)
        ld_for_all_methods[m] = summed_ld
    ld_df = pd.DataFrame(pd.Series(ld_for_all_methods), columns=["summed_ld"])
    ld_df = handle_multiple_random_cmocs(ld_df.T).T
    ld_df = ld_df.sort_values("summed_ld", ascending=True)

    return ld_df


def ld_df_varying_weights(
    cost_missed_cp_range=[-100, 100],
    cost_declared_cp_range=[-100, 100],
    interval_size=1.0,
    verbose=False,
    methods=["bocpd_pg_h1000_a0.01_b10"],
):
    # Load Data
    gtmoc_strings, cmoc_strings, _ = create_all_moc_strings()
    user_ids = list(gtmoc_strings.keys())

    # Constrct cost grid ranges
    cost_missed_cp_grid = list(
        np.arange(cost_missed_cp_range[0], cost_missed_cp_range[1], interval_size)
    )
    cost_declared_cp_grid = list(
        np.arange(cost_declared_cp_range[0], cost_declared_cp_range[1], interval_size)
    )

    ld_grid_for_all_methods = {}
    df_ld_grid_for_all_methods = {}
    for m in methods:
        ld_grid_for_all_methods[m] = {}
        for cost_missed_cp in cost_missed_cp_grid:
            ld_grid_for_all_methods[m][cost_missed_cp] = {}
            for cost_declared_cp in cost_declared_cp_grid:
                user_distances = []
                for u in user_ids:
                    cmoc_string = cmoc_strings[m][u]
                    gtmoc_string = gtmoc_strings[u]

                    ld = weighted_levenshtein_distance(
                        cmoc_string,
                        gtmoc_string,
                        print_distance_matrix=False,
                        change_point_setting=True,
                        cost_missed_cp=cost_missed_cp,
                        cost_declared_cp=cost_declared_cp,
                        verbose=verbose,
                    )

                    user_distances.append(ld)
                summed_ld = np.sum(user_distances)
                ld_grid_for_all_methods[m][cost_missed_cp][cost_declared_cp] = summed_ld

        df_ld_grid_for_all_methods[m] = pd.DataFrame(ld_grid_for_all_methods[m])

    return df_ld_grid_for_all_methods


def visualize_grid(
    grids, style="surface", savefig=False, figsize=(8, 7),
):

    plt.figure(figsize=figsize)
    methods = list(grids.keys())
    if style == "heatmap":
        method = methods[0]
        sns.heatmap(grids[method], cbar_kws={"label": "Edit Distance"})
        plt.xlabel("False Negative Cost \n(incorrectly missed a GTMoC)")
        plt.ylabel("False Positive Cost \n(incorrectly predicted a GTMoC)")

        plt.title("Weighted Edit Distances for method:\n{}".format(method))
    else:
        ax = plt.axes(projection="3d")
        for method in methods:
            df = grids[method]

            X = df.index
            Y = df.columns
            Z = df.values

            if style.lower() == "contour":
                ax.contour3D(X, Y, Z, label=method)

                ax.legend()
                ax.set_xlabel("False Negative Cost \n(incorrectly missed a GTMoC)")
                ax.set_ylabel("False Positive Cost \n(incorrectly predicted a GTMoC)")
                ax.set_zlabel("Edit Distance")
            elif style.lower() == "surface":
                # ax.plot_surface(X, Y, Z, label=method)
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
                fig.update_layout(
                    autosize=False,
                    width=500,
                    height=500,
                    # margin=dict(l=65, r=50, b=65, t=90)
                )
                fig.show()

        plt.title("Weighted Edit Distances")

    plt.tight_layout()
    plt.show()


def weighted_gtmoc_levenshtein_distance(
    s,
    t,
    user_string_stats,
    print_distance_matrix=False,
    change_point_setting=True,
    verbose=True,
    false_positive_cost="posts",  # or 'density_gtmoc'
    false_negative_cost="Switch/Escalation",
    false_negative_multiplier=1.0,
    false_positive_multiplier=1.0,
):

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

    for j, c2 in enumerate(t):  # True string (GTMoCs)
        distances_ = [j + 1]
        for i, c1 in enumerate(s):  # Predicted string
            if c1 == c2:  # Correct prediction
                cost = distances[i]
            elif (c1 == "x") & (c2 == "-"):  # False positive cost
                gtmoc_cost = (
                    user_string_stats.iloc[j][false_positive_cost]
                    * false_positive_multiplier
                )
                cost = gtmoc_cost + min(
                    (distances[i], distances[i + 1], distances_[-1])
                )
            elif (c1 == "-") & (c2 == "x"):  # False negative cost
                gtmoc_cost = (
                    user_string_stats.iloc[j][false_negative_cost]
                    * false_negative_multiplier
                )
                cost = gtmoc_cost + min(
                    (distances[i], distances[i + 1], distances_[-1])
                )
            else:
                raise TypeError(
                    "Input string has wrong characters. Must be either 'x', or '-'."
                )
            distances_.append(cost)
        distances = distances_
        if print_distance_matrix:
            distance_matrix.append(distances)

    if print_distance_matrix:
        distance_matrix = np.array(distance_matrix)
        distance_matrix = pd.DataFrame(
            distance_matrix, columns=[""] + list(s), index=[""] + list(t)
        )
        # print(distance_matrix)
        return distances[-1], distance_matrix

    return distances[-1]


def full_experiment_gtmoc_weighting(
    print_distance_matrix=False,
    verbose=True,
    false_positive_cost="posts",  # or 'density_gtmoc'
    false_negative_cost="Switch/Escalation",
    false_negative_multiplier=1.0,
    false_positive_multiplier=1.0,
):
    """
    Returns a dataframe containing the Levenshtein Distances (LDs), summed 
    across all users, and displayed in a dataframe for each method. The table 
    is then sorted where methods with the lowest LD rank at the top 
    (which is our goal).

    Returns:
        [type]: [description]
    """

    # Load Data
    gtmoc_strings, cmoc_strings, string_stats = create_all_moc_strings()
    methods = list(cmoc_strings.keys())
    user_ids = list(gtmoc_strings.keys())

    ld_for_all_methods = {}
    for m in methods:
        user_distances = []
        for u in user_ids:
            cmoc_string = cmoc_strings[m][u]
            gtmoc_string = gtmoc_strings[u]
            string_stats_user = string_stats[u]

            # Get weightings per day (i.e. number of GTMoCs per day)
            ld = weighted_gtmoc_levenshtein_distance(
                s=cmoc_string,
                t=gtmoc_string,
                user_string_stats=string_stats_user,
                print_distance_matrix=False,
                change_point_setting=True,
                false_positive_cost=false_positive_cost,  # or 'density_gtmoc'
                false_negative_cost=false_negative_cost,
                false_negative_multiplier=false_negative_multiplier,
                false_positive_multiplier=false_positive_multiplier,
            )

            user_distances.append(ld)
        summed_ld = np.sum(user_distances)
        ld_for_all_methods[m] = summed_ld
    ld_df = pd.DataFrame(pd.Series(ld_for_all_methods), columns=["summed_ld"])
    ld_df = ld_df.sort_values("summed_ld", ascending=True)

    return ld_df


def levenshtein_to_test(s, t, cost_value=1):
    """ 
    From Wikipedia article; Iterative with two matrix rows. 
    # Christopher P. Matthews
    # christophermatthews1985@gmail.com
    # Sacramento, CA, USA
    """
    if s == t:
        return 0
    elif len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else cost_value
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(t)]


def modified_levenshtein_distance(
    s,
    t,
    user_string_stats,
    print_distance_matrix=False,
    change_point_setting=True,
    verbose=True,
    false_positive_cost="posts",  # or 'density_gtmoc'
    false_negative_cost="Switch/Escalation",
    false_negative_multiplier=1.0,
    false_positive_multiplier=1.0,
    normalize_by_n_substitutions=False,
):

    if change_point_setting:
        if len(s) != len(t):
            raise TypeError(
                "The predicted change-points and true change-point sequences must be equal in length."
            )

    if false_positive_cost == None:
        false_positive_cost = 1
    if false_negative_cost == None:
        false_negative_cost = 1

    if len(s) > len(t):
        s, t = t, s

    distances = range(len(s) + 1)
    if print_distance_matrix:
        distance_matrix = [distances]

    n_operations = 0  # 1, rather than 0 - to avoid division by zero

    # True string (GTMoCs)
    for j, c2 in enumerate(t):
        distances_ = [j + 1]

        # Predicted string
        for i, c1 in enumerate(s):

            # Correct prediction (true positive or true negative)
            if c1 == c2:
                cost = distances[i]

            # False positive
            elif (c1 == "x") & (c2 == "-"):
                n_operations += 1
                if type(false_positive_cost) == str:
                    cost = (
                        user_string_stats.iloc[j][false_positive_cost]
                        * false_positive_multiplier
                    )
                else:
                    cost = false_positive_cost
                # Otherwise, the cost should be a numerical value
                cost = cost + min((distances[i], distances[i + 1], distances_[-1]))

            # False negative
            elif (c1 == "-") & (c2 == "x"):
                n_operations += 1
                if type(false_negative_cost) == str:
                    cost = (
                        user_string_stats.iloc[j][false_negative_cost]
                        * false_negative_multiplier
                    )
                # Otherwise, the cost should be a numerical value
                else:
                    cost = false_negative_cost
                cost = cost + min((distances[i], distances[i + 1], distances_[-1]))
            else:
                raise TypeError(
                    "Input string has wrong characters. Must be either 'x', or '-'."
                )
            distances_.append(cost)
        distances = distances_
        if print_distance_matrix:
            distance_matrix.append(distances)

    edit_distance = distances[-1]
    if normalize_by_n_substitutions:
        if n_operations > 0:
            edit_distance = edit_distance / n_operations
        else:
            edit_distance = edit_distance  # Can't divide by zero

    if print_distance_matrix:
        distance_matrix = np.array(distance_matrix)
        distance_matrix = pd.DataFrame(
            distance_matrix, columns=[""] + list(s), index=[""] + list(t)
        )
        # print(distance_matrix)
        return edit_distance, distance_matrix

    return edit_distance


def full_experiment_modified_levenshtein_distance(
    print_distance_matrix=False,
    verbose=True,
    false_positive_cost="posts",  # or 'density_gtmoc'
    false_negative_cost="Switch/Escalation",
    false_negative_multiplier=1.0,
    false_positive_multiplier=1.0,
    normalize_by_n_substitutions=False,
    prior_across_all_users=None,
    prior_day_level=False,  # Whether the p(G) should be on the day level, or the post level
):
    """
    Returns a dataframe containing the Levenshtein Distances (LDs), summed 
    across all users, and displayed in a dataframe for each method. The table 
    is then sorted where methods with the lowest LD rank at the top 
    (which is our goal).

    Returns:
        [type]: [description]
    """

    # Load Data
    gtmoc_strings, cmoc_strings, string_stats = create_all_moc_strings()
    methods = list(cmoc_strings.keys())
    user_ids = list(gtmoc_strings.keys())

    # Probability of GTMoCs
    if prior_across_all_users != None:
        prior_g_all = prior_moc(
            across_all_users=prior_across_all_users, prior_day_level=prior_day_level
        )

    ld_for_all_methods = {}
    for m in methods:
        user_distances = []
        for u in user_ids:
            cmoc_string = cmoc_strings[m][u]
            gtmoc_string = gtmoc_strings[u]
            string_stats_user = string_stats[u]

            # Whether to multiply costs by priors of GTMoCs
            if prior_across_all_users:
                prior_g = prior_g_all
            elif prior_across_all_users == False:  # Prior is for each user individually
                prior_g = prior_g_all[u]

            if prior_across_all_users != None:
                false_positive_multiplier = prior_g
                false_negative_multiplier = 1 - prior_g

            # Get weightings per day (i.e. number of GTMoCs per day)
            ld = modified_levenshtein_distance(
                s=cmoc_string,
                t=gtmoc_string,
                user_string_stats=string_stats_user,
                print_distance_matrix=False,
                change_point_setting=True,
                false_positive_cost=false_positive_cost,  # or 'density_gtmoc'
                false_negative_cost=false_negative_cost,
                false_negative_multiplier=false_negative_multiplier,
                false_positive_multiplier=false_positive_multiplier,
                normalize_by_n_substitutions=normalize_by_n_substitutions,
            )

            user_distances.append(ld)
        summed_ld = np.sum(user_distances)
        ld_for_all_methods[m] = summed_ld
    ld_df = pd.DataFrame(pd.Series(ld_for_all_methods), columns=["summed_ld"])

    ld_df = handle_multiple_random_cmocs(ld_df.T).T

    ld_df = ld_df.sort_values("summed_ld", ascending=True)

    return ld_df


def prior_moc(
    across_all_users=True, prior_day_level=False
):  # Whether the p(G) should be on the day level, or the post level):
    """
    Should return 0.155 across all users. It is the number of GTMoC divided by 
    the number of posts. 
    
    The prior of p(G) for a given day containing a GTMoC is 0.135. I did a 
    groupby on the user level and day level.

    Args:
        across_all_users (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    df_annotations = my_pickler("i", "df_annotations")
    timeline_length = 15
    n_timelines = 500
    df_annotations["date"] = df_annotations["datetime"].dt.date

    if across_all_users:
        if prior_day_level:
            prior_g = (
                df_annotations.groupby(["user_id", "date"])["Switch/Escalation"].sum()
                > 0
            ).sum() / (n_timelines * timeline_length)
        else:
            prior_g = df_annotations["Switch/Escalation"].mean()
    else:  # Compute priors on a user basis
        if prior_day_level:
            df_n_mocs_per_day = (
                df_annotations.groupby(["user_id", "date"])["Switch/Escalation"]
                .sum()
                .reset_index()
            )
            df_n_mocs_per_day["binary_at_least_1_gtmoc"] = (
                df_n_mocs_per_day["Switch/Escalation"] > 0
            ).astype(int)

            prior_g = (
                df_n_mocs_per_day.groupby("user_id")["binary_at_least_1_gtmoc"].sum()
                / timeline_length
            )
        else:
            prior_g = df_annotations.groupby("user_id")["Switch/Escalation"].mean()
    return prior_g


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
