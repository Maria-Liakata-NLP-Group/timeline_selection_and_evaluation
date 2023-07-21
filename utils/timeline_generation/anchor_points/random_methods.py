import numpy as np
from utils.io.my_pickler import my_pickler
from datetime import datetime


def random_single_day(
    full_user_histories,
    user_ids,
    N_RANDOM_METHODS=100,  # How many random methods to create CMoCs for)
    require_posts=False,
    data_cmocs={},
    require_within_annotated_spans=False,  # Whether the random days must be within annotated spans
    verbose=True,
):

    # Random single day across whole user history, which has a post
    for i in range(0, N_RANDOM_METHODS):
        if verbose:
            print("Progress: {}/{} complete".format(i, N_RANDOM_METHODS))
        seed = i  # range(0, 100)
        np.random.seed(seed)  # Set the seed as the user id

        # Write method name, including seed
        m = "random single day"
        if require_posts:
            m += " with posts"
        m += " {}".format(seed)

        # Save CMoCs for this method
        data_cmocs[m] = {}
        for u_id in user_ids:
            user_feature_data = full_user_histories[u_id]

            # Filter to only days which have posts, if desired
            if require_posts:
                user_feature_data = full_user_histories[u_id]["posts"][
                    full_user_histories[u_id]["posts"] > 0
                ]  # Filter only to days where there are posts

            if len(user_feature_data) > 0:
                user_cmoc = [
                    post_processing(
                        cmoc=np.random.choice(list(user_feature_data.index))
                    )
                ]
            else:  # If user has no days with posts, then don't return any CMoCs for this user
                user_cmoc = []

            data_cmocs[m][u_id] = user_cmoc

    return data_cmocs


def post_processing(cmoc):

    # Convert to datetimes
    cmoc = cmoc.to_pydatetime().date()

    return cmoc


def full_experiment_random_methods(
    data_cmocs={},
    user_ids=None,
    full_user_histories=None,
    N_RANDOM_METHODS=100,  # How many random methods to create CMoCs for)
    require_posts=False,
    require_within_annotated_spans=False,
    verbose=True,
):
    if user_ids == None:
        # Load Data
        df_annotations = my_pickler(
            "i", "df_annotations"
        )  # Used to only create CMoCs for annotated users

        # Pre-processing
        user_ids = list(list(df_annotations["user_id"].unique()))
        user_ids = sorted(
            user_ids
        )  # Sort user ids, so we go through them in numerical order

    if full_user_histories == None:
        full_user_histories = my_pickler("i", "data_daily_interactions")

    data_cmocs = random_single_day(
        full_user_histories,
        user_ids,
        N_RANDOM_METHODS=N_RANDOM_METHODS,  # How many random methods to create CMoCs for)
        require_posts=require_posts,
        data_cmocs=data_cmocs,
        require_within_annotated_spans=require_within_annotated_spans,  # Whether the random days must be within annotated spans
        verbose=verbose,
    )

    return data_cmocs
