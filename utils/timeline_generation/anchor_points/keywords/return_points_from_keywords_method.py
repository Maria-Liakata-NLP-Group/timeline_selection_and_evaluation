import numpy as np
from ....io.my_pickler import my_pickler

# DATA_PATH = "../../../../datasets/processed_data/talklife/pickle/"


def return_cps_from_keywords_method(
    style="keywords_all", only_observed_users=True, to_days=False, process_into="dates"
):

    raw_cps = load_stored_cps_from_keywords_method(
        style=style, stored_path="./datasets/raw_data/talklife/keywords_cps/"
    )
    cps = postprocess_keywords_cps(raw_cps, to_days=to_days)

    return cps


def load_stored_cps_from_keywords_method(
    style="all_keywords", stored_path="./datasets/raw_data/talklife/keywords_cps/"
):
    """
    Returns raw CPs. These need to be postprocessed.
    """
    # Pre-process to fit file-name.
    if style == "keywords_all":
        style = "all_keywords"
    elif style == "keywords_three_categories":
        style = "three_category_keywords"

    path = stored_path + style + "_anomalies.p"
    cps = np.load(path, allow_pickle=True)

    return cps


def postprocess_keywords_cps(
    raw_cps, to_days=False, only_observed_users=True, process_into="dates"
):

    """
    Takes as input a dictionary, where the keys are the user id and the values are the change-points
    in datetimes.
    """

    if only_observed_users:
        data_daily_interactions = my_pickler(
            "i", "observed_data_daily_interactions", verbose=False
        )
    else:
        data_daily_interactions = my_pickler(
            "i", "data_daily_interactions", verbose=False
        )

    cps = {}
    raw_keys = list(raw_cps.keys())

    # Convert user keys to integers (removing the .p extension)
    for rk in raw_keys:
        pk = int(rk[:-2])  # Processed keys (remove the .p extension)

        # Only use the users that are observed in the dataset
        if pk in data_daily_interactions.keys():

            # Convert from set into list
            dt_cps = list(raw_cps[rk])

            # Handle missing values
            if len(dt_cps) < 1:
                cps[pk] = []  # Return an empty list, if no change-points

            else:
                # Round CPs to days since start of the user
                if to_days:
                    days_cps = []
                    start_date = data_daily_interactions[pk].ne(0).idxmax()["posts"]

                    for dt_cp in dt_cps:
                        delta = dt_cp - start_date
                        days_cps.append(delta.days)

                    cps[pk] = days_cps

                # Else, return as datetimes
                else:
                    if process_into == "dates":

                        anchor_points = []
                        for cp in dt_cps:
                            anchor_points.append(cp.date())

                        # Remove duplicate anchor points
                        anchor_points = list(set(anchor_points))

                        # Sort in ascending order
                        anchor_points.sort()

                        dt_cps = anchor_points

                    cps[pk] = dt_cps

        # Disregard users that don't appear in the dataset
        else:
            pass

    return cps
