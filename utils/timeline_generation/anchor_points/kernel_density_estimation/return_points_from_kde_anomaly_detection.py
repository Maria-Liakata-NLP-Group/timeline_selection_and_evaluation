import numpy as np
import scipy
from datetime import timedelta as td


def return_points_from_kde_anomaly_detection(
    user_data, user_id, style, cmoc_days_prior=0
):
    u_id = user_id

    feature_style = style.split("_")[-1]
    if feature_style == "received":
        feature = "comments_received"
    elif feature_style == "posts":
        feature = "posts"
    user_data = user_data[feature]

    # Apply pre-processing
    user_data = preprocess_kde(user_data, user_id)

    anomalies = []
    # Get spans for different methods
    if "_".join(style.split("_")[:2]) == "high_activity":
        spans = find_high_activity(users=user_data, toa=feature)[u_id]
        if len(spans) > 0:
            for span in spans:
                anomalies.append(span[7])

    elif "_".join(style.split("_")[:2]) == "low_activity":
        spans = find_no_activity(users=user_data, toa=feature)[u_id]

        # Extract the first element of the span as the CMoC
        if len(spans) > 0:
            for span in spans:
                cmoc = span[0]
                days = td(cmoc_days_prior)
                cmoc -= days  # Optionally return a few days away as the CMoC (e.g. cmoc_start=-8 returns the date 8 days prior as the CMoC)
                anomalies.append(cmoc)

    elif "_".join(style.split("_")[:4]) == "high_and_low_activity":
        spans = find_high_activity(users=user_data, toa=feature)[u_id]
        if len(spans) > 0:
            for span in spans:
                anomalies.append(span[7])

        spans = find_no_activity(users=user_data, toa=feature)[u_id]
        # Extract the first element of the span as the CMoC
        if len(spans) > 0:
            for span in spans:
                cmoc = span[0]
                days = td(cmoc_days_prior)
                anomalies.append(cmoc)

    return anomalies


"""
--- The code below is almost entirely from Dr. Adam Tsakalidis ---
"""


def find_high_activity(users, toa="posts"):
    """
    Finds high activity anomalies.
    """

    from itertools import groupby

    anom_dates, anom_posts = dict(), dict()
    cnt = 0
    overall_anomalies = 0
    for user in users.keys():
        anom_dates[user], anom_posts[user] = [], []
        cnt += 1
        posts, dates = users[user][toa], users[user]["dates"]

        window_size = 90
        num_days = 14  # minimum number of days

        idx = 0
        if len(posts) > 30:  # we target users with >=30 days of activity
            for i in range(
                30, len(posts)
            ):  # we ignore the first 30 days of these users; "i" is the day counter since they first posted
                data = posts[max(0, i - window_size) : i]
                if (np.std(data) != 0) & (posts[i] > np.average(posts)):
                    kde = scipy.stats.gaussian_kde(data)
                    proba = kde.pdf(posts[i])
                    if proba < 0.01:
                        overall_anomalies += 1
                        prev_dates, prev_posts = anom_dates[user], anom_posts[user]

                        # if i<len(posts)-3:
                        if i < len(posts) - 7:
                            # prev_dates.append([dates[i]-td(days=p) for p in range(3,-4,-1)])
                            # prev_posts.append([posts[p] for p in range(i-3, i+4)])
                            prev_dates.append(
                                [dates[i] - td(days=p) for p in range(7, -8, -1)]
                            )
                            prev_posts.append([posts[p] for p in range(i - 7, i + 8)])
                        else:
                            # prev_dates.append([dates[i]-td(days=p) for p in range(3,i-len(posts),-1)])
                            # prev_posts.append([posts[p] for p in range(i-3, len(posts))])
                            prev_dates.append(
                                [
                                    dates[i] - td(days=p)
                                    for p in range(7, i - len(posts), -1)
                                ]
                            )
                            prev_posts.append(
                                [posts[p] for p in range(i - 7, len(posts))]
                            )

                        anom_dates[user] = prev_dates
                        anom_posts[user] = prev_posts
        if cnt % 1000 == 0:
            print(cnt, "\t", len(anom_dates[user]), overall_anomalies)

        high_dates, high_posts = anom_dates[user], anom_posts[user]
    return anom_dates


def find_no_activity(users, toa="posts"):
    """
    Finds low activity anomalies
    """

    from itertools import groupby

    anom_dates, anom_posts = dict(), dict()
    cnt = 0
    overall_anomalies = 0
    for user in users.keys():
        anom_dates[user], anom_posts[user] = [], []
        cnt += 1
        #         posts, dates = users[user][toa][0], users[user]['dates']
        posts, dates = np.array(users[user][toa]), users[user]["dates"]
        # Anthony: these are two lists of CONSECUTIVE dates
        # here and the corresponding numPosts on each day

        num_days = 14  # minimum number of days to consider as "silence"
        window_size = 90  # number of days to consider when calculating priors
        prob_active = [
            len(np.where(posts[max(0, i - window_size) : i] > 0)[0])
            / min(i, window_size)
            for i in range(1, len(posts) + 1)
        ]

        idx = 0
        # we target users with at least 30 days of activity and 10% active
        if (len(posts) > 30) & (len(np.where(posts > 0)[0]) / len(posts) > 0.1):
            for val, lista in groupby(posts):
                lista = list(lista)
                if (val == 0) & (len(lista) >= num_days) & (idx > 14):
                    proba = (1 - prob_active[idx]) ** len(lista)
                    if proba < 0.01:
                        # update dates
                        preval = anom_dates[user]
                        update = [dates[i] for i in range(idx, idx + len(lista))]
                        preval.append(update)
                        anom_dates[user] = preval

                idx += len(lista)
        overall_anomalies += len(anom_dates[user])
        if cnt % 1000 == 0:
            print(cnt, "\t", len(anom_dates[user]), overall_anomalies)

    return anom_dates


def preprocess_kde(user_data, user_id):
    feature = user_data.name
    preprocessed_users = {}
    preprocessed_users[user_id] = {"dates": list(user_data.index)}
    preprocessed_users[user_id][feature] = list(user_data)

    return preprocessed_users
