import sys

sys.path.insert(0, "../../../../")  # Adds higher directory to python modules path

from utils.io import data_handler
from utils.timeline_generation.anchor_points.keywords import risk_severity


def identify_cmocs_for_reddit_timelines_using_keywords():
    """
    Returns a dictionary where keys are user ids, and values are a list of
    sorted datetimes corresponding to CMoCs identified via keywords, but only
    within annotated timelines.
    """
    KeywordsSuicideRiskSeverity = risk_severity.KeywordsSuicideRiskSeverity()
    RedditDataset = data_handler.RedditDataset()

    # Load annotated posts
    all_annotations = RedditDataset.concatenate_all_annotations()

    # Check if the content of each post contains a CMoC
    all_annotations["cmoc"] = all_annotations["content"].apply(
        lambda x: KeywordsSuicideRiskSeverity.input_text_contains_suicide_risk(x)
    )

    user_ids = list(all_annotations["user_id"].unique())

    cmocs = {}
    for u_id in user_ids:
        user_annotations = all_annotations[all_annotations["user_id"] == u_id]
        user_cmocs = user_annotations[user_annotations["cmoc"] == True]
        if len(user_cmocs) > 0:
            user_cmocs = list(
                set(user_annotations[user_annotations["cmoc"] == True]["date"])
            )
            user_cmocs.sort()
        else:
            user_cmocs = []  # Handle case, where no CMoCs are detected for user.

        cmocs[u_id] = user_cmocs

    return cmocs
