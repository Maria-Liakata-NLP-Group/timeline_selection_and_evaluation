"""
Creates histories dataframes, for annotated users. Then concatenates this with 
the timelines dataframe. 
"""

import pickle
import pandas as pd

import sys
sys.path.insert(0, "../../../timeline_generation/")  # Adds higher directory to python modules path

from utils.io import data_handler


if __name__ == '__main__':
    TalkLifeDataset = data_handler.TalkLifeDataset(include_embeddings=True, 
                                                save_processed_timelines=False, 
                                                load_timelines_from_saved=True)

    df = TalkLifeDataset.create_unaggregated_all_user_history_dataframe(verbose=False, columns=[], only_annotated_users=True, save_as_pickle=True)
