"""
Creates histories dataframes, for annotated users. Then concatenates this with 
the timelines dataframe. 
"""

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, "../../../timeline_generation/")  # Adds higher directory to python modules path

from utils.io import data_handler

sys.path.insert(0, "../../../global_utils/")  # Adds higher directory to python modules path
from notification_bot import send_email


def main():
    embedding_type = 'sentence-bert'
    
    
    TalkLifeDataset = data_handler.TalkLifeDataset(include_embeddings=True, 
                                            save_processed_timelines=False, 
                                            load_timelines_from_saved=True)
    # Create histories
    tl_histories = TalkLifeDataset.create_raw_history_dataframe_for_annotated_users(save=True, load=False)
    
    # Combine histories with timelines
    df_full = TalkLifeDataset.combine_histories_with_timelines(save=True, load=False)
    
    # Get embeddings for full dataset, and save
    df_full_sentence_bert = TalkLifeDataset.return_combined_history_and_timelines_with_embeddings_df(load_from_file=False, save_data=True, embedding_type=embedding_type)

if __name__ == "__main__":
    try:
        main()
        send_email(subject='[{}] Finished running with no errors'.format(os.path.basename(__file__)),
               message='See pickles in /datasets: talklife_sentence-bert_full_history_embeddings',
               receiving_email='angryasparagus@hotmail.com')
    except Exception as error_message:
        send_email(subject='[{}] Error, script failed.'.format(os.path.basename(__file__)),
            message=str(error_message),
            receiving_email='angryasparagus@hotmail.com')
