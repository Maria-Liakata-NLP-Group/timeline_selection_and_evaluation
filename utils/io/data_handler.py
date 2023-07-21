import datetime as python_datetime
import json
import os
import pickle
import random
import sys
import xdrlib
from cgi import test
from os import listdir, walk
from os.path import isfile, join
from re import S
from tkinter import X
from unittest import TextTestRunner

import numpy as np
import pandas as pd
import torch
from utils.io import text_vectorizer
from utils.io.my_pickler import my_pickler

sys.path.insert(0, "../../timeline_selection_and_evaluation/")
import global_parameters

"""
Loads the datasets from /datadrive/ on the vm.
"""


class RedditDataset:
    """
    CLPsych 2022 Reddit timeline dataset.

    255 user timelines in total. There are 185 unique users in the dataset.
    """

    def __init__(
        self,
        include_embeddings=True,
        load_timelines_from_saved=True,
        save_processed_timelines=False,
    ):

        # Paths
        self.path_dataset = global_parameters.path_clpsych_2022_raw_dataset
        self.path_train_users = self.path_dataset + "train_users.json"
        self.path_test_users = self.path_dataset + "test_users.json"
        self.path_unhashed_user_data = (
            global_parameters.path_clpsych_2022_unhashed_user_data
        )

        self.path_jenny_csv = global_parameters.path_clpsych_2022_jenny_csv

        self.path_fede_datasets = global_parameters.path_clpsych_2022_fede_datasets

        # Load JSONs
        # Maps the timeline keys to the hashed users
        self.path_mapper_hash = global_parameters.path_clpsych_2022_mapper_hash

        # Dictionaries, with keys that are hashes that represent the user IDs
        # self.train_users = json.load(open(self.path_train_users, "r"))
        # self.test_users = json.load(open(self.path_test_users, "r"))

        # The hashes in the above annotated timelines can be used with this mapper
        # self.mapper_hash = json.load(open(self.path_mapper_hash, "r"))

        # Aggregate user history on the daily basis
        self.aggregate = "daily"
        # self.aggregate = "raw"

        # Create datasets
        # self.user_timelines, self.linked_users = self.deanonymize_annotated_timelines()
        # self.user_histories = self.extract_all_annotated_users_histories()
        # self.user_timelines, self.linked_users = self.deanonymize_annotated_timelines()
        self.timelines = self.return_all_timelines_as_df(
            include_embeddings=include_embeddings,
            load_from_file=load_timelines_from_saved,
            save_data=save_processed_timelines,
            embedding_type="sentence-bert",
            assign_folds=True,
            post_process=True,
        )
        self.prototype_timelines = self.return_prototype_dataset(
            save=save_processed_timelines
        )
        # self.histories = self.extract_all_annotated_users_histories()

    def post_process_annotations_df(self, df):
        """
        * Orders the columns by datetime -> timeline_id
        * Resets the index, based on chronology
        """

        # Sort by datetime, then timeline id
        df = df.sort_values(["datetime", "timeline_id"], ascending=[True, True])
        df = df.reset_index().drop("index", axis=1)

        # Order, and select columns
        if "fold" in df.columns:

            df = df[
                [
                    "datetime",
                    "time_epoch_days",
                    "content",
                    # "label_5",
                    "label_3",
                    "label_2",
                    "user_id",
                    "timeline_id",
                    "postid",
                    "train_or_test",
                    "fold",
                ]
            ]
        else:
            df = df[
                [
                    "datetime",
                    "time_epoch_days",
                    "content",
                    # "label_5",
                    "label_3",
                    "label_2",
                    "user_id",
                    "timeline_id",
                    "postid",
                    "train_or_test",
                ]
            ]

        return df

    def add_embeddings_to_dataframe(
        self,
        df,
        embedding_type="sentence-bert",
        load_from_file=False,
        project_name="reddit",
        save_data=True,
    ):
        """
        Adds additional column to timeline dataframe, which are the
        sentence-bert embeddings of the corresponding posts (content)
        """
        full_df = df.copy()

        # Set postid as index, and then join embeddings to it later
        df = df[["content", "postid"]]
        df = df.set_index("postid")

        if load_from_file:
            embedding = my_pickler(
                "i",
                "{}_{}_embeddings".format(project_name, embedding_type),
                folder="datasets",
            )

        # Otherwise, recreate these embeddings
        else:
            # Embed these posts
            print(
                "Creating {} embeddings for {} posts...".format(
                    embedding_type, df.shape[0]
                )
            )
            TextVectorizer = text_vectorizer.TextVectorizer()
            text = df["content"]
            embedding = text.apply(
                lambda x: TextVectorizer.vectorize(x, embedding_type=embedding_type)
            )

            print("Finished creating embeddings.")

            # Save the embeddings to datadrive
            if save_data:
                my_pickler(
                    "o",
                    "{}_{}_embeddings".format(project_name, embedding_type),
                    embedding,
                    folder="datasets",
                )

        # Add column to dataframe
        df[embedding_type] = embedding

        # Join the embeddings on the postid
        df = df.reset_index()
        df = df[["postid", embedding_type]]
        full_df = full_df.merge(df, on="postid", how="left")

        return full_df

    def extract_hashed_timeline(self, hashed, train_or_test="train"):
        """
        Returns a DataFrame containing the information for all the annotated
        posts, for a given hashed timeline.
        """

        # Path to the stored timeline, for the given hash
        path = self.path_dataset + train_or_test + "/" + hashed + ".tsv"

        # Read the stored data
        df = pd.read_csv(path, sep="\t")

        # Process the raw stored data (e.g. convert strings to DateTimes)
        df = self.process_timeline_dataframe(df)

        return df

    def process_timeline_dataframe(self, user_timeline, binary_labels=False):
        """
        Processes an input user timeline DataFrame.

        * Converts string like dates into DateTimes.
        * Concatenates titles with content, with a space in between.
        Any missing text is converted to ''.
        * Convert MoCs to binary labels (0 if no MoC present, 1 otherwise)
        * Chronologically orders by the datetime.
        """

        # Convert to DateTimes
        user_timeline["datetime"] = pd.to_datetime(
            user_timeline["date"], format="%d-%m-%Y, %X"
        )

        # Concatenate title with content, replacing NaNs with empty strings
        user_timeline["content"] = (
            user_timeline["title"].fillna("")
            + " "
            + user_timeline["content"].fillna("")
        )

        # Aggregate label 0 and '0'
        user_timeline.loc[(user_timeline.label == 0), "label"] = "0"

        # Binarize the MoC labels (1 if MoC present, 0 otherwise)
        if binary_labels:
            one_hot_annotations = pd.get_dummies(user_timeline["label"])
            one_hot_annotations["0"] = 0  # Keep this to zero, so sum doesn't become 1
            is_moc = one_hot_annotations.sum(axis=1)  # 1 if moc, 0 otherwise
            user_timeline["label"] = is_moc  # Replace with binary labels
        else:
            user_timeline["label_3"] = user_timeline["label"].copy()

            # 3 Labels
            user_timeline.loc[
                (user_timeline["label_3"] == "IEP")
                | (user_timeline["label_3"] == "IE"),
                "label_3",
            ] = "E"
            user_timeline.loc[
                (user_timeline["label_3"] == "ISB")
                | (user_timeline["label_3"] == "IS"),
                "label_3",
            ] = "S"

            # Binary
            user_timeline["label_2"] = user_timeline[
                "label_3"
            ].copy()  # Binary (1, 0) if is GTMoC
            user_timeline.loc[
                (user_timeline["label_2"] == "E") | (user_timeline["label_3"] == "S"),
                "label_2",
            ] = 1
            user_timeline.loc[(user_timeline["label_2"] == "0"), "label_2"] = 0

            user_timeline = user_timeline.drop(
                "label", axis=1
            )  # Remove raw original label

        # Remove unnecessary columns
        user_timeline = user_timeline.drop(columns=["date", "title"])

        # Chronologically order DataFrame, by datetime
        user_timeline = user_timeline.sort_values(by="datetime", ascending=True)
        user_timeline = user_timeline.reset_index().drop(columns=["index"])

        return user_timeline

    def extract_ground_truth_timelines(self):
        """
        Returns the annotated timelines.
        """
        pass

    def deanonymize_hashed_timelines(self):
        """
        Converts a hashed timeline id to the user's actual name.

        Returns a dictionary where the key corresponds to the hashed timeline
        id, and the value is the actual username
        (previously was the path to the stored user timelineIn the path
        is contained the usernames id, just before the pickle extension.)

        Note that the way Fede stored it is that there was a quick hack, where
        the keys and values were reversed for some users. This is why I do
        another set of for loops.

        * 138 keys in total
        * 82 unique users
        * Note there are 255 user timelines, but only 138 keys match Fede's
        hashed database.
        """

        # Extract mapping
        mapper = {}  # Key is hash, value is username file path

        for users in [self.train_users, self.test_users]:
            # Loop over each hashed user ID
            for user in users.keys():

                # Extract all annotated timelines for each user
                timelines = users[user]["timelines"]

                # Loop over each annotated timeline, for given user
                for timeline in timelines:

                    # Map this timeline to the hash mapper
                    for key, value in self.mapper_hash.items():
                        if key == timeline:
                            # Process value, to just contain username
                            username = self.extract_username_from_raw(value)
                            # username = value.split("/")[-1].split(".p_")[0]

                            mapper[timeline] = username
                        elif value == timeline:
                            # Process key, to just contain username
                            username = self.extract_username_from_raw(key)
                            # username = key.split("/")[-1].split(".p_")[0]

                            mapper[timeline] = username

        return mapper

    def extract_unhashed_user_data(self, user_name):
        """
        Returns the full user history for a deanonymized (unhashed) user.
        """

        path = self.path_unhashed_user_data + user_name + ".p"

        with open(path, "rb") as handle:
            user_history = pickle.load(handle)

        # Process the dataset (filter columns, fix data types)
        user_history = self.process_user_history(user_history)

        return user_history

    def process_user_history(self, user_history):
        """
        Processes the raw user history for a single user.
        Also chronologically orders it.

        Returns only the:
        * num_comments
        * datetime
        * content
        """
        # Convert list into a DataFrame
        user_history = pd.DataFrame(user_history)

        # Convert to DateTimes
        user_history["datetime"] = pd.to_datetime(user_history["created_utc"], unit="s")

        # Concatenate title with content
        user_history["content"] = (
            user_history["title"].fillna("") + " " + user_history["selftext"].fillna("")
        )

        # Ensure the user history is chronologically ordered
        user_history = user_history.sort_values(by="datetime", ascending=True)
        user_history = user_history.reset_index().drop(columns=["index"])

        # Filter to only necessary columns
        user_history = user_history[["datetime", "content", "num_comments"]]

        # Aggregate to get the number of daily posts, and comments received
        if self.aggregate == "daily":
            user_history = self.aggregate_daily_interactions(user_history)

        return user_history

    def aggregate_daily_interactions(self, user_history):
        """
        Processes the raw history of a user, to return a time-series containing
        the number of daily interactions for each of the features recorded for
        that history.

        Returns a DataFrame with the following columns:
        * comments_received
        * posts

        The index is "datetime" and is just a date, which is chronologically ordered.
        """

        per_day = user_history.drop("content", axis=1)
        per_day = per_day.rename(columns={"num_comments": "comments_received"})
        per_day["posts"] = 1  # Initialize each timestamp to count as as single post

        # Aggregate, to get daily posts and comments received
        per_day = per_day.set_index("datetime")
        per_day = per_day.resample("D").apply(sum)

        return per_day

    def extract_all_annotated_users_histories(self):
        """
        Only the linked users.

        185 unique users.
        """
        data_daily_interactions = {}
        for user_id in self.linked_users:
            data_daily_interactions[user_id] = pd.DataFrame(
                RedditDataset.extract_unhashed_user_data(self, user_id)
            )

        return data_daily_interactions

    def extract_all_hashed_timeline_data(self):
        """
        Dictionary where key is the username (first) concatenated with their
        timeline id (after), which are both hashed. The value is the dataframe
        containing the annotated timeline.

        * 255 keys (hashed timelines) in total
        """
        all_user_timelines = {}

        # Loop over each hashed username in both the train and test set
        for users in [self.train_users, self.test_users]:
            if users == self.train_users:
                train_or_test = "train"
            else:
                train_or_test = "test"
            for user in users:
                current_users_timelines = users[user]["timelines"]
                for timeline_id in current_users_timelines:
                    user_timeline_id = "{}_{}".format(user, timeline_id)
                    # user_timeline_id = "{}".format(timeline_id)
                    all_user_timelines[user_timeline_id] = self.extract_hashed_timeline(
                        timeline_id, train_or_test=train_or_test
                    )

                    # Post process DataFrame

        return all_user_timelines

    def deanonymize_annotated_timelines(self):
        """
        Returns a dictionary where the key corresponds to the deanoynmized
        username concatenated with the timeline id. The values are DataFrames
        containing the anotated timeline for that user's timeline.

        'exampleusername_12235408u01238' will be the key for example.

        138 hashed timelines, which can be linked to deanonymized user histories.
        82 unique users which can be linked. These 138 will be returned.

        Note there are 255 user timelines in the training and test sets combined.

        """

        # Deanonymize hashed usernames
        hashed_timeline_to_username = self.deanonymize_hashed_timelines()

        deanonymized_timeline_data = {}
        usernames = []
        linked_hashed_timelines = list(hashed_timeline_to_username.keys())

        # Get all the hashed user data, then deanonymize it
        all_hashed_timeline_data = self.extract_all_hashed_timeline_data()
        user_timeline_ids = list(all_hashed_timeline_data.keys())
        for hashed_timeline in linked_hashed_timelines:
            for utl_id in user_timeline_ids:
                u_id, tl_id = utl_id.split("_")
                if hashed_timeline == tl_id:
                    actual_username = hashed_timeline_to_username[hashed_timeline]
                    # Concatenate deanonymized user id with timeline id
                    user_timeline_id = actual_username + "_" + tl_id

                    # Link this to the anonymous annotated timeline
                    deanonymized_timeline_data[
                        user_timeline_id
                    ] = all_hashed_timeline_data[utl_id]
                    usernames.append(actual_username)

        return deanonymized_timeline_data, usernames

    def load_reddit_tsv_files(self, train_or_test="train"):

        path_dataset = global_parameters.path_raw_clpsych_dataset
        path_train = path_dataset + "train/"
        path_test = path_dataset + "test/"

        train_file_names = next(walk(path_train), (None, None, []))[2]  # [] if no file
        test_file_names = next(walk(path_test), (None, None, []))[2]  # [] if no file

        if train_or_test == "train":
            file_names = train_file_names
            p = path_train
        else:
            file_names = test_file_names
            p = path_test

        initialized = False
        for file_name in file_names:
            timeline_id = file_name[:-4]  # anonymized, hashed

            df_user_timeline = pd.read_csv(p + file_name, sep="\t")
            df_user_timeline["timeline_id"] = timeline_id

            if initialized:
                df_all = pd.concat([df_all, df_user_timeline], axis=0)
            else:
                df_all = df_user_timeline
                initialized = True
        df_all["train_or_test"] = train_or_test

        return df_all

    def load_all_reddit_tsv_files(self, apply_processing=True):
        """
        Returns a dataframe containing all user timelines in the reddit dataset.
        Contains a column indicating whether it is part of the train or test set.

        255 timelines in total. There is no user id column, as these are all
        anonyomous.
        """
        df_train = self.load_reddit_tsv_files(train_or_test="train")
        df_test = self.load_reddit_tsv_files(train_or_test="test")
        df = pd.concat([df_train, df_test], axis=0)

        df["user_id"] = df[
            "timeline_id"
        ]  # This is a hack, which isn't correct. It should be fine though

        if apply_processing:
            df = self.process_timeline_dataframe(df)

        return df

    def get_username_and_timeline_id_from_concatenation(self, utl):
        """
        Get username from concatenated username and timeline id
        """

        split_utl = utl.split("_")  # Extract username and timeline id
        tl = split_utl[-1]
        if len(split_utl) > 2:
            u = "_".join(split_utl[:-1])  # If username has "_" inside
        else:
            u = split_utl[0]

        return u, tl

    def user_timeline_sub_dictionary(self):
        """
        Returns a two tier hierachical dictionary, where first key is user,
        followed by a list of more dictionaries where subkeys are timeline ids,
        and values are the annotated timelines.
        """

        timelines, _ = self.deanonymize_annotated_timelines()
        u_tl_ids = list(timelines.keys())  # 'username_timelineid'

        user_timelines = {}

        # Loop over each user timeline id
        for utl in u_tl_ids:
            u, tl = self.get_username_and_timeline_id_from_concatenation(utl)

            if u not in user_timelines:
                user_timelines[u] = {}
            user_timelines[u][tl] = timelines[utl]

        return user_timelines

    def read_jenny_csv(self, postprocess=True):
        """
        Loads the mapper which Jenny provided, linking hashes to usernames.
        """
        path = self.path_jenny_csv

        dataset = pd.read_csv(path)
        if postprocess:
            dataset["username"] = dataset["raw"].apply(self.extract_username_from_raw)

            dataset = dataset.dropna()
            dataset = dataset.drop("Unnamed: 0", axis=1)
            dataset = dataset.reset_index().drop(columns=["index"])

        return dataset

    def extract_username_from_raw(self, raw_text):
        split_list = raw_text.split("/")
        if len(split_list) > 1:
            username = split_list[-1].split(".p")[0]
        else:
            username = np.NaN

        return username

    def read_fede_all_dataset(self):

        path_dataset = global_parameters.path_clpsych_2022_fede_hashed_dataset
        path_erisk = path_dataset + "erisk_hashed.json"
        path_reddit_new_hashed = path_dataset + "reddit_new_hashed.json"
        path_reddit_umd_hashed = path_dataset + "reddit_umd_hashed.json"

        datasets = {
            "erisk": json.load(open(path_erisk, "r")),
            "reddit_new_hashed": json.load(open(path_reddit_new_hashed, "r")),
            "reddit_umd_hashed": json.load(open(path_reddit_umd_hashed, "r")),
        }

        return datasets

    # Adam's code
    def get_all_mappings(self):
        """Returns hashed_id (timeline/post) to actual_id"""
        hashed_to_tlid = dict()
        for dataset in ["reddit_umd", "reddit_new", "erisk"]:
            data_mapping = self.get_mapping_hashed_to_original(
                dataset
            )  # hashed->actual
            for key in data_mapping.keys():
                hashed_to_tlid[key] = data_mapping[key]
        return hashed_to_tlid

    def get_mapping_hashed_to_original(self, dataset):
        mapping = json.load(
            open(
                global_parameters.path_fede_hashed_dataset + dataset + "_hashed.json",
                "r",
            )
        )
        toreturn = dict()
        for key in mapping.keys():
            if dataset != "erisk":
                if (
                    mapping[key] != "f0a01960ce"
                ):  # the duplicate timeline (removing from reddit_new)
                    toreturn[mapping[key]] = key
            else:
                toreturn[key] = mapping[key]
                # toreturn[key] = mapping[key]
        return toreturn

    def concatenate_all_annotations(self):
        """
        Returns a DataFrame with all annotations concatenated in a single
        DataFrame.
        """
        timelines = self.user_timelines  # All user timelines
        user_timeline_ids = timelines.keys()

        initialized = False
        for utl in user_timeline_ids:
            (
                username,
                timeline_id,
            ) = self.get_username_and_timeline_id_from_concatenation(utl)

            tl = timelines[utl]
            tl["user_id"] = username
            tl["timeline_id"] = timeline_id

            if initialized:
                all_annotations = pd.concat([all_annotations, tl], axis=0)
            else:
                all_annotations = tl
                initialized = True

        all_annotations = all_annotations.reset_index().drop("index", axis=1)

        # Post-processing
        all_annotations["date"] = all_annotations["datetime"].dt.date

        return all_annotations

    def return_only_positive_annotations(self, label="label_2"):
        all_annotations = self.concatenate_all_annotations()

        only_positive_annotations = all_annotations[all_annotations[label] == 1]
        only_positive_annotations = only_positive_annotations.reset_index().drop(
            "index", axis=1
        )

        return only_positive_annotations

    def return_all_timelines_as_df(
        self,
        include_embeddings=True,
        load_from_file=False,
        save_data=True,
        embedding_type="sentence-bert",
        assign_folds=True,
        post_process=True,
        include_epoch_time=True,
    ):
        """
        Returns the annotated dataset for Reddit, including sentence-bert
        embeddings.
        """

        if load_from_file:
            all_timelines = my_pickler(
                "i", "reddit_timelines_{}".format(embedding_type), folder="datasets"
            )

            return all_timelines
        else:

            all_timelines = self.load_all_reddit_tsv_files(apply_processing=True)

            all_timelines = all_timelines.reset_index().drop("index", axis=1)

            all_timelines["fold"] = np.NaN  # Default value, before folds assigned

            if include_epoch_time:
                all_timelines["time_epoch_days"] = all_timelines["datetime"].apply(
                    convert_datetime_to_epoch_time
                )

            if post_process:
                all_timelines = self.post_process_annotations_df(all_timelines)

            # Include sentence-bert embeddings
            if include_embeddings:
                all_timelines = self.add_embeddings_to_dataframe(
                    all_timelines,
                    embedding_type=embedding_type,
                    load_from_file=False,
                )

            if assign_folds:
                all_timelines = self.apply_folds(
                    all_timelines, only_on_training_set=True, seed=1, n_folds=5
                )

            if save_data:
                my_pickler(
                    "o",
                    "reddit_timelines_{}".format(embedding_type),
                    all_timelines,
                    folder="datasets",
                )

            return all_timelines

    def return_all_timelines_as_df_reddit_new_only(
        self,
        include_embeddings=True,
        load_from_file=False,
        save_data=True,
        embedding_type="sentence-bert",
        assign_folds=True,
        post_process=True,
        include_epoch_time=True,
    ):
        """
        Returns the annotated dataset for Reddit, including sentence-bert
        embeddings.
        """

        if load_from_file:
            all_timelines = my_pickler(
                "i", "reddit_timelines_{}".format(embedding_type), folder="datasets"
            )

            return all_timelines
        else:
            all_timelines = self.read_raw_timelines(apply_processing=True)

            all_timelines["fold"] = 0  # Just for processing

            if include_epoch_time:
                all_timelines["time_epoch_days"] = all_timelines["datetime"].apply(
                    convert_datetime_to_epoch_time
                )

            if post_process:
                all_timelines = self.post_process_annotations_df(all_timelines)

            # Include sentence-bert embeddings
            if include_embeddings:
                all_timelines = self.add_embeddings_to_dataframe(
                    all_timelines,
                    embedding_type=embedding_type,
                    load_from_file=False,
                )

            if assign_folds:
                all_timelines = self.apply_folds(all_timelines)

            if save_data:
                my_pickler(
                    "o",
                    "reddit_timelines_{}".format(embedding_type),
                    all_timelines,
                    folder="datasets",
                )

            return all_timelines

    def find_list_index(self, list_of_lists, search_term):
        """
        Finds the list index of a search term, in a list of lists.
        """
        folds_df = pd.DataFrame(list_of_lists).T
        tl_id = search_term
        ind = folds_df.loc[:, (folds_df == tl_id).any()].columns.values[0]

        return ind

    def apply_folds(self, df, only_on_training_set=True, seed=1, n_folds=5):
        """
        Assigns folds to each timeline in the dataframe.
        """

        # Select only training set ids, if desired
        if only_on_training_set:
            timeline_ids = list(
                df[df["train_or_test"] == "train"]["timeline_id"].unique()
            )
        else:
            timeline_ids = list(df["timeline_id"].unique())

        # Shuffle the ids
        np.random.seed(seed)
        np.random.shuffle(timeline_ids)
        folded_ids = [timeline_ids[i::n_folds] for i in range(n_folds)]

        # Assign fold number to timeline ids in DataFrame
        for tl_id in timeline_ids:
            fold = self.find_list_index(folded_ids, search_term=tl_id)
            df.loc[(df["timeline_id"] == tl_id), "fold"] = fold

        return df

    def return_prototype_dataset(self, save=True):
        """
        Returns the smaller, prototyping version of the full dataset. This
        contains just 10 timelines, across 5 folds (2 timelines each) - so the
        dataset is significantly smaller than the full dataset.

        Allows for rapid prototyping for training/ testing and evaluation.
        """

        # timeline_ids = [
        #     "Plushine_6c9677b482",  # Fold 0
        #     "bitchnumber24_30c9e21337",
        #     "gotja_c15449b99f",  # Fold 1
        #     "1man_factory_d1bd8bfe45",
        #     "davegri_621817aa9a",  # Fold 2
        #     "MithrandirTheIstari_6fad2f8e78",
        #     "Flaxmoore_8a1fccf030",  # Fold 3
        #     "Spartancupcakez_78125e9967",
        #     "lonelylosercreep_2d36711c90",  # Fold 4
        #     "CheesyBennington_77090d29cc",
        # ]

        timeline_ids = [
            # Fold 0
            "b47445df66",
            "5ad6c8c557",
            # Fold 1
            "0bbcbb4634",
            "a5dbc36a3c",
            # Fold 2
            "d7c53b9d2e",
            "f804849358",
            # Fold 3
            "f0a5e5ee1e",
            "de18045f25",
            # Fold 4
            "b6e248adc6",
            "8622775c33",
            # Test set (holdout)
            "5019e24631",
            "99da834134",
        ]
        df = self.timelines.copy()
        df = df[df["timeline_id"].isin(timeline_ids)]  # Filter to selected timelines

        if save:
            my_pickler("o", "prototype_reddit.pickle", folder="datasets")

        return df


class TalkLifeDataset:
    """
    Returns the annotated timelines and full user histories for the TalkLife
    dataset.
    """

    def __init__(
        self,
        include_embeddings=False,
        load_timelines_from_saved=True,
        save_processed_timelines=False,
    ):

        # r = "../" * 100  # Just to ensure you get to the root directory

        # Paths
        self.path_raw_annotations = global_parameters.path_talklife_raw_annotations
        self.path_raw_histories = global_parameters.path_talklife_raw_histories
        self.path_raw_individual_histories = (
            global_parameters.path_talklife_raw_individual_histories
        )
        self.path_processed_data_daily_interactions = (
            global_parameters.path_talklife_processed_data_daily_interactions
        )
        self.path_processed_annotations = (
            global_parameters.path_talklife_processed_annotations
        )

        # DataFrame containing annotated timelines, timestamps
        self.timelines = self.return_dataset_as_df(
            include_embeddings=include_embeddings,
            load_from_file=load_timelines_from_saved,
            save_data=save_processed_timelines,
        )
        self.prototype_timelines = self.return_prototype_dataset(
            save=save_processed_timelines
        )

        self.dataset_name = "talklife"

        self.observed_data_daily_interactions = my_pickler(
            "i", "observed_data_daily_interactions", folder="datasets"
        )

    def return_full_user_histories(
        self, only_annotated_users=True, aggregate_daily=True, load_from_pickle=True
    ):
        """
        Returns all the user histories, for all users in the TalkLife dataset.
        """

        if load_from_pickle:
            if aggregate_daily == False:
                histories = self.load_raw_talklife_data()
            else:
                with open(self.path_processed_data_daily_interactions, "rb") as f:
                    histories = pickle.load(f)
        else:
            if aggregate_daily:
                histories = self.create_all_history_dataframes()
            else:
                histories = self.load_raw_talklife_data()

        return histories

    def return_annotated_timelines(self, load_from_pickle=False):
        """
        Returns the 500 annotated timelines, selected by BOCPD.
        """

        if load_from_pickle:
            with open(self.path_processed_annotations, "rb") as f:
                annotations = pickle.load(f)
        else:
            annotations = self.return_dataset_as_df()

        return annotations

    def load_raw_talklife_data(self):
        """
        Returns the user histories from the raw pickle file.
        """
        path_histories = self.path_raw_histories
        with open(path_histories, "rb") as f:
            raw_histories = pickle.load(f)

        return raw_histories

    def create_all_history_dataframes(
        self, verbose=False, aggregate=True, only_annotated_users=True
    ):
        """
        Creates a dictionary of DataFrames for each user, where the
        id is the user id, and the dataframe has an index of
        consecutive datetimes for each day, and the columns are
        the number of activities recorded for the user per day.
        """

        # Load raw pickle file, then process it later
        all_users_raw_data = self.load_raw_talklife_data()

        if only_annotated_users:
            user_ids = list(all_users_raw_data.keys())
        processed_all_user_dataframes = {}
        for iter_num, i in enumerate(user_ids):
            if verbose:
                prop_computed = iter_num / len(user_ids)
                if iter_num % 10000:
                    print("{} %".format(100 * prop_computed))

            if aggregate:
                user_data = self.return_user_dataframe_of_features_per_day(
                    all_users_raw_data[i]
                )
            else:
                user_data = all_users_raw_data[i]

            processed_all_user_dataframes[i] = user_data

        # Apply post-processing
        if aggregate:
            processed_all_user_dataframes = self.post_processing(
                processed_all_user_dataframes
            )

        return processed_all_user_dataframes

    def create_unaggregated_all_user_history_dataframe(
        self,
        verbose=False,
        columns=[
            "datetime",
            "time_epoch_days",
            "content",
            "label_5",
            "label_3",
            "label_2",
            "user_id",
            "timeline_id",
            "postid",
            "fold",
            "sentence-bert",
        ],
        only_annotated_users=True,
        save_as_pickle=False,
    ):
        """
        Returns a dataframe which is the histories across all users. It is
        unaggregated, and thus does not contain daily activity information -
        rather the raw data with exact timestamps.
        """
        df = self.create_all_history_dataframes(
            self, aggregate=False, only_annotated_users=only_annotated_users
        )

        if save_as_pickle:
            my_pickler(
                "o",
                "talklife_raw_histories_for_annotated_users",
                df,
                folder="datasets",
            )

        return df

    def create_raw_history_dataframe_for_annotated_users(self, save=False, load=True):
        """
        Loads from Adam/Talia at datadrive, to create the unannotated histories
        dataframe, for just the 500 annotated users.

        Args:
            save (bool, optional): _description_. Defaults to False.
        """
        # Load from file, if desired
        if load:
            df_all_users = my_pickler(
                "i", "talklife_histories_df_for_annotated_users", folder="datasets"
            )
        # Otherwise, recreate it
        else:
            # Paths
            # r = "../" * 100  # Just to ensure you get to the root directory
            path_history_dataset = (
                global_parameters.path_talklife_raw_individual_histories
            )

            directory = os.fsencode(path_history_dataset)

            initialized = False
            # Loop over each user pickle in the directory
            for file in os.listdir(directory):
                filename = str(os.fsdecode(file))
                path_to_file = path_history_dataset + filename

                if filename.endswith(".p"):
                    with open(path_to_file, "rb") as f:
                        p = pickle.load(f)

                    user_id = int(filename.split(".")[0])
                    df_current_user = pd.DataFrame(p).T
                    df_current_user["user_id"] = user_id

                    # Sort each user by the date
                    df_current_user = df_current_user.sort_values(
                        by="date", ascending=True
                    )

                    if initialized:
                        df_all_users = pd.concat(
                            [df_all_users, df_current_user], axis=0
                        )
                    else:
                        df_all_users = df_current_user
                        initialized = True

                    continue
                else:
                    continue

            # Post-procesing

            # Sort the dataframe by user_id first, and then date
            df_all_users = df_all_users.sort_values(
                by=["user_id", "date"], ascending=True
            )
            df_all_users = df_all_users.reset_index().drop("index", axis=1)

            # Rename columns
            df_all_users = df_all_users.rename(
                columns={"question": "content", "date": "datetime", "post_id": "postid"}
            )

            # Get epoch time
            df_all_users["time_epoch_days"] = df_all_users["datetime"].apply(
                convert_datetime_to_epoch_time
            )

            # Order Columns (some columns removed, e.g. replies)
            df_all_users = df_all_users[
                ["datetime", "time_epoch_days", "content", "user_id", "postid"]
            ]

            if save:
                my_pickler(
                    "o",
                    "talklife_histories_df_for_annotated_users",
                    df_all_users,
                    folder="datasets",
                )

        return df_all_users

    def combine_histories_with_timelines(self, save=False, load=True):
        """
        Concatenates the history dataframe with the annotated timeline dataframe,
        deduplicating the same posts which are unannotated.

        Returns the full dataset, where columns such as labels for the unannotated
        history are left as NaNs.

        Shapes:
        * talklife:  424241 rows, 11 columns
        """

        # Load from file, if desired
        if load:
            df_full = my_pickler(
                "i",
                "df_talklife_history_and_timelines_combined_no_embeddings",
                folder="datasets",
            )
        else:
            # Get histories
            histories = self.create_raw_history_dataframe_for_annotated_users(
                save=False, load=True
            )

            # Get timelines
            timelines = self.timelines

            # Concatenate
            df_full = pd.concat([timelines, histories], axis=0)

            # Place annotated data at the top
            df_full = df_full.sort_values(["timeline_id"])

            # Remove duplicates, keeping the annotated data first
            df_full = df_full.drop_duplicates(["postid"], keep="first")

            # Post-processing
            # Sort the dataframe by user_id first, and then date
            df_full = df_full.sort_values(by=["user_id", "datetime"], ascending=True)
            df_full = df_full.reset_index().drop("index", axis=1)

            # Order Columns (some columns removed, e.g. replies)
            df_full = df_full[
                [
                    "datetime",
                    "time_epoch_days",
                    "content",
                    "label_5",
                    "label_3",
                    "label_2",
                    "user_id",
                    "timeline_id",
                    "postid",
                    "fold",
                ]
            ]

            # Convert user ids all to strings
            df_full["user_id"] = df_full["user_id"].apply(str)

            # Save as a pickle file, if desired
            if save:
                my_pickler(
                    "o",
                    "df_talklife_history_and_timelines_combined_no_embeddings",
                    df_full,
                    folder="datasets",
                )

        return df_full

    def return_combined_history_and_timelines_with_embeddings_df(
        self, load_from_file=True, save_data=False, embedding_type="sentence-bert"
    ):
        """
        Adds embeddings to the full dataset, which is the history concatenated
        with the annotated timelines.
        """

        # Combine timelines with histories, deduplicated
        df_full = self.combine_histories_with_timelines(save=False, load=True)

        # Create embeddings, and concatenate it to dataframe
        df_full_with_embeddings = self.add_embeddings_to_dataframe(
            df_full,
            embedding_type=embedding_type,
            load_from_file=False,
            project_name="talklife",
            save_data=save_data,
            custom_file_name="talklife_{}_full_history_embeddings".format(
                embedding_type
            ),
        )

        # Save as pickle, if desired
        if save_data:
            my_pickler(
                "o",
                "df_talklife_history_and_timelines_combined_{}".format(embedding_type),
                df_full_with_embeddings,
                folder="datasets",
            )

        return df_full_with_embeddings

    def create_user_activity_feature(self, user_raw_data):
        """
        Returns a sorted list of datetimes which record whenever
        an activity has been made by the user.

        Input:
        ======
        Raw data for that input user.
        """
        activities = []

        feature_names = [
            "posts",
            "comments_received",
            "comments_made",
            "follow_made",
            "reactions_made",
            "likes_made",
        ]

        for feature in feature_names:
            # Disregard comments received feature, as it is not a datetime, and is
            # not their activity
            if feature == "comments_received":
                continue
            else:
                activities.extend(user_raw_data[feature])

        # Sort the datetime activities in place
        activities.sort()

        return activities

    def post_processing(self, all_user_daily_data):
        """
        Set features to integers.
        Ensure that the datetime columns are padded.
        There should be no holes, and the datetime rows should be equal to the shape.
        """

        user_ids = list(all_user_daily_data.keys())

        # Post processing
        for user_id in user_ids:
            d = all_user_daily_data[user_id]

            # Convert features to integers (since they're counts)
            d = d.astype(int)

            # Pad the missing dates, as a sanity check
            max_date = max(d.index)
            min_date = min(d.index)
            idx = pd.date_range(min_date, max_date)
            d = d.reindex(idx, fill_value=0)

            # Update data
            all_user_daily_data[user_id] = d

        return all_user_daily_data

    def return_activity_per_day_for_user(self, user_raw_data):
        """
        Returns a pandas DataFrame time-series which records the
        number of activities per day for the input user.

        User is inputted as the full raw data for that user.
        """
        activity = self.create_user_activity_feature(user_raw_data)

        df = pd.DataFrame(activity, columns=["datetime"])
        df = df.set_index("datetime")

        # Set number of activity to 1, for each exact timestamp
        df["activity"] = 1

        # Get number of activity per day, for each day from start to end
        activity_per_day = df.resample("D").apply({"activity": "count"})

        return activity_per_day

    def return_independent_feature_per_day_for_user(
        self, user_raw_data, feature="posts"
    ):
        """
        Returns a pandas DataFrame time-series which records the
        number of activities per day for the input user.

        Inputs:
        =======
        user = all the raw data for that talklife user
        """

        if feature == "comments_received":
            df = pd.DataFrame(
                {
                    "datetime": user_raw_data["posts"],
                    "comments_received": user_raw_data["comments_received"],
                }
            )
            df = df.set_index("datetime")
            per_day = df.resample("D").apply({"comments_received": "sum"})

            return per_day

        # Error handling if no values for that feature (i.e. empty list)
        if len(user_raw_data[feature]) == 0:
            df = self.return_activity_per_day_for_user(user_raw_data).head(1)
            df = df.rename({"activity": feature}, axis="columns")
            df[feature] = 0

            return df

        # Otherwise, continue as normal
        else:
            df = pd.DataFrame(user_raw_data[feature], columns=["datetime"])
            df = df.set_index("datetime")

            # Set number of activity to 1, for each exact timestamp
            df[feature] = 1

            # Get number of activity per day, for each day from start to end
            per_day = df.resample("D").apply({feature: "count"})

            return per_day

    def return_user_dataframe_of_features_per_day(self, user):
        """
        Takes as input the raw data for a user in the dataset, and returns
        a dataframe containing the features per day where missing values are imputed
        with values of 0 (no activity). Returns activity per day which is the sum
        of all the other features, as well as other features such as posts per day.
        """

        feature_names = [
            "posts",
            "comments_received",
            "comments_made",
            "follow_made",
            "reactions_made",
            "likes_made",
        ]

        df_user = self.return_activity_per_day_for_user(
            user
        )  # Get activity scores per day

        # Get feature scores per day, for all features (except comments_received)
        for feature in feature_names:
            feature_per_day = self.return_independent_feature_per_day_for_user(
                user, feature
            )
            df_user = df_user.merge(
                feature_per_day, on="datetime", how="outer"
            )  # Merge together

        df_user = df_user.fillna(0)  # Fill missing values with 0 (no activity that day)

        return df_user

    """
    Below is code for loading the raw annotation data.
    """

    def get_timelines_for_fold(self, fold, return_timestamps=True):
        """
        Returns lists of different fields of all timelines IN the specified fold.
        Input:
            - fold (int): the fold we want to retrieve the timelines from
        Output (lists of posts):
            - timeline_ids: one tl_id per post
            - post_ids: the post_ids
            - texts: the text of each post
            - labels: the label of each post (5 possible labels)
        """
        FOLD_to_TIMELINE = self.create_fold_to_timeline()

        timelines_tsv = FOLD_to_TIMELINE[fold]
        timeline_ids, post_ids, texts, labels, timestamps = [], [], [], [], []
        for tsv in timelines_tsv:
            df = pd.read_csv(tsv, sep="\t")
            if (
                "374448_217" in tsv
            ):  # manually found (post 5723227 was not incorporated for some reason)
                df = pd.read_csv(tsv, sep="\t", quotechar="'")

            #         return df

            if return_timestamps:
                pstid, txt, lbl, ts = (
                    df.postid.values,
                    df.content.values,
                    df.label.values,
                    pd.to_datetime(df.date).values,
                )
            else:
                pstid, txt, lbl = df.postid.values, df.content.values, df.label.values

            for i in range(len(pstid)):
                timeline_ids.append(tsv.split("/")[-1][:-4])
                post_ids.append(pstid[i])
                texts.append(str(txt[i]))
                labels.append(lbl[i])
                timestamps.append(ts[i])

        if return_timestamps:
            return timeline_ids, post_ids, texts, np.array(labels), timestamps
        else:
            return timeline_ids, post_ids, texts, np.array(labels)

    def get_timelines_except_for_fold(self, fold, return_timestamps=True):
        """
        Returns lists of different fields of all timelines EXCEPT FOR the specified fold.
        Input:
            - fold (int): the fold we want to avoid retrieving the timelines from
        Output (lists of posts):
            - timeline_ids: one tl_id per post
            - post_ids: the post_ids
            - texts: the text of each post
            - labels: the label of each post (5 possible labels)
        """
        FOLD_to_TIMELINE = self.create_fold_to_timeline()
        timeline_ids, post_ids, texts, labels, timestamps = [], [], [], [], []
        for f in range(len(FOLD_to_TIMELINE)):
            if f != fold:
                if return_timestamps:
                    tlids, pstid, txt, lbl, ts = self.get_timelines_for_fold(
                        f, return_timestamps=return_timestamps
                    )
                else:
                    tlids, pstid, txt, lbl = self.get_timelines_for_fold(
                        f, return_timestamps=return_timestamps
                    )
                for i in range(len(pstid)):
                    timeline_ids.append(tlids[i])
                    post_ids.append(pstid[i])
                    texts.append(str(txt[i]))
                    labels.append(lbl[i])
                    if return_timestamps:
                        timestamps.append(ts[i])

        if return_timestamps:
            return timeline_ids, post_ids, texts, np.array(labels), ts
        else:
            return timeline_ids, post_ids, texts, np.array(labels)

    def get_three_labels(self, train_labels, test_labels):
        """
        Replaces our ground truth labels: IEP with IE & ISB with IS.
        """
        test_labels[test_labels == "ISB"] = "IS"
        test_labels[test_labels == "IEP"] = "IE"
        train_labels[train_labels == "ISB"] = "IS"
        train_labels[train_labels == "IEP"] = "IE"

        return train_labels, test_labels

    def get_2_and_3_labels_from_5(self, df):
        """
        Replaces our ground truth labels: IEP with IE & ISB with IS.
        """
        # Initialize labels
        raw_labels = df["label_5"]  # ISB, IEP, IE, IS, 0
        df["label_3"] = df["label_5"].copy()  # S, E, O

        # 3 Labels
        df.loc[(df["label_3"] == "IEP") | (df["label_3"] == "IE"), "label_3"] = "E"
        df.loc[(df["label_3"] == "ISB") | (df["label_3"] == "IS"), "label_3"] = "S"

        # Binary
        df["label_2"] = df["label_3"].copy()  # Binary (1, 0) if is GTMoC
        df.loc[(df["label_2"] == "E") | (df["label_3"] == "S"), "label_2"] = 1
        df.loc[(df["label_2"] == "0"), "label_2"] = 0

        return df

    def create_fold_to_timeline(self):
        """Initialising a list which contains all the timelines to be used in each fold"""
        NUM_folds = 5
        FOLD_to_TIMELINE = (
            []
        )  # list with NUM_folds sublists, each containing the paths to the corresponding fold's timelines
        for _fld in range(NUM_folds):
            _tmp_fldr = self.path_raw_annotations + str(_fld) + "/"
            FOLD_to_TIMELINE.append(
                [
                    _tmp_fldr + f
                    for f in listdir(_tmp_fldr)
                    if isfile(join(_tmp_fldr, f))
                ]
            )

        return FOLD_to_TIMELINE

    def get_user_id_from_timeline_id(self, timeline_ids):
        user_ids = timeline_ids.apply(lambda x: x.split("_")[0])

        return user_ids

    def post_process_annotations_df(self, df):
        """
        * Orders the columns by datetime -> timeline_id
        * Resets the index, based on chronology
        """

        # Sort by datetime, then timeline id
        df = df.sort_values(["datetime", "timeline_id"], ascending=[True, True])
        df = df.reset_index().drop("index", axis=1)

        # Order, and select columns
        df = df[
            [
                "datetime",
                "time_epoch_days",
                "content",
                "label_5",
                "label_3",
                "label_2",
                "user_id",
                "timeline_id",
                "postid",
                "fold",
            ]
        ]

        return df

    def add_embeddings_to_dataframe(
        self,
        df,
        embedding_type="sentence-bert",
        load_from_file=False,
        project_name="talklife",
        save_data=True,
        custom_file_name=None,
    ):
        """
        Adds additional column to timeline dataframe, which are the
        sentence-bert embeddings of the corresponding posts (content)
        """
        full_df = df.copy()

        # Set postid as index, and then join embeddings to it later
        df = df[["content", "postid"]]
        df = df.set_index("postid")

        if load_from_file:
            embedding = my_pickler(
                "i",
                "{}_{}_embeddings".format(project_name, embedding_type),
                folder="datasets",
            )

        # Otherwise, recreate these embeddings
        else:
            # Embed these posts
            print(
                "Creating {} embeddings for {} posts...".format(
                    embedding_type, df.shape[0]
                )
            )
            TextVectorizer = text_vectorizer.TextVectorizer()
            text = df["content"]
            embedding = text.apply(
                lambda x: TextVectorizer.vectorize(x, embedding_type=embedding_type)
            )

            print("Finished creating embeddings.")

            # Save just the embeddings to datadrive
            if save_data:
                if custom_file_name == None:
                    file_name = "{}_{}_embeddings".format(project_name, embedding_type)
                else:
                    file_name = custom_file_name
                my_pickler(
                    "o",
                    file_name,
                    embedding,
                    folder="datasets",
                )

        # Add column to dataframe
        df[embedding_type] = embedding

        # Join the embeddings on the postid
        df = df.reset_index()
        df = df[["postid", embedding_type]]
        full_df = full_df.merge(df, on="postid", how="left")

        return full_df

    def return_dataset_as_df(
        self,
        include_fold_column=True,
        include_processed_labels=True,
        include_user_id=True,
        post_process=True,
        include_embeddings=False,
        embedding_type="sentence-bert",
        load_from_file=True,
        save_data=False,
        include_epoch_time=True,
    ):
        """
        Place all data into a dataframe.
        """
        if load_from_file:
            print("Loading timeline dataframe:")
            # df = my_pickler("i", "", custom_path=self.path_processed_annotations)

            df = my_pickler("i", "talklife_timelines_all_embeddings", folder="datasets")

            return df
        else:

            N_FOLDS = 5
            initialized = False
            for i in range(N_FOLDS):
                d = self.get_timelines_for_fold(i, return_timestamps=True)
                df = pd.DataFrame(d).T.rename(
                    columns={
                        0: "timeline_id",
                        1: "postid",
                        2: "content",
                        3: "label_5",
                        4: "datetime",
                    }
                )
                if include_fold_column:
                    df["fold"] = i
                if not initialized:
                    full_dataset = df
                    initialized = True
                else:
                    full_dataset = pd.concat(
                        [full_dataset, df], axis=0, ignore_index=True
                    )

            if include_processed_labels:
                full_dataset = self.get_2_and_3_labels_from_5(full_dataset)

            if include_user_id:
                full_dataset["user_id"] = self.get_user_id_from_timeline_id(
                    full_dataset["timeline_id"]
                )

            if include_epoch_time:
                full_dataset["time_epoch_days"] = full_dataset["datetime"].apply(
                    convert_datetime_to_epoch_time
                )

            if post_process:
                full_dataset = self.post_process_annotations_df(full_dataset)

            if include_embeddings:
                full_dataset = self.add_embeddings_to_dataframe(
                    full_dataset,
                    embedding_type=embedding_type,
                    load_from_file=load_from_file,
                )

            # Save the embeddings to datadrive
            if save_data:
                my_pickler(
                    "o",
                    "{}_timelines_{}".format("talklife", embedding_type),
                    full_dataset,
                    folder="datasets",
                )

            return full_dataset

    def return_prototype_dataset(self, save=True):
        """
        Returns the smaller, prototyping version of the full dataset. This
        contains just 10 timelines, across 5 folds (2 timelines each) - so the
        dataset is significantly smaller than the full dataset.

        Allows for rapid prototyping for training/ testing and evaluation.
        """

        timeline_ids = [
            "35247_456",  # Fold 0
            "71157_326",
            "42034_232",  # Fold 1
            "35124_27",
            "2157_168",  # Fold 2
            "58945_413",
            "14953_9",  # Fold 3
            "15393_189",
            "34279_355",  # Fold 4
            "52067_225",
        ]
        df = self.timelines.copy()
        df = df[df["timeline_id"].isin(timeline_ids)]  # Filter to selected timelines

        if save:
            my_pickler("o", "prototype_talklife_timelines.pickle", folder="datasets")

        return df


def get_features_as_arrays(
    df, include_timestamps=True, embedding_type="sentence-bert", label="label_3"
):
    X_embed = get_embeddings_as_array_from_dataframe(df, embedding_type=embedding_type)

    if include_timestamps:
        X_timestamp = get_scalar_values_from_dataframe_as_array(df, column="datetime")
        X = [X_embed, X_timestamp]
    else:
        X = [X_embed]

    y = get_scalar_values_from_dataframe_as_array(df, column=label)

    return X, y  # X is  list of features, the 0th is embed, 1 is timestamps


def get_embeddings_as_array_from_dataframe(df, embedding_type="sentence-bert"):
    """
    (n_samples, embedding_size)
    """
    embedding_size = df[embedding_type].values[0].shape[1]
    X = np.stack(df[embedding_type].values).reshape(-1, embedding_size)

    return X


def get_scalar_values_from_dataframe_as_array(df, column="datetime"):
    """
    (n_samples, 1)
    """

    y = df[column].to_numpy().reshape(-1, 1)

    return y


def return_df_for_given_folds(df, folds=[0, 1, 2]):
    """
    Returns the dataframe of annotations for everything within the list
    of specified folds. To get a single fold, just use a list with a single
    fold number in it.
    """
    df_for_fold = df[df["fold"].isin(folds)]

    return df_for_fold


def return_df_except_for_folds(df, folds=[4]):
    """
    Returns a dataframe for everything except for the specified fold.
    """
    df_except_for_fold = df[~df["fold"].isin(folds)]

    return df_except_for_fold


def kfold_train_val_test(df, model, train_val_test_sizes=[3, 1, 1]):
    """
    Iterate the test-set. 4 Remaining folds used for training and validation.
    Each test set has it's own optimal model, from the validation set.
    The optimal model trained on 3 folds then makes predictions on the test set,
    where these predictions get saved - to make a final evaluation on the
    full list of test set predictions - using 5 different models.

    Returns 3 DataFrames, corresponding to train, val, test.
    """

    for test_folds_are in [[0], [1], [2], [3], [4]]:
        test_fold_is = test_folds_are[0]  # Test folds are only single folds

        df_test_fold = return_df_for_given_folds(df, test_fold_is)

    dict_return = {"train": train, "val": val, "test": test}

    return dict_return


def get_train_val_test_df(
    df, test_folds=[0], train_val_test_sizes=[3, 1, 1], train_val_sizes=[4, 1]
):
    """
    Returns the train, val, test set, for a specified test set.
    """

    # Check if there is a specific hold-out set. In this case only return train, val
    if "train_or_test" in df.columns:
        # Get train, val, test folds - based on test fold
        train_folds, val_folds = get_train_val_fold_numbers_from_val(
            val_folds=test_folds, train_val_sizes=train_val_sizes
        )

        test_df = df[df["train_or_test"] == "test"]
    elif -1 in df["fold"].values:
        # Get train, val, test folds - based on test fold
        train_folds, val_folds = get_train_val_fold_numbers_from_val(
            val_folds=test_folds, train_val_sizes=train_val_sizes
        )
        test_df = df[df["fold"] == -1]
    else:
        # Get train, val, test folds - based on test fold
        train_folds, val_folds, test_folds = get_train_val_fold_numbers_from_test(
            test_folds, train_val_test_sizes=train_val_test_sizes
        )

        test_df = get_df_for_specified_folds(df, folds=test_folds)

    # Split full dataframe into 3 DataFrames for: train, val, test
    train_df = get_df_for_specified_folds(df, folds=train_folds)
    val_df = get_df_for_specified_folds(df, folds=val_folds)

    # Return a dataframe containing both the training and validation sets. For final retraining.
    train_val_combined_folds = train_folds + val_folds
    train_val_combined_df = get_df_for_specified_folds(
        df, folds=train_val_combined_folds
    )

    return train_df, val_df, train_val_combined_df, test_df


def replace_fine_tuned_feature_with_relevant_test_fold_feature(features, test_fold):
    """
    Replace the fine-tuned feature with the relevant test fold feature.

    e.g. ['bert_focal_loss'] -> ['bert_focal_loss_test_fold=0']
    """
    features = features.copy()
    all_features = []
    for f in features:
        if check_if_feature_is_fine_tuned(f):
            f += "_test_fold={}".format(
                test_fold
            )  # Replace the fine-tuned feature with the relevant test fold feature
        all_features.append(f)

    return all_features


def get_train_val_test_dataloaders(
    df,
    test_folds=[0],
    train_val_test_sizes=[3, 1, 1],
    train_val_sizes=[
        4,
        1,
    ],  # In the scenario where a seperate hold-out test set is specified (e.g. reddit)
    features=["sentence-bert", "time_epoch_days"],
    target="label_3",
    batch_size=1,
    shuffle=False,
    num_workers=1,
    is_time_aware=False,
    assign_folds_to_nans=False,
    which_dataset="talklife",
    include_post_id=True
    # embedding_type='sentence-bert'
):
    """
    For a given DataFrame (e.g. the entire Reddit / TalkLife timelines dataset),
    will return the train/ val/ test dataloaders based on a specified test fold,
    features to train the model on, and target to predict.
    """
    if is_time_aware and "time_epoch_days" not in features:
        features.append("time_epoch_days")

    # Handle case where feature is fine-tuned on a given test_fold
    if which_dataset == "talklife":
        features = replace_fine_tuned_feature_with_relevant_test_fold_feature(
            features, test_folds[0]
        )
        
    # if include_post_id:
    #     features.append('post_id')

    # In the case where the fold has a NaN, assign the same fold of the user to
    # that
    if assign_folds_to_nans:
        df = assign_folds_to_combined_history_timelines_df(df)

    

    # Split input dataframe into 3 smaller dataframes, for train val test folds
    # - based on the input specified test fold and dataset sizes
    train_df, val_df, train_val_combined_df, test_df = get_train_val_test_df(
        df, test_folds=test_folds, train_val_test_sizes=train_val_test_sizes
    )

    # Create dataloaders for train/val/test dataframes
    train_dataloader = create_dataloader_from_df(
        train_df,
        features=features,
        target=target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        include_post_id=include_post_id
    )
    val_dataloader = create_dataloader_from_df(
        val_df,
        features=features,
        target=target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        include_post_id=include_post_id
    )
    train_val_dataloader = create_dataloader_from_df(
        train_val_combined_df,
        features=features,
        target=target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        include_post_id=include_post_id
    )
    test_dataloader = create_dataloader_from_df(
        test_df,
        features=features,
        target=target,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        include_post_id=include_post_id
    )

    return train_dataloader, val_dataloader, train_val_dataloader, test_dataloader


def kfold_model_on_df(df, model, as_torch=True):
    """
    Trains and evaluates a model, by iterating a test set and trains on the dataset.
    """


# def convert_x_y_to_torch_dataset(x, y):

#     dataset = torch.utils.data.Dataset(x, y)

#     return dataset
    

def convert_x_y_to_torch_dataset(x, y, post_id):
    """
    Converts an input tensor into a PyTorch dataset.
    """

    dataset = torch.utils.data.TensorDataset(x, y, post_id)

    return dataset


def convert_torch_dataset_to_dataloader(
    dataset, batch_size=1, shuffle=False, num_workers=1
):

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


def get_x_y_tensors_from_df(df, x=["sentence-bert", "time_epoch_days"], y="label_3"):

    x = get_array_features_from_df(df, features=x, as_torch=True)

    # Stack y as a one-hot Array, containing timeline-level labels
    # Check if input is already one-hot, as Tensors (from timeline-level aggregation)
    sample_y = df[y].values[0]
    if isinstance(sample_y, torch.Tensor):
        y = convert_series_of_arrays_to_stacked_array(df[y])
    else:
        y = convert_series_to_one_hot(df[y], as_torch=True)

    return x, y


def create_dataloader_from_df(
    df,
    features=["sentence-bert", "time_epoch_days"],
    target="label_3",
    batch_size=1,
    shuffle=False,
    num_workers=1,
    include_post_id=True
):
    """
    Takes an input DataFrame, and returns a dataloader for it, where the features
    are given by `x` and the targets are given by `y`.
    """
    # print("==== `create_dataloader_from_df` ====")

    # Ensure time feature is at end of the list (i.e last column of array)
    time_feature = "time_epoch_days"
    if time_feature in features:
        features.remove(time_feature)
        features.append(time_feature)

    # Check if features and targets are already (timeline-level) Tensors

    # Get Tensors X and y
    x, y = get_x_y_tensors_from_df(df, x=features, y=target)
    
    # PyTorch expects numerical values, not strings. Set post index to integer row index.
    post_index = convert_series_of_arrays_to_stacked_array(df['post_index'])
    
    # Convert Tensors to dataset
    dataset = convert_x_y_to_torch_dataset(x, y, post_index)

    # Convert Dataset to dataloader
    dataloader = convert_torch_dataset_to_dataloader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader

# def get_post_ids_from_df(df):
    
#     return post_id

def get_train_val_test_arrays_from_df(
    df,
    test_folds=[0],
    train_val_test_sizes=[3, 1, 1],
    embedding_type="sentence-bert",
    y_label="label_3",
    as_torch=True,
):
    """
    Returns all the arrays for train/val/test folds, for the specified iterated test set.
    This will then be used for training and evaluating the models, by iterating the test set.
    """
    train_df, val_df, test_df = get_train_val_test_df(
        df, test_folds=test_folds, train_val_test_sizes=train_val_test_sizes
    )

    x_embed_train, x_time_train, y_train = get_x_embed_x_time_y_from_df(
        train_df, embedding_type=embedding_type, y=y_label, as_torch=as_torch
    )
    x_embed_val, x_time_val, y_val = get_x_embed_x_time_y_from_df(
        val_df, embedding_type=embedding_type, y=y_label, as_torch=as_torch
    )
    x_embed_test, x_time_test, y_test = get_x_embed_x_time_y_from_df(
        test_df, embedding_type=embedding_type, y=y_label, as_torch=as_torch
    )

    return (
        x_embed_train,
        x_time_train,
        y_train,
        x_embed_val,
        x_time_val,
        y_val,
        x_embed_test,
        x_time_test,
        y_test,
    )


def get_train_val_fold_numbers_from_test(test_folds, train_val_test_sizes=[3, 1, 1]):
    """
    Returns the fold numbers for the remaining training and validation set, given
    an iterated test fold.
    """
    total_folds = sum(train_val_test_sizes)
    remaining_folds = list(range(total_folds))

    # Val fold is adjacent to test
    val_folds = test_folds[0] + train_val_test_sizes[1]
    if (
        val_folds < total_folds
    ):  # Set val_fold to 0, if we exceed the maximum fold number
        pass
    else:
        val_folds = 0

    # Remove the test and val folds from available folds
    remaining_folds.remove(test_folds[0])
    remaining_folds.remove(val_folds)

    val_folds = [val_folds]  # For consistency, keep each fold as a list.

    # Set training folds as the remaining folds left
    train_folds = remaining_folds

    return train_folds, val_folds, test_folds


def get_train_val_fold_numbers_from_val(val_folds, train_val_sizes=[4, 1]):
    """
    Returns the fold numbers for the remaining training and validation set, given
    an iterated test fold.
    """
    total_folds = sum(train_val_sizes)
    remaining_folds = list(range(total_folds))

    # Remove the val folds from available folds
    remaining_folds.remove(val_folds[0])

    # Set training folds as the remaining folds left
    train_folds = remaining_folds

    return train_folds, val_folds


def get_df_for_specified_folds(df, folds=[0]):
    """
    Filters the DataFrame for the specified folds.
    """

    df_for_specified_folds = df[df["fold"].isin(folds)]

    return df_for_specified_folds


def get_df_for_specified_timeline(df, timeline_id):
    """
    Filters the DataFrame for the specified timeline id.
    """
    df_output = df[df["timeline_id"] == timeline_id]

    return df_output


def get_array_features_from_df(
    df, features=["sentence-bert", "time_epoch_days"], as_torch=True
):
    """
    Embeddings first. Concatenates all the specified features together from an
    input DataFrame, and returns a single Tensor/ array as ouput. This will then
    be used to train the model.
    """
    embedding_type = features[0]  # Embedding first

    initialized = False
    for feature in features:
        if (
            feature == embedding_type or feature == "time_epoch_days"
        ):  # Treated differently, as array of arrays
            series_embeddings = df[feature]
            x = convert_series_of_arrays_to_stacked_array(series_embeddings)

            # For features with zero dimensions (e.g. time_epoch_days), add a dimension
            if len(x.shape) == 2:
                x = torch.unsqueeze(x, -1)
        else:
            df_flat_features = df[feature]  # Filter to specified features

            if isinstance(df_flat_features, torch.Tensor):
                x = convert_series_of_arrays_to_stacked_array(df_flat_features)
            else:
                x = df_flat_features.to_numpy().reshape(-1, 1)  # Convert to NumPy array

        # Concatenate all features together, with stacked embeddings first
        if initialized:

            # x_all = np.concatenate([x_all, x], axis=1)

            # Concatenating along last dimension
            x_all = np.concatenate([x_all, x], axis=-1)
        else:
            x_all = x
            initialized = True

    # Convert to PyTorch Tensor
    if as_torch:
        x_all = torch.Tensor(x_all)

    # print("x_all.shape: ", x_all.shape)

    return x_all


def convert_numpy_array_to_torch_tensor(X):
    X = torch.from_numpy(X)

    return X


def convert_tensor_to_data_loader(
    features_tensor, target_tensor, batch_size=1, shuffle=False
):

    dataset = torch.utils.data.TensorDataset(features_tensor, target_tensor)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def get_dataloader_from_df_for_folds(
    df,
    folds=[0],
    features=["sentence-bert"],
    target="label_3",
    batch_size=1,
    shuffle=False,
):
    """
    Returns a DataLoader object for a specified set of folds, features, and
    targets, and dataframe.
    """

    # Filter to specified folds
    df = get_df_for_specified_folds(df, folds=folds)

    x = get_array_features_from_df(df, features=features, as_torch=True)
    y = convert_series_to_one_hot(df[y], as_torch=as_torch)

    data_loader = convert_tensor_to_data_loader(x, y, batch_size=1, shuffle=False)

    return data_loader


def convert_series_of_arrays_to_stacked_array(s):
    """
    Used for converting the sentence-bert series, which consists of many
    arrays - into a single stacked array of shape (n_sample, 768). If
    timeline sensitive, then output is (n_timelines, 124, 768).
    """
    # Remove NaNs, in case of concatenation of unannotated history with timelines
    sample = s.dropna().values[0]

    # Convert series to an array
    if isinstance(sample, torch.Tensor):
        x = s
    else:
        x = s.to_numpy()

    # Return an embedding size of 1, if not sentence-bert
    if check_if_is_nlp_embedding(x[0]):
        embedding_size = x[0].shape[-1]  # 768 (sentence-bert)
    else:
        embedding_size = 1

    # Stack arrays/Tensors (each one represents a timeline to: (n_samples, 124, embedding_size)
    if isinstance(sample, torch.Tensor):
        stacked_array = torch.tensor(np.stack(s.values))
    else:
        stacked_array = np.array([x_s[0].reshape(embedding_size) for x_s in x])


    return stacked_array


def check_if_is_nlp_embedding(sample):
    """
    Checks if the feature is an NLP embedding.
    """
    embedding_shapes = [768]
    if sample.shape[-1] in embedding_shapes:
        is_nlp_embedding = True
    else:
        is_nlp_embedding = False

    return is_nlp_embedding


def convert_datetime_to_epoch_time(dt, units="days"):
    """
    Converts the datetime object to epoch time in miliseconds (time since
    Januray 1st, 1970 epoch).
    """
    epoch = python_datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    delta_seconds = delta.total_seconds()

    if units == "miliseconds":
        epoch_time = delta_seconds * 1000.0
    elif units == "seconds":
        epoch_time = delta_seconds
    elif units == "days":
        epoch_time = delta_seconds / 86400

    return epoch_time


def identify_embedding_feature(features=["sentence-bert"]):
    """
    Returns which features are embedding ones, and removes them from the list
    and returns 2 things a feature, and a list.
    """
    remaining_features = features

    embedding_list = [
        "sentence-bert",
        "bert_focal_loss",
        "bert_class_balanced_focal_loss",
    ]
    embedding_features_identified = []
    for f in remaining_features:
        if f in embedding_list:
            embedding_features_identified.append(f)
            remaining_features.remove(f)
    embedding_feature = embedding_features_identified[0]

    return embedding_feature, remaining_features


def convert_series_to_one_hot(s, as_torch=True, as_df=False, label="label_3"):
    """
    Convert 3_label to a one-hot encoded matrix.
    | S | E | 0 | is the ordering. The first index corresponds to switch,
    second to escalation, 3rd to no moc.
    """

    # Create a dataframe where, the column orders are S, E, O. Ensure they are
    # consistent
    if label == "label_3":
        df_one_hot = pd.DataFrame(columns=["S", "E", "0"])
        df_one_hot["S"] = return_binary_if_labels_in_series(s, target_labels=["S"])
        df_one_hot["E"] = return_binary_if_labels_in_series(s, target_labels=["E"])
        df_one_hot["0"] = return_binary_if_labels_in_series(s, target_labels=["0"])
    elif label == "label_5":
        df_one_hot = pd.DataFrame(columns=["IS", "ISB", "IE", "IEP", "0"])
        df_one_hot["IS"] = return_binary_if_labels_in_series(s, target_labels=["IS"])
        df_one_hot["ISB"] = return_binary_if_labels_in_series(s, target_labels=["ISB"])
        df_one_hot["IE"] = return_binary_if_labels_in_series(s, target_labels=["IE"])
        df_one_hot["IEP"] = return_binary_if_labels_in_series(s, target_labels=["IEP"])
        df_one_hot["0"] = return_binary_if_labels_in_series(s, target_labels=["0"])

    if as_df:
        return df_one_hot
    else:
        y = df_one_hot.to_numpy()

    if as_torch:
        y = torch.tensor(y)

    return y


def convert_one_hot_array_back_to_labels_df(y, reverse_one_hot=True):
    """
    Converts an input array of one-hot labels back to the actual labels. If
    one_hot=True, then will be returned as a labelled one-hot dataframe.
    Otherwise will be returned as a single series containing the actual labels.
    Ensures the order of the one-hot arrays is conserved.
    """

    if y.shape[1] == 3:
        column_names = ["S", "E", "0"]
    elif y.shape[1] == 5:
        column_names = ["IS", "ISB" "IE", "IEP" "0"]

    # Give label names to the array, and as dataframe.
    df = pd.DataFrame(y, columns=column_names)

    # Note, this returns a series
    if reverse_one_hot:
        df = df.idxmax(axis=1)

    return df


def return_binary_if_labels_in_series(s, target_labels=[]):
    """
    Used for creating a one-hot encoded array, for Switch / Escalation / O
    """
    binary_series = s.apply(lambda x: x in target_labels).astype(int)

    return binary_series


def get_x_embed_x_time_y_from_df(
    df, embedding_type="sentence-bert", y="label_3", as_torch=True
):
    """
    For a given DataFrame, will return 3 arrays (or PyTorch Tensors):
    x_embed - an array/ tensor for the embedding features
    x_time - the timestamp (as epoch time in days)
    y - a one-hot encoded array for the 'label_3' column. Has 3 columns.
    """

    x_embed = get_array_features_from_df(
        df, features=[embedding_type], as_torch=as_torch
    )
    x_time = get_array_features_from_df(
        df, features=["time_epoch_days"], as_torch=as_torch
    )
    y = convert_series_to_one_hot(df[y], as_torch=as_torch)

    return x_embed, x_time, y


def aggregate_over_loop(all_results, current_results):
    """
    Aggregate results over a for loop.
    """

    if all_results == None:
        all_results = current_results
    else:
        # Aggregate results
        all_results = torch.concat(all_results, current_results, axis=0)

    return all_results


def get_default_paths(which="models"):
    """
    Returns the default path, to a specified directory.
    """
    r = "../" * 100  # Just to ensure you get to the root directory

    if which == "models":
        path = r + "{}/ahills/models/".format(global_parameters.data_dir)
    elif which == "figures":
        path = r + "{}/ahills/figures/".format(global_parameters.data_dir)

    return path


def save_model(model, file_name, file_type="torch"):
    """
    Saves a PyTorch model.
    """
    model_save_path = get_default_paths(which="models")
    model_save_path += file_name

    if file_type == "torch":
        torch.save(model, model_save_path)  # Save full model, including topography

    elif file_type == "state_dict_torch":
        torch.save(model.state_dict(), model_save_path)  # Save best model so far


def load_model(model, file_name, file_type="torch"):
    """
    Loads a PyTorch model.
    """
    model_load_path = get_default_paths(which="models")
    model_load_path += file_name

    if file_type == "state_dict_torch":
        model.load_state_dict(torch.load(model_load_path))
    else:  # Load full model
        model = torch.load(model_load_path)

    return model


def return_fine_tuned_feature_as_list(feature, folds=[0, 1, 2, 3, 4]):
    """
    Embeddings created by fine-tuning a model on a given dataset. For example
    BERT focal loss. Here embeddings are created using a specific fold as a
    test set. We have multiple columns, where the test set is specified.

    In experiments, you should only use the column which has the same test fold
    as your current test fold.
    """

    is_fine_tuned = check_if_feature_is_fine_tuned(feature)
    if is_fine_tuned:
        fine_tuned_features = []
        for f in folds:
            f_feature = feature + "_test_fold={}".format(f)
            fine_tuned_features.append(f_feature)
    else:
        fine_tuned_features = [feature]

    return fine_tuned_features


def check_if_feature_is_fine_tuned(feature=""):
    """
    Retrns True if the feature is part of a fine-tuning process (e.g. bert focal loss)
    """
    fine_tuned_features = ["bert_focal_loss"]
    is_fine_tuned = False
    if feature in fine_tuned_features:
        is_fine_tuned = True

    return is_fine_tuned


def adjust_list_containing_features_to_handle_fine_tuned_embeddings(feature_list):
    """
    Adds all features needed, for the `aggregate_datafame_to_timeline_level` function.
    Will add fine-tuned features, as multiple features (containing the respective
    test fold).
    """

    all_features = []

    for f in feature_list:
        if check_if_feature_is_fine_tuned(f):
            fine_tuned_features = return_fine_tuned_feature_as_list(f)
            for ftf in fine_tuned_features:
                all_features.append(ftf)
        else:
            all_features.append(f)

    return all_features


def aggregate_dataframe_to_timeline_level(
    df,
    features=[
        "sentence-bert",
        "time_epoch_days",
        "tau",
        # "datetime",
        "label_5",
        "label_3",
        "label_2",
        # "user_id",
        # "postid",
        "post_index",
        "fold",
    ],
    datatype="torch",
    apply_padding=True,
    max_seq_length=124,
    padding_value=-123.0,
    embedding_type="sentence-bert",
    which_dataset="talklife",
):
    """
    Takes an input DataFrame, containing data for a given dataset (e.g. the
    whole TalkLife dataset), and returns a new dataframe indexed by the
    timeline id - and containing aggregate features for each post in that
    timeline as a single row (list of post embeddings, list of timestamps).

    Everything aggregated is sorted by the datetime of the post in the timeline.
    """
    # Add fine-tuned features, if there are any
    if which_dataset == "talklife":
        features = adjust_list_containing_features_to_handle_fine_tuned_embeddings(
            features
        )
    full_df = pd.DataFrame()
    for f in features:

        if f not in list(df.columns) + ["tau"]:
            print("WARNING: feature `{}` not in dataset. Skipping.".format(f))
        else:
            # One-hot-encode label features (as torch Tensors)
            if (f == "label_3") or (f == "label_5"):
                if datatype == "torch":
                    as_torch = True
                else:
                    as_torch = False
                aggregate_single_feature = df.groupby("timeline_id")[f].apply(
                    lambda x: convert_series_to_one_hot(
                        x, as_torch=as_torch, as_df=False, label=f
                    )
                )
            elif f == "fold":
                aggregate_single_feature = df.groupby("timeline_id")[f].apply(
                    lambda x: x.values[0]  # The first value
                )

            # Create time-delta matrix, tau, for each timeline if specified
            elif f == "tau":
                # First get all timestamps within each timeline
                timestamps = aggregate_single_feature = df.groupby("timeline_id")[
                    "time_epoch_days"
                ].apply(lambda x: torch.tensor(np.array(list(x))))
                # Then create self-pairwise deltas for all timestamps within timeline
                aggregate_single_feature = timestamps.apply(
                    lambda x: extract_time_delta_matrix(x)
                )
                
            elif f == 'postid':
                aggregate_single_feature = df.groupby("timeline_id")[f].apply(lambda x: np.array(list(x)))
                f = 'post_id'

            # Otherwise, aggregate the rest of the numeric features
            else:
                if datatype == "torch":
                    aggregate_single_feature = df.groupby("timeline_id")[f].apply(
                        lambda x: torch.tensor(np.array(list(x)))
                    )
                    if (f == embedding_type) or (
                        f[: len(embedding_type)] == embedding_type
                    ):  # Check if prefix is identical
                        aggregate_single_feature = aggregate_single_feature.apply(
                            lambda x: x.reshape(
                                -1, 768
                            )  # Reshape, so is (n_posts_in_timeline, 768)
                        )
                elif datatype == "numpy":
                    aggregate_single_feature = df.groupby("timeline_id")[f].apply(
                        lambda x: np.array(x)
                    )
                elif datatype == "list":
                    aggregate_single_feature = df.groupby("timeline_id")[f].apply(
                        lambda x: list(x)
                    )
            # Apply padding, if desired
            if apply_padding:
                if (f != "fold") and (f != "post_id"):
                    aggregate_single_feature = aggregate_single_feature.apply(
                        lambda x: pad_tensor(
                            x,
                            max_seq_length=max_seq_length,
                            padding_value=padding_value,
                        )
                    )
            aggregate_single_feature = pd.DataFrame(aggregate_single_feature)
            if len(full_df) > 0:
                full_df[f] = aggregate_single_feature
            else:
                full_df = aggregate_single_feature

    return full_df


def extract_time_delta_matrix(t, zero_future_posts=False):
    """
    Takes an input vector consisting of timestamps (epoch time, days), and
    returns a time-delta matrix which are pair-wise time-deltas where the i'th
    element in the input vector is compared to each other timestamp, j, in the
    vector, and the time-delta between those are measured for all i and j.

    Element (i, j) in the matrix measures the time-delta between the ith
    timestamp relative to the jth timestamp. It is positive where i >= j, and
    negative otherwise. If the optional parameter (zero_future_posts) is set to
    True, then all negative time-deltas (i.e. future j, relative to i) are set
    as zero.
    """
    # https://stackoverflow.com/questions/52780559/outer-sum-etc-in-pytorch
    tau = t.reshape(-1, 1) - t

    return tau


def pad_tensor(v, max_seq_length=124, padding_value=-123.0):
    """
    (zero) pads an input vector, to return a padded one of a desired length.

    Used for padding features in a timeline, to get a maximum length of 124 -
    which is the largest timeline sequence length. Helpful for training LSTMs.

    If batch dimension is true, this should be set in the first dimension.
    """
    # For multi-dimensional tensors (e.g. sentence-bert, label-5)
    if len(v.shape) > 1:
        n_samples = v.shape[0]
        embedding_dim = v.shape[-1]  # 768, for sentence-bert
        desired_size = (max_seq_length, embedding_dim)  # (124, 768) or (124, 3)

    # For 1D tensors (e.g. epoch time, label_2)
    else:
        n_samples = v.shape[0]
        desired_size = max_seq_length

    # Create desired tensor shape, with padded values
    v_padded = torch.ones(desired_size) * padding_value

    # Replace padded tensor with the values of the input tensor, where no
    # padding is needed
    if len(v.shape) > 1:
        v_padded[:n_samples, :embedding_dim] = v
    else:
        v_padded[:n_samples] = v  # For 1D tensors

    return v_padded


def pad_batched_tensor(v, max_seq_length=124, padding_value=-123.0):
    """
    Pads a batched tensor to the maximum sequence length, for all samples in the batch.
    """

    # Loop over each sample in the batch
    batch_size = v.shape[0]
    v_padded_list = []
    for b in range(0, batch_size):
        sample = v[b]

        # Replace the sample in the batch with the padded version
        sample_padded = pad_tensor(sample, max_seq_length, padding_value)

        # Store in a list, to be converted to a tensor later
        v_padded_list.append(sample_padded)

    # Convert to tensor
    v_padded = torch.stack(v_padded_list)

    return v_padded


def return_dims_with_no_padding(padded_v, padding_value=-123.0):
    """
    Returns a mask which is True for where there are no padding.
    """

    batch_first = False
    if len(padded_v.shape) == 3:  # Handle the case, where batch size is provided
        batch_first = True
        batch_size = padded_v.shape[0]
        # max_seq_length = padded_v.shape[1]
        embedding_size = padded_v.shape[2]

        dims_with_no_padding = padded_v[:, :, 0] != padding_value  # Check which rows

        # dims_with_no_padding = padded_v != padding_value  # Check which rows

        # print("dims_with_no_padding.sum()", dims_with_no_padding.sum())
        # print("dims_with_no_padding.shape", dims_with_no_padding.shape)
        # print("HERE")

    # Multi-dimensional Tensor
    elif len(padded_v.shape) > 1:
        dims_with_no_padding = padded_v[:, 0] != padding_value  # Check which rows

    # 1D tensor
    else:
        dims_with_no_padding = padded_v != padding_value

    return dims_with_no_padding


def remove_padding_from_tensor(padded_v, padding_value=-123.0, return_mask_only=False):
    """
    Removes padding from a tensor, and returns the unpadded tensor.
    """
    # print("=== `remove_padding_from_tensor` ===")

    # Select only the elements which have no padding
    batch_first = False
    if len(padded_v.shape) == 3:  # Handle the case, where batch size is provided
        batch_first = True
        batch_size = padded_v.shape[0]
        embedding_size = padded_v.shape[-1]
        dims_with_no_padding = (
            padded_v[:, :, 0] != padding_value
        )  # Check which rows have no padding, from embedding values
        # print("dims_with_no_padding.sum()", dims_with_no_padding.sum())

    # Multi-dimensional Tensor
    elif len(padded_v.shape) > 1:
        dims_with_no_padding = padded_v[:, 0] != padding_value  # Check which rows

    # 1D tensor
    else:
        dims_with_no_padding = padded_v != padding_value

    if return_mask_only:
        return dims_with_no_padding
    else:
        # Select the unpadded values
        unpadded_v = padded_v[dims_with_no_padding]

        # Keep the batch size and embedding size, while unrolling unpadded values
        if batch_first:
            unpadded_v = unpadded_v.reshape(batch_size, -1, embedding_size)

        return unpadded_v


def set_random_seeds(seed=0):
    """
    Used for reproducibility. Call this at the start of your experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assign_folds_to_combined_history_timelines_df(df):
    """
    The unannotated history contains NaNs in folds, but we require folds to
    do the train/val/test split. Thus here we assign fold numbers to the
    unannotated history, so they can be included within the dataloaders.

    Some users may have multiple folds, so in this case the fold for the
    unnannotated history will be a list containing the multiple possible fold
    numbers. For each possible fold, we should then include the whole
    history of that  user (i.e. we will be exposing different folds to
    each other - which may be problematic). This won't be a problem
    for the TalkLife dataset, since each user contains only 1 timeline at
    maximum.

    TODO:
    Note this may be a problem for Reddit, as we will lose the original fold
    for the annotated data, and will have multiple. Will need to adjust this
    later, before doing experiments on Reddit.
    """

    df = df.copy()

    # Assign folds to the unannotated data, so we can split the dataset into train/val/test
    series_folds = df.groupby("user_id")[
        "fold"
    ].unique()  # The unique fold numbers for this user
    series_folds_no_nans = series_folds.apply(
        lambda x: [int(y) for y in x if ~np.isnan(y)]
    )  # Remove NaNs, and convert fold to int
    series_folds = series_folds_no_nans.apply(
        lambda x: x[0] if len(x) == 1 else x
    )  # Get eleemnt of list, otherwise full lsit

    # Put fold into dataframe, using user ids
    df = df.set_index("user_id")
    df["fold"] = series_folds

    # Reset index, and reassert ordering
    df = df.reset_index()
    df = df.sort_values(by=["user_id", "datetime"], ascending=True)
    df = df.reset_index().drop("index", axis=1)

    return df
