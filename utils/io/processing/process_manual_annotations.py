import pandas as pd
import numpy as np
import pickle
from datetime import datetime


def return_annotatons_df(DATA_LOAD_PATH='./datasets/raw_data/talklife/', 
                         annotators=['anthony_hills', 'jenny_chim', 'jiayu_song']):
    """
    Accesses the raw data stored in the /annotations folder, and return a dataframe 
    containing the annotations (switch/escalation) for each TalkLife user observed by the annotators. 
    
    Inside the Annotations folder there should be folders for /Moods and /LikertMoods. 
    In Moods for instance you have a json file for each annotator. That json is a list of dictionaries, 
    each dictionary is a timeline with metadata information and the output of the annotations. 
    """
    
    # Load the post timestamp to ID mapper
    path = DATA_LOAD_PATH + 'post_to_ids.pickle'
    post_to_ids = np.load(path, allow_pickle=True)

    # Initialize empty dataframe. Data will be appended to this.
    df_annotations = pd.DataFrame(columns=['datetime', 'annotation', 'annotator', 'user_id', 'timeline_id'])

    # Collect data from all annotators
    for annotator in annotators:
        data_path_moods = DATA_LOAD_PATH + 'annotations/TalkLife/Moods/'
        file_path = data_path_moods + annotator + '.json'
        annotations = pd.read_json(file_path)

        # Collect data from all timelines annotated, for current annotator
        for timeline_idx in range(len(annotations['annotations'])):

            post_ids = annotations['annotations'][timeline_idx]['idPosts']
            annotated_dates = []
            annotated_local_dates = []
            annotated_labels = []

            for post_id in post_ids:
                dt = post_to_ids[int(post_id)]
                annotated_dates.append(dt[0])  # Corresponding date for the post

                # Local datetime, that was saved
                if dt[-1] != None:
                    tpl = dt[-1]
                    local_dt = datetime.strptime('-'.join(str(x) for x in tpl), '%Y-%m-%d-%H-%M-%S').strftime('%Y-%m-%d-%H-%M-%S')
                    local_dt = datetime.strptime(local_dt, '%Y-%m-%d-%H-%M-%S')
                    annotated_local_dates.append(local_dt)
                else:
                    annotated_local_dates.append(dt[0])  # No local date saved, just use original datetime

                # Corresponding label for this post Id
                annotation_label = annotations['annotations'][timeline_idx]['moodType_{}'.format(post_id)]
                annotated_labels.append(annotation_label)

            timeline_id = annotations['annotations'][timeline_idx]['annotate']
            user_id = int(timeline_id[:timeline_id.index("_")])

            dict_annotations = {}
            dict_annotations['datetime'] = annotated_dates
            dict_annotations['local_datetime'] = annotated_local_dates
            dict_annotations['annotation'] = annotated_labels
            dict_annotations['annotator'] = annotator
            dict_annotations['user_id'] = user_id
            dict_annotations['timeline_id'] = timeline_id

            # Append dataframe for each user.
            df_annotations = df_annotations.append(pd.DataFrame(dict_annotations))
            
    # Post-processing
    df_annotations = df_annotations.reset_index().drop('index', axis=1)
    one_hot_annotations = pd.get_dummies(df_annotations['annotation'])
    df_annotations = df_annotations.join(one_hot_annotations)

    # Either switch or escalation occured, label
    df_annotations['Switch/Escalation'] = df_annotations['Switch'] + df_annotations['Escalation']
    
    # Ad-hoc Post Processing. Some timelines were annotated annotated by all annotators. Skip these.    
    timelines_to_remove = ['604213_2', '604213_1', '414265_1']
    
    # Remove the identified timelines from the dataset
    df_annotations = df_annotations[~df_annotations['timeline_id'].isin(timelines_to_remove)]
    df_annotations = df_annotations.reset_index().drop(['index'], axis=1)
    
    
    return df_annotations