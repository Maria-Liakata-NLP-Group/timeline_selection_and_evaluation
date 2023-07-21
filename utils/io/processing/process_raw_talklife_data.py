import pandas as pd


# class Processer():
#     def __init__(self):
#         self.feature_names = []

def create_all_dataframes(all_users_raw_data, verbose=False):
    """
    Creates a dictionary of DataFrames for each user, where the 
    id is the user id, and the dataframe has an index of 
    consecutive datetimes for each day, and the columns are 
    the number of activities recorded for the user per day.
    """
    
    user_ids = list(all_users_raw_data.keys())
    processed_all_user_dataframes = {}
    for iter_num, i in enumerate(user_ids):
        if verbose:
            prop_computed = iter_num/len(user_ids)
            if iter_num % 10000:
                print("{} %".format(100 * prop_computed))
        
        processed_all_user_dataframes[i] = return_user_dataframe_of_features_per_day(all_users_raw_data[i])
        
    # Apply post-processing
    processed_all_user_dataframes = post_processing(processed_all_user_dataframes)
    
    
    return processed_all_user_dataframes


def create_user_activity_feature(user_raw_data):
    """
    Returns a sorted list of datetimes which record whenever 
    an activity has been made by the user.
    
    Input:
    ======
    Raw data for that input user.
    """
    activities = []

    feature_names = [
        'posts', 'comments_received', 'comments_made', 'follow_made',
        'reactions_made', 'likes_made'
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


def get_sample_data(all_users_raw_data, n_users=100, verbose=False):    
    user_ids = list(raw_data.keys())
    few_user_ids = user_ids[:n_users]
    few_users = {}
    for iter_num, i in enumerate(few_user_ids):
        few_users[i] = raw_data[i]
        
    sample_df = create_all_dataframes(few_users, verbose=verbose)
    
    
    return sample_df


def post_processing(all_user_daily_data):
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


def return_activity_per_day_for_user(user_raw_data):
    """
    Returns a pandas DataFrame time-series which records the 
    number of activities per day for the input user.
    
    User is inputted as the full raw data for that user. 
    """
    activity = create_user_activity_feature(user_raw_data)

    df = pd.DataFrame(activity, columns=['datetime'])
    df = df.set_index('datetime')

    # Set number of activity to 1, for each exact timestamp
    df['activity'] = 1

    # Get number of activity per day, for each day from start to end
    activity_per_day = df.resample('D').apply({'activity':'count'})

    return activity_per_day


def return_independent_feature_per_day_for_user(user_raw_data, feature='posts'):
    """
    Returns a pandas DataFrame time-series which records the 
    number of activities per day for the input user.
    
    Inputs:
    =======
    user = all the raw data for that talklife user
    """
    
    if feature == "comments_received":
        df = pd.DataFrame({'datetime': user_raw_data['posts'], 
                     'comments_received': user_raw_data['comments_received']})
        df = df.set_index('datetime')
        per_day = df.resample('D').apply({'comments_received':'sum'})
        
        return per_day
        
    
    # Error handling if no values for that feature (i.e. empty list)
    if len(user_raw_data[feature]) == 0:
        df = return_activity_per_day_for_user(user_raw_data).head(1)
        df = df.rename({'activity': feature}, axis='columns')
        df[feature] = 0
        
        return df
    
    # Otherwise, continue as normal
    else:
        df = pd.DataFrame(user_raw_data[feature], columns=['datetime'])
        df = df.set_index('datetime')

        # Set number of activity to 1, for each exact timestamp
        df[feature] = 1

        # Get number of activity per day, for each day from start to end
        per_day = df.resample('D').apply({feature:'count'})

        return per_day
    
    
def return_user_dataframe_of_features_per_day(user):
    """
    Takes as input the raw data for a user in the dataset, and returns
    a dataframe containing the features per day where missing values are imputed
    with values of 0 (no activity). Returns activity per day which is the sum
    of all the other features, as well as other features such as posts per day.
    """
    
    feature_names = [
        'posts', 'comments_received', 'comments_made', 'follow_made',
        'reactions_made', 'likes_made'
    ]
    
    df_user = return_activity_per_day_for_user(user)  # Get activity scores per day
    
    # Get feature scores per day, for all features (except comments_received)
    for feature in feature_names:
        feature_per_day = return_independent_feature_per_day_for_user(user, feature)
        df_user = df_user.merge(feature_per_day, on='datetime', how='outer')  # Merge together

    df_user = df_user.fillna(0)  # Fill missing values with 0 (no activity that day)
    
    
    return df_user