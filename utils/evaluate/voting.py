def vote_percent_length_of_time_series(timeline, timeline_id, low_proportion=0.1, high_proportion=-0.1):
    """
    +1 if positive and less than or equal to 10% of length of user time-series, 
    -1 if negative and greater or equal to 10% of length of user time-series, 
    0 otherwise
    """
    
    # Extract user id from timeline id 
    user_id = int(timeline_id[:timeline_id.index("_")])
    
    # Number of days in this user's time-series
    n_days = all_user_daily_data[user_id].shape[0]
    
    # Proprotion of days, based on number of days in user's time-sereis
    low = low_proportion * n_days
    high = high_proportion * n_days
    
    
    votes = {}
    for method in timeline.keys():
        score = timeline[method]
        
        # Negative (bad) centroid
        if score < 0:
            # Penalize if it's close
            if score >= high:
                vote = -1
            else:
                vote = 0 
        
        # Positive (good) centroid
        elif score >= 0:
            # Reward, if close
            if score <= low:
                vote = 1
            elif score >= higher_quantile:
                vote = -1
            else:
                vote = 0
             
        # Did not detect a CP
        else:
            vote = 0

        votes[method] = vote    
    df_votes = pd.DataFrame(pd.Series(votes)).T
    df_votes = df_votes.rename({0: timeline.name})
    
    
    return df_votes


all_votes = pd.DataFrame()
for index, row in method_scores.iterrows():
    if all_votes.shape[0] < 1:
        all_votes = quantile_vote(row, timeline_id=index, low_proportion=0.1, high_proportion=-0.1)
    else:
        all_votes = all_votes.append(quantile_vote(row, timeline_id=index, low_proportion=0.1, high_proportion=-0.1))
        
all_metric_votes['+1 if positive and less than or equal to 10% of length of user time-series, -1 if negative and greater or equal to 10% of length of user time-series, 0 otherwise'] = all_votes

