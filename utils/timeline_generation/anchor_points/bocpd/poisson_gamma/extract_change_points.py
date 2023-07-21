import numpy as np
import pandas as pd
from datetime import datetime, timedelta


from scipy.stats import norm
from scipy.special import logsumexp

# Imported code from Yannis' scripts
from .cp_probability_model import CpModel
from .detector import Detector
from .poisson_gamma_model import PGModel
from tqdm import tqdm


def poisson_bocpd(data, 
                  prior_hazard=100, 
                  prior_alpha=1, 
                  prior_beta=1,
                 visualize=False):
    """
    
    Inputs:
    =======
    Data is a list of equally spaced points with each time-step
    
    Outputs:
    =======
    Change-points, that are the indexes of the equally spaced points.
    """
    
    data = np.array(data)
    firstdateindex = 0      
            
    T, s1 = data.shape
    s2 = 1
    prior_means = 0 * np.ones(s1 * s2)
    pruning_threshold = None
    
    # Create hazard model object
    cp_model = CpModel(prior_hazard)
    
    # Change prior shape
    prior_alpha = prior_alpha * np.ones(s1 * s2)
    prior_beta = prior_beta * np.ones(s1 * s2)
    
    # Create model object(s)
    pg_model = PGModel(
        prior_alpha,
        prior_beta,
        prior_means,
        s1,
        s2,
        auto_prior_update=False,
    )
    
    # Create Detector object
    detector = Detector(data,
                        np.array([pg_model]),
                        np.array([1]),
                        cp_model,
                        s1,
                        s2,
                        T,
                        threshold=pruning_threshold)
    
    # Run detection algorithm
    for t in range(0, T):
        detector.next_run(data[t, :], t + 1)

    # Return MAP segmentation of CPS
    cps = detector.MAP_segmentation[-1][0]
    
    # Discard first CP, since it's initialized and arbitrary
    if len(cps) > 1:
        cps = cps[1:]
    else:
        cps = np.array([])
    
    
    return cps




def return_cps_from_poisson_gamma_bocpd_with_user_id(user_id, 
                                        data_daily_interactions, 
                                        feature='posts', 
                                        hazard=10, 
                                        alpha=1, 
                                        beta=1,
                                                    output_type='days'):
    """
    Takes as input the user id, and outputs the change-points for that user with the Poisson-Gamma BOCPD model.
    
    Outputs:
    ========
    A list of change-points in time-steps (days) or datetimes
    """
        
    # Get data
    poisson_input_data = np.reshape(list(data_daily_interactions[user_id][feature]), (-1,1))
    
    cps = poisson_bocpd(poisson_input_data, 
                      prior_hazard=hazard,  # Higher -> more CPs.
                      prior_alpha=alpha, 
                      prior_beta=beta)
    
    if output_type == 'days':
        return list(cps.astype(int))
    elif output_type == 'datetime':
        # Find the datetime of the change-points, from the days
        dt_cps = None
        
        
        
        
def preprocess_user_feature_data(user_feature_data):
    preprocessed_input_data = np.reshape(list(user_feature_data), (-1,1))

    return preprocessed_input_data
    
    
def postprocess_anchor_points(user_feature_data, anchor_points, style='default'):
    
    # Remove duplicate anchor points
    anchor_points = list(set(anchor_points))
    
    # Check if any anchor points exist
    if len(anchor_points) == 0:
        return anchor_points  # Return empty list of anchor points
    
    # Continue postprocessing, if points exist
    else:
        # Sort in ascending order
        anchor_points.sort()

        if style == 'default':
            return anchor_points
        
        # Check type of data
        data_type = type(anchor_points[0])
        
        # Convert datapoints into datetimes.
        if style == 'dates':
            return convert_days_since_to_datetimes(user_feature_data, anchor_points)

            
            
def convert_np_dt_to_dt(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
    
    return datetime.utcfromtimestamp(ts).date()

    
def convert_days_since_to_datetimes(user_feature_data, anchor_points):
    # Convert days to datetimes.
    dt_points = pd.Series(user_feature_data.index)[anchor_points].values
    dt_points = list(dt_points)

    # Convert NumPy DateTime object to a datetime Date object.
    converted_dts = []
    for np_dt in dt_points:
        converted = convert_np_dt_to_dt(np_dt)
        converted_dts.append(converted)

    # Update with dates
    return converted_dts
        
    
def return_cps_from_bocpd_poisson_gamma_user_feature_data(user_feature_data, prior_hazard, prior_alpha, prior_beta, post_process='dates'):
    """
    
    """
    # Pre-processing
    input_data = preprocess_user_feature_data(user_feature_data)
    
    # Extract change-points
    cps = poisson_bocpd(input_data,
                  prior_hazard=prior_hazard, 
                  prior_alpha=prior_alpha, 
                  prior_beta=prior_beta,
                  visualize=False)
    
    # Post-processing
    cps = postprocess_anchor_points(user_feature_data, cps, style=post_process)
        
    return cps