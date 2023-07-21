# import sys
# sys.path.append("../../") # Go to base utils path

import pandas as pd
import numpy as np

from ..io.my_pickler import my_pickler 
from .anchor_points.kernel_density_estimation.return_points_from_kde_anomaly_detection import return_points_from_kde_anomaly_detection
from .anchor_points.bocpd.poisson_gamma.extract_change_points import return_cps_from_bocpd_poisson_gamma_user_feature_data
from .anchor_points.keywords.return_points_from_keywords_method import return_cps_from_keywords_method


def return_anchor_points_for_method(method, user_data=None, user_id=None, process_into='dates', feature='posts'):
    """
    Inputs:
    =======
    method = String. The specified model to use, to create anchor points (e.g. change-points)
    user_feature_data = Pandas dataframe for the user, consisting of all their features.
    
    Outputs:
    ========
    An ordered, de-duplicated list of datetimes indicating the anchor points. Can specify to process it as date (days), 
    or keep as accurate timestamps.
    """
    # Extract just the relevant feature, as a Series
    user_feature_data = user_data[feature]
    
    # Identify the input method and hyper-parameters parameters 
    method_params = method.split('_')
    method = method_params[0]  # e.g. 'bocpd'
    method_params = method_params[1:]  # Remove the method name from the parameters 
    style = '_'.join(method_params)
    
    anchor_points = None
        
    # BOCPD
    if method == 'bocpd':
        
        # Specified distribution and priors, which we assume the data-generating process can be modelled by
        distribution = method_params[0]
        if distribution == 'pg':  # Poisson-Gamma model
            prior_hazard = float(method_params[1][1:])
            prior_alpha = float(method_params[2][1:])
            prior_beta = float(method_params[3][1:])
        
            # Extract change-points using the Poisson-Gamma model
            anchor_points = return_cps_from_bocpd_poisson_gamma_user_feature_data(user_feature_data,
                                                                  prior_hazard=prior_hazard,
                                                                  prior_alpha=prior_alpha,
                                                                  prior_beta=prior_beta,
                                                                  post_process='dates')
#         elif distribution == 'gaussian':
            
    
    # Kernel Density Estimation
    elif method == 'kde':
        anchor_points = return_points_from_kde_anomaly_detection(user_data, user_id, style)
           
    # Matrix Profile 
    elif method == 'mp':
        analysis_type = method_params[0]  # e.g. discords
        window_size = method_params[-1]
        anchor_points = return_points_from_matrix_profile(user_feature_data, analysis_type, window_size)
    
#     # Keywords Methods
#     elif method == 'keywords':
#         # Does not use any feature data. Will access this using the user id.
#         anchor_points = return_cps_from_keywords_method(style=style)[user_id]
           
    # Returns a CP for every day (over-generates) for that user
    elif method == 'every day':
        anchor_points = list(user_feature_data.index)
    elif method == 'no cps':
        anchor_points = []
    elif method == 'random single day':
        anchor_points = [np.random.choice(list(user_feature_data.index))]
    
    # Post-processing
    anchor_points = postprocess_anchor_points(user_feature_data, anchor_points, style=process_into)
            
        
    return anchor_points


def return_all_anchors_from_batch_method(method='keywords_all', user_ids=None, process_into='dates'):
    if method == 'keywords_three_categories':
        cps = return_cps_from_keywords_method(style=method)        
    elif method == 'keywords_all':
        cps = return_cps_from_keywords_method(style=method)
        
    return cps
        

def create_anchor_points(user_ids=[], 
                         methods=[], 
                         data_daily_interactions=None, 
                         only_observed_users=True,
                         process_into='dates',
                         feature='posts', verbose=False):
    """
    Takes as input a list of user ids, and a list of methods, and returns the anchor points 
    for those users and methods.
    """
    
    # Load data if not passed in.
    if data_daily_interactions == None:
        if only_observed_users:
            file_name = "observed_data_daily_interactions"
        else:
            file_name = "data_daily_interactions"
        data_daily_interactions = my_pickler('i', file_name, verbose=False)
    
    # Initialize empty dictionary, to store detected points across all users and methods
    anchor_points = {}
    
    # Run batch methods first, and pop them from the remaining methods
    batch_methods = ['keywords_three_categories',
                     'keywords_all']
    
    # Store the detected anchor points for all methods, for all users 
    for method in methods:
        if verbose:
            print(method, '...')
            
        if method in batch_methods:
            all_cps = return_all_anchors_from_batch_method(method=method, user_ids=user_ids)
            anchor_points[method] = all_cps
        else:
            anchor_points[method] = {}
            for u_id in user_ids:        
                user_data = data_daily_interactions[u_id]

                # Detect anchor-points for this time-series
                detected_points = return_anchor_points_for_method(method, 
                                                                  user_data=user_data, 
                                                                  user_id=u_id, 
                                                                  process_into=process_into, 
                                                                  feature=feature)

                # Store detected points
                anchor_points[method][u_id] = detected_points
        
    return anchor_points


def postprocess_anchor_points(user_feature_data, anchor_points=None, style='default'):
    # Check if any anchor points exist, and return empty list of anchor points
    if anchor_points == None:
        anchor_points = []
    if len(anchor_points) == 0:
        return anchor_points
    
    # Continue postprocessing, if points exist
    else:
        # Remove duplicate anchor points
        anchor_points = list(set(anchor_points))
        
        # Sort in ascending order
        anchor_points.sort()

        if style == 'default':
            return anchor_points
        
        # Check type of data
        data_type = type(anchor_points[0])
        
        if style == 'dates':
            if data_type == int or data_type == float:
                return convert_days_since_to_datetimes(user_feature_data, anchor_points)
            elif data_type == pd.Timestamp:
                anchor_points = convert_timestamp_to_datetime(anchor_points)
                return anchor_points
            else:
                return anchor_points
            
            
            
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

def convert_timestamp_to_datetime(anchor_points):
    processed = []
    for p in anchor_points:
        processed.append(p.to_pydatetime().date())
    
    return processed
    