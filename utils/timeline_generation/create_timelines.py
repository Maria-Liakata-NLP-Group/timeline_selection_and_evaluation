import datetime

def assert_span_for_single_point(anchor_point, span_radius=7):
    
    left = anchor_point - datetime.timedelta(span_radius)
    right = anchor_point + datetime.timedelta(span_radius)
    span = [left, right]
    
    return span


def return_anchor_points_for_user(multiple_anchor_points, span_radius=7):
    
    multiple_spans = []
    for ap in multiple_anchor_points:
        multiple_spans.append(assert_span_for_single_point(ap, span_radius))
        
    return multiple_spans


def create_timelines_from_anchor_points(anchor_points, span_radius=7, merge_overlapping=True):
    """
    Main function
    """
    methods = anchor_points.keys()
    timelines = {}
    for method in methods:
        timelines[method] = {}
        users = anchor_points[method].keys()
        for user in users:
            user_spans = return_anchor_points_for_user(anchor_points[method][user], span_radius)
            
            # Merge overlapping spans, if desired
            if merge_overlapping:
                user_spans = merge_overlapping_spans(user_spans)
            
            timelines[method][user] = user_spans

    
    return timelines

def merge_overlapping_spans(temp_tuple):
    """
    Merges overlapping spans in a list.
    """
    
    # No need to merge anything, if there is less than or equal to 1 span.
    if len(temp_tuple) <= 1:
        return temp_tuple
    else:
        temp_tuple.sort(key=lambda interval: interval[0])
        merged = [temp_tuple[0]]
        for current in temp_tuple:
            previous = merged[-1]
            if current[0] <= previous[1]:
                previous[1] = max(previous[1], current[1])
            else:
                merged.append(current)

            
    return merged