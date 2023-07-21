import pickle

def get_subreddits():
    '''The subreddits that are potentially linked with mental health'''
    reddits = ['depression', 'SuicideWatch', 'Anxiety', 'foreveralone', 
            'offmychest', 'socialanxiety', 'trueoffmychest', 'unsentletters',
            'rant', 'mentalhealth', 'traumatoolbox', 'bipolarreddit', 'BPD',
            'ptsd', 'psychoticreddit', 'EatingDisorders', 'StopSelfHarm', 
            'survivorsofabuse', 'rapecounseling', 'hardshipmates', 'panicparty']
    return reddits


def read_data(infile):
    '''Reads the data of a single user. Generates two lists of labels: one
    concerning the user posts and another concerning the user comments.'''
    user_data = pickle.load(open(infile, 'rb'))
    posts, comments = user_data['posts'], user_data['comments'] #each user dict has two lists: comments and posts
    
    posts_labels = get_labels(posts, reddits) #a list with pseudo-labels
    comments_labels = get_labels(comments, reddits) 
    
    #do something with the user_data and the posts_labels, comments_labels
    
    
def get_labels(list_of_dicts, reddits):
    labels = []
    for i in range(len(list_of_dicts)):
        try:
            sr = list_of_dicts[i]['subreddit']
        except KeyError:
            sr = ''
        lbl = 0
        if sr in reddits:
            lbl = 1
        labels.append(lbl)
    return labels