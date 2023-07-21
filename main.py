from utils.full_experiments import reddit_timelines, talklife_timelines

"""
TalkLife
"""

TalklifeTimelinesExperiments = talklife_timelines.TalklifeTimelinesExperiments()
talklife_results = TalklifeTimelinesExperiments.full_experiment()

"""
Reddit
"""
RedditTimelinesExperiments = reddit_timelines.RedditTimelinesExperiments()
reddit_results = RedditTimelinesExperiments.full_experiment()

# Verbose output
print()
print('tau = 5')
print("TalkLife results:", talklife_results)
print("Reddit results:", reddit_results)
print()
print("Please note that the subplots have been saved in the current folder.")