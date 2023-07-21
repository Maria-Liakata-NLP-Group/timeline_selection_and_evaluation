import os
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime

def create_directory_if_not_exists(path):
    # Check if directory exists for today
    if not os.path.exists(path):
        os.makedirs(path)

def save_figure(file_name=None, path='./figures', dpi=200, verbose=True):
    # Get time data, so we can store results neatly
    now = datetime.now()
    today = str(now.date())
    now = str(now)

    path += '/' + today
    
    # Create a folder for today, if it doesn't exist
    create_directory_if_not_exists(path)
    
    # Append information for now
#     path += '/' + file_name + ' (' + now + ')' +'.png'
    path += '/' + '(' + now + ') \n' + file_name  +'.png'
    
    # Create a folder with today's date
    plt.savefig(path, dpi=dpi)
    
    if verbose:
        print('Figure saved to \n {}'.format(path))