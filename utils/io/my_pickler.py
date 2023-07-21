import os
import pickle
import sys
from glob import glob

sys.path.insert(
    0, "../../timeline_selection_and_evaluation/"
)  # Adds higher directory to python modules path
import global_parameters


def my_pickler(
    io,
    file_name,
    export_data=None,
    custom_path=None,
    verbose=True,
    file_format="pickle",
    folder=None,
    display_warnings=True,
):
    """
    Handy data loader/ exporter using Pickles. Could potentially be extended to other formats.

    Input:
    ======
    * io = Either "i" or "o". If "i", then loads a pickle (i = "input"). If "o", loads a pickle.
    * custom_path = Custom absolute path. string. Can optionally specify a path to store/ load the pickle to.
    * file_format = File extension
    * verbose = Boolean. Optionally prints progress.

    DEFAULT_PATH = './datasets/processed_data/talklife/{}/.format(file_format)'
    "datadrive/ahills/timeline_generation/datasets/datasets/processed_data/talklife/{}/".format(
    file_format
    )
    "datadrive/ahills/pickles/"
    """
    DEFAULT_PATH = global_parameters.path_default_pickle
    if folder != None:
        DEFAULT_PATH += folder + "/"
    # Can optionally specify "input" or "output". Extract just the first letter, in lower-case.
    io = io[0].lower()

    # Declare path to the file
    if custom_path == None:
        path = DEFAULT_PATH
        path += file_name + "." + file_format
    else:
        path = custom_path
        
    # Remove path names with two pickles.
    if path[-14:] == ".pickle.pickle":
        path = path[:-7]

    # print("path: ", path)

    # Verbose outputs
    if verbose:
        io_prefix = "load" if io == "i" else "sav"
        print("{}ing data at `{}`".format(io_prefix, remove_prefix_to_root(path)))

    # I/O
    # Save the pickle
    if io == "o":
        if type(export_data) == None:
            print("Error: must specify the `export_data=` parameter you want to save!")
            print("No data has been saved.")
        else:
            with open(path, "wb") as handle:
                pickle.dump(export_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print("Saved!")

    # Load the pickle
    elif io == "i":  # Load Data
        try:
            # print("path: ", path)
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except:
            if display_warnings:
                print(
                    "Unable to find pickle at path `{}`. Searching across other folders...".format(
                        remove_prefix_to_root(path)
                    )
                )
            path = search_across_other_folders_when_loading(
                path, display_warnings=display_warnings
            )
            if path != None:
                if display_warnings:
                    print(
                        "Warning: Loading file found at different location: \n`{}`".format(
                            remove_prefix_to_root(path)
                        )
                    )
                with open(path, "rb") as handle:
                    return pickle.load(handle)


def search_across_other_folders_when_loading(
    file_path, verbose=True, display_warnings=True
):
    """
    If the pickle to be loaded is not in the specified file path, possibly
    because the incorrect folder was provided - then this function will search
    the other folders in this directory to find the file. It will then pass the
    correct file path to that file - the first one it finds.
    """
    DEFAULT_PATH = global_parameters.path_default_pickle
    
    # print("1. file_path: ", file_path)

    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0]
    # print("2. file_name: ", file_name)
    

    locations = glob(DEFAULT_PATH + "/**/{}*".format(file_name), recursive=True)
    
    # print("3. locations: ", locations)
    
    # TODO: This doesn't work when there is [] in the file name, so replace temporarily
    if len(locations) > 0:
        file_path = locations[0]
        if verbose:
            if display_warnings:
                print("Found file at the {} locations.".format(len(locations)))
    else:
        # file_path = '.'.join(file_path.split(".")[:-1])
        print("Warning: No file found at the specified path: `{}`".format(file_path))
        print("locations: ", locations)
        print("file_name: ", file_name)
        print("----")
        
    

    return file_path


def remove_prefix_to_root(path):
    path = path.split("../")[100]  # The first 100 ../ are repeating, to get to root.
    path = "/" + path

    return path
