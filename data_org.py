
"""
Med Associates Subject File Parser - Janak Lab

This script takes raw Med Associates data in the form of subject files (.txt) and organizes them into two inter-related structures
for data analysis: 
    1. metadata_df --> pandas DataFrame, holds metadata (subject, start date, group, MSN, etc.)
       multi-indexed by (rat, date) so that we can filter through sessions easily
    2. grid --> dictionary that's also keyed by (rat, date), where each value is a 1D Numpy array that holds the session's
       behavioral data (Lick, PortEnter, etc)
       ** Numpy arrays were used here because of differing variable lengths within a single session

Example use: The two structures are linked by the same (rat, date) key, so metadata_df can be used to identify which sessions meet some 
condition, and grid can be used to pull the corresponding behavioral data for those sessions.       

AI Disclosure: Portions of this script were written with the help of Claude (Anthropic), mainly for determining the best data
structures + logic to go about data organization
"""


import os
import pandas as pd
import numpy as np

# Variable definition (from what was sent earlier)
varList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'X', 'Y']
varSave = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
varName = ['rewID_L', 'rewID_R', 'Indeces', 'Timers', 'PortEnter', 'PortExit', 'rewON_L', 'rewON_R', 'rewLP_L', 'rewLP_R', 'Lick', 'CS_start', 'TrialID', 'lat_rewPE', 'lat_rewLP', 'allLP_L', 'allLP_R', 'rewPE', 'list_ITI', 'list_trialID']

# Matching variables to readable names (in dictionary)
letter_to_name = {}
for letter, save, name in zip(varList, varSave, varName):
    if save == 1:
        letter_to_name[letter] = name

def read_subject_file(filepath):
    """
    Opens + reads through a subject (.txt; med associates) file. 
    
    Returns 2 dictionaries, one w/ metadata (flat dict of strings; 
    date, subject, experiment, MSN, etc.) and one with 1D Numpy arrays of the data (1 key per behavioral variable, values are 
    Numpy arrays w/ all numerical data stored in the variable).

    """

    metadata = {}
    data = {}
    current_letter = None  # letter we're currently reading
    current_data = []      # values for that letter
 
    file = open(filepath, 'r') # read file + store in lines
    lines = file.readlines()
    file.close()
 
    for line in lines:
        stripped = line.strip()
 
        # skip empty lines
        if stripped == "":
            continue
 
        # storing metadata
        if stripped.startswith('Start Date:'):
            metadata['start date'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Subject:'):
            metadata['subject'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Group:'):
            metadata['group'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Box:'):
            metadata['box'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('MSN:'):
            metadata['MSN'] = stripped.split(':', 1)[1].strip()
 
        # storing variables
        elif len(stripped) >= 2 and stripped[0].isupper() and stripped[1] == ':':
            # save whatever we were collecting for the previous letter
            if current_letter is not None and current_letter in letter_to_name:
                data[letter_to_name[current_letter]] = np.array(current_data)   # STORING DATA AS NUMPY ARRAY
 
            current_letter = stripped[0]   # store just the letter
            current_data = []              # reset for new variable
 
            # check if there's a value on the same line (deals w/ A and B)
            remaining = stripped[2:].strip()
            if remaining:
                # dealing w/ single value on the same line
                if current_letter in letter_to_name:
                    data[letter_to_name[current_letter]] = np.array([float(remaining)])
                current_letter = None  # reset curr letter because we don't want further lines to be counted
 

        # storing data rows inside each variable/letter
        elif current_letter is not None and stripped[0].isdigit() and ':' in stripped:
            # check if this is a var we want to save, then store data
            if current_letter in letter_to_name:
                # split off the row index label and store all values after it
                values_part = stripped.split(':', 1)[1].strip()
                for val in values_part.split():
                    current_data.append(float(val))
 
    # account for last variable
    if current_letter is not None and current_letter in letter_to_name:
        data[letter_to_name[current_letter]] = np.array(current_data) # STORING AS NUMPY ARRAY
 
    return metadata, data


def load_data(data_dir):

    """
    Loops through every .subject file in the folder and calls read_subject_file() on each one. Returns 2 data structures:

    metadata_df - pandas data frame, each session is one row, multi-indexed (indexed by (subject, start date))
    grid - dict w/ (rat, date) tuples as the keys, and 1D Numpy arrays of behavioral data as the values

        example: grid = {
                            ('1', '02/06/25'): {
                                'rewID_L':   array([15.0]),
                                'Indeces':   array([72.0, 1000.0, 250.0, ...]),
                                'Lick':      array([14.1, 14.3, 46.0, ...]),
                                'TrialID':   array([1.0, 2.0, 3.0, ...])
                            },
                            
                            ('1', '02/07/25'): {
                                'rewID_L':   array([12.0]),
                                'Indeces':   array([80.0, 950.0, ...]),
                                'Lick':      array([10.5, 22.0, ...]),
                                'TrialID':   array([1.0, 2.0, ...])
                            }
                        }

    Aligned "keys" for both structures to be the same (tuple of (rat, date)) so that filtering is easier
    """
 
    metadata_rows = [] # will become pandas DF; one row per session (a list of dictionaries essentially)
    grid = {}  # holds (rat, date): data; dict of Numpy arrays (so that we account for variable lengths)
 
    for filename in sorted(os.listdir(data_dir)):
        if '.Subject' in filename:
            filepath = data_dir + '/' + filename
            metadata, data = read_subject_file(filepath)

            rat  = metadata['subject']
            date = metadata['start date']

            metadata_rows.append(metadata) # add metadata row for this session
            grid[(rat, date)] = data # store numpy arrays under the same key
    
            print(f"Loaded: {filename}  |  rat = {rat}, date = {date}")

    metadata_df = pd.DataFrame(metadata_rows) # CONVERTING TO DATA FRAME
    metadata_df = metadata_df.set_index(['subject', 'start date']) # setting index as multi-index

    return metadata_df, grid


if __name__ == '__main__':

    data_dir = './MPCdata_MJ'

    metadata_df, grid = load_data(data_dir)

    # --- Random test ---
    import random

    keys = list(grid.keys())
    test_key = random.choice(keys)

    rat = test_key[0]
    date = test_key[1]

    print("Random key picked: rat =", rat, ", date =", date)

    value = grid[test_key]['PortEnter'][49]
    print("50th value of E (PortEnter):", value)