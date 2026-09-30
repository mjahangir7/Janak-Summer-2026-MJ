
"""
Med Associates Subject File Parser + Ethanol Analysis - Janak Lab

This script takes raw Med Associates data in the form of subject files (.txt) and organizes them into two inter-related structures
for data analysis: 
    1. metadata_df --> pandas DataFrame, holds metadata (subject, start date, group, MSN, etc.)
       multi-indexed by (rat, date) so that we can filter through sessions easily
    2. grid --> dictionary that's also keyed by (rat, date), where each value is a 1D Numpy array that holds the session's
       behavioral data (Lick, PortEnter, etc)
       ** Numpy arrays were used here because of differing variable lengths within a single session

Additionally, the script uses the data to generate two different latency (from CS --> operant response) plots, one with raw data
and one with a log transformation to better visualize trends and differences between etoh and h2o.

AI Disclosure: Portions of this script were written with the help of Claude (Anthropic), mainly for determining the best data
structures + logic to go about data organization, as well as formatting of figures and code to make more concise
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# font for all plots
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11

# --- variable mapping setup ---
# Med-PC stores data under single letters (A, B, C...). these three lists line up so we can
# translate each letter into a readable name. varSave = 1 means we want to keep that variable,
# 0 means skip it
varList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'X', 'Y']
varSave = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
varName = ['rewID_L', 'rewID_R', 'Indeces', 'Timers', 'PortEnter', 'PortExit', 'rewON_L', 'rewON_R', 'rewLP_L', 'rewLP_R', 'Lick', 'CS_start', 'TrialID', 'lat_rewPE', 'lat_rewLP', 'allLP_L', 'allLP_R', 'rewPE', 'list_ITI', 'list_trialID']

# build dictionary that maps each letter to its readable name, but only for the ones we want
# e.g. {'A': 'rewID_L', 'B': 'rewID_R', 'E': 'PortEnter', ...}
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

    # open the file and read all lines into a list
    file = open(filepath, 'r')
    lines = file.readlines()
    file.close()

    # go through every line in the file
    for line in lines:
        stripped = line.strip()

        # skip empty lines
        if stripped == "":
            continue

        # --- grab metadata from the header lines (Subject, Start Date, etc.) ---
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

        # --- detect a new variable section (line starts with a capital letter then colon, like "E:") ---
        elif len(stripped) >= 2 and stripped[0].isupper() and stripped[1] == ':':
            # before moving to the new letter, save whatever data we collected for the previous one
            if current_letter is not None and current_letter in letter_to_name:
                data[letter_to_name[current_letter]] = np.array(current_data)

            current_letter = stripped[0] # store just the letter (e.g. 'E')
            current_data = []  # start fresh for new variable

            # some variables (like A and B) have their value on the same line as the letter
            remaining = stripped[2:].strip()
            if remaining:
                if current_letter in letter_to_name:
                    data[letter_to_name[current_letter]] = np.array([float(remaining)])
                current_letter = None  # done with this variable, don't collect more lines

        # --- read the numbered data rows that belong to the current variable ---
        # looks like "0:  1.234  5.678  9.012" (row index, then values)
        elif current_letter is not None and stripped[0].isdigit() and ':' in stripped:
            if current_letter in letter_to_name:
                # grab just the numbers after the colon
                values_part = stripped.split(':', 1)[1].strip()
                for val in values_part.split():
                    current_data.append(float(val))

    # save very last variable (the loop ends before it gets saved otherwise)
    if current_letter is not None and current_letter in letter_to_name:
        data[letter_to_name[current_letter]] = np.array(current_data)

    return metadata, data


def load_data(data_dir):

    """
    Loops through every .subject file in the folder and calls read_subject_file() on each one. Returns 2 things:

    metadata_df - pandas data frame, each session is one row, multi-indexed (indexed by (rat, start date))
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
 
    metadata_rows = []  # will become a pandas DF; each element is one session's metadata dict
    grid = {}  # will hold all the behavioral data, keyed by (rat, date)

    # loop through every file in the folder, only process .Subject files
    for filename in sorted(os.listdir(data_dir)):
        if '.Subject' in filename:
            filepath = data_dir + '/' + filename
            metadata, data = read_subject_file(filepath)

            rat  = metadata['subject']
            date = metadata['start date']

            metadata_rows.append(metadata)   # add this session's metadata to the list
            grid[(rat, date)] = data         # store this session's data arrays under (rat, date)

            print(f"Loaded: {filename}  |  rat = {rat}, date = {date}")

    # convert the list of metadata dicts into a pandas DataFrame, indexed by (subject, date)
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df = metadata_df.set_index(['subject', 'start date'])

    return metadata_df, grid




def plot_cs_to_response_latency(grid, etoh_id):
    """
    Histogram: Plots the latency (time difference) from CS onset --> operant response separately for surprise/forced and choice trials

    Surprise trials: operant response = first port entry after CS onset
    Choice trials: operant response = whichever lever press (L or R) occurred in that trial's window

    Two subplots/histograms:
        Left: Surprise trials — water vs ethanol latencies
        Right: Choice trials — water vs ethanol latencies

    x-axis: latency bins (in seconds)
    y-axis: # of trials/observations

    """

    # make lists for latencies (4 lists)
    
    etoh_surprise_lat = []
    water_surprise_lat = []
    etoh_choice_lat = []
    water_choice_lat = []


    # loop through each session
    
    for (rat, date), data in grid.items():

        # set important variables
        ids = data['TrialID']
        cues = data['CS_start']
        port_enter = data['PortEnter']
        lp_L = data['rewLP_L']   # left lever press timestamps
        lp_R = data['rewLP_R']   # right lever press timestamps
        n = len(ids) # number of trials to know how many times to loop


        # pull trial id (water vs etoh)
        if data['rewID_L'][0] == etoh_id:
            eth_trial = 1   # ethanol on left, trial 1 fires left
            wat_trial = 2   # water on right, trial 2 fires right
        else:
            eth_trial = 2
            wat_trial = 1

        # go through each trial
        for i in range(n):
            t = ids[i] #trial type

            # get time window (time stamps for beginning + end) for this trial
            cue = cues[i]
            nxt = cues[i + 1] if i < n - 1 else np.inf # inf is for last trial


        # SURPRISE TRIALS (id 1 or 2)
            if t == 1 or t == 2:

                # find first port entry after CS onset within this trial's window
                in_trial = (port_enter >= cue) & (port_enter < nxt)
                if not np.any(in_trial):
                    continue

                # first port entry after cue
                first_entry = port_enter[in_trial][0]
                latency = first_entry - cue

                # store under correct reward type
                if t == eth_trial:
                    etoh_surprise_lat.append(latency)
                else:
                    water_surprise_lat.append(latency)

        # CHOICE TRIALS (id 3)
            # elif t == 3:
            elif t == 3:

                # check which lever was pressed within this trial's window
                lp_L_in = lp_L[(lp_L >= cue) & (lp_L < nxt)]
                lp_R_in = lp_R[(lp_R >= cue) & (lp_R < nxt)]

                # figure out which one fired, calculate latency
                if len(lp_L_in) > 0 and len(lp_R_in) > 0:
                    # both fired, use whichever came first **NOT SURE IF NECESSARY
                    press_time = min(lp_L_in[0], lp_R_in[0]) #min val = quicker press
                    pressed_side = 'L' if lp_L_in[0] < lp_R_in[0] else 'R'
                elif len(lp_L_in) > 0:
                    press_time = lp_L_in[0]
                    pressed_side = 'L'
                elif len(lp_R_in) > 0:
                    press_time = lp_R_in[0]
                    pressed_side = 'R'
                else:
                    continue  # no lever press, skip trial

                latency = press_time - cue

                # figure out which reward the rat chose based on which side it pressed 
                # + whether that side is etoh/water for this rat (kind of confusing, might rewrite)
                if pressed_side == 'L':
                    chose_eth = (data['rewID_L'][0] == etoh_id)
                else:
                    chose_eth = (data['rewID_L'][0] != etoh_id)

                if chose_eth:
                    etoh_choice_lat.append(latency)
                else:
                    water_choice_lat.append(latency)

        
    # plot + formatting

    # histogram binning
    custom_bins_ax1 = np.linspace(0, 17, 80)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

    ax1.hist(etoh_surprise_lat, bins=custom_bins_ax1, alpha=0.5, label="EtOH")
    ax1.hist(water_surprise_lat, bins=custom_bins_ax1, alpha=0.5, label="Water")
    ax1.set_title("Surprise Trials: CS → Port Entry")
    ax1.set_xlabel('Latency (s)')
    ax1.set_ylabel('Number of trials')
    ax1.set_xlim(0, 17)
    ax1.legend()

    custom_bins_ax2 = np.linspace(0, 100, 160)
    ax2.hist(etoh_choice_lat, bins=custom_bins_ax2, alpha=0.5, label="EtOH")
    ax2.hist(water_choice_lat, bins=custom_bins_ax2, alpha=0.5, label="Water")
    ax2.set_title("Choice Trials: CS → Lever Press")
    ax2.set_xlabel('Latency (s)')
    ax2.set_ylabel('Number of trials')
    ax2.set_xlim(0, 100)
    ax2.legend()

    plt.show()




def plot_cs_to_log_of_response_latency(grid, etoh_id):
    """
    Histogram: Plots LOG (base 10) TRANSFORMATION of the latency (time difference) from CS onset --> operant response 
    separately for surprise/forced and choice trials

    Surprise trials: operant response = first port entry after CS onset
    Choice trials: operant response = whichever lever press (L or R) occurred in that trial's window

    Two subplots/histograms:
        Left: Surprise trials — log of water vs ethanol latencies
        Right: Choice trials — log of water vs ethanol latencies

    x-axis: latency bins (in seconds)
    y-axis: # of trials/observations

    """


    # make lists for latencies (4 lists)
    
    etoh_surprise_lat = []
    water_surprise_lat = []
    etoh_choice_lat = []
    water_choice_lat = []


    # loop through each session
    
    for (rat, date), data in grid.items():

        # set important variables
        ids = data['TrialID']
        cues = data['CS_start']
        port_enter = data['PortEnter']
        lp_L = data['rewLP_L']   # left lever press timestamps
        lp_R = data['rewLP_R']   # right lever press timestamps
        n = len(ids) # number of trials to know how many times to loop


        # pull trial id (water vs etoh)
        if data['rewID_L'][0] == etoh_id:
            eth_trial = 1   # ethanol on left, trial 1 fires left
            wat_trial = 2   # water on right, trial 2 fires right
        else:
            eth_trial = 2
            wat_trial = 1

        # go through each trial
        for i in range(n):
            t = ids[i] #trial type

            # get time window (time stamps for beginning + end) for this trial
            cue = cues[i]
            nxt = cues[i + 1] if i < n - 1 else np.inf # inf is for last trial


        # SURPRISE TRIALS (id 1 or 2)
            if t == 1 or t == 2:

                # find first port entry after CS onset within this trial's window
                in_trial = (port_enter >= cue) & (port_enter < nxt)
                if not np.any(in_trial):
                    continue

                # first port entry after cue
                first_entry = port_enter[in_trial][0]
                latency = first_entry - cue
                if latency == 0:
                    latency = 0.01

                # store under correct reward type
                if t == eth_trial:
                    etoh_surprise_lat.append(math.log10(latency))
                else:
                    water_surprise_lat.append(math.log10(latency))

        # CHOICE TRIALS (id 3)
            # elif t == 3:
            elif t == 3:

                # check which lever was pressed within this trial's window
                lp_L_in = lp_L[(lp_L >= cue) & (lp_L < nxt)]
                lp_R_in = lp_R[(lp_R >= cue) & (lp_R < nxt)]

                # figure out which one fired, calculate latency
                if len(lp_L_in) > 0 and len(lp_R_in) > 0:
                    # both fired, use whichever came first **NOT SURE IF NECESSARY
                    press_time = min(lp_L_in[0], lp_R_in[0]) #min val = quicker press
                    pressed_side = 'L' if lp_L_in[0] < lp_R_in[0] else 'R'
                elif len(lp_L_in) > 0:
                    press_time = lp_L_in[0]
                    pressed_side = 'L'
                elif len(lp_R_in) > 0:
                    press_time = lp_R_in[0]
                    pressed_side = 'R'
                else:
                    continue  # no lever press, skip trial

                latency = press_time - cue
                if latency == 0:
                    latency = 0.01

                # figure out which reward the rat chose based on which side it pressed 
                # + whether that side is etoh/water for this rat (kind of confusing, might rewrite)
                if pressed_side == 'L':
                    chose_eth = (data['rewID_L'][0] == etoh_id)
                else:
                    chose_eth = (data['rewID_L'][0] != etoh_id)

                if chose_eth:
                    etoh_choice_lat.append(math.log10(latency))
                else:
                    water_choice_lat.append(math.log10(latency))

    
    # plot + formatting

    # histogram binning

    #custom_bins_ax1 = np.linspace(0, 17, 80)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

    etoh_color = (0.18, 0.62, 0.18)
    water_color = (0.12, 0.47, 0.71)

    ax1.hist(etoh_surprise_lat, color=etoh_color, bins=80, alpha=0.5, label="EtOH")
    ax1.hist(water_surprise_lat, color=water_color, bins=80, alpha=0.5, label="Water")
    ax1.set_title("Surprise Trials: CS → Port Entry")
    ax1.set_xlabel('Log of Latency (s)')
    ax1.set_ylabel('Number of trials')
    #ax1.set_xlim(0, 17)
    ax1.legend()

    #custom_bins_ax2 = np.linspace(0, 100, 160)
    ax2.hist(etoh_choice_lat, color=etoh_color, bins=160, alpha=0.5, label="EtOH")
    ax2.hist(water_choice_lat, color=water_color, bins=160, alpha=0.5, label="Water")
    ax2.set_title("Choice Trials: CS → Lever Press")
    ax2.set_xlabel('Log of Latency (s)')
    ax2.set_ylabel('Number of trials')
    #ax2.set_xlim(0, 100)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.suptitle('CS Onset to Operant Response Latency', fontsize=14, fontweight='bold', x=0.52, y=0.93)
    plt.show()






if __name__ == '__main__':

    # --- latency plot ---
    data_dir = './Lotus_phase4_day2-end'
    metadata_df, grid = load_data(data_dir)
    plot_cs_to_response_latency(grid, 7)

    # --- latency plot (log transformation) ---
    data_dir = './Lotus_phase4_day2-end'
    metadata_df, grid = load_data(data_dir)
    plot_cs_to_log_of_response_latency(grid, 7)
