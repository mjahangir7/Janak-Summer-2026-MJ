
"""
Med Associates Subject File Parser - Janak Lab

This script takes raw Med Associates data in the form of subject files (.txt) and organizes them into two inter-related structures
for data analysis: 
    1. metadata_df --> pandas DataFrame, holds metadata (subject, start date, group, MSN, etc.)
       multi-indexed by (rat, date) so that we can filter through sessions easily
    2. grid --> dictionary that's also keyed by (rat, date), where each value is a 1D Numpy array that holds the session's
       behavioral data (Lick, PortEnter, etc)
       ** Numpy arrays were used here because of differing variable lengths within a single session

AI Disclosure: Portions of this script were written with the help of Claude (Anthropic), mainly for determining the best data
structures + logic to go about data organization, as well as formatting of figures and code to make more concise
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def plot_lick_rate_by_quarter(grid, sucrose_id, win_start=-2, win_end=13, bin_size=0.1):
    """
    Plots lick rate (licks/sec) using reward onset/time from reward for forced trials, split by session quarter.

    Two subplots: sucrose forced (left, orange shades) and water forced (right, blue shades).
    Each session's forced trials are split into 4 equal quarters. Lick times are centered on
    reward delivery, binned into a histogram, and converted to rate. Averaging is done first
    within each quarter per rat, then across rats.

    sucrose_id: the rewID value that corresponds to sucrose (e.g. 15 or 7).
                Used to figure out which side is sucrose for each rat,
                since the sides are counterbalanced.
    """

    # --- set up the histogram bins for the x-axis (time from reward) ---
    # edges = the bin boundaries, centers = the midpoint of each bin (used for plotting)
    edges = np.arange(win_start, win_end + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2

    # store each rat's lick rate arrays, organized by quarter
    # structure: suc_data[rat][quarter] = list of lick rate arrays, one per trial
    suc_data = {}
    wat_data = {}

    # --- loop through every session and process each forced trial ---
    for (rat, date), data in grid.items():

        # pull out the arrays we need from this session
        licks = data['Lick']  # all lick timestamps
        ids = data['TrialID']   # trial type for each trial (1, 2, or 3)
        cues = data['CS_start']  # cue start time for each trial
        rew_L = data['rewON_L']  # reward onset times on the left side
        rew_R = data['rewON_R']  # reward onset times on the right side
        n = len(ids)

        # --- figure out which trial ID is sucrose vs water for this rat ---
        # have to check which side has sucrose bc of counterbalance; check rewID_L to see if left side matches sucrose reward ID.
        if data['rewID_L'][0] == sucrose_id:
            suc_trial = 1  # sucrose on left side, so trial 1 = sucrose
            wat_trial = 2  # water on right side, so trial 2 = water
        else:
            suc_trial = 2
            wat_trial = 1 

        # get the indices of all sucrose surprise and all water surprise
        suc_idx = np.where(ids == suc_trial)[0]
        wat_idx = np.where(ids == wat_trial)[0]

        # --- split each type's trial indices into 4 equal quarter chunks ---
        def quarters(idx):
            size = len(idx) // 4
            return [set(idx[q * size : (q+1) * size if q < 3 else len(idx)]) for q in range(4)]

        suc_q = quarters(suc_idx)
        wat_q = quarters(wat_idx)

        # initialize storage for this rat if we haven't seen it before
        if rat not in suc_data:
            suc_data[rat] = [[] for i in range(4)]
            wat_data[rat] = [[] for i in range(4)]

        # --- go through each trial one by one ---
        for i in range(n):
            t = ids[i]

            # skip choice trials (trial id 3)
            if t == 3:
                continue

            # figure out time window for this trial (from this cue to next)
            cue = cues[i]
            nxt = cues[i + 1] if i < n - 1 else np.inf # infinity --> to avoid index out of bounds

            # --- find the reward delivery time for this trial ---
            # trial 1 always uses the left reward port, trial 2 uses the right?? double check at end
            if t == 1:
                in_trial = (rew_L >= cue) & (rew_L < nxt)
                if not np.any(in_trial):
                    continue
                rew_time = rew_L[in_trial][0]
            else:
                in_trial = (rew_R >= cue) & (rew_R < nxt)
                if not np.any(in_trial):
                    continue
                rew_time = rew_R[in_trial][0]

            # --- center lick times on reward delivery, then bin into histogram ---
            centered = licks - rew_time  # how many seconds each lick was before/after reward
            in_win = centered[(centered >= win_start) & (centered <= win_end)]
            hist, _ = np.histogram(in_win, bins=edges)
            rate = hist / bin_size  # convert counts to licks per second

            # --- figure out which quarter this trial belongs to, store it ---
            is_suc = (t == suc_trial)
            q_list = suc_q if is_suc else wat_q
            r_dict = suc_data if is_suc else wat_data
            for q in range(4):
                if i in q_list[q]:
                    r_dict[rat][q].append(rate)
                    break

    # --- 1st average: average all trials within each quarter for each rat ---
    # --- 2nd average: average across rats to get one line per quarter ---
    suc_mean = []
    wat_mean = []
    for q in range(4):
        rat_avgs = [np.mean(suc_data[r][q], axis=0) for r in suc_data if len(suc_data[r][q]) > 0]
        suc_mean.append(np.mean(rat_avgs, axis=0) if rat_avgs else np.zeros(len(centers)))

        rat_avgs = [np.mean(wat_data[r][q], axis=0) for r in wat_data if len(wat_data[r][q]) > 0]
        wat_mean.append(np.mean(rat_avgs, axis=0) if rat_avgs else np.zeros(len(centers)))

    # --- plot results: sucrose on left, water on right ---
    suc_colors = ['black', 'saddlebrown', 'chocolate', 'orange']
    wat_colors = ['black', 'darkblue', 'blue', 'purple']
    labels = ['Q1', 'Q2', 'Q3', 'Q4']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # sucrose subplot (left)
    for q in range(4):
        ax1.plot(centers, suc_mean[q], color = suc_colors[q], linewidth = 1.5, label = labels[q])
    ax1.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    ax1.set_xlabel('Time from reward delivery (s)')
    ax1.set_ylabel('Lick rate (licks/s)')
    ax1.set_title('Sucrose Surprise Trials', fontweight='bold')
    ax1.legend(title='Quarter')

    # water subplot (right)
    for q in range(4):
        ax2.plot(centers, wat_mean[q], color = wat_colors[q], linewidth = 1.5, label = labels[q])
    ax2.axvline(0, color = 'gray', linestyle = '--', linewidth = 0.8)
    ax2.set_xlabel('Time from reward delivery (s)')
    ax2.set_title('Water Surprise Trials', fontweight='bold')
    ax2.legend(title='Quarter')

    plt.suptitle('Surprise Trial Lick Rate', fontsize=14, fontweight='bold', y=0.93)
    fig.text(0.5, 0.86, 'Phase 4 Training  |  3 rats, 5 sessions',
             ha='center', fontsize=11, fontstyle='italic', fontweight='bold', color='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig('lick_rate_surprise.png', dpi=150)
    plt.show()


def segment_licks(lick_times, gap_threshold):
    """
    Takes a sorted list of lick timestamps and splits them into groups (bouts or clusters)
    wherever rats pause longer than gap_threshold seconds between licks
    Returns a list of arrays (each array = one group of licks)
    """
    if len(lick_times) == 0:
        return []
    # calculate the time gap between each consecutive pair of licks
    time_gap = np.diff(lick_times)
    # find where the gap is big enough to count as a break
    break_points = np.where(time_gap >= gap_threshold)[0]
    # split the lick times at those break points
    return np.split(lick_times, break_points + 1)



def plot_bout_cluster_analysis(grid, win_start = -2, win_end = 13):
    """
    Bout and cluster analysis on surprise trials, split by session quarter

    For each trial, licks within a window around reward onset are segmented into
    bouts (gap >= 1.0 s) and clusters (gap >= 0.5 s). Calculates metrics of number of bouts/clusters
    and mean licks per bout/cluster. Trials split into 4 equal quarters within each session.
    Averages within quarter per rat, then across rats.

    Uses rewID_L to account for counterbalancing (which side is sucrose vs water).
    """

    # bout = licks separated by < 1.0 s; cluster = licks separated by < 0.5 s
    BOUT_GAP = 1.0
    CLUSTER_GAP = 0.5

    # store per-trial bout/cluster metrics, organized by rat and quarter
    # structure is suc_data[rat][quarter] = list of metric dicts, one per trial
    suc_data = {}
    wat_data = {}

    # --- loop through every session ---
    for (rat, date), data in grid.items():

        # pull out arrays we need
        licks = data['Lick']
        ids = data['TrialID']
        cues = data['CS_start']
        rew_L = data['rewON_L']
        rew_R = data['rewON_R']
        n = len(ids)

        # --- figure out which trial ID is sucrose vs water for this rat (copied from lick rate plot) ---
        if data['rewID_L'][0] == 15:
            suc_trial = 1
            wat_trial = 2
        else:
            suc_trial = 2
            wat_trial = 1

        # get indices of all surprise sucrose and water trials
        suc_idx = np.where(ids == suc_trial)[0]
        wat_idx = np.where(ids == wat_trial)[0]

        # split each type's trial indices into 4 equal quarter chunks
        def quarters(idx):
            size = len(idx) // 4
            return [set(idx[q * size : (q + 1) * size if q < 3 else len(idx)]) for q in range(4)]

        suc_q = quarters(suc_idx)
        wat_q = quarters(wat_idx)

        # create storage for this rat if first time seeing it
        if rat not in suc_data:
            suc_data[rat] = [[] for i in range(4)]
            wat_data[rat] = [[] for i in range(4)]

        # --- go through each trial (similar logic to lick rate as well) ---
        for i in range(n):
            t = ids[i]

            # skip choice trials
            if t == 3:
                continue

            # figure out time window for this trial (from this cue to the next)
            cue = cues[i]
            nxt = cues[i + 1] if i < n - 1 else np.inf

            # find reward onset time (1 = left reward, 2 = right reward)
            if t == 1:
                in_trial = (rew_L >= cue) & (rew_L < nxt)
                if not np.any(in_trial):
                    continue
                rew_time = rew_L[in_trial][0]
            else:
                in_trial = (rew_R >= cue) & (rew_R < nxt)
                if not np.any(in_trial):
                    continue
                rew_time = rew_R[in_trial][0]

            # grab only the licks within window around reward onset
            centered = licks - rew_time
            win_licks = np.sort(licks[(centered >= win_start) & (centered <= win_end)])

            # split those licks into bouts (big gaps) and clusters (smaller gaps) w/ function from above
            bouts = segment_licks(win_licks, BOUT_GAP)
            clusters = segment_licks(win_licks, CLUSTER_GAP)

            # compute the 4 metrics for this trial
            trial_metrics = {
                'n_bouts': len(bouts),
                'mean_licks_bout': np.mean([len(b) for b in bouts]) if bouts else 0,
                'n_clusters': len(clusters),
                'mean_licks_cluster': np.mean([len(c) for c in clusters]) if clusters else 0,
            }

            # store in the correct reward type and quarter
            is_suc = (t == suc_trial)
            q_list = suc_q if is_suc else wat_q
            r_dict = suc_data if is_suc else wat_data
            for q in range(4):
                if i in q_list[q]:
                    r_dict[rat][q].append(trial_metrics)
                    break

    # --- to average --> first within each quarter per rat, then across rats ---
    metric_names = ['n_bouts', 'mean_licks_bout', 'n_clusters', 'mean_licks_cluster']

    def rat_quarter_means(data_dict):
        """For each rat, average all trials within each quarter into a single value per metric"""
        out = {}
        for rat in data_dict:
            out[rat] = []
            for q in range(4):
                trials = data_dict[rat][q]
                if trials:
                    out[rat].append({m: np.mean([t[m] for t in trials]) for m in metric_names})
                else:
                    out[rat].append({m: np.nan for m in metric_names})
        return out

    suc_rat_means = rat_quarter_means(suc_data)
    wat_rat_means = rat_quarter_means(wat_data)

    # --- plot a 2x2 grid of subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # each tuple: (axis, metric key, subplot title)
    plot_info = [
        (axes[0, 0], 'n_bouts', 'Number of Bouts per Trial'),
        (axes[0, 1], 'mean_licks_bout', 'Mean Licks per Bout'),
        (axes[1, 0], 'n_clusters', 'Number of Clusters per Trial'),
        (axes[1, 1], 'mean_licks_cluster', 'Mean Licks per Cluster'),
    ]

    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    x = np.arange(4)
    offset = 0.12  # pushes water + sucrose dots slightly apart **adjust

    for ax, metric, title in plot_info:

        # collect each rat's quarter mean for this metric
        suc_ind = {q: [] for q in range(4)}
        wat_ind = {q: [] for q in range(4)}

        for rat in suc_rat_means:
            for q in range(4):
                val = suc_rat_means[rat][q][metric]
                if not np.isnan(val):
                    suc_ind[q].append(val)

        for rat in wat_rat_means:
            for q in range(4):
                val = wat_rat_means[rat][q][metric]
                if not np.isnan(val):
                    wat_ind[q].append(val)

        # compute the group mean across all rats for each quarter
        suc_group = [np.mean(suc_ind[q]) if suc_ind[q] else np.nan for q in range(4)]
        wat_group = [np.mean(wat_ind[q]) if wat_ind[q] else np.nan for q in range(4)]

        # plot individual rat values as small dots
        for q in range(4):
            ax.scatter([x[q] - offset] * len(suc_ind[q]), suc_ind[q],
                       color='orange', alpha=0.5, s=25, zorder=2)
            ax.scatter([x[q] + offset] * len(wat_ind[q]), wat_ind[q],
                       color='blue', alpha=0.5, s=25, zorder=2)

        # plot group means as larger squares w/ black outlines
        ax.scatter(x - offset, suc_group, color='orange', s=80, zorder=3,
                   marker='s', edgecolors='black', linewidths=0.5, label='Sucrose')
        ax.scatter(x + offset, wat_group, color='blue', s=80, zorder=3,
                   marker='s', edgecolors='black', linewidths=0.5, label='Water')

        # label axes and add legend
        ax.set_xticks(x)
        ax.set_xticklabels(quarter_labels)
        ax.set_xlabel('Session Quarter')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend()

    # main title + subtitle with session info
    plt.suptitle('Bout & Cluster Analysis', fontsize=14, fontweight='bold', y=0.95)
    fig.text(0.5, 0.90, 'Surprise Trials by Quarter  |  Phase 4 Training',
             ha='center', fontsize=11, fontstyle='italic', color='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('bout_cluster_analysis.png', dpi=150)
    plt.show()


def plot_cs_to_response_latency(grid, etoh_id):
    """
    Plots the latency/time difference from CS onset --> operant response,
    separately for surprise and choice trials, with one line per reward type.

    Surprise trials: operant response = first port entry after CS onset
    Choice trials: operant response = whichever lever press (L or R) fired in that trial's window

    Two subplots:
        Left: Surprise trials — water vs ethanol latency over trials
        Right: Choice trials — water vs ethanol latency over trials

    x-axis: trial number (across all sessions, cumulative)
    y-axis: latency in seconds
    """

    # QUESTIONS FOR WEDNESDAY
    # 1. There are a couple data points that have super high latency, how do we handle those (could be non-trial related activity)
    # 


    # make lists for latencies (4 lists)
    
    etoh_forced_lat = []
    water_forced_lat = []
    etoh_choice_lat = []
    water_choice_lat = []


    # loop through each session
    
    for (rat, date), data in grid.items():

        # set important variables
        licks = data['Lick']
        ids = data['TrialID']
        cues = data['CS_start']
        rew_L = data['rewON_L']
        rew_R = data['rewON_R']
        port_enter = data['PortEnter']
        lp_L = data['rewLP_L']   # left lever press timestamps
        lp_R = data['rewLP_R']   # right lever press timestamps
        n = len(ids)


        # pull trial id (water vs etoh)
        if data['rewID_L'][0] == etoh_id:
            eth_trial = 1   # ethanol on left, trial 1 fires left
            wat_trial = 2   # water on right, trial 2 fires right
        else:
            eth_trial = 2
            wat_trial = 1

        # go through each trial
        for i in range(n):
            t = ids[i]

            # get time window (time stamps for beginning + end) for this trial
            cue = cues[i]
            nxt = cues[i + 1] if i < n - 1 else np.inf


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
                    etoh_forced_lat.append(latency)
                else:
                    water_forced_lat.append(latency)

        # CHOICE TRIALS (id 3)
            # elif t == 3:
            elif t == 3:

                # check which lever was pressed within this trial's window
                lp_L_in = lp_L[(lp_L >= cue) & (lp_L < nxt)]
                lp_R_in = lp_R[(lp_R >= cue) & (lp_R < nxt)]

                # figure out which one fired, calculate latency
                if len(lp_L_in) > 0 and len(lp_R_in) > 0:
                    # both fired — use whichever came first
                    press_time = min(lp_L_in[0], lp_R_in[0])
                    pressed_side = 'L' if lp_L_in[0] < lp_R_in[0] else 'R'
                elif len(lp_L_in) > 0:
                    press_time = lp_L_in[0]
                    pressed_side = 'L'
                elif len(lp_R_in) > 0:
                    press_time = lp_R_in[0]
                    pressed_side = 'R'
                else:
                    continue  # no lever press found, skip trial

                latency = press_time - cue

                # figure out which reward the rat chose based on which side it pressed 
                # + whether that side is etoh/water for this rat
                if pressed_side == 'L':
                    chose_eth = (data['rewID_L'][0] == etoh_id)
                else:
                    chose_eth = (data['rewID_L'][0] != etoh_id)

                if chose_eth:
                    etoh_choice_lat.append(latency)
                else:
                    water_choice_lat.append(latency)

        
    # plot + formatting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # surprise trials subplot
    ax1.scatter(range(len(etoh_forced_lat)), etoh_forced_lat,
                color='darkorange', s=10, alpha=0.5, label='Ethanol')
    ax1.scatter(range(len(water_forced_lat)), water_forced_lat,
                color='blue', s=10, alpha=0.5, label='Water')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Latency (s)')
    ax1.set_title('Surprise Trials: CS → Port Entry', fontweight='bold')
    ax1.legend()

    # choice trials subplot
    ax2.scatter(range(len(etoh_choice_lat)), etoh_choice_lat,
                color='darkorange', s=10, alpha=0.5, label='Ethanol')
    ax2.scatter(range(len(water_choice_lat)), water_choice_lat,
                color='blue', s=10, alpha=0.5, label='Water')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Latency (s)')
    ax2.set_title('Choice Trials: CS → Lever Press', fontweight='bold')
    ax2.legend()

    plt.suptitle('CS Onset to Operant Response Latency', fontsize=14, fontweight='bold', y=0.93)
    fig.text(0.5, 0.85, 'Ethanol vs Water', ha='center', fontsize=11, fontstyle='italic', color='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig('cs_to_operant_latency.png', dpi=150)
    plt.show()






if __name__ == '__main__':

    # load all subject files from the data folder
    data_dir = './MPCdata_MJ'
    metadata_df, grid = load_data(data_dir)

    # --- debugging ---
    import random
    keys = list(grid.keys())
    test_key = random.choice(keys)
    rat = test_key[0]
    date = test_key[1]
    print("Random key picked: rat =", rat, ", date =", date)
    value = grid[test_key]['PortEnter'][49]
    print("50th value of E (PortEnter):", value)

    # --- lick rate plot (forced/surprise trials only, skipping choice trials) ---
    # sucrose_id = 7 means rewID 7 is the sucrose reward
    # (this handles the counterbalancing — each rat's sides are checked automatically)
    plot_lick_rate_by_quarter(grid, sucrose_id=15)

    # --- bout & cluster analysis ---
    plot_bout_cluster_analysis(grid)

    # --- latency plot ---
    plot_cs_to_response_latency(grid, 7)