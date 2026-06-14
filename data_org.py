

import os

# Variable definition (from what was sent earlier)
varList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'X', 'Y']
varSave = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
varName = ['rewID_L', 'rewID_R', 'Indeces', 'Timers', 'PortEnter', 'PortExit', 'rewON_L', 'rewON_R', 'rewLP_L', 'rewLP_R', 'Lick', 'CS_start', 'TrialID', 'lat_rewPE', 'lat_rewLP', 'allLP_L', 'allLP_R', 'rewPE', 'list_ITI', 'list_trialID']

# Matching variables to readable names (in dictionary)
letter_to_name = {}
for letter, save, name in zip(varList, varSave, varName):
    if save == 1:
        letter_to_name[letter] = name

# Reads through a .subject file line by line
# Returns a dictionary w/ metadata (date, subject, experiment, group, box, MSN) and data (one key per variable, value is list of floats)
def read_subject_file(filepath):
 
    session = {}           # everything for this session
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
            session['start date'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Subject:'):
            session['subject'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Group:'):
            session['group'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('Box:'):
            session['box'] = stripped.split(':', 1)[1].strip()
 
        elif stripped.startswith('MSN:'):
            session['MSN'] = stripped.split(':', 1)[1].strip()
 
        # storing variables
        elif len(stripped) >= 2 and stripped[0].isupper() and stripped[1] == ':':
            # save whatever we were collecting for the previous letter
            if current_letter is not None and current_letter in letter_to_name:
                session[letter_to_name[current_letter]] = current_data
 
            current_letter = stripped[0]   # store just the letter
            current_data = []              # reset for new variable
 
            # check if there's a value on the same line (deals w/ A and B)
            remaining = stripped[2:].strip()
            if remaining:
                # dealing w/ single value on the same line
                session[letter_to_name[current_letter]] = [float(remaining)]
                current_letter = None  # reset curr letter because we don't want further lines to be counted
 
        # *** CHECK THIS PART LATER
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
        session[letter_to_name[current_letter]] = current_data
 
    return session


# Loops through every .subject file in the folder and calls read_subject_file() on each one
# Organizes all returned session dicts into a nested dictionary (grid = {'rat': {'date': all data}})

def load_data(data_dir):
 
    grid = {}  # outer key --> rat, inner key --> date, value --> session dict
 
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.subject'):
            filepath = data_dir + '/' + filename
            session = read_subject_file(filepath)

            rat  = session['subject']
            date = session['date']
    
            if rat not in grid:
                grid[rat] = {}
    
            grid[rat][date] = session
            print(f"Loaded: {filename}  |  rat = {rat}, date = {date}")
 
    return grid