import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import textwrap
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    f1_score
)
from sklearn.linear_model import LogisticRegression


def read_patient_files(data_dir):
    """
    Reads the files for each patient and exploration in the folder structure.

    Parameters:
    data_dir (str): The directory containing patient data folders.

    Returns:
    dict: A nested dictionary where the top-level keys are patient folder names,
          the next level keys are exploration folder names, and the leaves are dictionaries
          containing the five dataframes: levent, revent, lts, rts, and avgts.
    """
    patient_data = {}
    
    # List all patient folders
    patient_folder_list = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    # Iterate over each patient folder
    for patient_folder in patient_folder_list:
        patient_data[patient_folder] = {}
        patient_folder_path = os.path.join(data_dir, patient_folder)
        
        # List all exploration folders for the current patient
        exploration_folders = [f for f in os.listdir(patient_folder_path) if os.path.isdir(os.path.join(patient_folder_path, f))]
        
        # Iterate over each exploration folder
        for exploration_folder in exploration_folders:
            patient_exploration_folder_path = os.path.join(patient_folder_path, exploration_folder)
            
            # Read and process the .dat files
            levent = read_dat_file(os.path.join(patient_exploration_folder_path, "LEvent.dat"))
            levent.columns = ['start_time', 'end_time', 'event']
            levent['event'] = levent['event'].map({0: 'saccade', 1: 'fixation', 2: 'blink'})

            lts = read_dat_file(os.path.join(patient_exploration_folder_path, "LTS.dat"))
            lts.columns = ['x', 'y']

            revent = read_dat_file(os.path.join(patient_exploration_folder_path, "REvent.dat"))
            revent.columns = ['start_time', 'end_time', 'event']
            revent['event'] = revent['event'].map({0: 'saccade', 1: 'fixation', 2: 'blink'})

            rts = read_dat_file(os.path.join(patient_exploration_folder_path, "RTS.dat"))
            rts.columns = ['x', 'y']
            
            # Create avgts DataFrame for the average of both eyes
            avgts = pd.DataFrame()
            avgts['x'] = (lts['x'] + rts['x']) / 2
            avgts['y'] = (lts['y'] + rts['y']) / 2

            # Ignore NA values
            avgts.dropna(inplace=True)
            
            # Filter out blinks
            lts_noblinks = filter_blinks(lts, levent)
            rts_noblinks = filter_blinks(rts, revent)
            
            # Store the dataframes in the nested dictionary
            patient_data[patient_folder][exploration_folder] = {
                'levent': levent,
                'revent': revent,
                'lts': lts,
                'rts': rts,
                'avgts': avgts,
                'lts_noblinks': lts_noblinks,
                'rts_noblinks': rts_noblinks
            }
    
    return patient_data

def read_dat_file(filepath):
    """
    Read a .dat file into a pandas DataFrame.

    Parameters:
    filepath (str): The path to the .dat file.

    Returns:
    pd.DataFrame or None: The loaded DataFrame, or None if an error occurs.
    """
    try:
        return pd.read_csv(filepath, delimiter=',', header=None)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None  
    
def filter_blinks(ts_df, event_df):
    # Get the blink events
    blink_events = event_df[event_df['event'] == 'blink']
    
    # Get the indices to be excluded
    indices_to_exclude = []
    for _, row in blink_events.iterrows():
        indices_to_exclude.extend(range(row['start_time'], row['end_time'] + 1))
    
    # Filter the rts dataframe
    ts_filtered = ts_df.drop(indices_to_exclude, errors='ignore')
    return ts_filtered

def get_blink_metrics(patient_data):
    """
    Calculate blink-related metrics and percentile-based dummies for all patients and their explorations.
    
    This function processes the event dataframes for both eyes of each patient to compute the following "counted" metrics:
        - total_blinks: Count the number of rows where the event is 'blink'.
        - total_events: Count the total number of rows.
        - time_blinking: Sum the durations (end_time - start_time) for all 'blink' events.
        - total_time: Sum the durations (end_time - start_time) for all events.
    
    From these, it derives the following "calculated" metrics:
        - pct_blinks: total_blinks / total_events.
        - pct_time_blinking: time_blinking / total_time.
    
    The function then analyzes the percentiles for the calculated metrics to create dummies if the values fall below the 1st percentile or above the 99th percentile, as follows:
        - pct_01_total_time_too_little_time: 1 if total_measured_time < 1st percentile, else 0.
        - pct_99_total_blinks_too_many_blinks: 1 if total_blinks > 99th percentile, else 0.
        - pct_99_pct_blinks_pct_of_blinks_vs_saccadesfixations_too_high: 1 if blinks_as_pct_events > 99th percentile, else 0.
        - pct_99_time_blinking_too_much_time_blinking: 1 if total_blinking_time > 99th percentile, else 0.
        - pct_99_pct_time_blinking_pct_of_time_blinking_too_high: 1 if pct_blinking_time > 99th percentile, else 0.
    
    Finally, it creates an additional dummy:
        - problematic_observation: 1 if any of the above dummies is 1, else 0.
    
    The function returns a DataFrame with the columns:
        'patient_id', 'exploration_id', 'eye', 'total_measured_time', 
        'total_blinking_time', 'pct_blinking_time', 'total_events', 
        'total_blinks', 'blinks_as_pct_events', 'pct_01_total_time_too_little_time', 
        'pct_99_total_blinks_too_many_blinks', 'pct_99_pct_blinks_pct_of_blinks_vs_saccadesfixations_too_high',
        'pct_99_time_blinking_too_much_time_blinking', 'pct_99_pct_time_blinking_pct_of_time_blinking_too_high',
        'problematic_observation'.

    Parameters:
    patient_data (dict): Nested dictionary containing the patient data. The structure is expected to be:
        patient_data[patient_id][exploration_id]['revent'] for right eye events
        patient_data[patient_id][exploration_id]['levent'] for left eye events
    
    Returns:
    pd.DataFrame: DataFrame containing the calculated metrics and dummies.
    """
    # Initialize the dataframe to store the blink metrics results
    blink_metrics_df = pd.DataFrame(columns=[
        'patient_id', 'exploration_id', 'eye', 
        'total_measured_time', 'total_blinking_time', 
        'pct_blinking_time', 'total_events', 'total_blinks', 'blinks_as_pct_events'
    ])

    # Iterate over each patient and exploration to calculate the blink metrics
    for patient_id in patient_data.keys():
        for exploration_id in patient_data[patient_id].keys():
            # Calculate metrics for the right eye
            metrics_right = calculate_completeness_and_blink_metrics(
                patient_data[patient_id][exploration_id]['revent']
            )
            new_row_right = pd.DataFrame([{
                'patient_id': patient_id,
                'exploration_id': exploration_id,
                'eye': 'right',
                'total_measured_time': metrics_right['total_measured_time'],
                'total_blinking_time': metrics_right['total_blinking_time'],
                'pct_blinking_time': metrics_right['pct_blinking_time'],
                'total_events': metrics_right['total_events'],
                'total_blinks': metrics_right['total_blinks'],
                'blinks_as_pct_events': metrics_right['blinks_as_pct_events']
            }])
            blink_metrics_df = pd.concat([blink_metrics_df, new_row_right], ignore_index=True)

            # Calculate metrics for the left eye
            metrics_left = calculate_completeness_and_blink_metrics(
                patient_data[patient_id][exploration_id]['levent']
            )
            new_row_left = pd.DataFrame([{
                'patient_id': patient_id,
                'exploration_id': exploration_id,
                'eye': 'left',
                'total_measured_time': metrics_left['total_measured_time'],
                'total_blinking_time': metrics_left['total_blinking_time'],
                'pct_blinking_time': metrics_left['pct_blinking_time'],
                'total_events': metrics_left['total_events'],
                'total_blinks': metrics_left['total_blinks'],
                'blinks_as_pct_events': metrics_left['blinks_as_pct_events']
            }])
            blink_metrics_df = pd.concat([blink_metrics_df, new_row_left], ignore_index=True)

    # Ensure all calculated columns are numeric
    for col in ['total_measured_time', 'total_blinking_time', 'pct_blinking_time', 'total_events', 'total_blinks', 'blinks_as_pct_events']:
        blink_metrics_df[col] = pd.to_numeric(blink_metrics_df[col])
    
    # Calculate percentiles
    pct_01_total_time = blink_metrics_df['total_measured_time'].quantile(0.01)
    pct_01_total_events = blink_metrics_df['total_events'].quantile(0.01)
    pct_99_total_events = blink_metrics_df['total_events'].quantile(0.99)
    pct_99_total_blinks = blink_metrics_df['total_blinks'].quantile(0.99)
    pct_99_blinks_as_pct_events = blink_metrics_df['blinks_as_pct_events'].quantile(0.99)
    pct_99_total_blinking_time = blink_metrics_df['total_blinking_time'].quantile(0.99)
    pct_99_pct_blinking_time = blink_metrics_df['pct_blinking_time'].quantile(0.99)
    
    # Create dummies based on percentiles
    blink_metrics_df['pct_01_total_time_too_little_time'] = (blink_metrics_df['total_measured_time'] < pct_01_total_time).astype(int)
    blink_metrics_df['pct_99_total_blinks_too_many_blinks'] = (blink_metrics_df['total_blinks'] > pct_99_total_blinks).astype(int)
    blink_metrics_df['pct_01_total_events_too_few_events'] = (blink_metrics_df['total_events'] < 2) # at least 4 events =~ 2 fixations and 2 saccades
    blink_metrics_df['pct_99_total_events_too_many_events'] = (blink_metrics_df['total_events'] > pct_99_total_events).astype(int)
    blink_metrics_df['pct_99_pct_blinks_pct_of_blinks_vs_saccadesfixations_too_high'] = (blink_metrics_df['blinks_as_pct_events'] > pct_99_blinks_as_pct_events).astype(int)
    blink_metrics_df['pct_99_time_blinking_too_much_time_blinking'] = (blink_metrics_df['total_blinking_time'] > pct_99_total_blinking_time).astype(int)
    blink_metrics_df['pct_99_pct_time_blinking_pct_of_time_blinking_too_high'] = (blink_metrics_df['pct_blinking_time'] > pct_99_pct_blinking_time).astype(int)
    
    # Create the extra dummy for problematic observations
    blink_metrics_df['problematic_observation'] = blink_metrics_df[
        ['pct_01_total_time_too_little_time', 'pct_99_total_blinks_too_many_blinks',
         'pct_01_total_events_too_few_events','pct_99_total_events_too_many_events',
         'pct_99_pct_blinks_pct_of_blinks_vs_saccadesfixations_too_high', 'pct_99_time_blinking_too_much_time_blinking',
         'pct_99_pct_time_blinking_pct_of_time_blinking_too_high']
    ].max(axis=1)
    
    return blink_metrics_df

def get_saccade_number_and_excursion(event_df, ts_df):
    # Initialize the new dataframe
    saccade_data = {
        'saccade_id': [],
        'start_time': [],
        'end_time': [],
        'x_0': [],
        'y_0': [],
        'x_1': [],
        'y_1': [],
        'distance': []
    }

    # Iterate through the events dataframe to find saccades
    saccade_id = 1
    for index, row in event_df.iterrows():
        if row['event'] == 'saccade':
            start_time = row['start_time'] - 1  # Adjust for the index offset
            end_time = row['end_time'] - 1  # Adjust for the index offset
            x_0 = ts_df.iloc[start_time]['x'] # find start coordinates
            y_0 = ts_df.iloc[start_time]['y']
            x_1 = ts_df.iloc[end_time]['x'] # find end coordinates
            y_1 = ts_df.iloc[end_time]['y']

            saccade_data['saccade_id'].append(saccade_id)
            saccade_data['start_time'].append(row['start_time'])
            saccade_data['end_time'].append(row['end_time'])
            saccade_data['x_0'].append(x_0)
            saccade_data['y_0'].append(y_0)
            saccade_data['x_1'].append(x_1)
            saccade_data['y_1'].append(y_1)

            # Add a column that calculates distance
            saccade_data['distance'].append(np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))

            saccade_id += 1

    # Convert the dictionary to a dataframe, for ease of handling
    saccade_df = pd.DataFrame(saccade_data)
    
    total_saccadic_excursion = saccade_df['distance'].sum()
    total_saccades = saccade_df.shape[0]  # Fixed to get the number of rows
    
    return total_saccadic_excursion, total_saccades

def get_fixation_total_and_avg(event_df):
    fixation_rows = event_df[event_df['event'] == 'fixation']
    total_fixations = len(fixation_rows)
    average_fixation_time = fixation_rows['end_time'].sub(fixation_rows['start_time']).mean()
    return total_fixations, average_fixation_time

def get_scanned_area_and_diagonal(ts_df):
    points = ts_df[['x', 'y']].dropna().values  # Drop rows with NaN values
    if len(points) < 3:  # ConvexHull needs at least 3 points to form a hull
        return 0.0, 0.0
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Calculate the longest diagonal
    max_distance = 0
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            distance = np.linalg.norm(hull_points[i] - hull_points[j])
            if distance > max_distance:
                max_distance = distance
                
    return hull.area, max_distance

def get_saccade_metrics(patient_data):
    saccades_excursion_df = pd.DataFrame(columns=['patient_id', 'exploration_id', 'eye', 'total_saccades', 'total_saccadic_excursion'])
    
    for patient_folder in patient_data.keys():
        for exploration_folder in patient_data[patient_folder].keys():
            # Check if necessary data exists for the right eye
            if 'revent' in patient_data[patient_folder][exploration_folder] and 'rts' in patient_data[patient_folder][exploration_folder]:
                total_saccadic_excursion, total_saccades = get_saccade_number_and_excursion(
                    patient_data[patient_folder][exploration_folder]['revent'],
                    patient_data[patient_folder][exploration_folder]['rts']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'right',
                    'total_saccades': total_saccades,
                    'total_saccadic_excursion': total_saccadic_excursion
                }])
                saccades_excursion_df = pd.concat([saccades_excursion_df, new_row], ignore_index=True)

            # Check if necessary data exists for the left eye
            if 'levent' in patient_data[patient_folder][exploration_folder] and 'lts' in patient_data[patient_folder][exploration_folder]:
                total_saccadic_excursion, total_saccades = get_saccade_number_and_excursion(
                    patient_data[patient_folder][exploration_folder]['levent'],
                    patient_data[patient_folder][exploration_folder]['lts']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'left',
                    'total_saccades': total_saccades,
                    'total_saccadic_excursion': total_saccadic_excursion
                }])
                saccades_excursion_df = pd.concat([saccades_excursion_df, new_row], ignore_index=True)
    
    return saccades_excursion_df

def get_fixation_metrics(patient_data):
    fixation_analysis_df = pd.DataFrame(columns=['patient_id', 'exploration_id', 'eye', 'total_fixations', 'average_fixation_time'])
    
    for patient_folder in patient_data.keys():
        for exploration_folder in patient_data[patient_folder].keys():
            # For the right eye
            if 'revent' in patient_data[patient_folder][exploration_folder]:
                total_fixations, average_fixation_time = get_fixation_total_and_avg(
                    patient_data[patient_folder][exploration_folder]['revent']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'right',
                    'total_fixations': total_fixations,
                    'average_fixation_time': average_fixation_time
                }])
                fixation_analysis_df = pd.concat([fixation_analysis_df, new_row], ignore_index=True)

            # For the left eye
            if 'levent' in patient_data[patient_folder][exploration_folder]:
                total_fixations, average_fixation_time = get_fixation_total_and_avg(
                    patient_data[patient_folder][exploration_folder]['levent']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'left',
                    'total_fixations': total_fixations,
                    'average_fixation_time': average_fixation_time
                }])
                fixation_analysis_df = pd.concat([fixation_analysis_df, new_row], ignore_index=True)
    
    return fixation_analysis_df

def get_area_metrics(patient_data):
    scanned_area_df = pd.DataFrame(columns=['patient_id', 'exploration_id', 'eye', 'total_scanned_area', 'longest_diagonal'])
    
    for patient_folder in patient_data.keys():
        for exploration_folder in patient_data[patient_folder].keys():
            # For the right eye
            if 'rts_noblinks' in patient_data[patient_folder][exploration_folder]:
                total_scanned_area, longest_diagonal = get_scanned_area_and_diagonal(
                    patient_data[patient_folder][exploration_folder]['rts_noblinks']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'right',
                    'total_scanned_area': total_scanned_area,
                    'longest_diagonal': longest_diagonal
                }])
                scanned_area_df = pd.concat([scanned_area_df, new_row], ignore_index=True)

            # For the left eye
            if 'lts_noblinks' in patient_data[patient_folder][exploration_folder]:
                total_scanned_area, longest_diagonal = get_scanned_area_and_diagonal(
                    patient_data[patient_folder][exploration_folder]['lts_noblinks']
                )
                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'left',
                    'total_scanned_area': total_scanned_area,
                    'longest_diagonal': longest_diagonal
                }])
                scanned_area_df = pd.concat([scanned_area_df, new_row], ignore_index=True)
    
    return scanned_area_df

def calculate_completeness_and_blink_metrics(event_df):
    """
    Calculate blink-related metrics for a given eye's event DataFrame.
    
    Parameters:
    event_df (pd.DataFrame): DataFrame containing the events for an eye.
    
    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    metrics = {
        'total_measured_time': 0,
        'total_blinking_time': 0,
        'pct_blinking_time': 0,
        'total_blinks': 0,
        'total_events': 0,
        'blinks_as_pct_events': 0
    }
    
    if event_df.empty:
        return metrics
    
    # Calculate total measured time
    total_measured_time = (event_df['end_time'] - event_df['start_time']).sum()
    metrics['total_measured_time'] = total_measured_time
    
    # Calculate metrics for blink events
    blink_events = event_df[event_df['event'] == 'blink']
    total_blinking_time = (blink_events['end_time'] - blink_events['start_time']).sum()
    total_blinks = blink_events.shape[0]
    total_events = event_df.shape[0]
    
    metrics['total_blinking_time'] = total_blinking_time
    metrics['pct_blinking_time'] = total_blinking_time / total_measured_time if total_measured_time > 0 else 0
    metrics['total_blinks'] = total_blinks
    metrics['total_events'] = total_events
    metrics['blinks_as_pct_events'] = total_blinks / total_events if total_events > 0 else 0
    
    return metrics

def get_crosscorr_lags(patient_data):
    """
    Calculate cross-correlation lags for x and y coordinates between left and right eyes, 
    and generate percentile-based dummies for all patients and their explorations.
    
    This function processes the "ts_noblinks" dataframes for both eyes of each patient to perform a cross-correlation analysis.
    Due to potential differences in the number of data points after removing blinks, the function allows some leniency with the lag.
    
    The function computes the following for each patient and exploration:
        - crosscorr_lag_x: Lag at which the maximum cross-correlation occurs for the x coordinates.
        - crosscorr_lag_y: Lag at which the maximum cross-correlation occurs for the y coordinates.
    
    From these, it derives the following percentile-based dummy variables:
        - pct_01_crosscorr_lag_x: 1 if crosscorr_lag_x < 1st percentile, else 0.
        - pct_99_crosscorr_lag_x: 1 if crosscorr_lag_x > 99th percentile, else 0.
        - pct_01_crosscorr_lag_y: 1 if crosscorr_lag_y < 1st percentile, else 0.
        - pct_99_crosscorr_lag_y: 1 if crosscorr_lag_y > 99th percentile, else 0.
    
    Finally, it creates an additional dummy:
        - problematic_synch: 1 if any of the above dummies is 1, else 0.
    
    The function returns a DataFrame with the columns:
        'patient_id', 'exploration_id', 'crosscorr_lag_x', 'crosscorr_lag_y', 
        'pct_01_crosscorr_lag_x', 'pct_99_crosscorr_lag_x', 
        'pct_01_crosscorr_lag_y', 'pct_99_crosscorr_lag_y', 'problematic_synch'.
    
    Parameters:
    patient_data (dict): Nested dictionary containing the patient data. The structure is expected to be:
        patient_data[patient_id][exploration_id]['lts_noblinks'] for left eye time series without blinks
        patient_data[patient_id][exploration_id]['rts_noblinks'] for right eye time series without blinks
    
    Returns:
    pd.DataFrame: DataFrame containing the calculated cross-correlation lags and dummies.
    """
    def compute_cross_correlation(x1, x2):
        """Compute cross-correlation between two time series and find the lag with the maximum correlation."""
        n = len(x1)
        xcorr = np.correlate(x1 - np.mean(x1), x2 - np.mean(x2), mode='full')
        lags = np.arange(-n + 1, n)
        lag = lags[np.argmax(xcorr)]
        return lag
    
    # Initialize the dataframe to store the cross-correlation lag results
    crosscorr_lags_df = pd.DataFrame(columns=[
        'patient_id', 'exploration_id', 'crosscorr_lag_x', 'crosscorr_lag_y'
    ])
    
    # Iterate over each patient and exploration to calculate the cross-correlation lags
    for patient_id in patient_data.keys():
        for exploration_id in patient_data[patient_id].keys():
            # Extract the time series for left and right eyes
            left_eye_x = patient_data[patient_id][exploration_id]['lts_noblinks']['x']
            left_eye_y = patient_data[patient_id][exploration_id]['lts_noblinks']['y']
            right_eye_x = patient_data[patient_id][exploration_id]['rts_noblinks']['x']
            right_eye_y = patient_data[patient_id][exploration_id]['rts_noblinks']['y']
            
            # Compute cross-correlation lags
            crosscorr_lag_x = compute_cross_correlation(left_eye_x, right_eye_x)
            crosscorr_lag_y = compute_cross_correlation(left_eye_y, right_eye_y)
            
            # Add the results to the dataframe
            new_row = pd.DataFrame([{
                'patient_id': patient_id,
                'exploration_id': exploration_id,
                'crosscorr_lag_x': crosscorr_lag_x,
                'crosscorr_lag_y': crosscorr_lag_y
            }])
            crosscorr_lags_df = pd.concat([crosscorr_lags_df, new_row], ignore_index=True)
    
    # Ensure all calculated columns are numeric
    for col in ['crosscorr_lag_x', 'crosscorr_lag_y']:
        crosscorr_lags_df[col] = pd.to_numeric(crosscorr_lags_df[col])
    
    # Calculate percentiles
    pct_01_crosscorr_lag_x = crosscorr_lags_df['crosscorr_lag_x'].quantile(0.01)
    pct_99_crosscorr_lag_x = crosscorr_lags_df['crosscorr_lag_x'].quantile(0.99)
    pct_01_crosscorr_lag_y = crosscorr_lags_df['crosscorr_lag_y'].quantile(0.01)
    pct_99_crosscorr_lag_y = crosscorr_lags_df['crosscorr_lag_y'].quantile(0.99)
    
    # Create dummies based on percentiles
    crosscorr_lags_df['pct_01_crosscorr_lag_x'] = (crosscorr_lags_df['crosscorr_lag_x'] < pct_01_crosscorr_lag_x).astype(int)
    crosscorr_lags_df['pct_99_crosscorr_lag_x'] = (crosscorr_lags_df['crosscorr_lag_x'] > pct_99_crosscorr_lag_x).astype(int)
    crosscorr_lags_df['pct_01_crosscorr_lag_y'] = (crosscorr_lags_df['crosscorr_lag_y'] < pct_01_crosscorr_lag_y).astype(int)
    crosscorr_lags_df['pct_99_crosscorr_lag_y'] = (crosscorr_lags_df['crosscorr_lag_y'] > pct_99_crosscorr_lag_y).astype(int)
    
    # Create the extra dummy for problematic synchronization
    crosscorr_lags_df['problematic_synch'] = crosscorr_lags_df[
        ['pct_01_crosscorr_lag_x', 'pct_99_crosscorr_lag_x',
         'pct_01_crosscorr_lag_y', 'pct_99_crosscorr_lag_y']
    ].max(axis=1)
    
    return crosscorr_lags_df

# Function to normalize data to be between 0 and 1, clamping outliers
def normalize_column(column, min_value, max_value):
    normalized = (column - min_value) / (max_value - min_value)
    # Clamp values to the range [0, 1]
    normalized = normalized.clip(0, 1)
    return normalized

def get_t_tests(metrics_df, each_eye=False, group_level='group'):
    # Identify the metrics to be tested
    exclude_columns = ['patient_id', 'exploration_id', 'eye', 'group', 'group_general']
    metrics = [col for col in metrics_df.columns if col not in exclude_columns]
    
    # Convert all metrics columns to numeric, coercing errors to NaN
    for metric in metrics:
        metrics_df[metric] = pd.to_numeric(metrics_df[metric], errors='coerce')
    
    # Drop any metrics columns that do not contain numeric data
    metrics = [col for col in metrics if pd.api.types.is_numeric_dtype(metrics_df[col])]
    
    # Identify unique groups based on the group_level column
    groups = metrics_df[group_level].dropna().unique()
    explorations = metrics_df['exploration_id'].unique()
    
    # Initialize the results list
    results = []

    if each_eye:
        # Perform pairwise t-tests for each metric, each eye, and each exploration
        for exploration in explorations:
            exploration_data = metrics_df[metrics_df['exploration_id'] == exploration]
            for metric in metrics:
                for eye in ['right', 'left']:
                    eye_data = exploration_data[exploration_data['eye'] == eye]
                    for i in range(len(groups)):
                        for j in range(i + 1, len(groups)):
                            group_A = groups[i]
                            group_B = groups[j]

                            data_A = eye_data[eye_data[group_level] == group_A][metric]
                            data_B = eye_data[eye_data[group_level] == group_B][metric]

                            t_statistic, p_value = ttest_ind(data_A, data_B, equal_var=False)  # Welch's t-test
                            significant = p_value < 0.05  # Assuming 0.05 as the significance level

                            results.append({
                                'exploration_id': exploration,
                                'metric': metric,
                                'group_A': group_A,
                                'group_B': group_B,
                                'eye': eye,
                                't_statistic': t_statistic,
                                'p_value': p_value,
                                'significant': significant
                            })
    else:
        # Perform pairwise t-tests for each metric and each exploration
        for exploration in explorations:
            exploration_data = metrics_df[metrics_df['exploration_id'] == exploration]
            for metric in metrics:
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        group_A = groups[i]
                        group_B = groups[j]

                        data_A = exploration_data[exploration_data[group_level] == group_A][metric]
                        data_B = exploration_data[exploration_data[group_level] == group_B][metric]

                        t_statistic, p_value = ttest_ind(data_A, data_B, equal_var=False)  # Welch's t-test
                        significant = p_value < 0.05  # Assuming 0.05 as the significance level

                        results.append({
                            'exploration_id': exploration,
                            'metric': metric,
                            'group_A': group_A,
                            'group_B': group_B,
                            't_statistic': t_statistic,
                            'p_value': p_value,
                            'significant': significant
                        })
    
    # Create the results DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out rows where group_B is NaN
    results_df = results_df.dropna(subset=['group_B'])
    
    return results_df

def calculate_state_standard_metrics(viterbi_path):
    """
    Calculate HMM metrics for a given Viterbi path.

    This function computes the following metrics:
    - Fractional Occupancy (FO): The proportion of time spent in each state.
    - Mean Lifetime (MLT): The average duration that the system stays in each state once it is entered.
    - Mean Interval Length (MIL): The average time between recurring visits to each state.

    Parameters:
    ----------
    viterbi_path : np.array
        A NumPy array representing the sequence of states in a Viterbi path.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the calculated metrics (FO, MLT, MIL) for each unique state.
    """

    # Calculate unique states in the Viterbi path
    states = np.unique(viterbi_path)

    # Total time or length of the Viterbi path
    T = len(viterbi_path)

    # Calculate the time spent in each state
    time_in_state = [np.sum(viterbi_path == state) for state in states]

    # Initialize an array to store the number of occurrences of each state
    n_occurences_state = np.zeros(len(states))

    # Create a mapping from state to index
    state_to_index = {state: idx for idx, state in enumerate(states)}

    # Calculate the number of occurrences of each state
    for i in range(1, len(viterbi_path)):
        current_state = viterbi_path[i]
        previous_state = viterbi_path[i - 1]
        if current_state != previous_state:
            n_occurences_state[state_to_index[current_state]] += 1

    # Adjust the first state occurrence count
    first_state_index = state_to_index[viterbi_path[0]]
    n_occurences_state[first_state_index] += 1

    # Convert occurrences array to a list
    n_occurences_state = n_occurences_state.tolist()

    # Calculate Fractional Occupancy (FO)
    FO = [t / T for t in time_in_state]

    # Calculate Mean Lifetime (MLT)
    MLT = [t / n for t, n in zip(time_in_state, n_occurences_state)]

    # Calculate Mean Interval Length (MIL)
    MIL = [(T - t) / n for t, n in zip(time_in_state, n_occurences_state)]

    # Create a DataFrame to store the metrics
    result_df = pd.DataFrame({
        'state': states,
        'FO': FO,
        'MLT': MLT,
        'MIL': MIL
    })

    return result_df

# Function to calculate entropy for a probability distribution
def entropy(probabilities):
    # Filter out zero probabilities to avoid log(0) which is undefined
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def get_auc_weights(variables, chosen_models):
    """ Return AUC weights for the selected variables """
    auc_weights = []
    for var in variables:
        exploration_name = "_".join(var.split("_")[:2])  # Extract Exploration_x
        auc_value = chosen_models.loc[chosen_models['exploration'] == exploration_name, 'mean_test_roc_auc'].values
        auc_weights.append(auc_value[0] if len(auc_value) > 0 else 1)  # Default to 1 if missing
    return np.array(auc_weights) / np.sum(auc_weights)  # Normalize weights

def fit_logistic_regression(X_train, y_train, X_test, penalty, C=1.0):
    if penalty == 'l1':
        model = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=5000, random_state=42)
    elif penalty == 'l2':
        model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=5000, random_state=42)
    elif penalty == 'none':
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, random_state=42)
    else:
        raise ValueError(f"Invalid penalty: {penalty}")

    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]

# Function to compute performance metrics

def compute_metrics(df):
    accuracy = accuracy_score(df['target'], df['prediction'])
    sensitivity = recall_score(df['target'], df['prediction'])  # Sensitivity (Recall)

    # Compute specificity manually
    tn, fp, fn, tp = confusion_matrix(df['target'], df['prediction']).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    auc = roc_auc_score(df['target'], df['proba'])  
    f1 = f1_score(df['target'], df['prediction'], zero_division=0)

    return pd.Series({
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "f1": f1  
    })

def pivot_and_flatten(df, value_col, prefix):
    df_pivot = df.pivot_table(
        index=['patient_id', 'fold', 'target'],
        columns=['exploration_id', 'eye_id'],
        values=value_col,
        aggfunc='first'
    ).fillna(0.5)
    df_pivot.columns = [f"{expl}_{prefix}_eye{eye}" for expl, eye in df_pivot.columns]
    return df_pivot.reset_index()
    
def wrap_label(model, variables_str, width=80):
    wrapped_vars = "\n".join(textwrap.wrap(variables_str, width=width))
    return f"{model}\n{wrapped_vars}"

def get_variance_metrics(patient_data):
    variance_metrics_df = pd.DataFrame(
        columns=[
            'patient_id',
            'exploration_id',
            'eye',
            'var_x',
            'var_y',
            'sd_x',
            'sd_y',
            'rms_x',
            'rms_y',
            'avg_abs_dist_x_center',
            'avg_abs_dist_y_center',
            'avg_dist_xy_center'
        ]
    )

    for patient_folder in patient_data.keys():
        for exploration_folder in patient_data[patient_folder].keys():

            # Right eye
            if 'rts_noblinks' in patient_data[patient_folder][exploration_folder]:
                (
                    var_x, var_y,
                    sd_x, sd_y,
                    rms_x, rms_y,
                    avg_abs_dx, avg_abs_dy, avg_dxy
                ) = get_variance_rms_and_center_dists(
                    patient_data[patient_folder][exploration_folder]['rts_noblinks']
                )

                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'right',
                    'var_x': var_x,
                    'var_y': var_y,
                    'sd_x': sd_x,
                    'sd_y': sd_y,
                    'rms_x': rms_x,
                    'rms_y': rms_y,
                    'avg_abs_dist_x_center': avg_abs_dx,
                    'avg_abs_dist_y_center': avg_abs_dy,
                    'avg_dist_xy_center': avg_dxy
                }])

                variance_metrics_df = pd.concat([variance_metrics_df, new_row], ignore_index=True)

            # Left eye
            if 'lts_noblinks' in patient_data[patient_folder][exploration_folder]:
                (
                    var_x, var_y,
                    sd_x, sd_y,
                    rms_x, rms_y,
                    avg_abs_dx, avg_abs_dy, avg_dxy
                ) = get_variance_rms_and_center_dists(
                    patient_data[patient_folder][exploration_folder]['lts_noblinks']
                )

                new_row = pd.DataFrame([{
                    'patient_id': patient_folder,
                    'exploration_id': exploration_folder,
                    'eye': 'left',
                    'var_x': var_x,
                    'var_y': var_y,
                    'sd_x': sd_x,
                    'sd_y': sd_y,
                    'rms_x': rms_x,
                    'rms_y': rms_y,
                    'avg_abs_dist_x_center': avg_abs_dx,
                    'avg_abs_dist_y_center': avg_abs_dy,
                    'avg_dist_xy_center': avg_dxy
                }])

                variance_metrics_df = pd.concat([variance_metrics_df, new_row], ignore_index=True)

    return variance_metrics_df


def get_variance_rms_and_center_dists(data_df, center_x=0.5, center_y=0.5):
    x = data_df['x']
    y = data_df['y']

    var_x = x.var()
    var_y = y.var()

    sd_x = x.std()
    sd_y = y.std()

    rms_x = np.sqrt(np.mean(x ** 2))
    rms_y = np.sqrt(np.mean(y ** 2))

    avg_abs_dist_x_center = np.mean(np.abs(x - center_x))
    avg_abs_dist_y_center = np.mean(np.abs(y - center_y))
    avg_dist_xy_center = np.mean(
        np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    )

    return (
        var_x,
        var_y,
        sd_x,
        sd_y,
        rms_x,
        rms_y,
        avg_abs_dist_x_center,
        avg_abs_dist_y_center,
        avg_dist_xy_center
    )
