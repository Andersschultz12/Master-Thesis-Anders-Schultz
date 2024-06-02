import pandas as pd
import numpy as np
import mne

def load_filtered_eeg_data_using_mne(set_file_path, obs_rate, low_freq, high_freq, num_electrodes):
    
    # Load the .set file
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
    
    # Get names of first x channels
    number_of_channels = raw.ch_names[:num_electrodes]
    
    # filter with respect to Hz and sample w.r.t obs_rate.
    raw.pick_channels(number_of_channels)
    raw.filter(l_freq=0.1, h_freq=100)
    raw.resample(obs_rate)

    ## Setting average reference
    #raw.set_eeg_reference('average', projection=False)  # This sets the reference to the average of all selected channels

    # Retrieve channel names from the info structure
    channel_names = raw.info['ch_names']

    # Get numpy array of raw data.
    data = raw.get_data().astype(np.float32)
    del raw

    return data, channel_names

def remove_high_nan_rows_cols(data, channel_names, threshold=0.4):

    print(f'Original NaN count: {np.sum(np.isnan(data))}')
    print(f'Original shape data: {data.shape}, original shape channel_names: {len(channel_names)}')
    
    # Calculate the number of NaNs in each row and each column
    nan_row_counts = np.isnan(data).sum(axis=1)  # Number of NaNs in each row
    nan_col_counts = np.isnan(data).sum(axis=0)  # Number of NaNs in each column

    # Find rows and columns where the number of NaNs exceeds the threshold
    rows_to_delete = nan_row_counts > (data.shape[1] * threshold)
    cols_to_delete = nan_col_counts > (data.shape[0] * threshold)
    del nan_col_counts, nan_row_counts

    print(f'Columns to delete: {np.sum(cols_to_delete)}, Rows to delete: {np.sum(rows_to_delete)}')

    # Get the indices of the rows to be deleted
    row_indices_to_delete = np.where(rows_to_delete)[0]
    print(f'Indices of rows to be deleted: {row_indices_to_delete}')

    # Remove the indices for channels that exceed the threshold
    channel_names = [item for idx, item in enumerate(channel_names) if idx not in row_indices_to_delete]

    # Remove the rows and columns that exceed the threshold
    data_cleaned = np.delete(data, np.where(rows_to_delete), axis=0)
    data_cleaned = np.delete(data_cleaned, np.where(cols_to_delete), axis=1)
    del rows_to_delete, cols_to_delete

    print(f'New data shape post cleaning: {data_cleaned.shape}, New channel_name shape post cleaning: {len(channel_names)}')

    return data_cleaned, np.array(channel_names)

def select_subsection_of_data(data, filter_type, filter_amount=200, interval_start = 0, interval_end = 300):
    '''
    Select either an interval or sample of the data.
    Input: Data, filter type and keyword arguments for interval or filter amount. 
    Returns: Filtered pandas dataset.
    '''
    
    if filter_type != "interval":
        data = data[:, ::filter_amount] 
    else:
        data = data[:, interval_start:interval_start+interval_end]

    return data

def transform_dataset_for_visualization(data, filter_type, channel_names, filter_amount = 200, interval_start = 0, interval_end = 300, obs_rate = 200):

    # Sub-select data if specified
    if (interval_end-interval_start) < data.shape[1]:
        data = select_subsection_of_data(data, filter_type, filter_amount=filter_amount, interval_start=interval_start, interval_end=interval_end)
    
    # Create a pandas dataframe of the data.
    data = pd.DataFrame(data.T, columns=channel_names)
    data = data.reset_index().rename(columns={'index': 'x'})

    # If data is proportionately sampled, adjust x values to match.
    if filter_type == 'sample':
        data['x'] = data['x']*filter_amount

    # Get data into Python-Altair supported structure.
    data = data.melt(id_vars=['x'], var_name='Electrode', value_name='y')

    # Sub-select based on electrode input.
    #data = data.loc[data['Electrode'].isin(channel_names[0:num_electrodes])]

    # Create a datetime column based on x
    data['time_hms'] = pd.to_datetime(data['x'] / obs_rate, unit='s')
    
    return data

def interpolate_nan_values(data):

    for i in range(data.shape[0]):  
        channel_data = data[i]
        nans = np.isnan(channel_data)
        not_nans = ~nans
        interpolated = np.interp(x=np.where(nans)[0], xp=np.where(not_nans)[0], fp=channel_data[not_nans])
        data[i, nans] = interpolated

    return data





def make_timestamps_for_xaxis(data, filter_amount, obs_rate=200):

    data['time_hms'] = data['x']*filter_amount
    data['time_hms'] = pd.to_datetime(data['time_hms'] / obs_rate, unit='s')

    return data

def make_timescale_for_yaxis(data, filter_amount, obs_rate=200):

    data['time_scale'] = data['y'] * filter_amount / obs_rate

    return data

def compute_electrode_means_data(array_2d):

    column_means = np.mean(array_2d, axis=0)
    result_array = array_2d - column_means

    return result_array