import pandas as pd
import numpy as np
import os
import shutil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate

def create_sign_difference_column(data):
    data['difference'] = data['y'].diff().fillna(0)
    signs = np.sign(data['difference'])
    data['sign_change'] = (signs.diff() != 0).fillna(False).astype(int)
    return data

def mark_threshold_sequences(df, thresholds, sequence_lengths):

    df['abs_diff'] = df['difference'].abs()
    
    # Initialize columns for each threshold and sequence length combination outside the loops
    for sequence_length in sequence_lengths:
        for threshold in thresholds:
            column_name = f'SEQ: {sequence_length}, T: {threshold}'
            df[column_name] = 0
    
    # Process each threshold and sequence length
    for sequence_length in sequence_lengths:
        for threshold in thresholds:
            # Vectorized condition check for threshold
            threshold_met = (df['abs_diff'] > threshold).astype(int)
            
            # Efficient rolling sum calculation
            rolling_sum = threshold_met.rolling(window=sequence_length, min_periods=1).sum()
            
            # Update the specific column based on rolling sum
            df[f'SEQ: {sequence_length}, T: {threshold}'] = np.where(rolling_sum >= sequence_length, 1, 0).astype(int)
    
    return df
    
def calculate_envelope_diff(data, window, num_std):
    # Assuming data['y'] is a Pandas Series
    moving_std = data['y'].rolling(window=window, min_periods=1).std()
    
    # Calculate envelope width as 2 times the moving standard deviation times the number of standard deviations
    envelope_width = 2 * moving_std * num_std
    
    # Assign the result to the data DataFrame and handle missing values
    data['envelope'] = envelope_width.fillna(0)
    
    return data

def calculate_metrics(group):
    if len(group) > 1:
        X = group['x'].values.reshape(-1, 1)  # Predictor
        y = group['y'].values  # Response
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        me = np.mean(y - y_pred)
        slope = model.coef_[0]  # Extract the slope of the regression line#
    else:
        mse = me = slope = pd.NA  # Simultaneous assignment for clarity and slight performance boost#
    return pd.Series({'MSE': mse, 'ME': me, 'Slope': slope})

def analyze_correlations(df, electrode_column, signal_columns, agg_df):
    # Preprocess and group by interval first
    grouped = df.groupby('x_interval')
    
    # Initialize a list to store the correlation results for each signal column
    results = {col: [] for col in signal_columns}

    # Loop over each group
    for interval, data in grouped:
        electrodes = data[electrode_column].unique()
        n_electrodes = len(electrodes)

        # Initialize correlation matrix for each signal column 
        corr_matrices = {col: np.zeros((n_electrodes, n_electrodes)) for col in signal_columns}

        # Compute correlation for each pair of electrodes for each signal column
        for i, elec1 in enumerate(electrodes):
            for j in range(i + 1, n_electrodes):
                elec2 = electrodes[j]
                
                for col in signal_columns:
                    signal1 = data[data[electrode_column] == elec1][col].to_numpy()
                    signal2 = data[data[electrode_column] == elec2][col].to_numpy()

                    # Correlate and normalize
                    corr = correlate(signal1, signal2, mode='full')
                    corr = corr.astype(np.float64)  # Convert to float
                    max_corr = np.max(np.abs(corr))
                    if max_corr != 0:
                        corr /= max_corr
                    
                    # Store mean absolute correlation
                    mean_abs_corr = np.mean(np.abs(corr))
                    corr_matrices[col][i, j] = mean_abs_corr
                    corr_matrices[col][j, i] = mean_abs_corr

        # Compute mean across correlations, excluding self-correlation (diagonal)
        for col in signal_columns:
            mean_abs_corrs_interval = np.mean(corr_matrices[col], axis=0)
            results[col].append(mean_abs_corrs_interval)

    # Format the results into a structured output
    for col in results.keys():
        # Convert list of arrays into a single 2D array and flatten in electrode-wise order
        # results[col] = np.array(results[col]).T.flatten()
        agg_df[f'{col} corr'] = np.array(results[col]).T.flatten()

    return agg_df

def aggregate_for_specific_columns(df, num_int, columns, num_electrodes, obs_rate):
    scale_factor = int(len(df) / (num_int * num_electrodes))
    df['x_interval'] = pd.cut(df['x'], bins=num_int)
    agg_df = df.groupby(['Electrode', 'x_interval'])[columns].sum().reset_index()
    agg_df['y'] /= scale_factor
    agg_df[['MSE', 'ME', 'Slope']] = df.groupby(['Electrode', 'x_interval']).apply(calculate_metrics).reset_index()[['MSE', 'ME', 'Slope']]
    stats_df = df.groupby(['Electrode', 'x_interval'])['y'].agg(['var', 'mean', 'std', 'median', 'min', 'max']).reset_index()
    agg_df = agg_df.join(stats_df.set_index(['Electrode', 'x_interval']), on=['Electrode', 'x_interval'])
    agg_df['range'] = agg_df['max'] - agg_df['min']
    agg_df.drop(columns=['min', 'max'], inplace=True)

    agg_df['x_start'] = agg_df['x_interval'].apply(
        lambda x: int(round((x.left + x.right) / 2))).astype(int)
    #agg_df['x_start'] = agg_df.reset_index()['x_interval'].apply(
    #    lambda x: float(str(x).split(',')[0].strip('([]]'))).astype(float)
    agg_df['time_hms'] = pd.to_datetime(agg_df['x_start'] / obs_rate, unit='s').dt.floor('S')
    
    return agg_df.reset_index()

def create_aggregated_data_csv(aggregated_df, subject_number):
    file_path = f'aggregated_test_df_streamlit_{subject_number}.csv'
    aggregated_df.to_csv(file_path, index=False)

def create_aggregated_data_parquet(aggregated_df, subject_number):
    file_path = f'parquet_aggregated_data_{subject_number}'
    aggregated_df.to_parquet(f"{file_path}", compression='gzip')

def create_parquet_partitioned_data(filtered_df_sample, subject_number):
    directory_path = f"parquet_partitioned_{subject_number}"

    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

    DF = filtered_df_sample[['Electrode', 'y', 'x', 'time_hms']]
    DF['Electrode'] = DF['Electrode'].astype('category')
    DF['y'] = DF['y'].astype('float32')
    DF['x'] = DF['x'].astype('int32')
    DF.to_parquet(directory_path, partition_cols=['Electrode'], compression='gzip')