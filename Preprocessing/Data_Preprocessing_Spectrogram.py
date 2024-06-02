import pandas as pd
import numpy as np
#import initial_preprocessing as sf

def spectrogram_matlabCopy(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided'):

    """
    light weight approach to spectrogram calculation. basically a slightly polished version of scipy.signal._fft_helper
    Syntax: times,  freqs, Sxx = spectrogram_lightweight(x=None, win=None, noverlap=None, nfft=None,fs=None, sides='onesided')
    sides='onesided' or 'twosided'
    noverlap is in samples
    nfft is in samples (if  nfft is larger than win, then the signal is zero padded)
    """
    assert x is not None
    assert win is not None
    assert noverlap is not None
    assert nfft is not None
    assert fs is not None

    if np.isscalar(win):
        nperseg=win
        win=np.ones(nperseg)
    else:
        nperseg=len(win)

    assert len(win)==nperseg

    # make sure x is a 1D array, with optional singular dimensions
    assert len(x)==x.shape[-1]

    #make sure data fits cleanly into segments#
    assert (len(x)-nperseg) % (nperseg-noverlap) == 0

    # Created strided array of data segments
    # https://stackoverflow.com/a/5568169

    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step,nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1],x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x,shape=shape,strides=strides)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically

    if sides == 'twosided':
        func = np.fft.fft
        freqs = np.fft.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        result = result.real
        func = np.fft.rfft
        freqs = np.fft.rfftfreq(nfft,1/fs)
    else:
        raise ValueError('sides must be twosided or onesided')

    result = func(result,nfft)
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1, nperseg - noverlap)/float(fs)

    return result,freqs,time

def create_spectrogram_dataframe_all_electrodes(data, obs_rate):
    df_list = []
    for electrode in data['Electrode'].unique():

        test_data = data.loc[data['Electrode'] == electrode]
        input_data = test_data['y'].to_numpy()
        result, freqs, time = spectrogram_matlabCopy(x=input_data, win=obs_rate*2, noverlap=obs_rate, nfft=512, fs=obs_rate, sides='onesided')

        value = np.abs(result)**2

        # Generate mesh grid for time and frequency
        time_grid, frequency_grid = np.meshgrid(time, freqs, indexing='ij')

        # Flatten the meshes and the values array
        time_flat = time_grid.flatten()
        frequency_flat = frequency_grid.flatten()
        values_flat = value.flatten()

        # Create DataFrame
        df = pd.DataFrame({
            'time': time_flat,
            'frequency': frequency_flat,
            'value': values_flat,
            'Electrode': [electrode]*len(time_flat)
        })
        df_list.append(df)
        print(f'Finished electrode: {electrode}')

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df

def aggregate_dataframe(df, num_bins):
    num_bins = 3000
    df['time_interval'] = pd.cut(df['time'], bins=num_bins)
    df['time'] = df['time_interval'].apply(lambda x: int(round((x.left + x.right) / 2))).astype(int)
    df = df[['Electrode', 'frequency', 'time', 'value']]
    df = df.groupby(['Electrode', 'time', 'frequency']).sum().reset_index()
    return df
    
def create_spectrogram_dataframe_all_electrodes(data, obs_rate, channel_names):#
    
    df_list = []
    
    for i in range(data.shape[0]):
        
        # Spectrogram calculation
        input_data = data[i, :7000000]
        value, freqs, time = spectrogram_matlabCopy(x=input_data, win=obs_rate*2, noverlap=obs_rate, nfft=512, fs=obs_rate, sides='onesided')
        value = np.abs(value)**2

        # Generate mesh grid for time and frequency
        time_grid, frequency_grid = np.meshgrid(time, freqs, indexing='ij')

        # Flatten the meshes and the values array
        time_flat = time_grid.flatten()
        frequency_flat = frequency_grid.flatten()
        values_flat = value.flatten()

        # Create and aggregate the Pandas dataFrame
        df = pd.DataFrame({'time': time_flat,'frequency': frequency_flat,'value': values_flat, 'Electrode': [channel_names[i]]*len(time_flat)})
        df = aggregate_dataframe(df, 3000)
        df = df.loc[(df['frequency'] <= 100) & (df['frequency'] >= 0.1)]
        df['time_hms'] = pd.to_datetime(df['time'], unit='s')
        
        df_list.append(df)
        print(f'Finished electrode: {i}')

    final_df = pd.concat(df_list, ignore_index=True)
    return final_df