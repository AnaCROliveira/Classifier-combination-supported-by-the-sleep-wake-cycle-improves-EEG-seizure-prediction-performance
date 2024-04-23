"""
Code with functions to import data, 
re-reference from monopolar to bipolar montage,
and import seizures information

@author: AnaCROliveira
"""

import os
import numpy as np
import pandas


def read_metadata(path):
    
    filename = 'all_seizure_information.pkl'
    file_path = os.path.join(path,filename)

    # Import metadata information
    information = np.load(file_path,allow_pickle=True)
    
    description = ['eeg_onset', 'eeg_offset', 'pattern', 'classification', 'vigilance', 'medicament', 'dosage']
    metadata_list = []
    for seizure in range(len(information)):
        info_per_seizure = {}
        for field in range(len(description)):
            if description[field]=='eeg_onset' or description[field]=='eeg_offset':
                info_per_seizure[description[field]] = pandas.to_datetime(information[seizure][field], unit='s')
            else:
                info_per_seizure[description[field]] = information[seizure][field]
        metadata_list.append(info_per_seizure)
        
    return metadata_list


def import_data(path, seizure, bipolar_channels):
    
    print('\nLoading data...')
    data_path = os.path.join(path, f'seizure_{seizure}_data.npy')
    times_path = os.path.join(path, f'feature_datetimes_{seizure}.npy')
    
    # Load signals & information
    data = np.load(data_path) # data: nºwindows x nºsamples per window (fs*old_window_size) x nºchannels
    channels = np.load('../EPILEPSIAE/channel_names.pkl',allow_pickle=True) # monopolar channels
    fs = 256 # sampling frequency
    times = np.load(times_path) # vector of times
    
    # Compute window size (#seconds)
    times = np.array([pandas.to_datetime(time, unit='s') for time in times])
    window_size = 30
    
    # Delete windows with gaps in the middle
    old_window_size = 5 # seconds
    step = int(window_size/old_window_size) 
    idx_deleted = []
    times_seizure = []
    for i in range(0, len(times)-step+1, step):
        seconds_diff = (times[i+step-1]-times[i]).seconds
        if seconds_diff>window_size:
            idx_deleted.append(np.arange(i,i+step))
        else:
            times_seizure.append(times[i+int(step/2)]) # set middle time of each new window
    data = np.delete(data, idx_deleted, axis=0)
    # Set structures
    data = np.reshape(data, (data.shape[0]*data.shape[1],data.shape[2])) # data: nºsamples x nºchannels
    data = data.T # data: nºchannels x nºsamples
    times_seizure = np.array(times_seizure)

    # Monopolar to bipolar montage
    data_seizure = []
    for channel in bipolar_channels:
        first_channel = channel.split('-')[0]
        second_channel = channel.split('-')[1]
        data_seizure.append(data[channels.index(first_channel)]-data[channels.index(second_channel)])
    data_seizure = np.array(data_seizure)
    
    print('Data loaded')
    
    return data_seizure, times_seizure, fs, window_size