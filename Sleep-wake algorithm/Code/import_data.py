"""
Code with a function to import data (edf files, txt annotations) 
and construct the target

@author: AnaCROliveira
"""

import pyedflib as edf
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as t


def import_data(patient, chosen_channels):
    
    print(f'\n\n--- Patient {patient} ---')

    # %% LOAD DATA (edf)
    
    print('\nLoading data...')
    
    # Import edf file
    path_edf = f'../CAP SLEEP/Data/nfle{patient}.edf'
    f = edf.EdfReader(path_edf)
        
    # Get information
    n_channels = f.signals_in_file
    all_channels = f.getSignalLabels()
    all_fs = f.getSampleFrequencies()
    begin_signal = f.getStartdatetime()
    duration_signal = f.getFileDuration()
    end_signal = begin_signal + t(seconds=duration_signal)
    
    # Load signals
    data = []
    for i in range(n_channels):
        if all_channels[i].upper() in chosen_channels:
            fs = int(all_fs[i])
            signal = f.readSignal(i)
            data.append(signal)
    # data: nºchannels x nºsamples
    data = np.array(data)

    f.close()
    print('Data loaded')
    
    
    # %% IMPORT ANNOTATIONS (txt)
    
    # Import txt file
    path_txt = f'../CAP SLEEP/Annotations/nfle{patient}.txt'
    annotations = np.loadtxt(path_txt,dtype=str,skiprows=20,delimiter="\t")
    
    # Find columns
    column_time = np.argwhere(annotations[0]=='Time [hh:mm:ss]')[0][0]
    column_duration = np.argwhere(annotations[0]=='Duration[s]')[0][0]
    column_stage = np.argwhere(annotations[0]=='Sleep Stage')[0][0]
    column_event = np.argwhere(annotations[0]=='Event')[0][0]
    # Delete description row
    annotations = annotations[1:]
    
    # Collect annotations only from sleep stages
    idx = np.argwhere([annotations[i,column_event].startswith('SL') for i in range(annotations.shape[0])])
    sleep_annotations = np.concatenate([annotations[i] for i in idx])
    last_idx = idx[-1][0]
    
    # Determine duration of annotations
    begin_annot = dt.strptime(annotations[0,column_time],'%H:%M:%S')
    end_annot = dt.strptime(annotations[last_idx,column_time],'%H:%M:%S')
    duration_annot = (end_annot-begin_annot).seconds
           
    
    # %% CONSTRUCT TARGET (0 - awake | 1 - sleep)
    
    target = []
    label = 0
    duration_label = int(sleep_annotations[0,column_duration])
    for i in range(sleep_annotations.shape[0]):
              
        # Compute duration between i-1 and i
        begin_time = dt.strptime(sleep_annotations[i-1,column_time],'%H:%M:%S')
        end_time = dt.strptime(sleep_annotations[i,column_time],'%H:%M:%S')
        duration = (end_time-begin_time).seconds

        # Add missing labels (between i-1 and i)
        if duration!=duration_label and i!=0:
            [target.append(label) for n_labels in range(duration_label,duration,duration_label)]
        
        # Add current label
        if sleep_annotations[i,column_stage]=='W':
            label = 0 # awake
        elif sleep_annotations[i,column_stage] in ['S1','S2','S3','S4','R']:
            label = 1 # sleep
        target.append(label)
    target = np.array(target)
    
    
    # %% MATCH DATA AND ANNOTATIONS
    
    # Compute difference between begin/end time of annotations and signal
    begin_annot = dt.combine(begin_signal.date(),begin_annot.time())
    end_annot = dt.combine(end_signal.date(),end_annot.time())

    diff_begin_time = (begin_annot-begin_signal).seconds
    diff_end_time = (end_annot+t(seconds=duration_label)-end_signal).seconds
    
    # Delete signal parts without annotations
    begin_idx_signal = diff_begin_time*fs
    end_idx_signal = begin_idx_signal + (duration_annot+duration_label)*fs
    data = data[:,begin_idx_signal:end_idx_signal]
    
    # Delete additional annotations existing beyond the signal
    if end_signal < end_annot+t(seconds=duration_label):
        n_extra_labels = int(np.ceil(diff_end_time/duration_label))
        target = target[:-n_extra_labels]

    return data, target, fs, duration_label