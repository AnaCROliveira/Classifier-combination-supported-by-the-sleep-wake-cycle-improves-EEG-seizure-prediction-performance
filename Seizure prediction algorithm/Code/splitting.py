"""
Code with a function to split data into train and test

@author: AnaCROliveira
"""

def splitting(data_list, times_list, metadata_list, vigilance_list, n_seizures_train):
    
    data = {}
    times = {}
    metadata = {}
    vigilance = {}
    
    # Train
    data['train'] = data_list[:n_seizures_train]
    times['train'] = times_list[:n_seizures_train]
    metadata['train'] = metadata_list[:n_seizures_train]
    vigilance['train'] = vigilance_list[:n_seizures_train]
    
    # Test
    data['test'] = data_list[n_seizures_train:]
    times['test'] = times_list[n_seizures_train:]
    metadata['test'] = metadata_list[n_seizures_train:]
    vigilance['test'] = vigilance_list[n_seizures_train:]
    
    return data, times, metadata, vigilance