"""
Code with a function to split data into train and test

@author: AnaCROliveira
"""

import numpy as np

def splitting(data_list, target_list, patients_list, percentage_train):
    
    n_patients_train = int(percentage_train/100 * len(patients_list))

    # --- Divide patients randomly into train and test ---
    idx_rand = np.random.permutation(len(patients_list))
    
    idx_train = idx_rand[:n_patients_train]
    idx_test = idx_rand[n_patients_train:]
    
    data = {}
    target = {}
    patients = {}
  
    # --- Train ---
    data['train'] = [data_list[idx] for idx in idx_train]
    target['train'] = [target_list[idx] for idx in idx_train]
    patients['train'] = [patients_list[idx] for idx in idx_train]
    
    # --- Test ---
    data['test'] = [data_list[idx] for idx in idx_test]
    target['test'] = [target_list[idx] for idx in idx_test]
    patients['test'] = [patients_list[idx] for idx in idx_test]
    
    return data, target, patients