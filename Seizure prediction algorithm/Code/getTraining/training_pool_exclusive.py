"""
Code with functions to train the seizure prediction model
(Approach Pool_exclusive)

@author: AnaCROliveira
"""

import pickle
import numpy as np
import time as t
import os

# Scripts
from auxiliary_func import construct_target, cut_sph, class_balancing, standardization, select_features, classifier, performance, cross_validation


def main_train_pool_exclusive(patient, data_list, times_list, metadata_list, vigilance_list, SPH, SOP_list):
    
    # --- Configurations ---
    n_seizures = len(data_list)
    onset_list = [seizure['eeg_onset'] for seizure in metadata_list]

    models_awake = {}
    models_sleep = {}
    info_train = []
    for SOP in SOP_list:
        
        print(f'\n\nSOP={SOP}')
        
        # --- Construct target (0 - interictal | 1 - preictal) ---
        target_list = [construct_target(times_list[seizure], onset_list[seizure], SOP, SPH) for seizure in range(n_seizures)]

        # --- Cut SPH ---
        target = [cut_sph(target) for target in target_list]
        data = [data_list[seizure][:len(target[seizure])] for seizure in range(n_seizures)]
        vigilance = [vigilance_list[seizure][:len(target[seizure])] for seizure in range(n_seizures)] # 0 - awake | 1 - sleep

        # --- Concatenate seizures ---
        target = np.concatenate(target)
        data = np.concatenate(data)
        vigilance = np.concatenate(vigilance)
        
        # --- Divide data (awake & sleep) ---
        data_awake = data[vigilance==0]
        data_sleep = data[vigilance==1]
        target_awake = target[vigilance==0]
        target_sleep = target[vigilance==1]

        # --- Grid search | Train [Classifier Awake] ---
        best_k_awake, best_C_awake, ss_awake, sp_awake, best_metric_awake = grid_search(data_awake, target_awake, n_seizures)
        model_awake = train(data_awake, target_awake, best_k_awake, best_C_awake)

        # --- Grid search | Train [Classifier Sleep] ---
        best_k_sleep, best_C_sleep, ss_sleep, sp_sleep, best_metric_sleep = grid_search(data_sleep, target_sleep, n_seizures)
        model_sleep = train(data_sleep, target_sleep, best_k_sleep, best_C_sleep)

        # --- Save parameters & results ---
        models_awake[SOP] = model_awake
        models_sleep[SOP] = model_sleep
        info_train.append([SOP, best_k_awake, best_C_awake, ss_awake, sp_awake, best_metric_awake, best_k_sleep, best_C_sleep, ss_sleep, sp_sleep, best_metric_sleep])
     
    # --- Save models ---
    path = '../Results/Models/Pool_exclusive'
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(models_awake, open(f'{path}/model_awake_patient{patient}', 'wb'))   
    pickle.dump(models_sleep, open(f'{path}/model_sleep_patient{patient}', 'wb'))    
    
    return info_train
        

def grid_search(data, target, n_seizures):
    
    print('\nGrid search...')
    
    # --- Configurations ---
    k = np.arange(start = 10, stop = 51, step = 10) # number of features
    c_pot = np.arange(start = -20, stop = 13, step = 2, dtype=float)
    C = 2**c_pot # parameter of SVM classifier
    n_classifiers = 31 # number of classifiers (odd to avoid ties)
    
    performances = []
    for k_i in k:
        for C_i in C:
            print(f'\n--------- k = {k_i} | C = {C_i:.2g} ---------')
            start_time_combination = t.time() # to compute time that each combination k,C lasts
    
            ss_per_combination = []
            sp_per_combination = []
            metric_per_combination = []     
            
            # --- Cross validation ---
            folds = cross_validation(data, target, n_seizures)
            
            for idx_train, idx_validation in folds:
                for classifier_i in range(n_classifiers):
                    
                    # --- Data splitting (train & validation) ---
                    set_validation = data[idx_validation]
                    set_train = data[idx_train]
                    target_validation = target[idx_validation]
                    target_train = target[idx_train]
                    
                    # --- Class balancing ---
                    idx_selected = class_balancing(target_train)
                    set_train = set_train[idx_selected]
                    target_train = target_train[idx_selected]

                    # --- Standardization ---
                    scaler = standardization(set_train)
                    set_train = scaler.transform(set_train)
                    set_validation = scaler.transform(set_validation)

                    # --- Feature selection ---
                    selector = select_features(set_train, target_train, k_i)      
                    set_train = selector.transform(set_train)
                    set_validation = selector.transform(set_validation) 
                    
                    # --- Train (training set) ---
                    svm_model = classifier(set_train, target_train, C_i)
                    
                    # --- Test (validation set) ---
                    prediction_validation = svm_model.predict(set_validation)
                     
                    # --- Performance ---
                    ss, sp, metric = performance(target_validation, prediction_validation)
                    
                    ss_per_combination.append(ss)
                    sp_per_combination.append(sp)
                    metric_per_combination.append(metric)
                    
            ss_avg = np.mean(ss_per_combination)
            sp_avg = np.mean(sp_per_combination)
            metric_avg = np.mean(metric_per_combination)
            print(f'Average performance: SS = {ss_avg:.2f} | SP = {sp_avg:.2f} | metric: {metric_avg:.2f}')
    
            end_time_combination = t.time()
            run_time = end_time_combination - start_time_combination
            print(f'Running time per combination = {run_time:.2f}')
            
            performances.append([k_i, C_i, ss_avg, sp_avg, metric_avg, run_time])
            
    performances = np.array(performances)
        
    # --- Select best parameters ---
    # Best performance (maximum metric)
    best_performance = max(performances[:,4])
    idx_best = np.where(performances[:,4] == best_performance)[0] # get array of indexes
    # Tiebreaker (minimum running time)          
    if len(idx_best)>1:
        tiebreaker_performance = min(performances[idx_best,5])
        idx_best_tie = np.where(performances[idx_best,5] == tiebreaker_performance)[0][0] # get index
        idx_best = idx_best[idx_best_tie]
    else:
        idx_best = idx_best[0]
    
    # --- Save selected parameters & results ---
    best_k = performances[idx_best,0]
    best_C = performances[idx_best,1]
    ss = performances[idx_best,2]
    sp = performances[idx_best,3]
    best_metric = performances[idx_best,4]  # best_performance
    
    print('\nGrid search completed')
    print(f'\n --------------- GRID SEARCH (best result) --------------- \nk = {best_k:.2f} | C = {best_C:.2f} | SS = {ss:.2f} | SP = {sp:.2f} | metric = {best_metric:.2f}')
        
    return best_k, best_C, ss, sp, best_metric


def train(data, target, best_k, best_C):
    
    print('\n\nTraining classifier...')
    start_time = t.time()

    n_classifiers = 31
    scaler_list = []
    selector_list = []
    svm_list = []
    for classifier_i in range(n_classifiers):
    
        # --- Class balancing ---
        idx_selected = class_balancing(target)
        data_train = data[idx_selected]
        target_train = target[idx_selected]
        
        # --- Standardization ---
        scaler = standardization(data_train)
        data_train = scaler.transform(data_train)

        # --- Feature selection ---
        selector = select_features(data_train, target_train, int(best_k))
        data_train = selector.transform(data_train)
    
        # --- Train (training set) ---
        svm_model = classifier(data_train, target_train, best_C)
        
        # --- Save --- 
        scaler_list.append(scaler)
        selector_list.append(selector)
        svm_list.append(svm_model)
                    
    # --- Model ---
    model = {}
    model['scaler'] = scaler_list
    model['selector'] = selector_list
    model['svm'] = svm_list

    end_time = t.time()
    run_time = end_time - start_time
    print(f'running time: {run_time:.2f}')
    print('Classifier trained')

    return model
    