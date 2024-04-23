"""
Code with functions to test the sleep-wake model, 
including a majority voting system and performance computation

@author: AnaCROliveira
"""

import numpy as np

# Scripts
from auxiliary_func import performance


def testing(data, target, model):
    
    show = False
    if type(data)==list:
        # --- Concatenate patients ---
        data = np.concatenate(data)
        target = np.concatenate(target)
        show = True
        print('\nTesting...')

    # --- Test ---
    prediction = test(data, model)
    
    # --- Performance ---
    ss, sp, metric = performance(target, prediction)
    
    # --- Save results ---
    info_test = [ss, sp, metric]
    
    if show:
        print(f'Tested\n\n--- TEST PERFORMANCE --- \n SS = {ss:.3f} | SP = {sp:.3f}')

    return info_test


def test(data, model):
    
    # --- Model ---
    scaler_list = model['scaler']
    selector_list = model['selector']
    svm_list = model['svm']
    
    # --- Ensemble classifiers ---
    predictions = []
    n_classifiers = len(svm_list)
    for classifier_i in range(n_classifiers):
        
        scaler = scaler_list[classifier_i]
        selector = selector_list[classifier_i]
        svm_model = svm_list[classifier_i]
            
        # --- Standardization ---
        data_test = scaler.transform(data)
        
        # --- Feature selection ---
        data_test = selector.transform(data_test)
        
        # --- Test ---
        prediction = svm_model.predict(data_test)
    
        predictions.append(prediction)
    predictions = np.array(predictions)
    
    # --- Classifiers vote ---
    final_prediction = []            
    min_n_classifiers = int(n_classifiers/2)
    for sample_i in range(predictions.shape[1]):
        
        if np.count_nonzero(predictions[:,sample_i]==0) > min_n_classifiers:
            final_prediction.append(0)
        else:
            final_prediction.append(1)            
    final_prediction = np.array(final_prediction)

    return final_prediction