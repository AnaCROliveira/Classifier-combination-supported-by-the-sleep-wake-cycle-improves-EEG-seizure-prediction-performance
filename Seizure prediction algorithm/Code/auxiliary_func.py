"""
Code with auxiliary functions used in training and testing steps

@author: AnaCROliveira
"""

import numpy as np
from sklearn import preprocessing, feature_selection, svm, metrics, model_selection
from pandas import Timedelta as t


def construct_target(times_seizure, onset_seizure, SOP, SPH):
    
    begin_sph = onset_seizure - t(minutes = SPH)
    begin_sop = begin_sph - t(minutes = SOP)
    
    idx_sop = np.where(times_seizure>=begin_sop)
    idx_sph = np.where(times_seizure>=begin_sph)
    
    target = np.zeros(len(times_seizure), dtype=int)
    target[idx_sop] = 1
    target[idx_sph] = 2
        
    return target


def cut_sph(target):
    
    idx_sph = np.where(target==2)
    target = np.delete(target, idx_sph)
    
    return target


def class_balancing(target):
    
    # Define majority & minority classes (class with more samples vs. class with less samples)
    idx_class0 = np.where(target==0)[0]
    idx_class1 = np.where(target==1)[0]
    if len(idx_class1)>=len(idx_class0):
        idx_majority_class = idx_class1
        idx_minority_class = idx_class0
    elif len(idx_class1)<len(idx_class0):
        idx_majority_class = idx_class0
        idx_minority_class = idx_class1
    
    # Define number of samples of each group
    n_groups = len(idx_minority_class)
    n_samples = len(idx_majority_class)
    min_samples = n_samples//n_groups
    remaining_samples = n_samples%n_groups
    n_samples_per_group = [min_samples+1]*remaining_samples + [min_samples]*(n_groups-remaining_samples)
    
    # Select one sample from each group of the majority class
    idx_selected = []
    begin_idx = 0
    for i in n_samples_per_group:
        end_idx = begin_idx + i
        
        idx_group = idx_majority_class[begin_idx:end_idx]
        idx = np.random.choice(idx_group)
        idx_selected.append(idx)

        begin_idx = end_idx
        
    # Add samples from the minority class
    [idx_selected.append(idx) for idx in idx_minority_class]

    # Sort selected indexes to keep samples order
    idx_selected = np.sort(idx_selected)
    
    return idx_selected


def standardization(data):
    
    # Define scaler
    scaler = preprocessing.StandardScaler()
    # Apply fit
    scaler.fit(data)
    
    return scaler


def select_features(data, target, n_features):
        
    # Define feature selection
    selector = feature_selection.SelectKBest(score_func = feature_selection.f_classif, k = n_features)  
    # Apply feature selection
    selector.fit(data, target)
    
    return selector


def classifier(data, target, c_value):
    
    # Define svm model
    svm_model = svm.LinearSVC(C = c_value, dual = False)
    # Appy fit
    svm_model.fit(data, target)
    
    return svm_model


def classifier_weighted(data, target, weight, c_value): # Approach Pool_weights
    
    # Define svm model
    svm_model = svm.LinearSVC(C = c_value, dual = False)
    # Appy fit
    svm_model.fit(data, target, sample_weight = weight)
    
    return svm_model


def performance(target, prediction):
    
    tn, fp, fn, tp = metrics.confusion_matrix(target, prediction).ravel()
    sensitivity = tp/(tp+fn)  
    specificity = tn/(tn+fp)
    metric = np.sqrt(sensitivity*specificity)

    return sensitivity, specificity, metric


def performance_weighted(target, prediction, weight): # Approach Pool_weights
    
    tn, fp, fn, tp = metrics.confusion_matrix(target, prediction, sample_weight = weight).ravel()
    sensitivity = tp/(tp+fn)  
    specificity = tn/(tn+fp)
    metric = np.sqrt(sensitivity*specificity)
   
    return sensitivity, specificity, metric


def cross_validation(data, target, n_folds): # Approach Pool_exclusive
    
    # Define cross-validator
    skf = model_selection.StratifiedKFold(n_splits = n_folds)
    # Apply split
    folds = skf.split(data, target)
    
    return folds


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
    n_classes = 2
    for sample_i in range(predictions.shape[1]):
        n_predictions_per_class = [np.count_nonzero(predictions[:,sample_i]==i) for i in range(n_classes)]
        majority_class = np.argmax(n_predictions_per_class)
        final_prediction.append(majority_class)
    final_prediction = np.array(final_prediction)

    return final_prediction