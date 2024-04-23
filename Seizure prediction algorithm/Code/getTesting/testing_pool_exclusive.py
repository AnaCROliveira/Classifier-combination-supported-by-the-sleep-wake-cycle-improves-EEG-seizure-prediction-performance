"""
Code with functions to test the seizure prediction model
(Approach Pool_exclusive)

@author: AnaCROliveira
"""

import pickle
import numpy as np

# Scripts
from auxiliary_func import construct_target, cut_sph, test, performance
from regularization import get_firing_power, alarm_generation, alarm_processing
from evaluation import alarm_evaluation, sensitivity, FPR_h, statistical_validation
from plot_results import fig_test, fig_temporality


def main_test_pool_exclusive(patient, data_list, times_list, metadata_list, vigilance_list, SPH, window_size):
    
    print('\nTesting...')

    # --- Load models ---  
    models_awake = pickle.load(open(f'../Results/Models/Pool_exclusive/model_awake_patient{patient}', 'rb'))
    models_sleep = pickle.load(open(f'../Results/Models/Pool_exclusive/model_sleep_patient{patient}', 'rb'))  
    
    # --- Configurations ---
    SOP_list = models_awake.keys()
    onset_list = [seizure['eeg_onset'] for seizure in metadata_list]
    n_seizures = len(data_list)
    firing_power_threshold = 0.5
    
    info_test = []
    for SOP in SOP_list:
        
        # --- Model (awake & sleep) ---
        model_awake = models_awake[SOP]
        model_sleep = models_sleep[SOP]
        
        target_per_seizure = []
        prediction_per_seizure = []
        alarm_per_seizure = []
        refractory_samples_per_seizure = []
        for seizure in range(n_seizures):
        
            # --- Construct target (0 - interictal | 1 - preictal) ---
            target = construct_target(times_list[seizure], onset_list[seizure], SOP, SPH)
            
            # --- Cut SPH ---
            target = cut_sph(target)
            data = data_list[seizure][:len(target)]
            times = times_list[seizure][:len(target)]
            vigilance = vigilance_list[seizure][:len(target)]
        
            # --- Divide data (awake & sleep) ---
            idx_awake = np.where(vigilance==0)[0]
            idx_sleep = np.where(vigilance==1)[0]
            data_awake = data[idx_awake]
            data_sleep = data[idx_sleep]
    
            # --- Test (awake & sleep) ---
            prediction = np.full(len(data), float('nan'))
            if len(idx_awake)!=0:
                prediction_awake = test(data_awake, model_awake)
                prediction[idx_awake] = prediction_awake
            if len(idx_sleep)!=0:
                prediction_sleep = test(data_sleep, model_sleep)
                prediction[idx_sleep] = prediction_sleep
                
            # --- Regularization [Firing Power + Alarms] ---
            firing_power = get_firing_power(prediction, times, SOP, window_size)
            alarm = alarm_generation(firing_power, firing_power_threshold)
            alarm, refractory_samples = alarm_processing(alarm, times, SOP, SPH)
            
            # --- Save per seizures ---
            target_per_seizure.append(target)
            prediction_per_seizure.append(prediction)
            alarm_per_seizure.append(alarm)
            refractory_samples_per_seizure.append(refractory_samples)
            
            # --- Figure: test ---
            fig_test(patient, seizure, SOP, times, target, prediction, firing_power, firing_power_threshold, alarm, vigilance, 'Pool_exclusive')
            fig_temporality(patient, seizure, SOP, SPH, times, onset_list[seizure], target, prediction, firing_power, firing_power_threshold, alarm, vigilance, window_size, 'Pool_exclusive')

        # --- Concatenate seizures ---
        target = np.concatenate(target_per_seizure)
        prediction = np.concatenate(prediction_per_seizure)
        alarm = np.concatenate(alarm_per_seizure)
        refractory_samples = np.concatenate(refractory_samples_per_seizure)
        
        # --- Performance [samples] ---
        ss_samples, sp_samples, metric = performance(target, prediction)
        
        # --- Performance [alarms] ---
        true_alarm, false_alarm = alarm_evaluation(target, alarm)
        ss = sensitivity(true_alarm, n_seizures)
        FPRh = FPR_h(target, false_alarm, refractory_samples, window_size)

        # --- Statistical validation ---
        surr_ss_mean, surr_ss_std, tt, pvalue = statistical_validation(target_per_seizure, alarm, ss, firing_power_threshold)

        # --- Save parameters & results ---
        info_test.append([ss_samples, sp_samples, firing_power_threshold, true_alarm, false_alarm, ss, FPRh, surr_ss_mean, surr_ss_std, tt, pvalue])
        print(f'\n--- TEST PERFORMANCE [SOP={SOP}] --- \nSS = {ss:.3f} | FPR/h = {FPRh:.3f}')
        print(f'--- Statistical validation ---\nSS surr = {surr_ss_mean:.3f} ± {surr_ss_std:.3f} (p-value = {pvalue:.4f})')

    print('\nTested\n\n')
        
    return info_test