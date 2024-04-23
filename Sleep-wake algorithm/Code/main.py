"""
Code to execute all steps of the sleep-wake detection pipeline

@author: AnaCROliveira
"""

import numpy as np
import time as t
ti = t.time()

# Scripts
from import_data import import_data
from pre_processing import filtering, downsampling
from feature_extraction import feature_extraction
from splitting import splitting
from training import training
from testing import testing
from save_results import save_results, save_model, save_best_model
from plot_results import fig_performance, fig_performance_per_patient, fig_feature_selection


# %% CONFIGURATIONS

patients_list = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40]
chosen_fs = 128 
channels = ['FP2-F4','F4-C4','C4-P4','P4-O2','FP1-F3','F3-C3','C3-P3','P3-O1','F8-T4','T4-T6','F7-T3','T3-T5']
n_runs = 30
percentage_train = 70 # 0-100 (%)


# %% IMPORT DATA | FILTERING | DOWNSAMPLING | FEATURE EXTRACTION

data_list = []
target_list = []
for patient in patients_list:
    
    # --- Import data --- 
    data, target, fs, window_size = import_data(patient, channels)

    # --- Filtering data ---
    data = filtering(data, fs)
    
    # --- Downsampling data ---
    data = downsampling(data, fs, chosen_fs)
    
    # --- Feature extraction ---
    data, features_names = feature_extraction(data, chosen_fs, window_size)
    
    data_list.append(data)
    target_list.append(target)

    
# %% DATA SPLITTING | TRAIN | TEST

info_per_run = []
info_per_patient = {patient:[] for patient in patients_list}
for run in range(1, n_runs+1): 
    
    print(f'\n\n==================== RUN {run} ====================')
    
    # --- Splitting data ---
    data, target, patients = splitting(data_list, target_list, patients_list, percentage_train)

    # --- Train ---
    model, info_train = training(data['train'], target['train'])
    
    # --- Test ---
    info_test = testing(data['test'], target['test'], model)
     
    # --- Save model & results per run ---
    save_model(model, f'model{run}')
    info_general = [run, str(patients['train']), str(patients['test'])]
    info_per_run.append(info_general + info_train + info_test) # info for excel


    # %% PER PATIENT
    performance_per_patient = []
    for i in range(len(patients['test'])):   
        info_test = testing(data['test'][i], target['test'][i], model)
        
        patient = patients['test'][i]
        info_per_patient[patient].append(info_test)
        
        performance_per_patient.append(info_test)

    # --- Figure: performance per patient (run i) ---
    labels = [f'patient {i}' for i in patients['test']]
    fig_performance(performance_per_patient, labels, name = f'patient_run{run}', option = '')
    

# %% SAVE FINAL RESULTS

# --- Select & save best model from all runs ---
idx_best_run = save_best_model(info_per_run)
info_per_run[idx_best_run][0] = f'{idx_best_run+1}*'

# --- Save final results  (excel) ---
save_results(info_per_run)

# --- Figure: performance per run ---
labels = [f'run {i}' for i in np.array(info_per_run)[:,0]]
fig_performance(info_per_run, labels, name = 'run', option = 'final')

# --- Figure: final performance per patient (all runs) ---
fig_performance_per_patient(info_per_patient)

# --- Figure: selected features ---
fig_feature_selection(n_runs, idx_best_run)


tf = t.time()
run_time = tf-ti
print(f'\n\nRunning time: {run_time:.2f}')
