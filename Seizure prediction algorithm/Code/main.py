"""
Code to execute all steps of the seizure prediction pipeline

@author: AnaCROliveira
"""

import os
import time as t
ti = t.time()

# Scripts
from import_data import import_data, read_metadata
from pre_processing import filtering, downsampling
from feature_extraction import feature_extraction
from vigilance import get_vigilance
from splitting import splitting
from main_train import main_train
from main_test import main_test
from save_results import save_train_results, read_train_results, select_final_result, save_results, save_final_results
from plot_results import fig_performance, fig_performance_per_patient, fig_final_performance, fig_feature_selection


# %% CONFIGURATIONS

# --- Approach option ---
approach = 'Control' # Standard 
#approach = 'Feature_state' # Extra feature (vigilance)
#approach = 'Pool_weights' # 2 classifiers (awake & sleep) with weights
#approach = 'Pool_exclusive' # 2 classifiers (awake & sleep)
#approach = 'Threshold_state' # 2 different firing power thresholds (awake & sleep)
#approach = 'Threshold_transitions' # Firing power threshold varying in vigilance transitions

# --- Mode option ---
#mode = 'Train'
#mode = 'Test'
mode = 'Train&Test'

# --- Pre-processing options ---
chosen_fs = 128 
bipolar_channels = ['FP2-F4','F4-C4','C4-P4','P4-O2','FP1-F3','F3-C3','C3-P3','P3-O1','F8-T4','T4-T6','F7-T3','T3-T5'] # from ASSC
# Translate channels nomenclature [T3->T7; T4->T8; T5->P7; T6->P8]
bipolar_channels = [channel.replace('T3','T7').replace('T4','T8').replace('T5','P7').replace('T6','P8') for channel in bipolar_channels]

# --- Train options ---
n_seizures_train = 3
SPH = 5
SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# --- Patients ---
patients = [202, 11002, 12702, 30802, 46702, 50802, 53402, 58602, 75202, 81402, 85202, 93902, 102202, 103802, 104602, 109202, 113902]
# approach Pool_exclusive (all except 202, 12702, 30802, 50802, 102202)
#patients = [11002, 46702, 53402, 58602, 75202, 81402, 85202, 93902, 103802, 104602, 109202, 113902]


# %% IMPORT DATA | FILTERING | DOWNSAMPLING | FEATURE EXTRACTION

print(f'\n\n==================== {approach.upper()} ====================')

information_general = {}
information_train = {}
information_test = {}
final_information = []
directory = '../EPILEPSIAE/Data'
for patient in patients:
    folder = f'pat_{patient}_splitted'
    path = os.path.join(directory,folder)
    patient = folder.split('_')[1]
    print(f'\n\n----- Patient {patient} -----')

    # --- Import seizures information ---
    metadata_list = read_metadata(path)
    
    data_list = []
    times_list = [] 
    vigilance_list = []
    for seizure in range(len(metadata_list)): # each file corresponds to a seizure
        print(f'\n\n- seizure {seizure} -')
        
        # --- Import data ---
        data, times, fs, window_size = import_data(path, seizure, bipolar_channels)

        # --- Filtering data ---
        data = filtering(data, fs)
        
        # --- Downsampling data ---
        data = downsampling(data, fs, chosen_fs)
        
        # --- Feature extraction ---
        data, features_names = feature_extraction(data, chosen_fs, window_size)
    
        # --- Vigilance state (ASSC classifier) ---
        vigilance = get_vigilance(data, times, patient, seizure)
    
        data_list.append(data)
        times_list.append(times)
        vigilance_list.append(vigilance)
        

# %% DATA SPLITTING | TRAIN | TEST

    # --- Splitting data ---
    data, times, metadata, vigilance = splitting(data_list, times_list, metadata_list, vigilance_list, n_seizures_train)
       
    # --- Save general information ---
    info_general = [patient, n_seizures_train, len(data['test']), SPH]
    
    if mode=='Train':   
        
        # --- Train ---
        info_train = main_train(approach, patient, data['train'], times['train'], metadata['train'], vigilance['train'], SPH, SOP_list)
        
        # --- Save results (patient) ---
        information_general[patient] = info_general
        information_train[patient] = info_train
        
    if mode=='Test':
        
        # --- Test ---
        info_test = main_test(approach, patient, data['test'], times['test'], metadata['test'], vigilance['test'], SPH, window_size)
        
        # --- Select & save final result ---
        info_train = read_train_results(approach, patient, len(SOP_list), len(info_general))
        idx_final = select_final_result(info_train, approach)
        final_information.append(info_general + info_train[idx_final] + info_test[idx_final])
        info_train[idx_final][0] = f'{info_train[idx_final][0]}*' # mark final result
        
        # --- Save results (patient) ---
        information_general[patient] = info_general
        information_train[patient] = info_train
        information_test[patient] = info_test
    
        # --- Figure: performance (patient) ---
        fig_performance(patient, info_train, info_test, idx_final, approach)
        
    if mode=='Train&Test':
        
        # --- Train ---
        info_train = main_train(approach, patient, data['train'], times['train'], metadata['train'], vigilance['train'], SPH, SOP_list)
        
        # --- Test ---
        info_test = main_test(approach, patient, data['test'], times['test'], metadata['test'], vigilance['test'], SPH, window_size)
        
        # --- Select & save final result ---
        idx_final = select_final_result(info_train, approach)
        final_information.append(info_general + info_train[idx_final] + info_test[idx_final])
        info_train[idx_final][0] = f'{info_train[idx_final][0]}*' # mark final result

        # --- Save results (patient) ---
        information_general[patient] = info_general
        information_train[patient] = info_train
        information_test[patient] = info_test
    
        # --- Figure: performance (patient) ---
        fig_performance(patient, info_train, info_test, idx_final, approach)
    
    
# %% SAVE FINAL RESULTS

if mode=='Train' or mode=='Train&Test':
    # --- Save train results (excel) ---
    save_train_results(information_general, information_train, approach)

if mode=='Test' or mode=='Train&Test':    
    # --- Save results (excel) ---
    save_results(information_general, information_train, information_test, approach)
    
    # --- Figure: performance per patient (all SOPs) ---
    fig_performance_per_patient(information_test, approach)
    
    # --- Save final results (excel) ---
    save_final_results(final_information, approach)
    
    # --- Figure: final performance per patient (selected SOPs) ---
    fig_final_performance(final_information, approach)
    
    # --- Figure: selected features ---
    fig_feature_selection(final_information, approach)
    

tf = t.time()
print(f'Running time: {tf-ti}')