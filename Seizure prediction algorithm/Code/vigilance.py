"""
Code with functions to compute patients' vigilance states
by applying the sleep-wake detection model and post-processing them

@author: AnaCROliveira
"""

import pickle
import numpy as np
from datetime import timedelta as t

# Scripts
from auxiliary_func import test
from plot_results import fig_hypnogram


def get_vigilance(data, times, patient, seizure):
    
    # --- Import sleep-awake model (trained classifiers from CAP SLEEP Database) ---
    model_ASSC = pickle.load(open('../../Sleep-wake algorithm/Results/Models/best_model','rb'))
    
    # --- Get viligance prediction ---
    prediction = test(data, model_ASSC)
    
    # --- Post-processing ---
    vigilance = post_processing(prediction, times)
    
    # --- Figure: hypnogram ---
    fig_hypnogram(prediction, times, f'patient {patient} - seizure {seizure}', '')
    fig_hypnogram(vigilance, times, f'patient {patient} - seizure {seizure}', '_final')
    
    return vigilance


def post_processing(prediction, times):
    
    # Moving average filter (10 minutes window)
    
    final_prediction = []
    for i in range(len(prediction)):
        
        begin_time = times[i] - t(minutes = 10)
        end_time = times[i]
        
        window_values = prediction[(times>begin_time) & (times<=end_time)]
        min_n_classifications = int(len(window_values)/2)
        
        if np.count_nonzero(window_values==0) > min_n_classifications:
            final_prediction.append(0) # awake
        else:
            final_prediction.append(1) # sleep
            
    final_prediction = np.array(final_prediction)
    
    return final_prediction