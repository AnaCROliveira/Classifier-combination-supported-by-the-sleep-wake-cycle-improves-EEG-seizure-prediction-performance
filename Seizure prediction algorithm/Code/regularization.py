"""
Code with functions to implement the Firing Power regularization method
and generate alarms 

@author: AnaCROliveira
"""

import numpy as np
from datetime import timedelta as t

def get_firing_power(prediction, times, SOP, window_size):
    
    firing_power = []
    for i in range(len(prediction)):
        
        begin_time = times[i] - t(minutes = SOP)
        end_time = times[i]
        window_values = prediction[(times>begin_time) & (times<=end_time)]
        
        if len(window_values)<int((SOP*60/window_size)/2):
            window_size_samples = int(SOP*60/window_size)
        else:
            window_size_samples = len(window_values)
        
        firing_power_value = np.sum(window_values)/window_size_samples
        firing_power.append(firing_power_value)
        
    firing_power = np.array(firing_power)
    
    return firing_power


def alarm_generation(firing_power, threshold):
    
    # Raise alarm if firing power >= threshold
    alarm = np.zeros(len(firing_power))
    for i in range(len(firing_power)):
        if firing_power[i]>=threshold:
            alarm[i] = 1
            
    return alarm
        

def alarm_processing(alarm, times, SOP, SPH):
    
    refractory_samples = np.full(len(alarm), False)
    for i in range(len(alarm)):
        if alarm[i]==1:
            
            # --- Refractory period (SOP + SPH) ---
            begin_time = times[i]
            end_time = times[i] + t(minutes = SOP+SPH)
            refractory_period = (times>begin_time) & (times<=end_time)
                        
            # --- Next alarm can only be raised after the refractory period ---
            alarm[refractory_period] = 0

            # --- Refractory samples ---
            refractory_samples[refractory_period] = True

    return alarm, refractory_samples


def alarm_generation_2thresholds_transitions(firing_power, threshold_default, threshold_transition, times, vigilance): # Approach Threshold_transitions
    
    # Find vigilance transitions (sleep->awake | awake->sleep) 
    idx_transition = np.where(abs(np.diff(vigilance))==1)[0]
    # Consider that a transition period lasts 30 minutes
    idx_transition = [np.where((times>=times[i]) & (times<=times[i]+t(minutes=30)))[0] for i in idx_transition]
    if len(idx_transition)>0:
        idx_transition = np.concatenate(idx_transition)
        
    alarm = np.zeros(len(firing_power))
    thresholds = []
    for i in range(len(firing_power)):
        
        # Choose threshold
        if i in idx_transition:
            threshold = threshold_transition
        else:
            threshold = threshold_default
        thresholds.append(threshold)
        
        # Raise alarm if firing power >= threshold
        if firing_power[i]>=threshold:
            alarm[i] = 1
        
    return alarm, thresholds


def alarm_generation_2thresholds(firing_power, threshold_awake, threshold_sleep, times, vigilance): # Approach Threshold_state
    
    # Find sleep samples 
    idx_sleep = np.where(vigilance==1)[0]
        
    alarm = np.zeros(len(firing_power))
    thresholds = []
    for i in range(len(firing_power)):
        
        # Choose threshold
        if i in idx_sleep:
            threshold = threshold_sleep
        else:
            threshold = threshold_awake
        thresholds.append(threshold)
        
        # Raise alarm if firing power >= threshold
        if firing_power[i]>=threshold:
            alarm[i] = 1
        
    return alarm, thresholds