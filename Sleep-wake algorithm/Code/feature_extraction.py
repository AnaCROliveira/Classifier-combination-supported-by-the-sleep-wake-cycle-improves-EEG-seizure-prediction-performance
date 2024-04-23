"""
Code with a function to extract features

@author: AnaCROliveira
"""

import numpy as np
import sys

# Scripts
sys.path.append('getFeatures')
from get_statistics import statistical_moments
from get_power import obtain_PSD, relative_power, spectral_edge_freq_power
from get_hjorth import hjorth_parameters
from get_wavelet import coefficients_energy


def feature_extraction(data, fs, window_size_seconds):
    
    print('\nExtracting features...')
    
    # --- Define step and window size samples ---
    overlap = 0 # 0-100 (%) 
    step = window_size_seconds * (100-overlap)/100
    step_samples = int(step * fs)
    window_size_samples = window_size_seconds * fs
    
    i = 0
    features_per_window_list = []
    while i + window_size_samples <= data.shape[1]:
        
        begin_window = i
        end_window = begin_window + window_size_samples
        
        window_values = data[:,begin_window:end_window]
        
        i = i + step_samples
        

        # --- STATISTICAL MOMENTS ---
        
        mean, variance, skewness, kurt = statistical_moments(window_values)
        
        
        # --- RELATIVE SPECTRAL POWER ---
        
        f_values, psd_values = obtain_PSD(window_values, fs)   
        
        # Frequency bands
        delta_freq = [1, 4] # 1-4Hz
        theta_freq = [4, 8] # 4-8Hz
        alpha_freq = [8, 12] # 8-12Hz
        beta_freq = [12, 25] # 12-25Hz
    
        # Relative powers
        rsp_delta = relative_power(delta_freq, f_values, psd_values)
        rsp_theta = relative_power(theta_freq, f_values, psd_values)
        rsp_alpha = relative_power(alpha_freq, f_values, psd_values)
        rsp_beta = relative_power(beta_freq, f_values, psd_values)
    
               
        # --- SPECTRAL EDGE POWER/FREQUENCY ---
    
        sep_50, sef_50 = spectral_edge_freq_power(50, psd_values, f_values) # percentage = 50
        sep_75, sef_75 = spectral_edge_freq_power(75, psd_values, f_values) # percentage = 75
        sep_90, sef_90 = spectral_edge_freq_power(90, psd_values, f_values) # percentage = 90
        
        
        # --- HJORTH PARAMETERS ---
       
        mobility, complexity = hjorth_parameters(window_values)
    
        
        # --- ENERGY OF WAVELET COEFFICIENTS ---
        
        mother_wavelet = 'db4'
        decomposition_level = 5
        
        e_d1 = []
        e_d2 = []
        e_d3 = []
        e_d4 = []
        e_d5 = []
        e_a5 = []
        for signal_values in window_values:
            energy = coefficients_energy(signal_values, mother_wavelet, decomposition_level)
            e_d1.append(energy[0])
            e_d2.append(energy[1])
            e_d3.append(energy[2])
            e_d4.append(energy[3])
            e_d5.append(energy[4])
            e_a5.append(energy[5])
                        
        
        # --- SAVE FEATURES ---

        # features per window: (nºfeatures * nºchannels)     
        features_per_window = np.concatenate([mean, variance, skewness, kurt, rsp_delta, rsp_theta, rsp_alpha, rsp_beta, sep_50, sef_50, sep_75, sef_75, sep_90, sef_90, mobility, complexity, e_d1, e_d2, e_d3, e_d4, e_d5, e_a5])
        features_names = ['mean','var','skew','kurt',r'rsp $\delta$',r'rsp $\theta$',r'rsp $\alpha$',r'rsp $\beta$','sep50','sef50','sep75','sef75','sep90','sef90','mobili','complex','ed1','ed2','ed3','ed4','ed5','ea5']
        
        features_per_window_list.append(features_per_window) 
        
    # features: nªsamples x (nºfeatures * nºchannels)
    features = np.array(features_per_window_list)
    
    print('Features extracted')
    
    return features, features_names