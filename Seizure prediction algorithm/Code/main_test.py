"""
Code with a function to forward the testing phase to the chosen approach

@author: AnaCROliveira
"""

import sys

# Scripts
sys.path.append('getTesting')
from testing_control import main_test_control
from testing_feature_state import main_test_feature_state
from testing_pool_weights import main_test_pool_weights
from testing_pool_exclusive import main_test_pool_exclusive
from testing_threshold_state import main_test_threshold_state
from testing_threshold_transitions import main_test_threshold_transitions

def main_test(approach, patient, data, times, metadata, vigilance, SPH, window_size):
    
    if approach=='Control':
        info_test = main_test_control(patient, data, times, metadata, vigilance, SPH, window_size)
    elif approach=='Feature_state':
        info_test = main_test_feature_state(patient, data, times, metadata, vigilance, SPH, window_size)
    elif approach=='Pool_weights':            
        info_test = main_test_pool_weights(patient, data, times, metadata, vigilance, SPH, window_size)
    elif approach=='Pool_exclusive':
        info_test = main_test_pool_exclusive(patient, data, times, metadata, vigilance, SPH, window_size)
    elif approach=='Threshold_state':
        info_test = main_test_threshold_state(patient, data, times, metadata, vigilance, SPH, window_size)
    elif approach=='Threshold_transitions':
        info_test = main_test_threshold_transitions(patient, data, times, metadata, vigilance, SPH, window_size)
    
    return info_test