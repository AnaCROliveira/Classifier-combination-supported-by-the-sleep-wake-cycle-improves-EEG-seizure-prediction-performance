"""
Code with a function to forward the training step to the chosen approach

@author: AnaCROliveira
"""

import sys

# Scripts
sys.path.append('getTraining')
from training_control import main_train_control
from training_feature_state import main_train_feature_state
from training_pool_weights import main_train_pool_weights
from training_pool_exclusive import main_train_pool_exclusive
from training_threshold_state import main_train_threshold_state
from training_threshold_transitions import main_train_threshold_transitions

def main_train(approach, patient, data, times, metadata, vigilance, SPH, SOP_list):
    
    if approach=='Control':
        info_train = main_train_control(patient, data, times, metadata, SPH, SOP_list)
    elif approach=='Feature_state':
        info_train = main_train_feature_state(patient, data, times, metadata, vigilance, SPH, SOP_list)
    elif approach=='Pool_weights':            
        info_train = main_train_pool_weights(patient, data, times, metadata, vigilance, SPH, SOP_list)
    elif approach=='Pool_exclusive':
        info_train = main_train_pool_exclusive(patient, data, times, metadata, vigilance, SPH, SOP_list)
    elif approach=='Threshold_state':
        info_train = main_train_threshold_state(patient, data, times, metadata, SPH, SOP_list)
    elif approach=='Threshold_transitions':
        info_train = main_train_threshold_transitions(patient, data, times, metadata, SPH, SOP_list)

    return info_train