"""
Code with a function to get Hjörth parameters

@author: AnaCROliveira
"""

import numpy as np

def hjorth_parameters(data):
    
    first_derivative = np.diff(data, axis=1) 
    second_derivative = np.diff(first_derivative, axis=1)
    
    # Activity
    variance = np.var(data, axis=1)
    
    # Mobility
    num_m = np.var(first_derivative, axis=1)
    den_m = np.var(data, axis=1)
    mobility = np.sqrt(num_m/den_m)
    
    # Complexity
    num_c = np.var(second_derivative, axis=1)*np.var(data, axis=1)
    den_c = np.square(np.var(first_derivative, axis=1))
    complexity = np.sqrt(num_c/den_c)
    
    return mobility, complexity