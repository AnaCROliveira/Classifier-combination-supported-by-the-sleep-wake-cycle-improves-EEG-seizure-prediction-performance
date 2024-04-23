"""
Code with a function to get statistical moments

@author: AnaCROliveira
"""

import numpy as np
from scipy import stats

def statistical_moments(data):
    
    # Mean
    mean = np.mean(data, axis=1) 
    # Variance
    variance = np.var(data, axis=1)
    # Skewness
    skewness = stats.skew(data, axis=1)
    # Kurtosis
    kurt = stats.kurtosis(data, axis=1)
        
    return mean, variance, skewness, kurt