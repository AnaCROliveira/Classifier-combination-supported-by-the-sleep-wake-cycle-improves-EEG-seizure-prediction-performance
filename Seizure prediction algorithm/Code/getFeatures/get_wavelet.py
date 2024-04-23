"""
Code with functions to get energy of wavelet coefficients

@author: AnaCROliveira
"""

import pywt
import numpy as np


def coefficients_energy(data, mother_wavelet, decomposition_level):
    
    # Extraction of the aproximation(a) and details(d) reconstructed
    all_d = decomposite(data, 'd', mother_wavelet, decomposition_level)
    a = decomposite(data, 'a', mother_wavelet, decomposition_level)[-1]

    # Reconstruc = all d coefficients (d1,d2,...,df) + coeff af
    reconstruc = all_d
    reconstruc.append(a)
  
    # Same size (ignore last extra elements)
    reconstruc = [array_d[0:len(data)] for array_d in reconstruc]

    # Energy values
    energy_values = [energy(reconstruc, k) for k in range(len(reconstruc))]
        
    return energy_values


def decomposite(data, coeff_type, mother_wavelet, decomposition_level):
    ca = []
    cd = []
    for i in range(decomposition_level):
        (a, d) = pywt.dwt(data, mother_wavelet)
        ca.append(a)
        cd.append(d)
        data=a # the next decomposition is performed in the aproximation (a) of the previous level
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, mother_wavelet))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, mother_wavelet))
    if coeff_type == 'd':
        return rec_d
    elif coeff_type == 'a':
        return rec_a


def energy(array_coeff, k):
    return np.sum(array_coeff[k]**2)