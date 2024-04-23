"""
Code with a function to get spectral edge power and frequency

@author: AnaCROliveira
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
 
def obtain_PSD(data, fs):
    
    f_values, psd_values = sc.signal.welch(data, fs, window='hann', scaling='spectrum', nperseg=fs, noverlap=0.5)
    
    return f_values, psd_values


def fig_PSD(f_values, psd_values, signal_number):
    
    plt.figure()
    plt.plot(f_values, psd_values[signal_number])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD [V^2/Hz]')
    plt.title('Spectrum of the signal')
    

def relative_power(freq_band, f_values, psd_values):
   
    # Band power
    band_powers = psd_values[:,((f_values>=freq_band[0]) & (f_values<freq_band[1]))]
    band_power = sc.integrate.simps(band_powers,axis=1)

    # Total power
    total_power = sc.integrate.simps(psd_values,axis=1)

    # Relative power
    rsp = band_power/total_power

    return rsp


def spectral_edge_freq_power(percentage, psd_values, f_values):

    # Total power 
    total_power = np.sum(psd_values, axis=1)
    
    # Spectral edge power (SEP)
    sep = percentage/100 * total_power
           
    # Spectral edge frequency (SEF)
    cumsum_values = np.cumsum(psd_values, axis=1)    
    idx_sef = [np.where(cumsum_values[i]<sep[i])[0][-1] for i in range(len(sep))] # last idx where cumsum_value < sep value
    sef = f_values[idx_sef]
    
    return sep, sef