"""
Code with functions to pre-process signals
(filtering and downsampling)

@author: AnaCROliveira
"""

from scipy import signal

def filtering(data, fs):
        
    print('\nFiltering signal...')
    
    # --- High-pass filter ---
    filter_type = 'high'
    cutoff = 1  # desired cutoff frequency (Hz)
    order = 4
    
    data = butter_filter(data, fs, filter_type, cutoff, order)
    
    
    # --- Low-pass filter ---
    filter_type = 'low'
    cutoff = 30  # desired cutoff frequency (Hz)
    order = 4 
    
    data = butter_filter(data, fs, filter_type, cutoff, order)
        
    print('Signal filtered')
      
    return data


def butter_filter(data, fs, filter_type, cutoff, order):
    
    nyq = 0.5*fs # Nyquist Frequency
    normal_cutoff = cutoff/nyq
    
    # Get the filter coefficients
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    
    filtered_signal = signal.filtfilt(b, a, data)
    
    return filtered_signal


def downsampling(data, fs, chosen_fs):

    if fs!=chosen_fs:
        print(f'\nDownsampling signal [{fs}->{chosen_fs}]...')
        reduce_factor = int(fs/chosen_fs)
        data = signal.decimate(data, reduce_factor)
        print('Signal downsampled')
    
    return data
