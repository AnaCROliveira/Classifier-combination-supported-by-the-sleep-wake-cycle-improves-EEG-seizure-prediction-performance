# Classifier-combination-supported-by-the-sleep-wake-cycle-improves-EEG-seizure-prediction-performance

This is the code used for the paper “Classifier combination supported by the sleep-wake cycle improves EEG seizure prediction performance”.
It includes a subject-independent sleep-wake algorithm based on epileptic patients from CAP Sleep database. The sleep-wake model is then used to ascertain the vigilance states of patients from EPILEPSIAE database. It also contains a patient-specific EEG seizure prediction algorithm, with novel approaches that integrate sleep-wake information.

## Main Folders
- **Sleep-wake algorithm**: folder with code for the sleep-wake detection pipeline.
- **Seizure prediction algorithm**: folder with code for the seizure prediction pipeline, including all approaches.


## Sleep-wake algorithm

It is possible to execute this code after downloading patient files from CAP Sleep database (available online at [PhysioNet](https://doi.org/10.13026/C2VC79)).

### CAP Sleep
- **Data**: folder where the data (edf files) should be located.
- **Annotations**: folder where the annotations (txt files) should be located.

### Code 
- [main.py] - code to execute all steps of the sleep-wake detection algorithm (data preprocessing, training, and testing).
- [import_data.py] - code with a function to import data.
- [pre_processing.py] - code with functions to pre-process signals (filtering and downsampling).
- [feature_extraction.py] - code with a function to extract features.
  - **getFeatures**: folder with code to get features (statistical moments, energy of wavelet coefficients, relative spectral powers, spectral edge power and frequency, and Hjörth parameters)
- [splitting.py] - code with a function to split data into train and test.
- [training.py] - code with functions to train the sleep-wake model.
- [testing.py] - code with functions to test the sleep-wake model.
- [auxiliary_fun.py] - code with auxiliary functions used in training and testing.
- [save_results.py] - code with functions to save models and results.
- [plot_results.py] - code with functions to plot results.


## Seizure prediction algorithm

Is it not possible to execute this code since it requires the raw data from EEG recordings. As we used data from EPILEPSIAE, we can not make it publicly available online due to ethical concerns.

### Code



