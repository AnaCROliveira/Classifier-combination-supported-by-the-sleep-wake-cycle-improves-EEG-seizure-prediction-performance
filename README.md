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
- [main.py] - code to execute all steps of the sleep-wake detection algorithm (data pre-processing, training, and testing).
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
Approaches brief description:
- _Control_: Standard seizure prediction algorithm, the control method, that does not use sleep-wake information. 
- _Feature<sub>state</sub>_: Seizure prediction algorithm with an extra feature of the vigilance state in model training.
- _Pool<sub>weights</sub>_: Seizure prediction algorithm with a pool of two distinct models, each trained with all samples according to weights, assigned based on the vigilance state.
- _Pool<sub>exclusive</sub>_: Seizure prediction algorithm with a pool of two distinct models, each trained with samples exclusively from the respective vigilance state.
- _Threshold<sub>state</sub>_: Seizure prediction algorithm with a different threshold for each vigilance state in the post-processing phase.
- _Threshold<sub>transitions</sub>_: Seizure prediction algorithm with a different threshold for vigilance state transitions in the post-processing phase.

### EPILEPSIAE
- **Data**: folder where the data (npy files) should be located.
  
### Code

- [main.py] - code to execute all steps of the seizure prediction algorithm (data pre-processing, training, and testing).
- [import_data.py] - code with functions to import data and metadata.
- [pre_processing.py] - code with functions to pre-process signals (filtering and downsampling).
- [feature_extraction.py] - code with a function to extract features.
  - **getFeatures**: folder with code to get features (statistical moments, energy of wavelet coefficients, relative spectral powers, spectral edge power and frequency, and Hjörth parameters)
- [vigilance.py] - code with functions to compute patients' vigilance states with the CAP Sleep sleep-wake detection model.
- [splitting.py] - code with a function to split data into train and test.
- [main_train.py] - code with a function to forward the training step to the chosen approach.
  - **getTraining**: folder with code of each approach to train the seizure prediction model (Approaches: Control, Feature<sub>state</sub>, Pool<sub>weights</sub>, Pool<sub>exclusive</sub>, Threshold<sub>state</sub>, Threshold<sub>transitions</sub>)
- [main_test.py] - code with a function to forward the testing step to the chosen approach.
  - **getTesting**: folder with code of each approach to test the seizure prediction model (Approaches: Control, Feature<sub>state</sub>, Pool<sub>weights</sub>, Pool<sub>exclusive</sub>, Threshold<sub>state</sub>, Threshold<sub>transitions</sub>)
- [auxiliary_fun.py] - code with auxiliary functions used in training and testing.
- [regularization.py] - code with functions to implement the Firing Power regularization method and generate alarms.
- [evaluation.py] - code with functions to compute performance (SS and FPR/h) and perform statistical validation (surrogate analysis).
- [save_results.py] - code with functions to save models and results.
- [plot_results.py] - code with functions to plot results.
