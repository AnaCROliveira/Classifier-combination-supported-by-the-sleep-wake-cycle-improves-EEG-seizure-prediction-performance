"""
Code with functions to plot results

@author: AnaCROliveira
"""

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import numpy as np
import pickle
import os
from datetime import timedelta as t


def fig_performance(patient, info_train, info_test, idx_final, approach):
    
    # Information
    ss = [info[-6] for info in info_test]
    fprh = [info[-5] for info in info_test]
    labels = [info[0] for info in info_train]
    labels[idx_final] = f'{labels[idx_final]}*' # mark final result
        
    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(ss))
    width = 0.3
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='c', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.xlabel('SOP')
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    
    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='y', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    
    plt.xticks(x, labels=labels) #, rotation=90)
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.annotate('* final result', xy=(1,0), xycoords='axes fraction', xytext=(-60,-50), textcoords='offset points')

    plt.title(f'Performance (patient {patient})')

    path = f'../Results/Approaches/{approach}/Patients'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/Performance (patient {patient})')
    plt.close()    


def fig_final_performance(final_information, approach):
    
    # Information
    ss = [info[-6] for info in final_information]
    fprh = [info[-5] for info in final_information]
    sop = [info[4] for info in final_information]
    
    labels = [f'patient {info[0]}' for info in final_information]
    p_values = [info[-1] for info in final_information]
    labels = [f'*{labels[info]}'  if p_values[info]<0.05 else f'{labels[info]}' for info in range(len(final_information))] # validated patients

    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(final_information))
    width = 0.3
    spare_width = 0.5
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='c', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xticks(x, labels=labels, rotation=90)
    plt.xlim(x[0]-spare_width,x[-1]+spare_width)

    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='y', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    
    plt.table(cellText=[sop], rowLabels=['SOP'], cellLoc='center', bbox=[0, -0.25, 1, 0.05], edges='horizontal')
    plt.subplots_adjust(bottom=0.25)
    
    ss_final = [np.mean(ss), np.std(ss)]
    fprh_final = [np.mean(fprh), np.std(fprh)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    ss_final = [np.median(ss), np.percentile(ss, 75) - np.percentile(ss, 25)]
    fprh_final = [np.median(fprh), np.percentile(fprh, 75) - np.percentile(fprh, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (median) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.88, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    plt.annotate('* statistically validated', xy=(1,0), xycoords='axes fraction', xytext=(-115,-130), textcoords='offset points')

    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient')
    
    path = f'../Results/Approaches/{approach}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/Performance')
    plt.close()
    
    
    
def fig_performance_per_patient(information_test, approach):
    
    # Information
    patients = information_test.keys()
    ss_mean = []
    ss_std = []
    fprh_mean = []
    fprh_std = []
    for patient in patients:
        ss = [info[5] for info in information_test[patient]]
        fprh = [info[6] for info in information_test[patient]]
        
        ss_mean.append(np.mean(ss))
        ss_std.append(np.std(ss))
        fprh_mean.append(np.mean(fprh))
        fprh_std.append(np.std(fprh))
    labels = patients

    # Figure
    fig = plt.figure(figsize=(20, 10))
    
    x = np.arange(len(patients))
    width = 0.3
    
    # Sensitivity
    plt.bar(x-width/2, ss_mean, width, yerr=ss_std, color='c', label = 'SS', error_kw=dict(elinewidth=0.5, capsize=5))
    plt.ylabel('SS', color='yellowgreen', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xlabel('Patient')
    plt.xticks(x, labels=labels)
    [plt.annotate(str(round(ss_mean[i],2)),(x[i]-width,ss_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(ss_mean))]

    # FPR/h
    plt.twinx() # second axis
    plt.bar(x+width/2, fprh_mean, width, yerr=fprh_std, color='y', label = 'FPR/h', error_kw=dict(elinewidth= 0.5, capsize=5))
    plt.ylabel('FPR/h', color='sandybrown', fontweight='bold')
    plt.ylim(bottom=0)
    [plt.annotate(str(round(fprh_mean[i],2)),(x[i],fprh_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(fprh_mean))]
 
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient')    

    ss_final = [np.mean(ss_mean), np.std(ss_mean)]
    fprh_final = [np.mean(fprh_mean), np.std(fprh_mean)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    ss_final = [np.median(ss_mean), np.percentile(ss_mean, 75) - np.percentile(ss_mean, 25)]
    fprh_final = [np.median(fprh_mean), np.percentile(fprh_mean, 75) - np.percentile(fprh_mean, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.9, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    path = f'../Results/Approaches/{approach}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/Performance per patient')
    plt.close()

    
def fig_feature_selection(final_information, approach):
    
    if approach=='Pool_weights' or approach=='Pool_exclusive':
        fig_fs_formation(final_information, approach, '_awake')
        fig_fs_formation(final_information, approach, '_sleep')
    else:
        fig_fs_formation(final_information, approach, '')    
 
    
def fig_fs_formation(final_information, approach, mode):
        
    patients = [info[0] for info in final_information]
    sop = [info[4] for info in final_information]
    
    idx_selected_features_list = []
    idx_final_selected_features_list = []
    for i in range(len(patients)):
        model = pickle.load(open(f'../Results/Models/{approach}/model{mode}_patient{patients[i]}','rb'))
        
        final = model[sop[i]]['selector']
        idx_final_selected_features = np.concatenate([selector.get_support(indices=True) for selector in final]) # concatenate features from 31 models of best SOP
        idx_final_selected_features_list.append(idx_final_selected_features)

        idx_selected_features = np.concatenate([selector.get_support(indices=True) for sop in model.keys() for selector in model[sop]['selector']]) # concatenate features from 31 models of all SOPs
        idx_selected_features_list.append(idx_selected_features)

        # Figure: selected features (patient)
        fig_fs(idx_selected_features, f'patient{patients[i]}{mode}', approach)
        
    # Figure: final selected features (all patients)
    fig_fs(np.concatenate(idx_final_selected_features_list),f'final{mode}', approach)
    # Figure: all selected features (all patients & SOPs)
    fig_fs(np.concatenate(idx_selected_features_list),f'all{mode}', approach)
    
    
def fig_fs(idx_selected_features, fig_name, approach):

    # Configurations
    features = ['mean','var','skew','kurt',r'rsp $\delta$',r'rsp $\theta$',r'rsp $\alpha$',r'rsp $\beta$','sep50','sef50','sep75','sef75','sep90','sef90','mobili','complex','ed1','ed2','ed3','ed4','ed5','ed6']
    channels = ['FP2-F4','F4-C4','C4-P4','P4-O2','FP1-F3','F3-C3','C3-P3','P3-O1','F8-T4','T4-T6','F7-T3','T3-T5']
    if approach=='Feature_state':
        features = ['mean','var','skew','kurt',r'rsp $\delta$',r'rsp $\theta$',r'rsp $\alpha$',r'rsp $\beta$','sep50','sef50','sep75','sef75','sep90','sef90','mobili','complex','ed1','ed2','ed3','ed4','ed5','ed6','vigil']

    
    features_freq = {feature:0 for feature in features}
    channels_freq = {channel:0 for channel in channels}
    features_channels_freq = {feature+'_'+channel:0 for feature in features for channel in channels}
    for i in idx_selected_features:
        idx_feature = i//len(channels)
        feature_name = features[idx_feature]
        features_freq[feature_name] += 1 
        
        idx_channel = i%len(channels)
        channel_name = channels[idx_channel]
        channels_freq[channel_name] += 1
        
        features_channels_freq[feature_name+'_'+channel_name] += 1   
        
    # Translate from number of occurrences to relative frequency
    features_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in features_freq.items()}
    channels_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in channels_freq.items()}
    features_channels_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in features_channels_freq.items()}

    path = f'../Results/Approaches/{approach}/Feature selection'
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Figure: selected frequency & channels
    plt.figure(figsize=(20, 10))
    plt.bar(features_channels_freq.keys(), features_channels_freq.values(), color='g')
    plt.title('Relative frequency of selected features and channels')
    plt.xlim([0, len(features_channels_freq)])
    plt.tick_params(axis='x', labelsize=5) # set the x-axis label size
    plt.xticks(rotation=90)
    plt.savefig(f'{path}/fs_detailed_{fig_name}', bbox_inches='tight')
    plt.close()

    # Figure: selected frequency + channels
    fig, ax = plt.subplots(2,1,figsize=(20, 10))
    ax[0].bar(features_freq.keys(), features_freq.values(), color='y')
    ax[0].set_title('Relative frequency of selected features')
    ax[1].bar(channels_freq.keys(), channels_freq.values(), color='g')
    ax[1].set_title('Relative frequency of selected channels')
    plt.savefig(f'{path}/fs_{fig_name}')
    plt.close()
    

def fig_test(patient, seizure, preictal, times, target, prediction, firing_power, threshold, alarm, vigilance, approach):
    
    fig, ax = plt.subplots(5, 1, figsize=(20, 10))
    
    # Target
    ax[0].plot(times, target, 'o', markersize=2, color='orange')
    ax[0].set_xlim([times[0], times[-1]])
    ax[0].set_xticks([])
    ax[0].set_ylabel('Target', color='orange')
    ax[0].set_yticks([0,1])
    ax[0].set_yticklabels(['Interictal','Preictal'])

    # Prediction
    ax[1].plot(times, prediction, 'o', markersize=2)
    ax[1].set_xlim([times[0], times[-1]])
    ax[1].set_xticks([])
    ax[1].set_ylabel('SVM Prediction', color='blue')
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels(['Interictal','Preictal'])
    
    # Firing power
    ax[2].plot(times, firing_power, color='red')
    ax[2].plot(times, np.full(len(times),threshold), color='green', label='Threshold')
    ax[2].set_xlim([times[0], times[-1]])
    ax[2].set_xticks([])
    ax[2].set_ylabel('Firing Power', color='red')
    ax[2].set_ylim([-0.05, 1.05]) 
    ax[2].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[2].grid(axis='y')
    ax[2].legend(loc='upper right')
    
    # Alarms
    ax[3].plot(times, alarm, color='yellow')
    ax[3].set_xlim([times[0], times[-1]])
    ax[3].set_xticks([])
    ax[3].set_ylabel('Alarms', color='yellow')
    ax[3].set_yticks([0,1])
    ax[3].set_yticklabels(['Interictal','Preictal'])

    # Vigilance
    ax[4].plot(times, vigilance, 'o', markersize=2, color='c')
    ax[4].set_xlim([times[0], times[-1]])
    ax[4].set_ylabel('Vigilance', color='c')
    ax[4].set_yticks([0,1])
    ax[4].set_yticklabels(['Awake','Sleep'])    

    ax[0].set_title(f'Test: patient {patient} - seizure {seizure+3} | preictal={preictal}')
    ax[4].set_xlabel('Time')
    #fig.align_labels()

    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

    path = f'../Results/Approaches/{approach}/Patients'
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f'{path}/test_pat{patient}_seiz{seizure+3}_preictal{preictal}')
    plt.close(fig)


def fig_hypnogram(prediction, times, name, mode):
    
    plt.figure(figsize=(20, 10))
    
    #plt.plot(times, prediction, color='c') # lines
    plt.plot(times, prediction, 'o', color='c', markersize=4) # points
    plt.yticks([0,1], labels=['Awake','Sleep'])

    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))    
    plt.gca().set_xlim(times[0],times[-1])
    
    plt.title(f'Hypnogram: {name} ({times[-1].date()})')
    plt.xlabel('Time')
    plt.ylabel('Vigilance stage')
    
    path = '../Results/Hypnogram'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/Hypnogram_{name}{mode}')
    plt.close()
    
    

def fig_temporality(patient, seizure, preictal, SPH, times, onset_seizure, target, prediction, firing_power, threshold, alarm, vigilance, window_size, approach):
    
    # Fill the missing data
    times_new = []
    firing_power_new = []
    target_new = []
    alarm_new = []    
    vigilance_new = []
    threshold_new = []
    for i in range(len(times)):
        
        times_new.append(times[i])
        firing_power_new.append(firing_power[i])
        target_new.append(target[i])
        alarm_new.append(alarm[i])
        vigilance_new.append(vigilance[i])
        if approach in ['Threshold_transitions', 'Threshold_state']:
            threshold_new.append(threshold[i])
        else:
            threshold_new.append(threshold)
        
        time_diff = (times[i+1]-times[i]).seconds if i!=len(times)-1 else (onset_seizure-t(minutes = SPH)-times[i]).seconds
        new_time = times[i]+t(seconds=window_size)
        
        while(time_diff>window_size):
            times_new.append(new_time)
            firing_power_new.append(np.NaN)
            target_new.append(np.NaN)
            alarm_new.append(np.NaN)
            vigilance_new.append(np.NaN)
            threshold_new.append(np.NaN)

            time_diff = (times[i+1]-new_time).seconds if i!=len(times)-1 else (onset_seizure-t(minutes = SPH)-new_time).seconds
            new_time = new_time+t(seconds=window_size)
    
    times_new = np.array(times_new)
    firing_power_new = np.array(firing_power_new)
    target_new = np.array(target_new)
    alarm_new = np.array(alarm_new)
    vigilance_new = np.array(vigilance_new)
    threshold_new = np.array(threshold_new)
    

    # Figure
    plt.figure(figsize=(20, 10))

    plt.xlim(times_new[0], times_new[-1])        
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    plt.xticks(fontsize=20)
    
    if approach=='Control':
        name = r'$\bfApproach: \ Control$'
    elif approach=='Pool_weights':
        name = r"$\bfApproach: \ Pool_{weights}$"
    else:
        name = approach
    plt.title(f'{name}\nPatient {patient} - Seizure {seizure+4}', fontsize=23)

    # Plot Firing power throughtout time
    plt.plot(times_new, firing_power_new, 'k', alpha=0.8, label='Firing power')
    plt.plot(times_new, np.full(len(times_new), threshold_new), linestyle='--', color='k', alpha=0.8, label='Alarm threshold')
    
    # Mark alarms
    idx_preictal = np.argwhere(target_new==1)
    idx_alarms = np.argwhere(alarm_new==1)
    label = 'False alarm'
    for idx in idx_alarms:
        if idx in idx_preictal:
            plt.plot(times_new[idx], firing_power_new[idx], marker='^', color='green', markersize=22, label='True alarm')
        else:
            plt.plot(times_new[idx], firing_power_new[idx], marker='^', color='maroon', markersize=22, label=label)
            label = ''
    plt.fill_between(times_new, threshold_new, firing_power_new, where=firing_power_new>=threshold_new, facecolor='brown', alpha=0.5, label='Firing power>threshold')

    # Color preictal
    plt.fill_between(times_new, 0, 1, where=target_new==1, facecolor='moccasin', alpha=0.5, label='Preictal')
    plt.axvline(onset_seizure-t(minutes = SPH)-t(minutes = preictal), color='k', alpha=0.8, linestyle='--', linewidth=0.8, label='Preictal onset')

    # Vigilance
    vigilance_new[vigilance_new==0] = -0.075
    vigilance_new[vigilance_new==1] = 1.075
    plt.plot(times_new, vigilance_new, alpha=0.7, label='Vigilance state')

    plt.ylim(-0.115, 1.115)
    plt.gca().yaxis.set_ticks([0,0.25,0.5,0.75,1])
    plt.yticks(fontsize=20)
    plt.grid(which='major', alpha=0.5)          
    plt.gca().yaxis.set_ticks([-0.075, 1.075], minor=True)     
    plt.gca().yaxis.set_ticklabels(['awake','sleep'], minor=True, fontsize=20)
    plt.legend(bbox_to_anchor =(0.5,-0.31), loc='lower center', ncol=4, fontsize=22)  
    
    path = f'../Results/Approaches/{approach}/Patients'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/pat{patient}_seiz{seizure+4}_preictal{preictal}')
    plt.close()