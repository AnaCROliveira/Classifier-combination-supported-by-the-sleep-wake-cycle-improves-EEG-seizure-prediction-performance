"""
Code with functions to plot results

@author: AnaCROliveira
"""

import matplotlib.pyplot as plt     
import numpy as np
import pickle
import os


def fig_performance(info, labels, name, option):
    
    # Information
    ss = [i[-3] for i in info]
    sp = [i[-2] for i in info]
    
    # Figure
    plt.figure(figsize=(20, 10))
    
    x = np.arange(len(ss))
    width = 0.3
    ss_bar = plt.bar(x-width/2, ss, width, color='c', label = 'SS')
    sp_bar = plt.bar(x+width/2, sp, width, color='y', label = 'SP')
    
    plt.bar_label(ss_bar, fmt='%.2f', padding=3, fontsize=7.5)
    plt.bar_label(sp_bar, fmt='%.2f', padding=3, fontsize=7.5)
    
    plt.xticks(x, labels=labels, rotation=90)
    plt.ylim([0, 1.05])
    plt.legend(loc='upper right')
    plt.title(f'Performance per {name}')

    if option=='final':
        ss_final = [np.mean(ss), np.std(ss)]
        sp_final = [np.mean(sp), np.std(sp)]    
        print(f'\n--- FINAL TEST PERFORMANCE (all runs) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | SP = {sp_final[0]:.3f} ± {sp_final[1]:.3f}')

        text = f'  --- Final result --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nSP = {sp_final[0]:.3f} ± {sp_final[1]:.3f}'
        plt.text(plt.xlim()[1], plt.ylim()[1], text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold')
        plt.annotate('* best run', xy=(1,0), xycoords='axes fraction', xytext=(-55,-65), textcoords='offset points')

        path = '../Results'
    else:
        path = '../Results/Runs'
    
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/performance per {name}')
    plt.close()
    
    
def fig_performance_per_patient(info_per_patient):
    
    # Information
    patients = sorted(info_per_patient.keys())
    ss_mean = []
    ss_std = []
    sp_mean = []
    sp_std = []    
    n_test = [] #nr times tested
    for patient in patients:
        ss = [info[-3] for info in info_per_patient[patient]]
        sp = [info[-2] for info in info_per_patient[patient]]
        
        ss_mean.append(np.mean(ss))
        ss_std.append(np.std(ss))
        sp_mean.append(np.mean(sp))
        sp_std.append(np.std(sp))
        n_test.append(len(ss))
    
    labels = [f'patient {patient}' for patient in patients]

    # Figure
    plt.figure(figsize=(20, 10))
    
    x = np.arange(len(patients))
    width = 0.3
    spare_width = 0.5

    plt.bar(x-width/2, ss_mean, width, yerr=ss_std, color='c', label = 'SS',error_kw=dict(elinewidth= 0.5, capsize=5)) 
    plt.bar(x+width/2, sp_mean, width, yerr=sp_std, color='y', label = 'SP',error_kw=dict(elinewidth= 0.5, capsize=5))
    
    for i in range(len(ss_mean)):
        plt.annotate(str(round(ss_mean[i],2)),(x[i]-width,ss_mean[i]+plt.ylim()[1]/100),fontsize=7.5)
        plt.annotate(str(round(sp_mean[i],2)),(x[i],sp_mean[i]+plt.ylim()[1]/100),fontsize=7.5)

    plt.table(cellText=[n_test],rowLabels=['Nr.times tested'],cellLoc='center',bbox=[0, -0.2, 1, 0.05],edges='horizontal')
    plt.subplots_adjust(bottom=0.25)    
    
    plt.xticks(x, labels=labels, rotation=90)
    plt.xlim(x[0]-spare_width,x[-1]+spare_width)
    plt.ylim([0, 1.05])
    plt.legend(loc='upper right')
    plt.title('Final performance per patient')
    
    path = '../Results'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/performance per patient_final')
    plt.close()


def fig_feature_selection(n_runs, idx_best_run):
    
    idx_selected_features_per_run = []
    for run in range(1, n_runs+1): 
        model = pickle.load(open(f'../Results/Models/model{run}','rb'))    
        
        idx_selected_features = np.concatenate([selector.get_support(indices=True) for selector in model['selector']])
        idx_selected_features_per_run.append(idx_selected_features)
        # Figure: selected features per run
        fig_fs(idx_selected_features,f'run{run}')
    
    # Figure: selected features of best run
    fig_fs(idx_selected_features_per_run[idx_best_run],'best_models')
    # Figure: selected features of all runs
    fig_fs(np.concatenate(idx_selected_features_per_run),'all_runs')  # concatenate features from runs
    
    
def fig_fs(idx_selected_features,fig_name):

    # Configurations
    features = ['mean','var','skew','kurt',r'rsp $\delta$',r'rsp $\theta$',r'rsp $\alpha$',r'rsp $\beta$','sep50','sef50','sep75','sef75','sep90','sef90','mobili','complex','ed1','ed2','ed3','ed4','ed5','ea5']
    channels = ['FP2-F4','F4-C4','C4-P4','P4-O2','FP1-F3','F3-C3','C3-P3','P3-O1','F8-T4','T4-T6','F7-T3','T3-T5']
        
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
    
    path = '../Results/Feature selection'
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Figure: selected frequency & channels
    plt.figure(figsize=(20, 10))
    plt.bar(features_channels_freq.keys(), features_channels_freq.values(), color='g')
    plt.title('Relative frequency of selected features and channels')
    plt.tick_params(axis='x', labelsize=5)
    plt.xticks(rotation=90)
    plt.savefig(f'{path}/feature_selection_detailed_{fig_name}', bbox_inches='tight')
    plt.close()

    # Figure: selected frequency + channels
    fig, ax = plt.subplots(2,1,figsize=(20, 10))
    ax[0].bar(features_freq.keys(), features_freq.values(), color='y')
    ax[0].set_title('Relative frequency of selected features')
    ax[1].bar(channels_freq.keys(), channels_freq.values(), color='g')
    ax[1].set_title('Relative frequency of selected channels')
    plt.savefig(f'{path}/feature_selection_{fig_name}')
    plt.close()


def fig_hypnogram(prediction, times, name, mode):
    
    plt.figure(figsize=(20, 10), constrained_layout=True)
    
    #plt.plot(times, prediction, color='c') # lines
    plt.plot(times, prediction, 'o', color='c', markersize=4) # points
    plt.yticks([0,1], labels=['Awake','Sleep'])

    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=10))    
    plt.gca().set_xlim(times[0],times[-1])
    
    plt.title(f'Hypnogram: {name} ({times[-1].date()})')
    plt.xlabel('Time')
    plt.ylabel('Vigilance stage')
    
    path = '../Results/Hypnogram'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/Hypnogram_{name}{mode}')
    plt.close()