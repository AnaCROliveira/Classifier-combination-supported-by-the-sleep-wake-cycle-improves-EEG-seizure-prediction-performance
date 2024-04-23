"""
Code with functions to save results (excel)

@author: AnaCROliveira
"""

import numpy as np
import xlsxwriter as xw
import xlrd as xr
import os


def save_train_results(information_general, information_train, approach):
    
    # Create xlsx
    path = f'../Results/Approaches/{approach}'
    if not os.path.exists(path):
        os.makedirs(path)
    wb = xw.Workbook(f'{path}/Results_train.xlsx', {'nan_inf_to_errors': True})

    patients = list(information_general.keys())
    for patient in patients:
        info_general = information_general[patient]
        info_train = information_train[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
        header_train = ['#Features','Cost','SS samples','SP samples','Metric']
        if approach=='Pool_weights' or approach=='Pool_exclusive':
            header_train = ['#Features (Awake)','Cost (Awake)','SS samples (Awake)','SP samples (Awake)','Metric (Awake)','#Features (Sleep)','Cost (Sleep)','SS samples (Sleep)','SP samples (Sleep)','Metric (Sleep)']
            
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
    
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, info_general)
        
        col = len(info_general)
        for i in info_train:
            ws.write_row(row, col, i)
            row += 1

    wb.close()
    
    
def read_train_results(approach, patient, n_rows, start_column):
    
    # Create xlsx
    path = f'../Results/Approaches/{approach}/Results_train.xlsx'
    wb = xr.open_workbook(path)
    
    ws = wb.sheet_by_name(f'pat_{patient}')
    
    info = []
    for i in range(1,ws.nrows):
        info.append(ws.row_values(i))
        
        # SOP & n_features to int
        if type(info[i-1][start_column])==str:
            info[i-1][start_column] = int(info[i-1][start_column][:-1]) # remove * from best sop
        else:
            info[i-1][start_column] = int(info[i-1][start_column])
        info[i-1][start_column+1] = int(info[i-1][start_column+1])
        
    info_train = [info_train[start_column:] for info_train in info]
                 
    return info_train
        

def save_results(information_general, information_train, information_test, approach):
    
    # Create xlsx
    path = f'../Results/Approaches/{approach}'
    if not os.path.exists(path):
        os.makedirs(path)
    wb = xw.Workbook(f'{path}/Results.xlsx', {'nan_inf_to_errors': True})

    patients = list(information_general.keys())
    for patient in patients:
        info_general = information_general[patient]
        info_train = information_train[patient]
        info_test = information_test[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
        header_train = ['#Features','Cost','SS samples','SP samples','Metric']
        header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
        if approach=='Pool_weights' or approach=='Pool_exclusive':
            header_train = ['#Features (Awake)','Cost (Awake)','SS samples (Awake)','SP samples (Awake)','Metric (Awake)','#Features (Sleep)','Cost (Sleep)','SS samples (Sleep)','SP samples (Sleep)','Metric (Sleep)']
            
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
        col = col + len(header_train)
        ws.write_row(row, col, header_test, format_test)
    
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, info_general)
        
        info = [info_train[i]+info_test[i] for i in range(len(info_train))]
        col = len(info_general)
        for i in info:
            ws.write_row(row, col, i)
            row += 1

    wb.close()
    
    
def save_final_results(final_information, approach):
    
    # Create xlsx
    path = f'../Results/Approaches/{approach}'
    if not os.path.exists(path):
        os.makedirs(path)
    wb = xw.Workbook(f'{path}/Final_results.xlsx', {'nan_inf_to_errors': True})
    ws = wb.add_worksheet('Final results')
    
    # Header format
    format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
    format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
    
    # Insert Header
    header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
    header_train = ['#Features','Cost','SS samples','SP samples','Metric']
    header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
    if approach=='Pool_weights' or approach=='Pool_exclusive':
        header_train = ['#Features (Awake)','Cost (Awake)','SS samples (Awake)','SP samples (Awake)','Metric (Awake)','#Features (Sleep)','Cost (Sleep)','SS samples (Sleep)','SP samples (Sleep)','Metric (Sleep)']
            
    row = 0
    col = 0
    ws.write_row(row, col, header_general, format_general)
    col = len(header_general)
    ws.write_row(row, col, header_train, format_train)
    col = col + len(header_train)
    ws.write_row(row, col, header_test, format_test)
    
    # Insert data
    row = 1
    col = 0
    for i in final_information:
        ws.write_row(row, col, i)
        row += 1

    wb.close()
    

def select_final_result(info_train, approach):
        
    # Find highest train metric
    if approach=='Pool_weights' or approach=='Pool_exclusive':
        metrics = [np.sqrt(info[5]*info[10]) for info in info_train] # metric (Classifier Awake) & metric (Classifier Sleep)
    else:
        metrics = [info[5] for info in info_train] # metric (Classifier)
    idx_best = np.argmax(metrics)
 
    return idx_best