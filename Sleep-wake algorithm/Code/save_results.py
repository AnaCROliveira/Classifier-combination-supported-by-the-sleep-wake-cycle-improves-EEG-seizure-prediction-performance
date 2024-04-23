"""
Code with functions to save models
and results (excel)

@author: AnaCROliveira
"""

import numpy as np
import xlsxwriter as xw
import pickle
import os


def save_results(info):
    
    # Create xlsx
    path = '../Results'
    if not os.path.exists(path):
        os.makedirs(path)
    wb = xw.Workbook(f'{path}/Final_results.xlsx', {'nan_inf_to_errors': True})
    ws = wb.add_worksheet('Final results')
    
    # Header format
    format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
    format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})

    # Insert header
    header_general = ['Run','Patients train','Patients test']
    header_train = ['#Features','Cost','SS','SP','Metric']
    header_test = ['SS','SP','Metric']

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
    for i in info:
        ws.write_row(row, col, i)
        row += 1

    wb.close()
    

def save_model(model, name):
    
    path = '../Results/Models'
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(model, open(f'{path}/{name}','wb'))    
    
    
def save_best_model(info):
        
    # Best model (highest train metric)
    metrics = [i[7] for i in info]
    idx_best_run = np.argmax(metrics)

    best_model = pickle.load(open(f'../Results/Models/model{idx_best_run+1}', 'rb'))
    pickle.dump(best_model, open('../Results/Models/best_model','wb'))
    
    print(f'\n\n--- FINAL TEST PERFORMANCE (best run) --- \nSS = {info[idx_best_run][-3]:.3f} | SP = {info[idx_best_run][-2]:.3f}')
   
    return idx_best_run