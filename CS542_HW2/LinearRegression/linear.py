import numpy as np
import scipy.io as sio
import math
from operator import itemgetter

def a():
    extract = sio.loadmat('detroit.mat')
    data = extract['data']
    
    all_design_matrices = []
    
    output_vector = []
    for i in range(len(data)):
        output_vector.append(data[i][-1])
    output = np.array(output_vector)
    
    for k in range(1,8):
        design_matrix = []
        for i in range(len(data)):
            row = []
            row.append(data[i][0])
            row.append(data[i][8])
            row.append(data[i][k])
            design_matrix.append(row)
        all_design_matrices.append(design_matrix)
    
    errors = []
    for i in range(7):
        design = np.array(all_design_matrices[i])
        
        phi_avg = [0.0, 0.0, 0.0]
        t_avg = 0.0
        for j in range(len(design)):
            phi_avg += design[j]
            t_avg += output[j]
        phi_avg /= len(design)
        t_avg /= len(output)
        
        weights = ((np.linalg.inv((design.T).dot(design))).dot(design.T)).dot(output)
        
        result = 0.0
        for j in range(len(phi_avg)):
            result += weights[j]*phi_avg[j]
        w0 = t_avg - result
        
        loss = 0.0
        for n in range(len(design)):
            third_term = (weights.T).dot(design[n])
            loss += (output[n] - w0 - third_term) ** 2
            '''
            msum = 0.0
            for m in range(len(design[0])):
                msum += design[n][m]*weights[m]
            loss += (output[n] - w0 - msum) ** 2
            '''
        loss /= 2
        errors.append(loss)
    
    print("Errors of each corresponding column, starting with Errors[0] = UEMP")
    print(errors)
    lowest_err_col = errors.index(min(errors))
    if lowest_err_col == 1:
        print("FTP, WE, + Third variable in determining HOM is UEMP")
    elif lowest_err_col == 2:
        print("FTP, WE, + Third variable in determining HOM is MAN")
    elif lowest_err_col == 3:
        print("FTP, WE, + Third variable in determining HOM is LIC")
    elif lowest_err_col == 4:
        print("FTP, WE, + Third variable in determining HOM is GR")
    elif lowest_err_col == 5:
        print("FTP, WE, + Third variable in determining HOM is NMAN")
    elif lowest_err_col == 6:
        print("FTP, WE, + Third variable in determining HOM is GOV")
    elif lowest_err_col == 7:
        print("FTP, WE, + Third variable in determining HOM is HE")

a()