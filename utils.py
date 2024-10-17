# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:37:43 2023

@author: zmzhai
"""

import numpy as np
import copy

def create_dataset(data, N):
    data_N = []
    for i in range(len(data)-N):
        data_N.append( np.squeeze(np.reshape(data[i:N+i], (-1, 1))))
    return np.array(data_N)


def count_grid(data, dt=0.02):
    # data = np.clip(data, 0., 1.)
    bins = np.arange(0., 1.01, dt)
    
    cell = np.zeros((len(bins), len(bins), len(bins)), dtype=float)
    data_copy = copy.deepcopy(data)
    
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y, data_z = data_copy[i, 0], data_copy[i, 1], data_copy[i, 2]
        
        if data_x < 0 or data_y < 0 or data_z < 0:
            data_copy[i, :] = 0
        
        if data_x > 1 or data_y > 1 or data_z > 1:
            data_copy[i, :] = 1
            
        if np.isnan(data_x) or np.isnan(data_y) or np.isnan(data_z):
            data_copy[i, :] = 1
            
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y, data_z = data_copy[i, 0], data_copy[i, 1], data_copy[i, 2]
        
        x_idx = int(np.floor(data_x / dt))
        y_idx = int(np.floor(data_y / dt))
        z_idx = int(np.floor(data_z / dt))
        
        cell[x_idx, y_idx, z_idx] += 1

    cell /= float(np.shape(cell)[0] ** 3)
    
    return cell

def loss_long_term(real, prediction, dt=0.02):
    
    real_cell = count_grid(real, dt=dt)
    pred_cell = count_grid(prediction, dt=dt)
    
    return np.sum( np.sqrt(np.square(real_cell - pred_cell)) )


def calculate_rmse(real, prediction, cal_length):
    real = np.array(real)
    prediction = np.array(prediction)
    
    rmse = np.sqrt(np.mean(np.square(real[:cal_length, :] - prediction[:cal_length, :])))
    
    return rmse

def pred_horizon(real, prediction, max_length=3000):
    
    horizon_matrix = np.zeros((1, max_length))
    for i in range(1, max_length):
        rmse_i = calculate_rmse(real, prediction, cal_length=i)
        
        horizon_matrix[0, i] = rmse_i
    
    return horizon_matrix














