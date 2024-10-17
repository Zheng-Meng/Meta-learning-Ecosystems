# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:33:31 2023

@author: zmzhai
"""
import os
import torch
import torch.nn as nn
import numpy as np
from ngrc_model import FNN
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import tqdm
import argparse
import copy
from load_tasks import ReadTasks, ExtractTasks
import matplotlib.pyplot as plt
import json
from utils import *
from scipy.spatial.distance import directed_hausdorff
# Parsing
parser = argparse.ArgumentParser('Train reptile on chaos')
parser.add_argument('--logdir', default='logdir', help='Folder to store everything/load')

# Training params
parser.add_argument('--meta-iterations', default=30, type=int, help='number of meta iterations')
parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
parser.add_argument('--iterations', default=10, type=int, help='number of base iterations')
parser.add_argument('--test-iterations', default=10, type=int, help='number of base iterations')
parser.add_argument('--batch', default=128, type=int, help='minibatch size in base task')
parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
parser.add_argument('--validate-every', default=5, type=int, help='Meta-evaluation every ... base-tasks')
parser.add_argument('--train-length', default=30000, type=int, help='inner training length')
parser.add_argument('--test-length', default=51000, type=int, help='testing length')
parser.add_argument('--train-scale', default=1.0, type=float, help='scale the training length')
parser.add_argument('--noise-level', default=0.003, type=float, help='add noise to the data')

# Transformer params
parser.add_argument('--d-obs', default=3, type=int, help='number of observations')
parser.add_argument('--embedding-time', default=1000, type=int, help='embedding time')
parser.add_argument('--hidden-size1', default=1024, type=int, help='hidden state 1')
parser.add_argument('--hidden-size2', default=512, type=int, help='hidden state 2')
parser.add_argument('--hidden-size3', default=128, type=int, help='hidden state 3')


args = parser.parse_args()
print(args)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load data
data_dir = "data"
read_tasks = ReadTasks(data_dir)

task_sample_num = 2

attractors = ['aizawa', 'bouali', 'chua', 'dadras', 'foodchain', 
              'four_wing', 'hastings', 'lorenz', 'lotka', 'rikitake', 'rossler',
              'sprott_00', 'sprott_01', 'sprott_02', 'sprott_03', 'sprott_04', 'sprott_05',
              'sprott_06', 'sprott_07','sprott_08','sprott_09','sprott_10','sprott_11',
              'sprott_12','sprott_13','sprott_14', 'sprott_15', 'sprott_16',
              'sprott_17', 'sprott_18', 'wang']

attractors_in  = ['aizawa', 'bouali', 'chua', 'sprott_03', 'sprott_14']
attractors_test = ['foodchain', 'hastings', 'lotka']

train_set = attractors_in
val_set = []
test_set = attractors_test

read_tasks.preprocess(random_tasks=False, train_set=train_set, val_set=val_set, test_set=test_set)
train_tasks, val_tasks, test_tasks, train_task_names, val_task_names, test_task_names = read_tasks.get_tasks()

extract_train_task = ExtractTasks(train_tasks, train_task_names, train_length=args.train_length, test_length=args.test_length)
extract_val_task = ExtractTasks(val_tasks, val_task_names, train_length=args.train_length, test_length=args.test_length)
extract_test_task = ExtractTasks(test_tasks, test_task_names, train_length=args.train_length, test_length=args.test_length)

input_size = 3 * args.embedding_time
output_size = 3
# Initialize transformer model
meta_model = FNN(input_size=input_size, hidden_size1=args.hidden_size1, hidden_size2=args.hidden_size2, 
            hidden_size3=args.hidden_size3, output_size=output_size)

meta_model = meta_model.to(device)
meta_optimizer = torch.optim.SGD(meta_model.parameters(), lr=1.)
info = {}
state = None

def create_dataset(data, N):
    data_N = []
    for i in range(len(data)-N):
        data_N.append( np.squeeze(np.reshape(data[i:N+i], (-1, 1))))
    return np.array(data_N)
    
def learning_process(model, data, optimizer, iterations):
    model.train()
    # add noise
    noise = np.random.normal(loc=0., scale=args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)
    
    data_N = create_dataset(data, N=args.embedding_time)
    # data = torch.Tensor(np.array(data))
    
    data_train = torch.Tensor(np.array(data_N[:args.train_length, :]))
    target_train = torch.Tensor(np.array(data[args.embedding_time:args.train_length+args.embedding_time, :]))

    dataset = TensorDataset(data_train, target_train)
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    loss_fn = nn.MSELoss()
    
    for iteration in range(iterations):
        for (data_train_epoch, target_train_epoch) in train_loader:
            data_train_epoch = data_train_epoch.to(device)
            target_train_epoch = target_train_epoch.to(device)
            
            loss = 0
            optimizer.zero_grad()
        
            output = model(data_train_epoch)
            loss = loss_fn(output, target_train_epoch)
            
            loss.backward()
            optimizer.step()
        
        # print(f"Inner loop Iteration {iteration}, Loss: {loss.item()}")

    return model, loss.item()

def validation_process(model, data):
    model.eval()
    
    noise = np.random.normal(loc=0., scale=args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)
    
    data_N = create_dataset(data, N=args.embedding_time)
    # data = torch.Tensor(np.array(data))
    
    data_train = torch.Tensor(np.array(data_N[:args.train_length, :]))
    target_train = torch.Tensor(np.array(data[args.embedding_time:args.train_length+args.embedding_time, :]))
    # data_test = torch.Tensor(np.array(data_N[args.train_length:, :]))
    # target_test = torch.Tensor(np.array(data[args.train_length+args.embedding_time:, :]))

    dataset = TensorDataset(data_train, target_train)
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    
    losses = []
    
    for (data_train_epoch, target_train_epoch) in train_loader:
        data_train_epoch = data_train_epoch.to(device)
        target_train_epoch = target_train_epoch.to(device)
        
        loss = 0
        # optimizer.zero_grad()
        with torch.no_grad():
            output = model(data_train_epoch)
        loss = loss_fn(output, target_train_epoch)
        losses.append(loss.item())
        
    loss_mean = np.mean(losses)
    
    return loss_mean, losses


def long_term_validation(model, data):
    model.eval()
    
    noise = np.random.normal(loc=0., scale=args.noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)
    
    data_N = create_dataset(data, N=args.embedding_time)
    # data = torch.Tensor(np.array(data))
    
    # data_train = torch.Tensor(np.array(data_N[:args.train_length, :]))
    # target_train = torch.Tensor(np.array(data[args.embedding_time:args.train_length+args.embedding_time, :]))
    data_test = torch.Tensor(np.array(data_N[:args.test_length, :]))
    target_test = torch.Tensor(np.array(data[args.embedding_time:args.test_length+args.embedding_time, :]))
    
    vali_length = args.test_length - 1000
    num_steps = vali_length  # the number of steps you want to predict
    predictions = []

    init_seq = data_test[0, :]

    for i in range(num_steps):
        data_input_i = init_seq
        data_input_i = data_input_i.to(device)
        
        # We don't need to compute gradients for validation, so wrap in no_grad to save memory
        with torch.no_grad():
            output = model(data_input_i)
        
        init_seq = torch.cat([init_seq.to(device), output.to(device)], dim=0)
        init_seq = init_seq[output_size:]
        
        # Store the prediction
        predictions.append(output.detach().cpu().numpy())   # Detach from the computation graph and move to cpu

    predictions = np.array(predictions)
    
    return target_test.cpu().numpy(), predictions
    
def get_optimizer(model, state=None):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Main loop
train_names = []
vali_losses = []
vali_names = []
for meta_iteration in tqdm.trange(args.start_meta_iteration, args.meta_iterations):
    # Update learning rate
    meta_lr = args.meta_lr * (1. - meta_iteration / float(args.meta_iterations))
    set_learning_rate(meta_optimizer, meta_lr)
    
    sum_gradients = None
    name_set = set()
    task_num = 0
    
    print('')
    while task_num < task_sample_num:
        # Clone model
        model = copy.deepcopy(meta_model)
        optimizer = get_optimizer(model, state)
        # Sample a task
        train_data, _, name = extract_train_task.get_random_data()
        if name in name_set:
            continue
        # if args.train_scale < 1.0:
        #     train_data = train_data[:round(args.train_scale * train_len), :]
        print('train task: ', name)
        # Run inner loop
        model_copy, loss = learning_process(model, train_data, optimizer, args.iterations)
        
        if sum_gradients is None:
            sum_gradients = [(param_copy.data - param.data) for param, param_copy in zip(meta_model.parameters(), model_copy.parameters())]
        else:
            sum_gradients = [(sum_grad + (param_copy.data - param.data)) for sum_grad, param, param_copy in zip(sum_gradients, meta_model.parameters(), model_copy.parameters())]
        
        name_set.add(name)
        task_num += 1
        
    # Update meta network
    for param, sum_grad in zip(meta_model.parameters(), sum_gradients):
        param.data += (meta_lr / task_sample_num) * sum_grad
    

# testing phase
test_losses = []
test_names = []
test_reals = []
test_predcitions = []
test_dv = []
for idx in range(extract_test_task.__len__()):
    # Clone model
    model = copy.deepcopy(meta_model)
    optimizer = get_optimizer(model, state=None)
    # Sample a task
    train_data, test_data, name = extract_test_task.get_specific_data(idx=idx)
    # if args.train_scale < 1.0:
    #     train_data = train_data[:round(args.train_scale * train_len), :]
    print('task: ', name)
    # Run inner loop
    model_copy, loss = learning_process(model, train_data, optimizer, args.iterations)
    
    loss_mean, losses = validation_process(model_copy, test_data)
    real, prediction = long_term_validation(model_copy, test_data)
    
    dv = loss_long_term(real, prediction, dt=0.04)
    
    test_losses.append(loss_mean)
    test_dv.append(dv)
    test_names.append(name)
    test_reals.append(real)
    test_predcitions.append(prediction)


for idx in range(extract_test_task.__len__()):
    real = test_reals[idx]
    prediction = test_predcitions[idx]
    name = test_names[idx]
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(real[:, 0], real[:, 1], real[:, 2], color='orange', label='real')
    ax.set_title(name)
    ax.legend()
    plt.show()
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color='blue', label='pred')
    ax.set_title(name)
    ax.legend()
    plt.show()













