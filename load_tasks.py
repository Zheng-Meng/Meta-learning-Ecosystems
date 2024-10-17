# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:32:04 2023

@author: zmzhai
"""

import os
import pickle
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ReadTasks:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        # self.preprocess()
    
    def preprocess(self, random_tasks=True, train_set=None, val_set=None, test_set=None, washdata=True):
        self.train_tasks = []
        self.train_task_names = []
        self.test_tasks = []
        self.test_task_names = []
        
        # read train tasks
        for train_task_name in train_set:
            file_path = './' + self.data_dir + '/data_{}'.format(train_task_name)
            # print(file_path)
            
            with open(file_path + '.pkl', 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            
            task_name = train_task_name
            self.train_task_names.append(task_name)
    
            # wash the transient and normalize the data
            if washdata:
                random_start = np.random.randint(30000, 80000)
            else:
                random_start = np.random.randint(0, 100)
            data = data[random_start:, :3]
            # normalize the data
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            
            self.train_tasks.append(data)
        
        # read test tasks
        for test_task_name in test_set:
            file_path = './' + self.data_dir + '/data_{}'.format(test_task_name)
            # print(file_path)
            
            with open(file_path + '.pkl', 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            
            task_name = test_task_name
            self.test_task_names.append(task_name)
    
            # wash the transient and normalize the data
            if washdata:
                random_start = np.random.randint(30000, 80000)
            else:
                random_start = np.random.randint(0, 100)
            data = data[random_start:, :3]
            # normalize the data
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
            
            self.test_tasks.append(data)
        
        # shuffle the training tasks
        paired = list(zip(self.train_task_names, self.train_tasks))
        random.shuffle(paired)
        self.train_task_names, self.train_tasks = zip(*paired)
        
        # we do not create validation tasks
        self.val_tasks = []
        self.val_task_names = []

    def get_tasks(self):
        return self.train_tasks, self.val_tasks, self.test_tasks, \
            self.train_task_names, self.val_task_names, self.test_task_names
        

class ExtractTasks:
    def __init__(self, tasks, task_names, train_length: int, test_length: int):
        self.tasks = tasks
        self.task_names = task_names
        self.tasks_len = self.__len__()
        self.train_length = train_length
        self.test_length = test_length
    
    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx], self.task_names[idx]
    
    def get_random_item(self):
        idx = np.random.randint(0, self.tasks_len)
        data = self.tasks[idx]
        name = self.task_names[idx]
        return data, name
    
    def get_random_data(self):
        idx = np.random.randint(0, self.tasks_len)
        data = self.tasks[idx]
        name = self.task_names[idx]
        
        random_start_train = 10000
        random_start_test = 120000

        train_data = data[random_start_train:random_start_train+self.train_length, :]
        test_data = data[random_start_test:random_start_test+self.test_length, :]
        
        return train_data, test_data, name
    
    def get_specific_data(self, idx):
        data = self.tasks[idx]
        name = self.task_names[idx]

        random_start_train = 10000
        random_start_test = 120000

        train_data = data[random_start_train:random_start_train+self.train_length, :]
        test_data = data[random_start_test:random_start_test+self.test_length, :]
        
        return train_data, test_data, name



if __name__ == '__main__':
    data_dir = "data"
    read_tasks = ReadTasks(data_dir)
    train_tasks, val_tasks, test_tasks, train_task_names, val_task_names, test_task_names = read_tasks.get_tasks()

    extract_train_task = ExtractTasks(train_tasks, train_task_names, train_length=5000, test_length=20000)
    train_data, test_data, name = extract_train_task.get_random_data()

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))

    ax[0].plot(range(len(train_data)), train_data[:, 0], color='blue')
    ax[1].plot(range(len(train_data)), train_data[:, 1], color='blue')
    ax[2].plot(range(len(train_data)), train_data[:, 2], color='blue')

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))

    ax[0].plot(range(len(test_data)), test_data[:, 0], color='orange')
    ax[1].plot(range(len(test_data)), test_data[:, 1], color='orange')
    ax[2].plot(range(len(test_data)), test_data[:, 2], color='orange')

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(train_data[:, 0], train_data[:, 1], train_data[:, 2], color='blue')

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(test_data[:, 0], test_data[:, 1], test_data[:, 2], color='orange')
    ax.set_title(name)







































