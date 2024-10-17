# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:47:43 2023

@author: zmzhai
"""
import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(hidden_size3, output_size)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        
        out = self.fc4(out)
        return out















































