# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:26:58 2023

@author: zmzhai
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from utils import *


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

attractors = ['aizawa', 'bouali', 'chua', 'dadras', 'foodchain', 
              'four_wing', 'hastings', 'lorenz', 'lotka', 'rikitake', 'rossler',
              'sprott_00', 'sprott_01', 'sprott_02', 'sprott_03', 'sprott_04', 'sprott_05',
              'sprott_06', 'sprott_07','sprott_08','sprott_09','sprott_10','sprott_11',
              'sprott_12','sprott_13','sprott_14', 'sprott_15', 'sprott_16',
              'sprott_17', 'sprott_18', 'wang']

attractor = attractors[6]
# attractor = attractors[4]
# attractor = attractors[8]
 
file_path = './data/data_{}.pkl'.format(attractor)
pkl_file = open(file_path, 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

time_embedding = 1000
output_size = 3
num_epochs = 20
batch_size = 128

# wash data first, and take the length of data
random_start = np.random.randint(30000, 80000)
train_length = 30000
test_length = 50000 + time_embedding
# test_length = 2100
data = data[random_start:random_start+train_length+test_length, :output_size]
# normalizae the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

noise = np.random.normal(loc=0., scale=0.003, size=data.shape)
data = np.clip(data + noise, 0, 1)

def create_dataset(data, N):
    data_N = []
    for i in range(len(data)-N):
        data_N.append( np.squeeze(np.reshape(data[i:N+i], (-1, 1))))
    return np.array(data_N)
    
data_N = create_dataset(data, N=time_embedding)
input_size = np.shape(data_N)[1]

data_train = torch.Tensor(np.array(data_N[:train_length, :]))
target_train = torch.Tensor(np.array(data[time_embedding:train_length+time_embedding, :]))
data_test = torch.Tensor(np.array(data_N[train_length:, :]))
target_test = torch.Tensor(np.array(data[train_length+time_embedding:, :]))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(data[:train_length, 0], data[:train_length, 1], data[:train_length, 2], color='blue')

ax.set_title('train data')

plt.show()

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


model = FNN(input_size=input_size, hidden_size1=1024, hidden_size2=512, 
            hidden_size3=128, output_size=output_size)

model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), betas=(0.9, 0.999))
loss_fn = nn.MSELoss()

dataset = TensorDataset(data_train, target_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# learning_rate: dict = {0: 1e-3}
for epoch in range(num_epochs):
    epoch_iterator = tqdm(train_loader)
    for (data_train_epoch, target_train_epoch) in epoch_iterator:
        data_train_epoch = data_train_epoch.to(device)
        target_train_epoch = target_train_epoch.to(device)
        
        epoch += 1
        loss = 0
        optimizer.zero_grad()
    
        output = model(data_train_epoch)
        loss = loss_fn(output, target_train_epoch)
        # loss = loss_fn(output[:, -1, :], target_epoch[:, -1, :])
        epoch_iterator.set_description(f"Loss={loss.item()}")
    
        loss.backward()
        optimizer.step()


vali_length = 50000
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
    predictions.append(output.detach().cpu().numpy())  # Detach from the computation graph and move to cpu

predictions = np.array(predictions)

def mse_calculation(A, B):
    # actual we calculate mse
    return (np.square(np.subtract(A, B)).mean())

mse = mse_calculation(target_test[:num_steps, :].cpu().numpy(), predictions)  # move to cpu before numpy operation

fig, ax = plt.subplots(3, 1, figsize=(8,6))

pred_length = 1000
ax[0].plot(predictions[:pred_length,  0], label='pred')
ax[0].plot(target_test[:pred_length, 0], label='real')

ax[1].plot(predictions[:pred_length, 1], label='pred')
ax[1].plot(target_test[:pred_length, 1], label='real')

ax[2].plot(predictions[:pred_length, 2], label='pred')
ax[2].plot(target_test[:pred_length, 2], label='real')

plt.legend()
plt.show()



dv = loss_long_term(target_test, predictions, dt=0.04)

print('dv: ', dv)

# pkl_file = open('./save_file/' + 'time_series_fnn'+ '.pkl', 'wb')
# pickle.dump(target_test, pkl_file)
# pickle.dump(predictions, pkl_file)
# pickle.dump(dv, pkl_file)
# pkl_file.close()


ax = plt.figure().add_subplot(projection='3d')
ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], color='blue', label='pred')
ax.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(target_test[:, 0].numpy(), target_test[:, 1].numpy(), target_test[:, 2].numpy(), color='orange', label='real')
ax.legend()
plt.show()


# print(predictions.min(), predictions.max())






