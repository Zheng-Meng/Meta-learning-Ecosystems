# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:46:45 2025

@author: zmzhai
"""

import torch
import torch.nn as nn
import numpy as np
from ngrc_model import FNN
from torch.utils.data import DataLoader, TensorDataset
import copy
from load_tasks import ReadTasks, ExtractTasks
from utils import loss_long_term
import matplotlib.pyplot as plt
import pickle



# -----------------------------
# Configurations
# -----------------------------
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Data directories and tasks
data_dir = "data"
read_tasks = ReadTasks(data_dir)

# One source task for pretraining (transfer learning definition)
source_task = ['aizawa']          # pretraining task
target_task = ['hastings']       # fine-tuning target

read_tasks.preprocess(random_tasks=False, train_set=source_task, val_set=[], test_set=target_task)
train_tasks, _, test_tasks, train_task_names, _, test_task_names = read_tasks.get_tasks()

extract_train_task = ExtractTasks(train_tasks, train_task_names, train_length=30000, test_length=51000)
extract_test_task = ExtractTasks(test_tasks, test_task_names, train_length=30000, test_length=51000)

# Model hyperparameters (same as meta-learning)
embedding_time = 1000
input_size = 3 * embedding_time
output_size = 3
hidden1, hidden2, hidden3 = 1024, 512, 128
lr = 1e-3
batch_size = 128
noise_level = 0.003
num_epochs = 20

# -----------------------------
# Helper functions
# -----------------------------
def create_dataset(data, N):
    data_N = []
    for i in range(len(data) - N):
        data_N.append(np.squeeze(np.reshape(data[i:N + i], (-1, 1))))
    return np.array(data_N)

def train_model(model, data, optimizer, train_all_layers=True):
    """Train model on given task."""
    model.train()
    noise = np.random.normal(loc=0., scale=noise_level, size=data.shape)
    data = np.clip(data + noise, 0, 1)
    data_N = create_dataset(data, N=embedding_time)

    data_train = torch.Tensor(np.array(data_N[:30000, :]))
    target_train = torch.Tensor(np.array(data[embedding_time:30000 + embedding_time, :]))

    dataset = TensorDataset(data_train, target_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()

    for _ in range(num_epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, data):
    """Evaluate and make long-term predictions."""
    model.eval()
    data_N = create_dataset(data, N=embedding_time)
    data_test = torch.Tensor(np.array(data_N[:51000, :])).to(device)
    target_test = torch.Tensor(np.array(data[embedding_time:51000 + embedding_time, :])).to(device)
    output_size = 3

    vali_length = 51000 - 1000
    init_seq = data_test[0, :]
    preds = []

    with torch.no_grad():
        for _ in range(vali_length):
            output = model(init_seq)
            init_seq = torch.cat([init_seq, output], dim=0)[output_size:]
            preds.append(output.cpu().numpy())
    preds = np.array(preds)
    return target_test.cpu().numpy(), preds


# -----------------------------
# Pretrain on source task
# -----------------------------
train_data, _, src_name = extract_train_task.get_specific_data(idx=0)
print(f"Pretraining on source task: {src_name}")
model = FNN(input_size, hidden1, hidden2, hidden3, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
model = train_model(model, train_data, optimizer, train_all_layers=True)

# Save pretrained weights
torch.save(model.state_dict(), f"pretrained_{src_name}.pth")

# -----------------------------
# Transfer learning on target task
# -----------------------------
_, test_data, tgt_name = extract_test_task.get_specific_data(idx=0)
print(f"Fine-tuning on target task: {tgt_name}")

model_all = copy.deepcopy(model)
for param in model_all.parameters():
    param.requires_grad = True

optimizer_all = torch.optim.Adam(model_all.parameters(), lr=lr, betas=(0.9, 0.999))
model_all = train_model(model_all, test_data, optimizer_all, train_all_layers=True)
real_all, pred_all = evaluate_model(model_all, test_data)
dv_all = loss_long_term(real_all, pred_all, dt=0.04)
print(f"DV: {dv_all:.4f}")

# -----------------------------
# Visualization and saving
# -----------------------------
for label, (real, pred, dv) in {
    "All-adjustable": (real_all, pred_all, dv_all),}.items():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(real[:, 0], real[:, 1], real[:, 2], color='orange', label='real')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color='blue', label=f'pred ({label})')
    ax.set_title(f"{tgt_name} ({label}) DV={dv:.3f}")
    ax.legend()
    plt.show()

    # with open(f'./save_file/transfer_{tgt_name}_{label}.pkl', 'wb') as f:
    #     pickle.dump(real, f)
    #     pickle.dump(pred, f)
    #     pickle.dump(dv, f)
