# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:38:00 2024

@author: zmzhai
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import copy
import tqdm
from typing import List, Tuple, Optional
import pandas as pd
from load_tasks import ReadTasks
import pickle

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.layer4 = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

class ReptileTimeSeriesPredictor:
    def __init__(self,
                 meta_iterations: int = 20,
                 iterations: int = 10,
                 embedding_time: int = 30,
                 hidden_size1: int = 128,
                 hidden_size2: int = 64,
                 hidden_size3: int = 16,
                 prediction_step: int = 4,
                 batch_size: int = 16,
                 meta_lr: float = 1.0,
                 lr: float = 1e-3,
                 train_length: int = 2000,
                 noise_level: float = 0.003,
                 interpolation_factor: int = 3):
        
        self.meta_iterations = meta_iterations
        self.iterations = iterations
        self.embedding_time = embedding_time
        self.hidden_sizes = (hidden_size1, hidden_size2, hidden_size3)
        self.prediction_step = prediction_step
        self.batch_size = batch_size
        self.meta_lr = meta_lr
        self.lr = lr
        self.train_length = train_length
        self.noise_level = noise_level
        self.interpolation_factor = interpolation_factor
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        
        # Initialize meta-model
        self.input_size = embedding_time
        self.output_size = 1
        self.meta_model = self._initialize_model()
        self.meta_optimizer = torch.optim.SGD(self.meta_model.parameters(), lr=self.meta_lr)

    def _initialize_model(self):
        model = FNN(
            input_size=self.input_size,
            hidden_size1=self.hidden_sizes[0],
            hidden_size2=self.hidden_sizes[1],
            hidden_size3=self.hidden_sizes[2],
            output_size=self.output_size
        ).to(self.device)
        return model

    def interpolate_data(self, data):
        """Interpolate data to create smoother transitions"""
        x_original = np.arange(len(data))
        x_interpolated = np.linspace(0, len(data)-1, len(data)*self.interpolation_factor)
        return np.interp(x_interpolated, x_original, data.flatten())

    def preprocess_data(self, data):
        """Preprocess the input data"""

        # Normalize
        data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        return data

    def create_dataset(self, data, N):
        """Create sequences for training"""
        data_N = []
        for i in range(len(data)-N):
            data_N.append(np.squeeze(np.reshape(data[i:N+i], (-1, 1))))
        return np.array(data_N)

    def _get_optimizer(self, model, state=None):
        """Initialize optimizer for inner loop"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        if state is not None:
            optimizer.load_state_dict(state)
        return optimizer

    def _set_learning_rate(self, optimizer, lr):
        """Update learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _inner_loop_training(self, model, data, optimizer):
        """Perform inner loop training"""
        model.train()
        # Add noise
        noise = np.random.normal(loc=0., scale=self.noise_level, size=data.shape)
        data = np.clip(data + noise, 0, 1)
        
        # Prepare data
        data_N = self.create_dataset(data, self.embedding_time)
        data_train = torch.Tensor(data_N[:self.train_length]).to(self.device)
        target_train = torch.Tensor(data[self.embedding_time:self.train_length+self.embedding_time]).to(self.device)
        
        dataset = TensorDataset(data_train, target_train)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        loss_fn = nn.MSELoss()
        
        for _ in range(self.iterations):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                
        return model, loss.item()

    def meta_train(self, train_tasks: List[np.ndarray], task_sample_num: int = 2):
        """Perform meta-training using Reptile algorithm"""
        print("Starting meta-training...")
        
        for meta_iteration in tqdm.trange(self.meta_iterations):
            # Update learning rate
            current_meta_lr = self.meta_lr * (1. - meta_iteration / float(self.meta_iterations))
            self._set_learning_rate(self.meta_optimizer, current_meta_lr)
            
            sum_gradients = None
            for task_idx in range(task_sample_num):
                # Clone model
                model = copy.deepcopy(self.meta_model)
                optimizer = self._get_optimizer(model)
                
                # Sample random task
                task_data = train_tasks[np.random.randint(len(train_tasks))]
                task_data = self.preprocess_data(task_data)
                
                # Inner loop training
                model_updated, _ = self._inner_loop_training(model, task_data, optimizer)
                
                # Accumulate gradients
                if sum_gradients is None:
                    sum_gradients = [(param_updated.data - param.data) 
                                   for param, param_updated in zip(self.meta_model.parameters(), 
                                                                model_updated.parameters())]
                else:
                    sum_gradients = [(sum_grad + (param_updated.data - param.data)) 
                                   for sum_grad, param, param_updated in zip(sum_gradients,
                                                                          self.meta_model.parameters(),
                                                                          model_updated.parameters())]
            
            # Update meta-model
            for param, sum_grad in zip(self.meta_model.parameters(), sum_gradients):
                param.data += (current_meta_lr / task_sample_num) * sum_grad

    def adapt_and_predict(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Adapt meta-model to test data and make predictions"""
        # Preprocess test data
        test_data = self.interpolate_data(test_data)
        test_data = self.preprocess_data(test_data)
        test_length = len(test_data)
        train_data = test_data[:int(test_length*0.7)]
        test_data = test_data[int(test_length*0.7):]
        
        # Adapt model
        adapted_model = copy.deepcopy(self.meta_model)
        optimizer = self._get_optimizer(adapted_model)
        self.noise_level = 0
        adapted_model, _ = self._inner_loop_training(adapted_model, train_data, optimizer)
        
        # Make predictions
        predictions, true_values = self._multi_step_predict(adapted_model, test_data)
        
        return predictions, true_values

    def _multi_step_predict(self, model, data):
        """Perform multi-step prediction"""
        model.eval()
        predictions_all = []
        # Adjust true values to match prediction length
        # true_values = data[self.embedding_time:-self.prediction_step]  
        true_values = []
        
        with torch.no_grad():
            for t in range(len(data) - self.embedding_time - self.prediction_step):
                current_input = torch.Tensor(data[t:t+self.embedding_time].reshape(1, -1)).to(self.device)
                step_predictions = []
                
                true_sequence = data[t+self.embedding_time:t+self.embedding_time+self.prediction_step]
                true_values.append(true_sequence)
                
                for _ in range(self.prediction_step):
                    output = model(current_input)
                    step_predictions.append(output.cpu().numpy())
                    current_input = torch.cat([
                        current_input[:, self.output_size:],
                        output
                    ], dim=1)
                
                predictions_all.append(np.array(step_predictions).reshape(self.prediction_step, -1))
        
        return np.array(predictions_all), np.array(true_values)
    
    def plot_predictions(self, predictions, true_values, title="Prediction Results"):
        """Plot the prediction results"""
        plt.figure(figsize=(12, 6))
        
        time_steps = np.arange(predictions[:].shape[0])
        selected_predictions = np.zeros((len(predictions), 1))
        
        for t in range(len(true_values)):
            if t < self.prediction_step:
                selected_predictions[t] = predictions[t, 0, 0]
            else:
                # Take prediction that was made 'prediction_step' steps ago
                step_idx = t % self.prediction_step
                sequence_idx = t - step_idx
                if sequence_idx < len(predictions):
                    selected_predictions[t] = predictions[sequence_idx, step_idx, 0]
        
        
        plt.plot(time_steps, true_values[:, 0, 0], 'k-', label='Ground truth', alpha=0.7)
        plt.plot(time_steps, selected_predictions, label='Predicted')
        
        # Calculate MSE only for the available predictions
        mse_per_step = np.mean((predictions - true_values)**2, axis=0)
        plt.title(f'{title}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return mse_per_step

# Example usage
if __name__ == "__main__":
    print(1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    