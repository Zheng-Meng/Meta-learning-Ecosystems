# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:02:40 2024

@author: zmzhai
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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

class TimeSeriesPredictor:
    def __init__(self, 
                 embedding_time=30,
                 hidden_size1=128,
                 hidden_size2=64,
                 hidden_size3=16,
                 prediction_step=4,
                 batch_size=1,
                 learning_rate=1e-3,
                 iterations=10,
                 interpolation_factor=3):
        
        self.embedding_time = embedding_time
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.prediction_step = prediction_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.interpolation_factor = interpolation_factor
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        
    def interpolate_data(self, data):
        """Interpolate data to create smoother transitions"""
        x_original = np.arange(len(data))
        x_interpolated = np.linspace(0, len(data)-1, len(data)*self.interpolation_factor)
        return np.interp(x_interpolated, x_original, data.flatten())
    
    def create_dataset(self, data, N):
        """Create sequences for training"""
        data_N = []
        for i in range(len(data)-N):
            data_N.append(np.squeeze(np.reshape(data[i:N+i], (-1, 1))))
        return np.array(data_N)
    
    def preprocess_data(self, data):
        """Preprocess the input data"""
        # Ensure data is numpy array
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.values
        
        # Interpolate
        data = self.interpolate_data(data)
        
        # Normalize
        data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        return data
    
    def train(self, data, verbose=True):
        """Train the model on the provided data"""
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Split into train/test
        test_length = data.shape[0]
        train_data = data[:int(test_length*0.7)]
        
        # Create sequences
        X_train = self.create_dataset(train_data, self.embedding_time)
        y_train = train_data[self.embedding_time:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Initialize model
        input_size = self.embedding_time
        output_size = 1
        self.model = FNN(input_size, self.hidden_size1, self.hidden_size2, 
                        self.hidden_size3, output_size).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        if verbose:
            print("Training started...")
        for epoch in range(self.iterations):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # if verbose and (epoch + 1) % 2 == 0:
            #     print(f'Epoch [{epoch+1}/{self.iterations}], Loss: {epoch_loss/len(train_loader):.6f}')
    
    def predict_multi_step(self, data):
        """Make multi-step predictions"""
        data = self.preprocess_data(data)
        test_length = data.shape[0]
        test_data = data[int(test_length*0.7):]
        
        predictions, true_values = self._multi_step_prediction(test_data)
        return predictions, true_values, data
    
    def _multi_step_prediction(self, data):
        """Internal method for multi-step prediction"""
        self.model.eval()
        predictions_all = []
        true_values = []
        
        with torch.no_grad():
            for t in range(len(data) - self.embedding_time - self.prediction_step):
                current_sequence = data[t:t+self.embedding_time].reshape(1, -1)
                current_sequence = torch.FloatTensor(current_sequence).to(self.device)
                
                true_sequence = data[t+self.embedding_time:t+self.embedding_time+self.prediction_step]
                true_values.append(true_sequence)
                
                step_predictions = []
                for _ in range(self.prediction_step):
                    pred = self.model(current_sequence)
                    step_predictions.append(pred.cpu().numpy())
                    current_sequence = torch.cat([
                        current_sequence[:, 1:],
                        pred
                    ], dim=1)
                
                predictions_all.append(np.array(step_predictions).reshape(self.prediction_step, -1))
        
        return np.array(predictions_all), np.array(true_values)
    
    def plot_results(self, predictions, true_values, title="Prediction Results"):
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

    
    
    
    