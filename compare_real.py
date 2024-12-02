# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:28:25 2024

@author: zmzhai
"""

import numpy as np
import torch
from fnn_real import TimeSeriesPredictor
from reptile_real import ReptileTimeSeriesPredictor
import matplotlib.pyplot as plt
from load_tasks import ReadTasks
import pickle
from sklearn.preprocessing import MinMaxScaler

def load_population_data(file_path: str = './real_data/long_series.pkl') -> dict:
    """Load the saved population data"""
    with open(file_path, 'rb') as f:
        return np.load(f, allow_pickle=True).item()

def test_models(data: np.ndarray, idx: int=9684, prediction_step: int=4):
    """Test both FNN and Reptile models on the same data"""
    # Initialize FNN predictor
    fnn_predictor = TimeSeriesPredictor(
        embedding_time=30,
        hidden_size1=128,
        hidden_size2=64,
        hidden_size3=16,
        prediction_step=prediction_step,
        iterations=10,
        interpolation_factor=3
    )
    
    # Initialize Reptile predictor
    reptile_predictor = ReptileTimeSeriesPredictor(
        meta_iterations=20,
        iterations=10,
        embedding_time=30,
        hidden_size1=128,
        hidden_size2=64,
        hidden_size3=16,
        prediction_step=prediction_step,
        interpolation_factor=3
    )
    
    # Train FNN
    print("Training FNN...")
    fnn_predictor.train(data, verbose=False)
    fnn_predictions, fnn_true, processed_data = fnn_predictor.predict_multi_step(data)
    
    # Train and adapt Reptile
    print("Training Reptile...")
    
    d_obs = 1
    data_shrink = 2
    
    data_dir = "data"
    read_tasks = ReadTasks(data_dir)
    train_set = ['aizawa', 'bouali', 'chua', 'sprott_03', 'sprott_14']
    
    train_tasks = []
    train_task_names = []
    # read train tasks
    for train_task_name in train_set:
        file_path = './' + data_dir + '/data_{}'.format(train_task_name)
        
        with open(file_path + '.pkl', 'rb') as pkl_file:
            data_c = pickle.load(pkl_file)

        random_start = np.random.randint(30000, 80000)

        data_c = data_c[random_start:, :d_obs]
        data_c = data_c[::data_shrink, :]
        # normalize the data
        scaler = MinMaxScaler()
        data_c = scaler.fit_transform(data_c)
        
        train_tasks.append(data_c)
        train_task_names.append(train_task_name)
        
    reptile_predictor.meta_train(train_tasks, task_sample_num=2)
    reptile_predictions, reptile_true = reptile_predictor.adapt_and_predict(data)
    
    # Plot comparison
    plt.figure(figsize=(15, 6))
    time_steps = np.arange(min(len(fnn_predictions), len(reptile_predictions)))
    
    # Plot true values
    plt.plot(time_steps, fnn_true[:len(time_steps), 0], 'k-', label='Ground truth', alpha=0.7)
    
    selected_predictions_fnn = np.zeros((len(fnn_predictions), 1))
    
    for t in range(len(fnn_true)):
        if t < prediction_step:
            selected_predictions_fnn[t] = fnn_predictions[t, 0, 0]
        else:
            # Take prediction that was made 'prediction_step' steps ago
            step_idx = t % prediction_step
            sequence_idx = t - step_idx
            if sequence_idx < len(fnn_predictions):
                selected_predictions_fnn[t] = fnn_predictions[sequence_idx, step_idx, 0]
    
    selected_predictions_reptile = np.zeros((len(reptile_predictions), 1))
    
    for t in range(len(fnn_true)):
        if t < prediction_step:
            selected_predictions_reptile[t] = reptile_predictions[t, 0, 0]
        else:
            # Take prediction that was made 'prediction_step' steps ago
            step_idx = t % prediction_step
            sequence_idx = t - step_idx
            if sequence_idx < len(reptile_predictions):
                selected_predictions_reptile[t] = reptile_predictions[sequence_idx, step_idx, 0]
    
    
    plt.plot(time_steps, selected_predictions_fnn, label='Predicted by FNN')
    plt.plot(time_steps, selected_predictions_reptile, label='Predicted by Meta-learning')
    # Calculate MSE only for the available predictions
    plt.title(f'Data ID {idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate and print MSE for both models
    fnn_mse = np.mean((fnn_predictions - fnn_true)**2, axis=0)
    reptile_mse = np.mean((reptile_predictions - reptile_true)**2, axis=0)

    # with open('population_prediction_results.pkl', 'wb') as f:
    #     pickle.dump(fnn_predictions, f)
    #     pickle.dump(fnn_true, f)
    #     pickle.dump(reptile_predictions, f)
    #     pickle.dump(reptile_true, f)
    #     pickle.dump(processed_data, f)
    #     pickle.dump(prediction_steps, f)
    #     pickle.dump(106, f)

    
    print("\nFNN MSE per step:", fnn_mse.flatten())
    print("Reptile MSE per step:", reptile_mse.flatten())

if __name__ == "__main__":
    # Load data
    population_data = load_population_data()
    
    # Get first series
    idx = list(population_data.keys())[575]
    time_series = population_data[idx]
    
    print(f"Testing on series ID: {idx}")
    print(f"Series length: {len(time_series)}")
    
    # Test both models
    test_models(time_series, idx)



















































