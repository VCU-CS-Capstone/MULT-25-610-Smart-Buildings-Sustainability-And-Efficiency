import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
import matplotlib.pyplot as plt
from scipy.fft import fft
import os

# Function to load and preprocess data
def load_and_preprocess(filepath, test_file_provided):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)

    df = df.dropna()
    
    # Ensure proper datetime parsing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)  # Remove rows where date parsing failed
    
    # Extract datetime features
    df['hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    
    # Split data into train and test
    if test_file_provided:
        train_df, test_df = df, None  # Use full dataset for training if a test file is provided
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Final Fault Status'], random_state=42)
    
    def process(df_subset, scaler=None):
        if df_subset is None:
            return None, None, None
        y = df_subset['Final Fault Status'].map({'N': 0, 'F': 1}).values  # Convert labels to binary
        df_subset = df_subset.drop(columns=['Final Fault Status', 'Date', 'week']).copy()
        df_subset = df_subset.apply(pd.to_numeric, errors='coerce')
        df_subset.dropna(inplace=True)  # Remove remaining NaNs
        
        if scaler is None:
            scaler = StandardScaler()
            train_X = scaler.fit_transform(df_subset)
        else:
            train_X = scaler.transform(df_subset)
        
        df_processed = train_X
        return df_processed, y, scaler
    
    X_train, y_train, scaler = process(train_df)
    X_test, y_test, _ = process(test_df, scaler)
    return X_train, y_train, X_test, y_test

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the last time step's output
        return torch.sigmoid(output)

# Train and evaluate the LSTM model
def train_and_evaluate_lstm(train_file, test_file=None, test_file_provided=False):
    # Load and preprocess data
    if test_file:
        X_train, y_train, _, _ = load_and_preprocess(train_file, test_file_provided=True)
        X_test, y_test, _, _ = load_and_preprocess(test_file, test_file_provided=True)
    else:
        X_train, y_train, X_test, y_test = load_and_preprocess(train_file, test_file_provided=False)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    # Compute class weights dynamically to balance normal and fault classes
    num_normal = (y_train == 0).sum().item()
    num_faults = (y_train == 1).sum().item()
    total = num_normal + num_faults
    
    weight_normal = total / (2.0 * num_normal)
    weight_fault = total / (2.0 * num_faults)
    
    weights = torch.tensor([weight_normal, weight_fault], dtype=torch.float32)
    
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]
        
        model = LSTMModel(X_t.shape[1], hidden_size=64, output_size=1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        for epoch in range(10):
            optimizer.zero_grad()
            y_pred = model(X_t).squeeze()
            loss = criterion(y_pred, y_t)
            loss.backward()
            optimizer.step()
        
        y_pred_val = (model(X_v).squeeze() > 0.5).int()
        fold_accuracies.append(accuracy_score(y_v.numpy(), y_pred_val.numpy()))
    
    print(f'Training Accuracy (5-Fold CV): {np.mean(fold_accuracies):.4f}')
    
    # Evaluate on test data if available
    if X_test is not None and y_test is not None:
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        # Train final model and predict on test data
        model = LSTMModel(X_train.shape[1], hidden_size=64, output_size=1)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        for epoch in range(2000):
            optimizer.zero_grad()
            y_pred = model(X_train).squeeze()
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        
        y_pred_test = (model(X_test).squeeze() > 0.3).int()
        
        # Calculate and print class distribution
        unique, counts = np.unique(y_test.numpy(), return_counts=True)
        actual_distribution = dict(zip(unique, counts))
        unique_pred, counts_pred = np.unique(y_pred_test.numpy(), return_counts=True)
        predicted_distribution = dict(zip(unique_pred, counts_pred))
        
        print(f'Actual Class Distribution: {actual_distribution}')
        print(f'Predicted Class Distribution: {predicted_distribution}')
    
if __name__ == "__main__":
    train_file = input("Enter the training data file path: ")
    use_test_file = input("Do you want to provide a separate testing file? (yes/no): ").strip().lower() == 'yes'
    test_file = input("Enter the testing data file path: ") if use_test_file else None
    
    try:
        train_and_evaluate_lstm(train_file, test_file=test_file, test_file_provided=use_test_file)
    except FileNotFoundError as e:
        print(e)
