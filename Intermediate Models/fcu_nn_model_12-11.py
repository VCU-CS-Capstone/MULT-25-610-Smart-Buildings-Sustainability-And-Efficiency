import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
df = pd.read_csv('complete_lab409_fc_feb22_feb23.csv', parse_dates=['Date'], index_col='Date')

# Step 1: Remove rows with missing values
df_cleaned = df.dropna()

# Step 2: Outlier detection using STL decomposition
def remove_outliers(df, columns, seasonal_period=24):
    df_cleaned = df.copy()
    for column in columns:
        series = df[column]
        # Directly specify the period for STL
        stl = STL(series, period=seasonal_period, seasonal=13)  # You can adjust 'seasonal' parameter as needed
        result = stl.fit()
        residual = result.resid
        residual_std = np.std(residual)
        residual_mean = np.mean(residual)
        outliers = np.abs(residual - residual_mean) > 3 * residual_std
        df_cleaned = df_cleaned[~outliers]
    return df_cleaned

df_cleaned = remove_outliers(df_cleaned, columns=['Lab 409 Room Temp (°F)', 'Discharge Air Temp (°F)'])

# Step 3: Resample the data to hourly averages
df_hourly = df_cleaned.resample('H').mean()

# Step 4: Prepare data for LSTM
def prepare_data_for_lstm(df, window=24):
    data = df[['Lab 409 Room Temp (°F)', 'Discharge Air Temp (°F)']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(window, len(data_scaled)):
        X.append(data_scaled[i-window:i, :])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

X, y, scaler = prepare_data_for_lstm(df_hourly, window=24)

# Step 5: TimeSeriesSplit for Cross Validation
tscv = TimeSeriesSplit(n_splits=5)
val_loss_per_fold = []

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 2
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 20
batch_size = 32
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cross-validation
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Training fold {fold + 1}/{tscv.get_n_splits()}...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

        if epoch == num_epochs - 1:
            val_loss_per_fold.append(val_loss.item())

# Step 6: Calculate the average validation loss across all folds
average_val_loss = np.mean(val_loss_per_fold)
print(f"Average Validation Loss: {average_val_loss}")