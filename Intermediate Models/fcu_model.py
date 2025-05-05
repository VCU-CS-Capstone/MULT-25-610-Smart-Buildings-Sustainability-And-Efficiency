'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import accuracy_score

# Load dataset (Assume CSV format)
data = pd.read_csv("fcu_training_data.csv")

# Debugging Step: Check column names
print("Columns in dataset:", data.columns)

# Ensure 'Date' column exists
if "Date" not in data.columns:
    raise ValueError("Error: 'Date' column not found in the dataset.")

# Convert 'Date' to string first (if needed), then to datetime
data["Date"] = pd.to_datetime(data["Date"].astype(str), errors='coerce')

# Drop rows where timestamp conversion failed
data = data.dropna(subset=["Date"])

# Set 'Date' as the index
data = data.set_index("Date")

print("Successfully loaded dataset with Date column as index.")

# Step 1: Preprocessing and Cleaning
print("Step 1: Preprocessing and Cleaning started...")
# Handle missing values
data = data.dropna()

# Seasonal Decomposition to capture seasonality for all features
seasonal_components = {}
for column in data.columns:
    result = seasonal_decompose(data[column], model='additive', period=96, extrapolate_trend='freq')
    data[f'{column}_Seasonal'] = result.seasonal
    seasonal_components[column] = result.seasonal
print("Step 1: Completed preprocessing and seasonal decomposition.")

# Normalize data
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# Step 2: Unsupervised Anomaly Detection (Learning Normal & Fault Behavior)
print("Step 2: Anomaly detection started...")
# Autoencoder Model in PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_dim = data_scaled.shape[1]
autoencoder = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

X_train, X_test = train_test_split(data_scaled.values, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Train the autoencoder
epochs = 100
batch_size = 32
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
print("Step 2: Autoencoder training completed.")

# Anomaly Score Calculation
reconstructions = autoencoder(torch.tensor(data_scaled.values, dtype=torch.float32)).detach().numpy()
mse = np.mean(np.power(data_scaled.values - reconstructions, 2), axis=1)
mse_threshold = np.percentile(mse, 95)  # Threshold for anomaly detection
anomalies_autoencoder = mse > mse_threshold
print("Step 2: Autoencoder-based anomaly detection completed.")

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train.astype(np.float32))
iso_preds = iso_forest.predict(data_scaled.values.astype(np.float32))
anomalies_iso = iso_preds == -1
print("Step 2: Isolation Forest anomaly detection completed.")

# Combine anomaly detections
final_anomalies = anomalies_autoencoder | anomalies_iso
data["Anomaly"] = final_anomalies
print("Step 2: Anomaly detection completed.")

# Compute anomaly detection accuracy (if ground truth exists)
if "True_Anomaly" in data.columns:  # Assuming you have a ground truth anomaly label
    accuracy = accuracy_score(data["True_Anomaly"], data["Anomaly"])
    print(f"Step 2: Anomaly Detection Accuracy: {accuracy:.4f}")

# Step 3: Forecasting with LSTM (Predicting Future Faults)
print("Step 3: Forecasting started...")
sequence_length = 4  # 1 hour ahead
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled.iloc[i:i+sequence_length].values)
    y.append(data_scaled.iloc[i+sequence_length].values)
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

lstm_model = LSTMModel(input_dim)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

# Train LSTM Model
epochs = 200
for epoch in range(epochs):
    optimizer_lstm.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_lstm.step()
print("Step 3: LSTM model training completed.")

# Evaluate LSTM Model
with torch.no_grad():
    test_predictions = lstm_model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor).item()
    print(f"Step 3: LSTM Model Testing Loss: {test_loss:.4f}")

# Save results
data.to_csv("anomaly_detected_fan_coil.csv")
print("Step 4: Results saved to anomaly_detected_fan_coil.csv.")
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import accuracy_score

# Load dataset (Assume CSV format)
train_file_path = input("Enter the path to the training dataset CSV file: ").strip()
test_file_path = input("Enter the path to the testing dataset CSV file: ").strip()
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Ensure test data columns match training data columns
test_data = test_data.rename(columns=lambda x: x.strip())  # Remove extra spaces
test_data = test_data.rename(columns={col: train_col for col, train_col in zip(test_data.columns, train_data.columns)})

# Ensure test data has the same columns as training data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
print(f"Loaded training dataset: {train_file_path}")
print(f"Loaded testing dataset: {test_file_path}")

# Ensure 'Date' column exists
if "Date" not in train_data.columns:
    raise ValueError("Error: 'Date' column not found in the training dataset.")
if "Date" not in test_data.columns:
    raise ValueError("Error: 'Date' column not found in the testing dataset.")

# Convert 'Date' to datetime
train_data["Date"] = pd.to_datetime(train_data["Date"].astype(str), errors='coerce')
test_data["Date"] = pd.to_datetime(test_data["Date"].astype(str), errors='coerce')

# Drop rows where timestamp conversion failed
train_data = train_data.dropna(subset=["Date"])
test_data = test_data.dropna(subset=["Date"])

# Set 'Date' as the index
train_data = train_data.set_index("Date")
test_data = test_data.set_index("Date")

# Remove faulty data from training dataset
train_data = train_data[(train_data['Lab 409 Room Temp (°F)'] >= train_data['Lab 409 Room Temp Setpoint (°F)'] - 2) & 
                        (train_data['Lab 409 Room Temp (°F)'] <= train_data['Lab 409 Room Temp Setpoint (°F)'] + 2)]

'''# Remove rows where any feature deviates more than 1 standard deviation from the previous week's moving average
window_size = 7 * 96  # 7 days * 96 readings per day (15 min intervals)
for column in train_data.columns:
    if column not in ['Lab 409 Room Temp Setpoint (°F)', 'Lab 409 Room Temp (°F)']:
        rolling_mean = train_data[column].rolling(window=window_size, min_periods=1).mean()
        rolling_std = train_data[column].rolling(window=window_size, min_periods=1).std()
        train_data = train_data[(train_data[column] >= rolling_mean - rolling_std) & (train_data[column] <= rolling_mean + rolling_std)]
        '''

# Ensure all columns are numeric
train_data = train_data.apply(pd.to_numeric, errors='coerce')
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# Drop NaN values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Normalize data
scaler = MinMaxScaler()
train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)
test_data_scaled = np.clip(test_data_scaled, 0, 1)

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_dim = train_data_scaled.shape[1]
autoencoder = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

X_train, X_valid = train_test_split(train_data_scaled.values, test_size=0.1, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(test_data_scaled.values, dtype=torch.float32)

# Train Autoencoder
for epoch in range(100):
    optimizer.zero_grad()
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        val_outputs = autoencoder(X_valid_tensor)
        val_loss = criterion(val_outputs, X_valid_tensor)
    print(f"Epoch {epoch+1}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}")

# LSTM Forecasting Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

lstm_model = LSTMModel(input_dim)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)

# Prepare data for LSTM
sequence_length = 96
X, y = [], []
for i in range(len(train_data_scaled) - sequence_length):
    X.append(train_data_scaled.iloc[i:i+sequence_length].values)
    y.append(train_data_scaled.iloc[i+sequence_length].values)
X, y = np.array(X), np.array(y)
X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.float32)

# Train LSTM Model
for epoch in range(50):
    optimizer_lstm.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_lstm.step()
    print(f"Epoch {epoch+1}, LSTM Training Loss: {loss.item():.6f}")

print("Training completed for Autoencoder and LSTM.")

# Step 3: Testing Autoencoder
print("Testing Autoencoder on test data...")
with torch.no_grad():
    test_reconstructions = autoencoder(X_test_tensor).detach().numpy()
test_mse = np.mean(np.power(test_data_scaled.values - test_reconstructions, 2), axis=1)
test_mse_threshold = np.percentile(test_mse, 98)
test_anomalies = test_mse > test_mse_threshold
test_data["Anomaly"] = test_anomalies
test_data["Anomaly_Score"] = test_mse
print(f"Total anomalies detected in test data: {test_anomalies.sum()} ({(test_anomalies.sum()/len(test_anomalies)) * 100:.2f}%)")

# Step 4: Testing LSTM Forecasting
print("Testing LSTM model for forecasting...")
X_test_lstm, y_test_lstm = [], []
for i in range(len(test_data_scaled) - sequence_length):
    X_test_lstm.append(test_data_scaled.iloc[i:i+sequence_length].values)
    y_test_lstm.append(test_data_scaled.iloc[i+sequence_length].values)
X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)
X_test_lstm_tensor = torch.tensor(X_test_lstm, dtype=torch.float32)
y_test_lstm_tensor = torch.tensor(y_test_lstm, dtype=torch.float32)

with torch.no_grad():
    lstm_predictions = lstm_model(X_test_lstm_tensor)
lstm_test_loss = criterion(lstm_predictions, y_test_lstm_tensor).item()
print(f"LSTM Model Test Loss: {lstm_test_loss:.6f}")

test_data.to_csv("anomaly_detected_fan_coil.csv")
print("Test results saved to anomaly_detected_fan_coil.csv.")

