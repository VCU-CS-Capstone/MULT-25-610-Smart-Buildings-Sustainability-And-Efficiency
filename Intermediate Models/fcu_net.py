"""
Predictive Maintenance Neural Network for Fan Coil Unit

This script trains and tests an autoencoder-based anomaly detection model for predictive maintenance of a fan coil unit in Lab 409. The neural network learns the normal operating behavior of
the system and flags anomalies in testing mode.

Usage:
--------------------------------
### Step 1: Open the Command Line
Before running the script, you need to **navigate to the folder where this file is located**.

#### On Mac/Linux:
1. Open **Terminal** (Press `Command + Space`, type `Terminal`, and press `Enter`).
2. Use the `cd` command to navigate to the folder where the script is located.


#### On Windows:
1. Open **Command Prompt** (Press `Windows + R`, type `cmd`, and press `Enter`).
2. Use the `cd` command to navigate to the folder where the script is located.


### Step 2: Run the Python Script
Once you are in the correct directory, run the script using Python.

#### Start the script (same command for both Mac/Linux and Windows):

python fcu_net.py


### Step 3: Provide User Input
After running the command, the program will prompt you to enter the **mode** and **CSV file path**.

#### Example Inputs:
- **For Training Mode:**
  
  Specify whether to train or test the model (train/test): train
  Provide path to the CSV data file: filepath
  


- **For Testing Mode:**
  ```
  Specify whether to train or test the model (train/test): test
  Provide path to the CSV data file: filepath
  ```

### Code Overview:
1. **`load_data(file_path)`**: Reads the CSV data file.
2. **`preprocess_training_data(df)`**: Cleans and normalizes training data.
3. **`preprocess_testing_data(df)`**: Prepares test data but does not filter anomalies.
4. **`Autoencoder(nn.Module)`**: Defines the neural network structure.
5. **`train_autoencoder(df, scaler)`**: Trains the autoencoder and saves the model.
6. **`test_autoencoder(df)`**: Loads the trained model and detects anomalies.
7. **`main()`**: Handles user input and runs either training or testing mode.

Requirements:
-------------
- Python 3
- PyTorch
- NumPy
- Pandas
- Scikit-learn



"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import os 
import numpy as np
import matplotlib.pyplot as plt
import pickle


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    return df

# Function to preprocess data
def preprocess_training_data(df):
    # Remove missing values
    df = df.dropna().copy()
    
    # Remove objects where Room Temp deviates beyond ±2°F from Setpoint
    df = df[abs(df["Lab 409 Room Temp (°F)"] - df["Lab 409 Room Temp Setpoint (°F)"]) <= 2.5]
    
    # Extract time-based features (but remove them before training)
    df.loc[:, 'Hour'] = df['Date'].dt.hour
    df.loc[:, 'DayOfWeek'] = df['Date'].dt.dayofweek
    df.loc[:, 'Month'] = df['Date'].dt.month
    df.loc[:, 'Week'] = df['Date'].dt.isocalendar().week
    
    # Define the expected training features
    numerical_features = ["Chilled Water Valve (%)", "Heating Coil Valve (%)", "Cooling Coil Valve (%)", 
                          "Outdoor Air Temp (°F)", "Lab 409 Room Temp (°F)", "Discharge Air Temp (°F)"]
    setpoint_features = ["Lab 409 Room Temp Setpoint (°F)"]
    binary_features = ["Reheat Valve Command (-)", "Occupancy Status (-)", "Fan Start/Stop Command (-)", "Heat Cool Mode (-)"]
    
    # Ensure only these 11 features are used
    final_features = numerical_features + setpoint_features + binary_features
    df = df[final_features]

    # Debugging: Print final feature count
    print("Final Training Features:", df.columns.tolist())  # Should print 11 features
    print("Final Number of Features:", df.shape[1])  # Should print 11

    # Normalize numerical features
    scaler = StandardScaler()
    df.loc[:, numerical_features] = scaler.fit_transform(df[numerical_features])

    return df, scaler


def preprocess_testing_data(df):
    df = df.copy()
    
    # Fill missing values with the column mean
    df.dropna()
    
    # Extract time-based features (for reference, but remove them before returning)
    df.loc[:, 'Hour'] = df['Date'].dt.hour
    df.loc[:, 'DayOfWeek'] = df['Date'].dt.dayofweek
    df.loc[:, 'Month'] = df['Date'].dt.month
    df.loc[:, 'Week'] = df['Date'].dt.isocalendar().week
    
    # Define numerical, binary, and setpoint features (same as training)
    numerical_features = ["Chilled Water Valve (%)", "Heating Coil Valve (%)", "Cooling Coil Valve (%)", 
                          "Outdoor Air Temp (°F)", "Lab 409 Room Temp (°F)", "Discharge Air Temp (°F)"]
    setpoint_features = ["Lab 409 Room Temp Setpoint (°F)"]
    binary_features = ["Reheat Valve Command (-)", "Occupancy Status (-)", "Fan Start/Stop Command (-)", "Heat Cool Mode (-)"]
    
    # Ensure the final dataset contains only the same features used in training
    final_features = numerical_features + setpoint_features + binary_features
    
    # Check if all required features exist in df before selecting them
    missing_features = [feature for feature in final_features if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Testing data is missing expected features: {missing_features}")

    df = df[final_features]

    return df


    

# Define Autoencoder Model using PyTorch
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
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Function to train the autoencoder
def train_autoencoder(df, scaler):
    # Save the input feature count
    input_dim = df.shape[1]
    with open("input_dim.txt", "w") as f:
        f.write(str(input_dim))
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    train_losses = []
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(data_tensor)
        loss = criterion(output, data_tensor)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    
    
    torch.save(model, "first_autoencoder_model.pth")
    torch.save(scaler, "first_scaler.pkl")


def test_autoencoder(df, model_path="autoencoder_model.pth", scaler_path="scaler.pkl", version_name="Model Version X", save_path=None):
    """
    Loads the trained autoencoder model and scaler, tests on the provided DataFrame, and plots reconstruction loss.

    Parameters:
    - df: DataFrame, the testing data
    - model_path: string, path to the saved model
    - scaler_path: string, path to the saved scaler
    - version_name: string, label for the plot
    - save_path: optional string, path to save the plot
    """
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Preprocess the testing data
    df_scaled = scaler.transform(df)

    # Convert to PyTorch tensor
    data_tensor = torch.tensor(df_scaled, dtype=torch.float32)

    # Load the model
    input_dim = data_tensor.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict (reconstruct)
    with torch.no_grad():
        reconstructions = model(data_tensor)

    # Compute reconstruction loss per sequence (MSE per sample)
    mse = torch.mean((data_tensor - reconstructions) ** 2, dim=1).numpy()

    # Plot reconstruction loss
    plot_reconstruction_loss(df_scaled, model, scaler, version_name, save_path)

    # Detect anomalies based on a threshold (e.g., 95th percentile)
    threshold = np.percentile(mse, 95)
    anomalies = df[mse > threshold]

    print("Detected Anomalies:")
    print(anomalies)




def plot_reconstruction_loss(X_test, model, scaler, version_name="Model Version X", save_path=None):
    """
    Plots reconstruction loss (MSE) per sequence in the test set.

    Parameters:
    - X_test: np.ndarray, shape (samples, features)
    - model: trained autoencoder model
    - scaler: the fitted StandardScaler used during training
    - version_name: string, label for the plot
    - save_path: optional string, path to save the plot
    """
    # Scale the test data
    X_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predict (reconstruct)
    model.eval()
    with torch.no_grad():
        X_pred = model(X_tensor)

    # Compute reconstruction loss per sequence (MSE per sample)
    losses = np.mean((X_scaled - X_pred.numpy())**2, axis=1)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Reconstruction Loss", color='steelblue')
    plt.title(f"Reconstruction Loss per Sequence - {version_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    while True:
        mode = input("Specify whether to train or test the model (train/test) or type 'quit' to exit: ").strip().lower()
        if mode == 'quit':
            print("Exiting program...")
            return
        if mode in ["train", "test"]:
            break
        print("Invalid mode. Please enter 'train' or 'test', or type 'quit' to exit.")

    while True:
        file_path = input("Provide path to the CSV data file (or type 'quit' to exit): ").strip()
        if file_path.lower() == 'quit':
            print("Exiting program...")
            return
        if os.path.isfile(file_path):
            break
        print("Invalid file path. Please enter a valid CSV file path or type 'quit' to exit.")

    df = load_data(file_path)

    if mode == "train":
        df, scaler = preprocess_training_data(df)
        train_autoencoder(df, scaler)
    else:
        version_name = input("Enter the model version name for the plot: ").strip()
        save_path = input("Enter the path to save the plot (leave blank to display without saving): ").strip()
        save_path = save_path if save_path else None
        df = preprocess_testing_data(df)
        test_autoencoder(df, version_name=version_name, save_path=save_path)


if __name__ == "__main__":
    main()
