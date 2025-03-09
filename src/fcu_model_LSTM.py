'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import os
import pickle
from sklearn.model_selection import KFold

# Prompt the user to choose between training or testing mode
mode = input("Enter mode (train/test): ").strip().lower()

if mode == "train":
    train_filepath = input("Enter filepath of training data: ")
    model_save_path = input("Enter filepath to save trained model: ")
elif mode == "test":
    model_load_path = input("Enter filepath of saved model: ")
    test_filepaths = input("Enter filepath(s) of testing dataset(s), comma separated: ").split(',')
    predictions_output_path = input("Enter filepath to save predictions CSV: ")
else:
    raise ValueError("Invalid mode. Please enter 'train' or 'test'.")

# Function to load dataset
def load_data(filepath, chunk_size=10000):
    df_chunks = pd.read_csv(filepath, parse_dates=['Date'], chunksize=chunk_size)
    df = pd.concat(df_chunks, ignore_index=True)  
    df.set_index('Date', inplace=True)
    return df

# Function to sample half the dataset with equal representation from each month
def sample_data(df):
    df['Month'] = df.index.month  # Extract month from the index
    sampled_df = df.groupby('Month').apply(lambda x: x.sample(frac=0.5, random_state=42), include_groups=False).reset_index(drop=True)
    return sampled_df

# Preprocessing function
def preprocess_data(df, scaler=None, fit_scaler=False):
    feature_cols = ["Chilled Water Valve (%)", "Heating Coil Valve (%)", "Cooling Coil Valve (%)", 
                    "Outdoor Air Temp (°F)", "Lab 409 Room Temp (°F)", "Discharge Air Temp (°F)", 
                    "Lab 409 Room Temp Deviation From Setpoint (°F)"]
    binary_cols = ["Occupancy Status (-)", "Fan Start/Stop Command (-)", "Heat Cool Mode (-)"]
    
    if fit_scaler:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour of the Day'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour of the Day'] / 24)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day of the Week'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day of the Week'] / 7)
    df['Season_sin'] = np.sin(2 * np.pi * df['Season'].factorize()[0] / 4)
    df['Season_cos'] = np.cos(2 * np.pi * df['Season'].factorize()[0] / 4)
    
    selected_features = feature_cols + binary_cols + ['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos', 'Season_sin', 'Season_cos']
    return df[selected_features].astype(np.float16), scaler

# DataLoader function
def create_dataloader(data, batch_size=128, sequence_length=1):
    sequences = np.array([data.values[i:i+sequence_length] for i in range(len(data) - sequence_length)])
    tensor_data = torch.tensor(sequences, dtype=torch.float32)  # Now optimized
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# LSTM Autoencoder class
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, sequence_length=1):
        super(LSTMAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.encoder = nn.LSTM(input_dim, 64, batch_first=True)
        self.decoder = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, input_dim)
    
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden.repeat(self.sequence_length, 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(hidden)
        decoded = self.fc(decoded.reshape(-1, 64)).reshape(-1, self.sequence_length, self.fc.out_features)
        return decoded

def train_model_kfold(train_data, model_save_path, scaler, k=5):
    input_dim = train_data.shape[1]
    device = torch.device("cpu")
    best_threshold = None


    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1

    for train_idx, val_idx in kfold.split(train_data):
        print(f"\n Training Fold {fold}/{k}...")

        # Split training and validation data
        train_subset = train_data.iloc[train_idx]
        val_subset = train_data.iloc[val_idx]

        train_loader = create_dataloader(train_subset, sequence_length=1)
        val_loader = create_dataloader(val_subset, sequence_length=1)

        # Initialize a fresh model for each fold
        model = LSTMAutoencoder(input_dim=input_dim, sequence_length=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        for epoch in range(125):  # Train for fewer epochs per fold
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch[0].to(device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Fold {fold}, Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}")

        # Evaluate on Validation Set to Find Best Threshold
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                output = model(batch)
                loss = criterion(output, batch).item()
                val_losses.append(loss)

       # Use a statistical approach to find the anomaly threshold
        mean_loss = np.mean(val_losses)
        std_loss = np.std(val_losses)
        threshold = mean_loss + std_loss  # Set threshold at mean + 2*std

        print(f" Selected threshold for Fold {fold}: {threshold}")
        fold += 1

    best_threshold = threshold

    # Train Final Model on Full Training Data
    train_loader = create_dataloader(train_data, sequence_length=1)
    model = LSTMAutoencoder(input_dim=input_dim, sequence_length=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(250):  # Train fully using the best threshold
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Final Training, Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}")

    # Save final model and threshold
    torch.save({'model_state_dict': model.state_dict(), 'threshold': best_threshold}, model_save_path)
    print(f"Model saved at {model_save_path}")

    # Save the scaler
    scaler_dir = os.path.dirname(model_save_path)
    scaler_save_path = os.path.join(scaler_dir, "scaler.pkl")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved at {scaler_save_path}")

    return model, best_threshold

def test_model(model_load_path, test_filepaths, predictions_output_path):
    print("Testing...")
    device = torch.device("cpu")

    # Load the trained model and threshold
    checkpoint = torch.load(model_load_path)
    model = LSTMAutoencoder(input_dim=16, sequence_length=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    best_threshold = checkpoint['threshold']  # Use the selected threshold

    # Load the scaler
    scaler_dir = os.path.dirname(model_load_path)
    scaler_load_path = os.path.join(scaler_dir, "scaler.pkl")
    if not os.path.exists(scaler_load_path):
        raise ValueError(f"Scaler file not found: {scaler_load_path}")
    
    with open(scaler_load_path, "rb") as f:
        scaler = pickle.load(f)

    all_results = []

    for test_filepath in test_filepaths:
        print(f"Testing on {test_filepath.strip()}...")
        test_df = load_data(test_filepath.strip())

        labels = test_df['Label'].map({'N': 0, 'F': 1}).values.astype(int)
        test_data, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)
        test_loader = create_dataloader(test_data, batch_size=1, sequence_length=1)

        total_loss = 0
        criterion = nn.MSELoss()
        predictions = []
        timestamps = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch[0].to(device)
                output = model(batch)
                loss = criterion(output, batch).item()
                predictions.append(loss)
                total_loss += loss
                timestamps.append(test_df.index[i])

        # Use the tuned threshold
        binary_preds = [1 if p > best_threshold else 0 for p in predictions]

        # Ensure all lists have the same length
        min_length = min(len(labels), len(predictions), len(timestamps))
        labels = labels[:min_length]
        timestamps = timestamps[:min_length]
        binary_preds = binary_preds[:min_length]

        # Compute Metrics
        accuracy = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds, zero_division=1)
        precision = precision_score(labels, binary_preds, zero_division=1)

        # Ensure confusion_matrix() always returns a 2x2 matrix
        cm = confusion_matrix(labels, binary_preds, labels=[0, 1])

        # Force a 2x2 format (handles cases where one class is missing)
        if cm.shape == (1, 1):  # Only one class present
            if labels[0] == 0:  # All normal data
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:  # All anomalous data
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        elif cm.shape == (2, 2):  # Normal 2x2 case
            tn, fp, fn, tp = cm.ravel()
        else:  # Fallback (should never happen)
            tn, fp, fn, tp = 0, 0, 0, 0

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0


        # Print metrics to terminal
        print("\n--- Test Results ---")
        print(f"Test File: {test_filepath.strip()}")
        print(f"Test Loss: {total_loss / len(test_loader):.6f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print("\nConfusion Matrix:")
        print(f"             Predicted Normal    Predicted Fault")
        print(f"Actual Normal      {tn}                 {fp}")
        print(f"Actual Fault       {fn}                 {tp}")

        # Save Confusion Matrix as a Rectangular CSV File
        confusion_matrix_path = predictions_output_path.replace(".csv", "_confusion_matrix.csv")
        confusion_df = pd.DataFrame([
            ["     ",        "Predicted Normal", "Predicted Fault"],
            ["Actual Normal",    tn,                     fp],
            ["Actual Fault",     fn,                     tp]
        ])
        confusion_df.to_csv(confusion_matrix_path, index=False, header=False)
        print(f"Confusion Matrix saved to {confusion_matrix_path}")


        # Save results to CSV (Metrics in SAME FILE)
        results_output_path = predictions_output_path.replace(".csv", "_metrics.csv")

        results_df = pd.DataFrame({
            "Test_File": [test_filepath.strip()],
            "Accuracy": [accuracy],
            "F1-score": [f1],
            "Precision": [precision],
            "Sensitivity (Recall)": [sensitivity],
            "Specificity": [specificity],
            "Threshold Used": [best_threshold]
        })

        results_df.to_csv(results_output_path, index=False)
        print(f"Metrics saved to {results_output_path}")



        df_predictions = pd.DataFrame({
            'Timestamp': timestamps,
            'True_Label': labels,
            'Predicted_Anomaly': binary_preds
        })
        all_results.append(df_predictions)


    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv(predictions_output_path, index=False)
    print(f"All predictions saved to {predictions_output_path}")


# Main logic
if mode == "train":
    train_df = load_data(train_filepath)
    train_df = sample_data(train_df)  # Reduce dataset size while maintaining equal distribution
    train_data, scaler = preprocess_data(train_df, fit_scaler=True)

    train_model_kfold(train_data, model_save_path, scaler, k=5)

elif mode == "test":
    test_model(model_load_path, test_filepaths, predictions_output_path)



'''