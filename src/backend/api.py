import sys
import os
import torch
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the '../' directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

app = Flask(__name__)
CORS(app)

# Define model paths
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"

device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Import the model class
from fcu_model_LSTM import LSTMAutoencoder

# Load model
def load_model():
    input_dim = 16  # Ensure this matches your feature count
    model = LSTMAutoencoder(input_dim=input_dim, sequence_length=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['threshold']

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        from fcu_model_LSTM import preprocess_data  # Import function

        model, best_threshold = load_model()

        # Get data from request
        data = request.json
        print("Received input data:", data)
        
        # Create dataframe from input
        df = pd.DataFrame([data])
        
        # Ensure all required columns exist
        if "Lab 409 Room Temp Deviation From Setpoint (°F)" not in df.columns:
            # Calculate or provide a default value
            df["Lab 409 Room Temp Deviation From Setpoint (°F)"] = 0.0
            
        # Preprocess data
        processed_data, _ = preprocess_data(df, scaler=scaler, fit_scaler=False)
        
        # Convert to tensor
        tensor_data = torch.tensor(processed_data.values, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Run model prediction
        with torch.no_grad():
            output = model(tensor_data)
            loss = torch.nn.functional.mse_loss(output, tensor_data).item()
        
        # Determine fault status
        prediction = 1 if loss > best_threshold else 0
        
        response = {
            "result": "Fault" if prediction == 1 else "Normal",
            "fault_probability": min(loss / (best_threshold * 2), 1.0),  # Normalized score between 0-1
            "mse_loss": loss,
            "threshold": best_threshold
        }
        
        print("Prediction result:", response)
        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# THIS IS THE CORRECT WAY TO START THE SERVER
if __name__ == "__main__":
    print("Starting server on port 5000...")
    app.run(debug=True, port=5000)